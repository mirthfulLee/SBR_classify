import json
import random
import re
from allennlp import data
import numpy as np
from collections import defaultdict
from itertools import permutations
from typing import Iterable, Iterator, Optional, Union, TypeVar, Dict, List
import logging
from datetime import datetime
from copy import deepcopy

from allennlp.data import Field
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField, MetadataField, SequenceLabelField, FlagField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
import pandas as pd

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("reader_tree")
class ReaderTree(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sample_neg: float = 0.05,
                 depth: int = 3,
                 cwe_info_file: str = "CWE_info.json") -> None:
        super().__init__()

        self._token_indexers = token_indexers  # token indexers for text
        self._tokenizer = tokenizer
        # self._sample_weights = {'pos': 1, 'neg': 1}
        self._choice_neg = [True, False]
        self._select_neg = [sample_neg, 1 - sample_neg]  # [True, False]
        self._cwe_info = json.load(open(cwe_info_file, "r"))
        self._depth = depth
        self._nsbr_path = ["neg", "neg" for _ in range(1, depth)]
        # transform the path info to corresponding depth:
        for cwe_id in self._cwe_info.keys():
            self._cwe_info[cwe_id]["Path"].insert(0, "SBR")
            while len(self._cwe_info[cwe_id]["Path"]) < depth: self._cwe_info[cwe_id]["Path"].append("neg")
            self._cwe_info[cwe_id]["Path"] = self._cwe_info[cwe_id]["Path"][:depth]
        self._dataset = dict()  # key is the file path

    def read_dataset(self, file_path):
        if self._dataset.get(file_path):
            return self._dataset[file_path]
        samples = pd.read_csv(file_path, header=0)
        samples.fillna("", inplace=True)
        dataset = {
            "pos": samples.loc[samples["CWE_ID"]!=""],
            "neg": samples.loc[samples["CWE_ID"]==""]
        }
        self._dataset[file_path] = dataset

        return self._dataset[file_path]

    def read_cwe_descriptions(self) -> Iterable[Instance]:
        # TODO: yield SBR and neg
        # SBR instance
        fields: Dict[str, Field] = {}
        description_of_SBR = "A security bug report that contains a vulnerability typically includes the following elements: Description of the vulnerability: The report should describe the vulnerability in detail, including how it was discovered, the affected component or feature, and the potential consequences of exploiting the vulnerability. Steps to reproduce: The report should include a detailed description of the steps required to reproduce the vulnerability, including any specific configuration or settings required. Impact assessment: The report should include an assessment of the potential impact of the vulnerability, including the severity of the potential consequences and the likelihood of exploitation. Recommendations: The report should include recommendations for addressing the vulnerability, such as applying a patch or upgrading to a newer version of the software. Contact information: The report should include contact information for the person submitting the report, in case the developer or administrator needs to follow up with additional questions or clarifications."
        fields["process"] = FlagField("update")
        fields["sample"] = TextField(self._tokenizer.tokenize(description_of_SBR), self._token_indexers)
        fields['metadata'] = MetadataField({"Depth": 0, "CWE_ID": "SBR"})
        yield Instance(fields)
        
        # neg instance
        for i in range(6):
            fields: Dict[str, Field] = {}
            fields["process"] = FlagField("update")
            fields["sample"] = TextField(self._tokenizer.tokenize(""), self._token_indexers) # empty context for neg
            fields['metadata'] = MetadataField({"Depth": i, "CWE_ID": "neg"})
            yield Instance(fields)

        # CWE instance
        for cwe_id, cwe in self._cwe_info:
            fields: Dict[str, Field] = {}
            fields["process"] = FlagField("update")
            fields["sample"] = TextField(self._tokenizer.tokenize(cwe["Description"]), self._token_indexers)
            # meta_ins = {"Issue_Url": ins["Issue_Url"], "label": ins[self._target]}
            fields['metadata'] = MetadataField({"Depth": cwe["Depth"], "CWE_ID": cwe_id})
            yield Instance(fields)

    @overrides
    def _read(self, file_path):
        if "CWE" in file_path:
            return self.read_cwe_token()
        dataset = self.read_dataset(file_path)
        classes_districution = {"pos": dataset["pos"].shape[0], "neg": dataset["neg"].shape[0]}
        sample_num = classes_districution["pos"] + classes_districution["neg"]
        logger.info(classes_districution)

        if "test_" in file_path:
            # provide test data
            logger.info("Begin predict------")

            for _, sample in dataset["pos"].iterrows():
                # positives come first and then the negatives
                sample["description"] = self._tokenizer.tokenize(sample["description"])
                yield self.text_to_instance((sample, sample), type_="unlabel")
            for _, sample in dataset["neg"].iterrows():
                sample["description"] = self._tokenizer.tokenize(sample["description"])
                yield self.text_to_instance((sample, sample), type_="unlabel")
            logger.info(f"Predict sample num is {sample_num}")

        elif "validation_" in file_path:
            # provide valdiation data
            logger.info("Begin testing------")
            for _, sample in dataset["pos"].iterrows():
                # positives come first and then the negatives
                sample["description"] = self._tokenizer.tokenize(sample["description"])
                yield self.text_to_instance((sample, sample), type_="unlabel")
            for _, sample in dataset["neg"].iterrows():
                sample["description"] = self._tokenizer.tokenize(sample["description"])
                yield self.text_to_instance((sample, sample), type_="unlabel")
            logger.info(f"Test sample num is {sample_num}")
            
        else:
            # training
            logger.info("loading training examples ...")
            shuffle_index = list(range(sample_num))
            random.shuffle(shuffle_index)
            same_num = 0
            diff_num = 0
            for index in shuffle_index:
                if index < classes_districution["pos"]:
                    # pos sample
                    sample = dataset["pos"].iloc[index, :]
                    sample["description"] = self._tokenizer.tokenize(sample["description"])
                    yield self.text_to_instance((sample), type_="train")
                    same_num += 1  # matched pairs
                # determine whether make neg sample or not p = select_neg (0.1)
                elif random.choices(self._choice_neg, weights=self._select_neg, k=1)[0]:
                    sample = dataset["neg"].iloc[index - classes_districution["pos"], :]
                    sample["description"] = self._tokenizer.tokenize(sample["description"])
                    yield self.text_to_instance((sample), type_="train")
                    diff_num += 1

            logger.info(f"Dataset Count: Same : {same_num} / Diff : {diff_num}")

    @overrides
    def text_to_instance(self, ins, type_="train") -> Instance:  # type: ignore
        '''
        1. process(constant): FlagField - whether it’s training process or not
        2. sample (batch, seq_len, embedding_dim): TextField - token from IR title & content
        3. path(batch, layer_num, 1): ListField - index of class correspond to each layer
        4. metadata()
        '''
        # share the code between predictor and trainer, hence the label field is optional
        fields: Dict[str, Field] = {}
        fields["process"] = FlagField(type_)
        fields["sample"] = TextField(ins["description"], self._token_indexers)

        # get path of instance
        path = self._nsbr_path if ins["CWE_ID"] == "" else self._cwe_info[ins["CWE_ID"]]["Path"]
        fields["path"] = ListField([LabelField(path[l], label_namespace=f"level{l}_labels") 
                                    for l in range(self._depth)])
        
        # meta_ins = {"Issue_Url": ins["Issue_Url"], "label": ins[self._target]}
        fields['metadata'] = MetadataField({"level": self._cwe_info[ins["CWE_ID"]]["Depth"]})

        return Instance(fields)