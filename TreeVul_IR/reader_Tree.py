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
                 sample_neg: float = 0.05) -> None:
        super().__init__()

        self._token_indexers = token_indexers  # token indexers for text
        self._tokenizer = tokenizer
        self._choice_neg = [True, False]
        self._select_neg = [sample_neg, 1 - sample_neg]  # [True, False]
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

    @overrides
    def _read(self, file_path):
        dataset = self.read_dataset(file_path)
        classes_districution = {"pos": dataset["pos"].shape[0], "neg": dataset["neg"].shape[0]}
        sample_num = classes_districution["pos"] + classes_districution["neg"]
        logger.info(classes_districution)

        if "test_" in file_path:
            # provide test data
            logger.info("Begin predict------")

            for _, sample in dataset["pos"].iterrows():
                # positives come first and then the negatives
                yield self.text_to_instance((sample, sample), type_="unlabel")
            for _, sample in dataset["neg"].iterrows():
                yield self.text_to_instance((sample, sample), type_="unlabel")
            logger.info(f"Predict sample num is {sample_num}")

        elif "validation_" in file_path:
            # provide valdiation data
            logger.info("Begin testing------")
            for _, sample in dataset["pos"].iterrows():
                # positives come first and then the negatives
                yield self.text_to_instance((sample, sample), type_="unlabel")
            for _, sample in dataset["neg"].iterrows():
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
                    yield self.text_to_instance((sample), type_="train")
                    same_num += 1  # matched pairs
                # determine whether make neg sample or not p = select_neg (0.1)
                elif random.choices(self._choice_neg, weights=self._select_neg, k=1)[0]:
                    sample = dataset["neg"].iloc[index - classes_districution["pos"], :]
                    yield self.text_to_instance((sample), type_="train")
                    diff_num += 1

            logger.info(f"Dataset Count: Same : {same_num} / Diff : {diff_num}")

    @overrides
    def text_to_instance(self, ins, type_="train") -> Instance:  # type: ignore
        '''
        1. process(constant): FlagField - whether it’s training process or not
        2. sample (batch, seq_len, embedding_dim): TextField - token from IR title & content
        3. metadata()
        |- path(batch, layer_num, 1): ListField - index of class correspond to each layer
        '''
        # share the code between predictor and trainer, hence the label field is optional
        fields: Dict[str, Field] = {}
        fields["process"] = FlagField(type_)
        fields["sample"] = TextField(self._tokenizer.tokenize(ins["description"]), self._token_indexers)
        # get path of instance
        fields['metadata'] = MetadataField({
            "CWE_ID": ins["CWE_ID"]
        })

        return Instance(fields)
    
@DatasetReader.register("embedding_reader")
class EmbeddingReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 level_num:int = 3,
                 cwe_info_file: str = "CWE_info.json") -> None:
        super().__init__()
        self._token_indexers = token_indexers  # token indexers for text
        self._tokenizer = tokenizer

        # TODO: build tree
        # 1. cwe_path: dict (cwe_id => path)
        # 2. cwe_description: dict (cwe_id => description)
        # 3. level info
        # |- level_node: list of level node (cwe_id)
        # |- num_class: node number
        # |- label2idx: level label => level idx
        # |- father_idx: the father of level node

        cwe_info = json.load(open(cwe_info_file, "r"))
        self._level_num = level_num
        self._cwe_path = {"neg": ["neg" for _ in range(level_num)],
                          "SBR": ["SBR", "neg" for _ in range(1, level_num)]}
        self._cwe_description = {"neg": "",
                                 "SBR": "A security bug report that contains a vulnerability typically includes the following elements: Description of the vulnerability: The report should describe the vulnerability in detail, including how it was discovered, the affected component or feature, and the potential consequences of exploiting the vulnerability. Steps to reproduce: The report should include a detailed description of the steps required to reproduce the vulnerability, including any specific configuration or settings required. Impact assessment: The report should include an assessment of the potential impact of the vulnerability, including the severity of the potential consequences and the likelihood of exploitation. Recommendations: The report should include recommendations for addressing the vulnerability, such as applying a patch or upgrading to a newer version of the software. Contact information: The report should include contact information for the person submitting the report, in case the developer or administrator needs to follow up with additional questions or clarifications."}
        self._level_node = {i : ["neg"] for i in range(level_num)}
        self._level_node[0].append("SBR")
        for cwe_id, cwe in cwe_info:
            if cwe["Depth"] >= level_num: 
                continue
            self._level_node[cwe["Depth"]].append(cwe_id)
            self._cwe_path[cwe_id] = cwe_info[cwe_id]["Path"].insert(0, "SBR")
            while len(self._cwe_path[cwe_id]) < level_num: self._cwe_path[cwe_id].append("neg")
        self._label2idx = dict()
        self._num_class = dict()
        self._node_father = dict()
        # the father idx of neg or the root level is 0
        for l in range(level_num): 
            self._num_class[l] = len(self._level_node[l])
            logger.info(f"level_{self._num_class[l]} has {self._num_class[l]} classes")
            self._label2idx[l] = {
                self._level_node[l][i]: i
                for i in range(self._num_class[l])
            }
            self._node_father[l] = [
                self._label2idx[self._cwe_path[self._level_node[l][i]][l-1]] if l>0 else 0
                for i in range(self._num_class[l])
            ]

    def _read(self, file_path) -> Iterable[Instance]:
        # file_path target the level, like l0, l1, l2...
        # neg node is always the 0-idx
        level = int(file_path[1:])
        for cwe_id in self._level_node[level]:
            node = {
                "sample": self._cwe_info[cwe_id]["Description"],
                "CWE_ID": cwe_id,
                "level": level
            }
            yield self.text_to_instance(node)

    def text_to_instance(self, node) -> Instance:
        fields: Dict[str, Field] = {}
        fields["process"] = FlagField("update")
        fields["sample"] = TextField(self._tokenizer.tokenize(node["sample"]), self._token_indexers)
        fields['metadata'] = MetadataField({"CWE_ID": node["CWE_ID"], "level": node["level"]})
        return Instance(fields)
    