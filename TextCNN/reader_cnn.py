import json
import random
import re
from allennlp import data
import numpy as np
from collections import defaultdict
from itertools import permutations
from typing import Dict, List, Optional
import logging
from datetime import datetime
from copy import deepcopy

from allennlp.data import Field
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
import pandas as pd
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("reader_cnn")
class ReaderCNN(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 sample_neg: float = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        
        super().__init__()

        self._token_indexers = token_indexers
        self._tokenizer = tokenizer
        
        self._choice_neg = [True, False]
        select_neg = sample_neg or 0.1
        self._select_neg = [select_neg, 1 - select_neg]

        self._dataset = dict()

    def read_dataset(self, file_path):
        if self._dataset.get(file_path):
            return self._dataset[file_path]
        samples = pd.read_csv(file_path, header=0)
        samples.fillna("", inplace=True)
        samples["description"] = samples.apply(lambda x: x["Issue_Title"]+". "+x["Issue_Body"], axis=1)
        dataset = {
            "pos": samples.loc[samples["CWE_ID"]!=""],
            "neg": samples.loc[samples["CWE_ID"]==""]
        }
        self._dataset[file_path] = dataset
        return dataset

    def _read(self, file_path):
        dataset = self.read_dataset(file_path)

        classes_districution = {"pos": dataset["pos"].shape[0], "neg": dataset["neg"].shape[0]}
        logger.info(classes_districution)

        if "test_" in file_path or "validation_" in file_path:
            # provide test data
            for _, sample in dataset["pos"].iterrows():
                # positives come first and then the negatives
                yield self.text_to_instance(sample, type_="unlabel")
            for _, sample in dataset["neg"].iterrows():
                yield self.text_to_instance(sample, type_="unlabel")
        else:
            # must shuffle for train
            neg_num = 0
            for _, sample in dataset["pos"].iterrows():
                yield self.text_to_instance(sample, type_="train")
            for _, sample in dataset["neg"].iterrows():
                if random.choices(self._choice_neg, weights=self._select_neg, k=1)[0]:
                    yield self.text_to_instance(sample, type_="train")
                    neg_num += 1

            logger.info("Dataset Count: POS : {} / NEG : {}".format(classes_districution["pos"], neg_num))
                
    def text_to_instance(self, ins, type_="train") -> Instance: 
        fields: Dict[str, Field] = {}

        fields["sample"] = TextField(self._tokenizer.tokenize(ins["description"]), self._token_indexers)
        label = "neg" if ins["CWE_ID"] == "" else "pos"
        fields['label'] = LabelField(label, label_namespace="class_labels")
        
        meta_ins = {"label": label}
        fields['metadata'] = MetadataField({"type": type_, "instance": meta_ins})

        return Instance(fields)