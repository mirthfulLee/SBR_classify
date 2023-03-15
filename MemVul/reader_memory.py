from dataclasses import field, fields
from enum import Flag
import json
import pandas as pd
from json import encoder
import random
import re
from allennlp import data
from allennlp.data.fields.text_field import TextFieldTensors
import numpy as np
from collections import defaultdict
from itertools import permutations
from typing import Dict, Iterable, List, Optional, Text
import logging
from datetime import datetime
from copy import deepcopy

from allennlp.data import Field
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    LabelField,
    TextField,
    ListField,
    MetadataField,
    SequenceLabelField,
)
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import (
    TokenIndexer,
    SingleIdTokenIndexer,
    PretrainedTransformerIndexer,
)
from transformers.utils.dummy_pt_objects import ElectraForMaskedLM

from .util import replace_tokens_simple

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("reader_memory")
class ReaderMemory(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        same_diff_ratio: Dict[str, int] = None,
        target: str = "Security_Issue_Full",
        anchor_path: str = "CWE_anchor_golden_project.json",
        cve_path: str = "CVE_dict.json",
        sample_neg: float = None,
        train_iter: int = None,
        token_indexers: Dict[str, TokenIndexer] = None,
    ) -> None:
        # super().__init__(cache_directory=cache_directory)
        super().__init__()

        self._token_indexers = token_indexers
        self._tokenizer = tokenizer
        self._same_diff_ratio = same_diff_ratio or {"diff": 6, "same": 2}
        self._choice_neg = [True, False]
        select_neg = sample_neg or 0.1
        self._train_iter = train_iter or 1
        self._select_neg = [select_neg, 1 - select_neg]
        self._target = target

        if sample_neg is None:
            # when used in the callbacks for custom validation (loading golden anchors)
            return

        # get the CWE ID from the correspoding CVE record
        # TODO: drop unused columns?
        # self._cve_info = json.load(open("data/CVE_dict.json", "r"))  # dict
        self._cve_info = pd.read_csv(cve_path, header=0).groupby("CWE_ID")  # dict
        # get the anchors
        self._anchor = json.load(
            open(anchor_path, "r")
        )  # used for constructing pairs during training
        for k, v in self._anchor.items():
            self._anchor[k] = self._tokenizer.tokenize(v)

        self._dataset = dict()

    def read_dataset(self, file_path):
        if "golden" in file_path:
            # for anchors in the external memory
            dataset = dict()
            anchors = json.load(open(file_path, "r", encoding="utf-8"))  # dict
            for cwe_id, description in anchors.items():
                dataset[cwe_id] = {
                    "CWE_ID": cwe_id,
                    "description": self._tokenizer.tokenize(description),
                }
            return dataset

        # FIXME: necessary to store the file data? (only consider MemVul)
        # No, this read_dataset() function only run once
        if self._dataset.get(file_path):
            return self._dataset[file_path]
        samples = pd.read_csv(file_path, header=0)
        samples.fillna("", inplace=True)
        samples["description"] = samples.apply(lambda x: self._tokenizer.tokenize(x["Issue_Title"]+". "+x["Issue_Body"]), axis=1)
        samples["CVE_Description"] = samples.apply(lambda x: self._tokenizer.tokenize(x["CVE_Description"]), axis=1)
        dataset = {
            "pos": samples.loc[samples["CWE_ID"]!=""],
            "neg": samples.loc[samples["CWE_ID"]==""]
        }
        self._dataset[file_path] = dataset
        return dataset

    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        # dataset key: CWE-ID or neg, value: samples combined with CVE_Description
        dataset = self.read_dataset(file_path)
        if "golden_" not in file_path:
            classes_districution = {"pos": dataset["pos"].shape[0], "neg": dataset["neg"].shape[0]}
            sample_num = classes_districution["pos"] + classes_districution["neg"]
            logger.info(classes_districution)

        same_num = 0
        diff_num = 0

        if "golden_" in file_path:
            # path may accidentally contain the keywords, hence adding the userline
            # provide golden instances
            logger.info("Begin loading golden instances------")
            for sample in dataset.values():
                yield self.text_to_instance((sample, sample), type_="golden")
            logger.info(f"Num of golden instances is {len(dataset)}")

        elif "test_" in file_path:
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
            # must shuffle for train
            shuffle_index = list(range(sample_num))
            random.shuffle(shuffle_index)
            anchor_classes = list(self._anchor.keys())
            cwe_index_dict = dataset["pos"].groupby("CWE_ID").groups

            same_per_sample = self._same_diff_ratio["same"]  # number of matched pairs (CIR)
            diff_per_sample = self._same_diff_ratio["diff"]  # number of mismatched pairs (NCIR)

            for index in shuffle_index:
                if index < classes_districution["pos"]:
                    # pos sample
                    sample = dataset["pos"].iloc[index, :]
                    yield self.text_to_instance((sample, sample), type_="train")  # always use the corresponding CVE to make pairs
                    for same in random.choices(cwe_index_dict[sample["CWE_ID"]], k=same_per_sample - 1):
                        yield self.text_to_instance(
                            (sample, dataset["pos"].loc[same, :]), type_="train"
                        )  # CVE pair

                    same_num += same_per_sample  # matched pairs
                # determine whether make neg sample or not p = select_neg (0.1)
                elif random.choices(self._choice_neg, weights=self._select_neg, k=1)[0]:
                    sample = dataset["neg"].iloc[index - classes_districution["pos"], :]
                    for diff in random.choices(anchor_classes, k=diff_per_sample):
                        # random sample k anchors from the external memory
                        yield self.text_to_instance(
                            (sample, {"CWE_ID": diff, self._target: "pos"}),
                            type_="train",
                        )

                    diff_num += diff_per_sample

            logger.info(f"Dataset Count: Same : {same_num} / Diff : {diff_num}")

    @overrides
    def text_to_instance(self, p, type_="train") -> Instance:
        fields: Dict[str, Field] = dict()
        ins1, ins2 = p
        # instance:Dict{id, intention, messages} mess:Dict{id, text, time, index, user}

        fields["sample1"] = TextField(ins1["description"], self._token_indexers)
        # ins1_class = ins1[self._target]
        # ins2_class = ins2[self._target]
        ins1_class = "neg" if ins1["CWE_ID"] == "" else "pos"
        ins2_class = "neg" if ins2["CWE_ID"] == "" else "pos"

        if type_ == "train":
            # always true
            if ins2_class == "pos":
                if ins1_class == "neg":
                    # use description of the anchor
                    fields["sample2"] = TextField(
                        self._anchor[ins2["CWE_ID"]], self._token_indexers
                    )
                elif ins1["Issue_Title"] == ins2["Issue_Title"]:
                    # use description of the corresponding CVE
                    fields["sample2"] = TextField(
                        ins2["CVE_Description"],
                        self._token_indexers,
                    )
                elif random.choices([True, False], [0.7, 0.3], k=1)[0]:
                    # use description of the other CVE that belong to the same category
                    fields["sample2"] = TextField(
                        ins2["CVE_Description"],
                        self._token_indexers,
                    )
                elif random.choices([True, False], [0.5, 0.5], k=1)[0]:
                    # use description of the anchor
                    anchor_id = ins2["CWE_ID"]
                    if anchor_id is not None:
                        fields["sample2"] = TextField(
                            self._anchor[anchor_id], self._token_indexers
                        )
                    else:
                        fields["sample2"] = TextField(
                            ins2["description"], self._token_indexers
                        )
                else:
                    # use description of other issue report that belong to the same category
                    fields["sample2"] = TextField(
                        ins2["description"], self._token_indexers
                    )

        if type_ in ["train"]:
            if ins1_class == ins2_class:
                fields["label"] = LabelField("same")
            else:
                fields["label"] = LabelField("diff")
        elif type_ in ["test", "unlabel"]:
            # pos == same (we only use CIR to make matched pairs)
            # neg == diff (we only use NCIR to make mismatched pairs)
            if ins1_class == "pos":
                fields["label"] = LabelField("same")
            else:
                fields["label"] = LabelField("diff")

        meta_ins1 = {"label": ins1_class}
        if type_ in ["train", "test", "unlabel"]:
            if ins1_class == "pos":
                meta_ins1["label"] = ins1["CWE_ID"]
            # meta_ins1["Issue_Url"] = ins1["Issue_Url"]

        fields["metadata"] = MetadataField(
            {"type": type_, "instance": [meta_ins1]}
        )  # only to record information
        return Instance(fields)
