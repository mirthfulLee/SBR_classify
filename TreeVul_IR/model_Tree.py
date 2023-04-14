import logging
from typing import Dict, List, Any, Optional

# from allennlp.modules.similarity_functions import BilinearSimilarity
from overrides import overrides

import torch
import numpy as np
from allennlp.data import Vocabulary, TextFieldTensors, Instance
from allennlp.models import Model
from allennlp.common import Params
from allennlp.modules import TextFieldEmbedder, FeedForward, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, LstmSeq2SeqEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder, BagOfEmbeddingsEncoder, BertPooler
from allennlp.nn import RegularizerApplicator, InitializerApplicator, Activation, util
from allennlp.nn.util import get_text_field_mask, get_final_encoder_states, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure, F1Measure, Metric, Auc
from allennlp.training.util import get_batch_size
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.modules.gated_sum import GatedSum
import random
from TreeVul_IR.custom_metric import PathFractionMetric, RootF1Metric

from torch import nn
import torch.nn.functional as F
from TreeVul_IR.reader_Tree import EmbeddingReader

import warnings
import json
from copy import deepcopy

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



class LevelClassifier(nn.Module):
    '''
    build a classifier response for a specific level:
    the level contains node_num kid-node and its upper level has upper_node_num node.
    '''
    def __init__(self, upper_node_num:int, node_num:int, embedding_dim:int=512, dropout:float=0.1, upper_level_result_awareness:bool=True):
        super().__init__()
        self._upper_node_num = upper_node_num
        self._node_num = node_num
        self._classifier = nn.Sequential(
            FeedForward(embedding_dim, 1, embedding_dim, torch.nn.ReLU(), dropout), 
            nn.Linear(embedding_dim, self._node_num, bias=False),  
        )
        self._upper_level_result_awareness = upper_level_result_awareness
        self._batch_norm = nn.BatchNorm1d(embedding_dim)
        # the embedding vector of upper-node
        self._upper_level_embeddings = nn.Parameter(torch.zeros(upper_node_num, embedding_dim),
                                                    requires_grad=False)
        self._gated_merge = GatedSum(input_dim=embedding_dim, activation=torch.nn.Sigmoid())
        

    def update_upper_level_embeddings(self, upper_level_embeddings):
        # upper_level_embeddings size: (upper_node_num, embedding_dim)
        self._upper_level_embeddings = nn.Parameter(upper_level_embeddings, requires_grad=False)

    def forward(self, x, upper_level_info):
        '''
        x(batch, embedding_dim): the feature of CLS token or AVG used for classifier
        upper_level_info(batch, upper_node_num): the upper level class of each sample in one-hot format
        '''
        # add batch norm to x
        x = self._batch_norm(x)
        # x = x + torch.matmul(upper_level_info, self._upper_level_embeddings) # (batch, upper_node_num) * (upper_node_num, embedding_dim)
        if self._upper_level_result_awareness:
            x = self._gated_merge(input_a = x, 
                                input_b = torch.matmul(upper_level_info, self._upper_level_embeddings))
        x = self._classifier(x)
        return x


class CustomCrossEntropyLoss(nn.Module):
    '''
    sum up cross entropy loss of multiple levels
    '''
    def __init__(self, level_num, weight=None):
        super().__init__()
        self.level_num = level_num
        self.weight = weight or [1/level_num for _ in range(level_num)]

    def forward(self, level_logits, level_label):
        loss_sum = 0
        for l in range(self.level_num):
            loss_sum += self.weight[l] * F.cross_entropy(level_logits[l], level_label[l])
        return loss_sum

@Model.register("model_tree")
class ModelTree(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 PTM: str = 'bert-base-uncased',
                 dropout: float = 0.1,
                 level_num: int = 3, # the first level is SBR and neg
                 cwe_info_file: str = "CWE_info.json",
                 loss_weight: List = None,
                 update_step: int = 64,
                 root_thres: float = 0.5,
                 level_lstm: bool = True, 
                 upper_level_result_awareness: bool = True,
                #  pooling_method: str = "bilstm+avg", # bilstm+avg or CLS
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)
        self._level_num = level_num
        reader = EmbeddingReader(tokenizer = PretrainedTransformerTokenizer(PTM, add_special_tokens=True, max_length=512),
                                 token_indexers = {"tokens": PretrainedTransformerIndexer(PTM, namespace="tags")},
                                 level_num=level_num, cwe_info_file=cwe_info_file)
        self._level_node = reader._level_node
        self._num_class = reader._num_class
        self._cwe_description = reader._cwe_description
        self._cwe_path = reader._cwe_path
        self._label2idx = reader._label2idx
        self._node_father = reader._node_father
        
        self._instance4update = list()
        for l in range(self._level_num):
            self._instance4update.append(list(reader.read(f"l{l}")))

        self._dropout = dropout
        self._PTM = PTM
        self._text_field_embedder = text_field_embedder
        
        self._embedding_dim = self._text_field_embedder.get_output_dim()
        self._root_classifier = nn.Sequential(
            # nn.BatchNorm1d(embedding_dim),
            FeedForward(self._embedding_dim, 1, self._embedding_dim, torch.nn.ReLU(), self._dropout), 
            nn.Linear(self._embedding_dim, self._num_class[0], bias=False),
        )
        self._root_pooler = BertPooler(PTM, requires_grad=True, dropout=dropout)
        self._level_pooler = BagOfEmbeddingsEncoder(embedding_dim=self._embedding_dim, averaged=True) # averaged vec may be too small
        self._level_lstm = level_lstm
        if self._level_lstm:
            self._seq2seq_encoder = nn.ModuleList([
                LstmSeq2SeqEncoder(
                    input_size = self._embedding_dim, 
                    hidden_size = self._embedding_dim // 2, 
                    bias = True,
                    num_layers = 1,
                    dropout = dropout,
                    bidirectional = True)
                for _ in range(1, level_num)
            ])
        self._level_classifiers = nn.ModuleList([
            LevelClassifier(self._num_class[l-1], self._num_class[l], 
                            self._embedding_dim, self._dropout, upper_level_result_awareness=upper_level_result_awareness)
            for l in range(1, level_num)
        ])
        self._root_metric = RootF1Metric(thres=root_thres)
        self._auc_metric = Auc()
        # init metrics for each level
        self._level_f1 = dict()
        for l in range(1, self._level_num):
            self._level_f1[l] = {
                "macro": FBetaMeasure(beta=1.0, average="macro", labels=range(1, self._num_class[l])),
                "weighted": FBetaMeasure(beta=1.0, average="weighted", labels=range(1, self._num_class[l])),
            }
        # init path fraction metric
        self._path_fraction_metric = PathFractionMetric(level_num=level_num)
        self._loss = CustomCrossEntropyLoss(level_num, weight=loss_weight)
        self._update_step = update_step
        # update embeddings when cnt == 0
        self._update_cnt = 0
        self._teacher_forcing_ratio = 1.0
        initializer(self)

    def update_embeddings(self):
        # update embeddings before between training and validation or at the beginning
        logger.info("updating golden embeddings")
        self._update_cnt = self._update_step
        with torch.no_grad():
            for l in range(self._level_num - 1):
                self.forward_on_instances(self._instance4update[l])
    
    def get_path(self, cwe_id):
        if cwe_id == "": cwe_id = "neg"
        elif cwe_id not in self._cwe_path.keys(): cwe_id = "SBR"
        return self._cwe_path[cwe_id]
    
    def get_path_idx(self, cwe_id):
        path = self.get_path(cwe_id)
        idx_path = [self._label2idx[l][path[l]] for l in range(self._level_num)]
        return idx_path

    def forward(self,
                process,
                sample: TextFieldTensors,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        '''
        1. process(constant): FlagField - whether it’s training process or not
        2. sample (batch, seq_len, embedding_dim): TextField - token from IR title & content
        3. metadata()
        '''
        if self._update_cnt == 0 or (process not in ["train", "update"] and self._update_cnt != self._update_step): 
            self.update_embeddings()
        elif process == "train": 
            self._update_cnt -= 1
        output_dict = dict()
        output_dict["process"] = process
        if process == "validation": self._root_metric.validation = True
        if metadata:
            output_dict["meta"] = metadata
        sample_mask = get_text_field_mask(sample)
        
        embedding = self._text_field_embedder(sample)
        root_emb = self._root_pooler(embedding, sample_mask)
        if process == "update":
            # update embedding for corresponding level classfier
            l = metadata[0]["level"]
            self._level_classifiers[l].update_upper_level_embeddings(root_emb)
            return output_dict
        level_logits = list()
        level_logits.append(self._root_classifier(root_emb))
        # build level label from CWE_ID
        level_label = list()
        for l in range(self._level_num):
            labels = list()
            for ins in metadata:
                labels.append(self.get_path_idx(ins["CWE_ID"])[l])
            level_label.append(torch.tensor(labels, dtype=torch.int64, device=self._get_prediction_device()))

        # classify process for level > 0 (use upper level info as previous knowledge)
        for l in range(self._level_num - 1):
            if self._level_lstm:
                embedding = self._seq2seq_encoder[l](embedding, sample_mask)
            level_emb = self._level_pooler(embedding, sample_mask)
            # Curriculum Learning: use teacher forcing (with ratio)
            if process == "train" and random.choices([True, False], weights=[self._teacher_forcing_ratio, 1 - self._teacher_forcing_ratio], k=1)[0]:
                # use ground truth
                upper_level_info = level_label[l]
            else:
                # use upper level predict result
                upper_level_info = torch.argmax(level_logits[l], dim=1)
            upper_level_info = F.one_hot(upper_level_info, num_classes=self._num_class[l]).to(torch.float)
            level_logits.append(self._level_classifiers[l](level_emb, upper_level_info))

            for metric_name, metric in self._level_f1[l+1].items():
                metric(predictions=level_logits[l+1], gold_labels=level_label[l+1], mask=(level_label[l] != 0))
        output_dict["logits"] = level_logits
        if process not in ["test", "predict"]:
            loss = self._loss(level_logits, level_label)
            output_dict["label"] = level_label
            output_dict['loss'] = loss
        # compute metric
        probs = F.softmax(level_logits[0], dim=1)[:, 1]
        self._root_metric(predictions=probs, gold_labels=level_label[0])
        self._auc_metric(predictions=probs, gold_labels=level_label[0])
        self._path_fraction_metric(predictions=level_logits, gold_labels=level_label)

        return output_dict
    
    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        # write the experiments during the test
        # beam search
        if output_dict["process"] not in ["predict"]:
            return output_dict
        out2file = list()
        # pred = np.argmax(output_dict["logits"][l], axis=1)
        for i in len(output_dict["logits"][0].shape[0]):
            # sample i
            obj = dict()
            cwe_id = output_dict["meta"]["CWE_ID"]
            obj["CWE_ID"] = cwe_id
            obj["PATH"] = self.get_path(cwe_id)
            p = dict()
            # record the level prob for further analyse
            for l in range(self._level_num):
                if l == 0:
                    p[l] = output_dict["logits"][l][i]
                else:
                    p[l] = [
                        p[l-1][self._node_father[k]] + output_dict["logits"][l][i][k]
                        for k in range(self._num_class[l])
                    ]
                
                obj[f"l{l}"] = {
                    self._level_node[k]: output_dict["logits"][l][i][k]
                    for k in range(self._num_class[l])
                }
            l = self._level_num - 1
            while l>0 and np.argmax(p[l]) == 0: l = l-1
            obj["PREDICT"] = self._cwe_path[self._level_node[l][np.argmax(p[l])]]
            out2file.append(obj)
        return out2file

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._root_metric.get_metric(reset)
        metrics["auc"] = self._auc_metric.get_metric(reset)
        metrics["path_fraction"] = self._path_fraction_metric.get_metric(reset)
        for l in range(1, self._level_num):
            # metrics[f'accuracy_l{l}'] = self._metrics[l]['accuracy'].get_metric(reset)
            for name, metric in self._level_f1[l].items():
                precision, recall, fscore = metric.get_metric(reset).values()
                metrics[f'{name}_precision_l{l}'] = precision
                metrics[f'{name}_recall_l{l}'] = recall
                metrics[f'{name}_f1-score_l{l}'] = fscore
        return metrics