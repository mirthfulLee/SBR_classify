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
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure, F1Measure, Metric
from allennlp.training.util import get_batch_size
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.data.batch import Batch

from torch import nn
from torch.nn import Dropout, PairwiseDistance, CosineSimilarity
import torch.nn.functional as F
from torch.autograd import Variable
from TreeVul_IR.reader_Tree import EmbeddingReader

import warnings
import json
from copy import deepcopy

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def token_level_avg(seq):
    return torch.mean(seq, dim=-2, keepdim=False) # (batch, token_num, embedding_dim) => (batch, embedding_dim)


class LevelClassifier(nn.Module):
    '''
    build a classifier response for a specific level:
    the level contains node_num kid-node and its upper level has upper_node_num node.
    '''
    def __init__(self, upper_node_num:int, node_num:int, embedding_dim:int=512, dropout:float=0.1):
        super().__init__()
        self._upper_node_num = upper_node_num
        self._node_num = node_num
        self._classifier = nn.Sequential(
            FeedForward(embedding_dim, 1, embedding_dim, torch.nn.ReLU(), dropout), 
            nn.Linear(embedding_dim, self._node_num, bias=False),  
        )
        # the embedding vector of upper-node
        self._upper_level_embeddings = nn.Parameter(torch.zeros(upper_node_num, embedding_dim),
                                                    requires_grad=False)

    def update_upper_level_embeddings(self, upper_level_embeddings):
        # upper_level_embeddings size: (upper_node_num, embedding_dim)
        self._upper_level_embeddings = nn.Parameter(upper_level_embeddings, requires_grad=False)

    def forward(self, x, upper_level_info):
        '''
        x(batch, embedding_dim): the feature of CLS token or AVG used for classifier
        upper_level_info(batch, upper_node_num): the upper level class of each sample in one-hot format
        '''
        x = x + torch.matmul(upper_level_info, self._upper_level_embeddings) # (batch, upper_node_num) * (upper_node_num, embedding_dim)
        x = self._classifier(x)
        return F.softmax(x, dim=-1)


class CustomCrossEntropyLoss(nn.Module):
    '''
    sum up cross entropy loss of multiple levels
    '''
    def __init__(self, level_num, weight=None):
        super().__init__()
        self.level_num = level_num
        self.weight = weight or [1 for _ in range(level_num)]

    def forward(self, level_prob, level_label):
        loss_sum = 0
        for i in range(self.level_num):
            loss_sum += F.cross_entropy(level_prob[i], level_label[i])
        return loss_sum

class PathFractionMetric(Metric):
    def __init__(self, level_num) -> None:
        super().__init__()
        self._level_num = level_num
        self.total_num: int = 0
        self.matched_num: int = 0
    
    def __call__(self, predictions, gold_labels, mask: Optional[torch.BoolTensor]=None):
        for l in range(self._level_num):
            pred = torch.argmax(predictions[l], dim=1)
            self.matched_num += torch.sum(pred == gold_labels[l]).item()
            self.total_num += predictions[l].shape[0]
    
    def reset(self):
        self.matched_num = 0
        self.total_num = 0
    
    def get_metric(self, reset: bool):
        result = self.matched_num / self.total_num
        if reset:
            self.reset()
        return result

@Model.register("model_tree")
class ModelTree(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 PTM: str = 'bert-base-uncased',
                 dropout: float = 0.1,
                 level_num: int = 3, # the first level is SBR and neg
                 cwe_info_file: str = "CWE_info.json",
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
            FeedForward(self._embedding_dim, 1, self._embedding_dim, torch.nn.ReLU(), self._dropout), 
            nn.Linear(self._embedding_dim, self._num_class[0], bias=False),
            nn.Softmax(dim=1)
        )
        self._pooler = token_level_avg
        self._bilstm = nn.ModuleList([
            nn.LSTM(
                input_size = self._embedding_dim, 
                hidden_size = self._embedding_dim // 2, 
                bias = True,
                num_layers = 1,
                batch_first = True, 
                dropout = dropout,
                bidirectional = True
            )
            for _ in range(1, level_num)
        ])
        self._level_classifiers = nn.ModuleList([
            LevelClassifier(self._num_class[l-1], self._num_class[l], self._embedding_dim, self._dropout)
            for l in range(1, level_num)
        ])
        # init metrics for each level
        self._metrics = list()
        for l in range(self._level_num):
            self._metrics.append({
                # "accuracy": CategoricalAccuracy(),
                "micro": FBetaMeasure(beta=1.0, average="micro", labels=range(self._num_class[l])),  # return list[float]
                "macro": FBetaMeasure(beta=1.0, average="macro", labels=range(self._num_class[l])),  # return list[float]
                "weighted": FBetaMeasure(beta=1.0, average="weighted", labels=range(self._num_class[l])),  # return list[float]
            })
        # init path fraction metric
        self._path_fraction_metric = PathFractionMetric(level_num=level_num)
        self._loss = CustomCrossEntropyLoss(level_num)
        initializer(self)

    def update_embeddings(self):
        # update embeddings before each epoch
        logger.info("updating golden embeddings")
        with torch.no_grad():
            for l in range(self._level_num - 1):
                self.forward_on_instances(self._instance4update[l])
    
    def get_path(self, cwe_id):
        if cwe_id not in self._cwe_path.keys(): cwe_id = "SBR"
        elif cwe_id == "": cwe_id = "neg"
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
        output_dict = dict()
        output_dict["process"] = process
        if metadata:
            output_dict["meta"] = metadata
        embedding = self._text_field_embedder(sample)
        root_emb = self._pooler(embedding)
        if process == "update":
            # update embedding for corresponding level classfier
            l = metadata[0]["level"]
            self._level_classifiers[l].update_upper_level_embeddings(root_emb)
            return output_dict
        level_prob = list()
        level_prob.append(self._root_classifier(root_emb))
        # build level label from CWE_ID
        level_label = list()
        for l in range(self._level_num):
            labels = list()
            for ins in metadata:
                labels.append(self.get_path_idx(ins["CWE_ID"])[l])
            level_label.append(torch.tensor(labels, dtype=torch.int64, device=self._get_prediction_device()))

        # add unlabel process
        for l in range(self._level_num - 1):
            embedding, _  = self._bilstm[l](embedding)
            level_emb = self._pooler(embedding)
            if process == "train":
                upper_level_info = level_label[l]
            else:
                # process for "unlabel"
                upper_level_info = torch.argmax(level_prob[l], dim=1)
            upper_level_info = F.one_hot(upper_level_info, num_classes=self._num_class[l])
            upper_level_info = upper_level_info.to(torch.float).to(level_emb.get_device())
            level_prob.append(self._level_classifiers[l](level_emb, upper_level_info))
            
        loss = self._loss(level_prob, level_label)
        output_dict["probs"] = level_prob
        output_dict["label"] = level_label
        output_dict['loss'] = loss
        # compute metric
        for l in range(self._level_num):
            for metric_name, metric in self._metrics[l].items():
                metric(predictions=level_prob[l], gold_labels=level_label[l])
        self._path_fraction_metric(predictions=level_prob, gold_labels=level_label)

        return output_dict
    
    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        # write the experiments during the test
        # beam search
        if "process" not in ["test", "unlabel"]:
            return output_dict
        out2file = list()
        # pred = np.argmax(output_dict["probs"][l], axis=1)
        for i in len(output_dict["probs"][0].shape[0]):
            # sample i
            obj = dict()
            cwe_id = output_dict["meta"]["CWE_ID"]
            obj["CWE_ID"] = cwe_id
            obj["PATH"] = self.get_path(cwe_id)
            p = dict()
            # record the level prob for further analyse
            for l in range(self._level_num):
                if l == 0:
                    p[l] = output_dict["probs"][l][i]
                else:
                    p[l] = [
                        p[l-1][self._node_father[k]] * output_dict["probs"][l][i][k]
                        for k in range(self._num_class[l])
                    ]
                
                obj[f"l{l}"] = {
                    self._level_node[k]: output_dict["probs"][l][i][k]
                    for k in range(self._num_class[l])
                }
            l = self._level_num - 1
            while l>0 and np.argmax(p[l]) == 0: l = l-1
            obj["PREDICT"] = self._cwe_path[self._level_node[l][np.argmax(p[l])]]
            out2file.append(obj)
        return out2file

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = dict()
        for l in range(self._level_num):
            # metrics[f'accuracy_l{l}'] = self._metrics[l]['accuracy'].get_metric(reset)
            for name, metric in self._metrics[l].items():
                precision, recall, fscore = metric.get_metric(reset).values()
                metrics[f'{name}_precision_l{l}'] = precision
                metrics[f'{name}_recall_l{l}'] = recall
                metrics[f'{name}_f1-score_l{l}'] = fscore
        metrics["path_fraction"] = self._path_fraction_metric.get_metric(reset)
        return metrics