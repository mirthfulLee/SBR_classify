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
from allennlp.data.batch import Batch

from torch import nn
from torch.nn import Dropout, PairwiseDistance, CosineSimilarity
import torch.nn.functional as F
from torch.autograd import Variable


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
        self._upper_level_embeddings = None

    def update_upper_level_embeddings(self, upper_level_embeddings):
        # upper_level_embeddings size: (upper_node_num, embedding_dim)
        self._upper_level_embeddings = upper_level_embeddings

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

# TODO: top k inference!!
class PathFractionMetric(Metric):
    def __init__(self, level_num) -> None:
        super().__init__()
        self._level_num = level_num
        self.total_num = 0
        self.matched_num = 0
    
    def __call__(self, predictions, gold_labels, mask: Optional[torch.BoolTensor]):
        for l in range(self._level_num):
            pred = np.argmax(predictions[l], axis=1)
            self.matched_num += np.sum(pred == gold_labels[l])
            self.total_num += predictions[l].shape[0]
    
    def get_metric(self, reset: bool):
        result = {"path_fraction": self.matched_num / self.total_num}
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

        self._dropout = Dropout(dropout)
        self._level_num = level_num
        self._PTM = PTM
        self._cwe_info = json.load(open(cwe_info_file, "r"))
        # get CWE num of each level
        self._num_class = {i : 0 for i in range(level_num)}
        for cwe_id, cwe in self._cwe_info:
            if cwe["Depth"] < level_num: self._num_class[cwe["Depth"]] += 1
        self._num_class[0] = 2 # "neg" and "SBR"
        for i in range(1, level_num): self._num_class[i] += 1 # add "neg"
        for l in range(level_num): 
            logger.info(f"level_{l} has {self._num_class[l]} classes")
        
        self._text_field_embedder = text_field_embedder
        
        self._embedding_dim = self._text_field_embedder.get_output_dim()
        self._root_classifier = nn.Sequential(
            FeedForward(self._embedding_dim, 1, self._embedding_dim, torch.nn.ReLU(), self._dropout), 
            nn.Linear(self._embedding_dim, self._node_num, bias=False),  
        )
        self._pooler = token_level_avg
        self._bilstm = nn.ModuleList([
            nn.LSTM(
                input_size = self._embedding_dim, 
                hidden_size = self._embedding_dim // 2, 
                num_levels = 1,
                batch_first = True, 
                dropout = dropout,
                bidirectional = 2
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
                "accuracy": CategoricalAccuracy(),
                "f1-score": FBetaMeasure(beta=1.0, average=None, labels=range(self._num_class[l])),  # return list[float]
            })
        # TODO: init path fraction metric
        self._path_fraction_metric = PathFractionMetric()
        self._loss = CustomCrossEntropyLoss()
        self._cwe_instance = list(list() for _ in range(self._level_num))
        initializer(self)

    def get_level_info(self, cwe_instances: List[Instance]):
        for ins in cwe_instances:
            if ins["meta"]["Depth"] < self._level_num:
                if ins["meta"]["CWE_ID"] == "neg": 
                    # make sure neg at 0-idx
                    self._cwe_instance[ins["meta"]["Depth"]].insert(0, ins)
                else: 
                    self._cwe_instance[ins["meta"]["Depth"]].append(ins)
        # sort to match with idx
        for l in range(self._level_num):
            list.sort(self._cwe_instance[level], key=lambda ins: self._token2idx(ins["meta"]["CWE_ID"]))

    def update_embeddings(self):
        # update embeddings before each epoch
        for l in range(self._level_num - 1):
            with torch.no_grad():
                cuda_device = self._get_prediction_device()
                dataset = Batch(self._cwe_instance[l])
                dataset.index_instances(self.vocab)
                model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
                embedding = self._pooler(self._text_field_embedder(model_input["sample"]))
                self._level_classifiers[l].update_upper_level_embeddings(embedding)

    def forward(self,
                process,
                sample: TextFieldTensors,
                path: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        '''
        1. process(constant): FlagField - whether it’s training process or not
        2. sample (batch, seq_len, embedding_dim): TextField - token from IR title & content
        3. path(batch, level_num): ListField - index of class correspond to each layer
        4. metadata()
        '''

        output_dict = dict()
        if metadata:
            output_dict["meta"] = metadata
        level_prob = list()
        level_label = [path[:, l] for l in range(self._level_num)]
        root_emb = self._pooler(self._text_field_embedder(sample))
        level_prob.append(self._root_classifier(root_emb))

        # TODO: add unlabel process
        for l in range(self._level_num - 1):
            if process == "train":
                upper_level_info = level_label[l]
            else:
                # process for "unlabel"
                upper_level_info = np.argmax(level_prob[l], axis=1)
            sample = self._bilstm(sample)
            level_emb = self._pooler(sample)
            level_prob.append(self._level_classifiers(level_emb, 
                                                      F.one_hot(upper_level_info, num_classes=self._num_class[l])))
            
        output_dict["prob"] = level_prob
        output_dict["label"] = level_label
        loss = self._loss(level_prob, level_label)
        output_dict['loss'] = loss
        # compute metric
        for l in range(self._level_num - 1):
            for metric_name, metric in self._metrics.items():
                metric(predictions=level_prob[l], gold_labels=level_label[l])
        metric(predictions=level_prob[l], gold_labels=level_label[l])

        return output_dict
    
    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        # write the experiments during the test
        out2file = list()
        for l in range(self._level_num):
            pred = np.argmax(output_dict["probs"][l], axis=1)
            for i, idx in enumerate(pred):
                out2file.append({f"label_l{l}": self._idx2token[l][output_dict["label"][l][i]],
                                 f"predict_l{l}": self._idx2token[l][idx],
                                 f"prob_l{l}": output_dict["probs"][l][i][output_dict["label"][l][i]]})
                             
        return out2file

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = dict()
        for l in range(self._level_num):
            metrics[f'accuracy_l{l}'] = self._metrics[l]['accuracy'].get_metric(reset)
            precision, recall, fscore = self._metrics[l]['f1-score'].get_metric(reset).values()
            metrics[f'precision_l{l}'] = precision
            metrics[f'recall_l{l}'] = recall
            metrics[f'f1-score_l{l}'] = fscore
        if reset:
            # only calculate this metric when the entire evaluation is done
            metrics["path_fraction"] = self._path_fraction_metric.get_metric(reset)

        return metrics