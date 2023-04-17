import logging
from typing import Dict, List, Any

# from allennlp.modules.similarity_functions import BilinearSimilarity
from overrides import overrides

import torch
import numpy as np
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.common import Params
from allennlp.modules import TextFieldEmbedder, FeedForward, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, LstmSeq2SeqEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder, BagOfEmbeddingsEncoder, BertPooler
from allennlp.nn import RegularizerApplicator, InitializerApplicator, Activation
from allennlp.nn.util import get_text_field_mask, get_final_encoder_states, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure, F1Measure, Metric, Auc
from allennlp.training.util import get_batch_size

from torch import nn
from torch.nn import Dropout, PairwiseDistance, CosineSimilarity
import torch.nn.functional as F
from torch.autograd import Variable
from TextCNN.custom_metric import RootF1Metric

import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def pad_sequence2len(tensor, dim, max_len) -> torch.LongTensor:
    shape = tensor.size()
    # print(shape)
    if shape[dim] >= max_len:
        return tensor
    
    pad_shape = list(shape)
    pad_shape[dim] = max_len - shape[dim]
    pad_tensor = torch.zeros(*pad_shape, device=tensor.device, dtype=tensor.dtype)
    new_tensor = torch.cat([tensor, pad_tensor], dim)
    return new_tensor


@Model.register("model_cnn")
class ModelCNN(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 dropout: float = 0.1,
                 label_namespace: str = "class_labels",
                 thres: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)

        self._label_namespace = label_namespace
        self._dropout = Dropout(dropout)
        
        self._idx2token_label = vocab.get_index_to_token_vocabulary(namespace=label_namespace)
        self._idx_pos = vocab.get_token_index(token="pos", namespace=label_namespace)
        self._text_field_embedder = text_field_embedder
        
        # seq2vec module
        embedding_dim = self._text_field_embedder.get_output_dim()
        self._text_cnn = CnnEncoder(embedding_dim, 256, ngram_filter_sizes=(2, 3, 4, 5))
        
        text_embedding_dim = self._text_cnn.get_output_dim()

        self._num_class = self.vocab.get_vocab_size(self._label_namespace)

        self._projector = nn.Sequential(
            FeedForward(text_embedding_dim, 1, [512], torch.nn.ReLU(), dropout),  # text_header
            nn.Linear(512, self._num_class, bias=False),  # classification layer
        )
        self._root_metric = RootF1Metric(thres=thres)
        self._auc_metric = Auc(self._idx_pos)
        
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,
                sample: TextFieldTensors,
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:

        output_dict = dict()
        if metadata:
            output_dict["meta"] = metadata
        # if metadata[0]["type"] == "unlabel": self._root_metric.validation = True

        # pad sequence length to 5（CNN filter size）
        sample["tokens"]["tokens"] = pad_sequence2len(tensor=sample["tokens"]["tokens"], dim=-1, max_len=5)

        mask = get_text_field_mask(sample, padding_id=0)
        sample = self._text_field_embedder(sample)
        # print(sample.shape)  # token embedding + pos embedding
        sample = self._text_cnn(sample, mask)

        sample = self._projector(sample)

        probs = F.softmax(sample, dim=-1)[:, self._idx_pos]
        loss = self._loss(sample, label)
        output_dict['loss'] = loss
        output_dict["probs"] = probs.tolist()
        self._root_metric(predictions=probs, gold_labels=label)
        self._auc_metric(predictions=probs, gold_labels=label)

        return output_dict
    
    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        # return dict()
        idx = np.argmax(output_dict["probs"], axis=1)
        out2file = list()
        for i, _ in enumerate(idx):
            out2file.append({"label": output_dict["meta"][i]["instance"]["label"],
                             "predict": self._idx2token_label[_],
                             "prob": output_dict["probs"][i][self._idx_pos]})
                             
        return out2file

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._root_metric.get_metric(reset)
        metrics["auc"] = self._auc_metric.get_metric(reset)

        return metrics