import torch
from typing import Optional
from allennlp.training.metrics import Metric
import numpy as np

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class PathFractionMetric(Metric):
    '''
    path accuracy regardless of "neg" labels and root classify task.
    '''
    def __init__(self, level_num, thres: float=0.5) -> None:
        super().__init__()
        self._level_num = level_num
        self.reset()

    def __call__(self, predictions, gold_labels, mask: Optional[torch.BoolTensor]=None):
        for l in range(1, self._level_num):
            pred = torch.argmax(predictions[l], dim=1)
            self.matched_num += torch.sum((pred == gold_labels[l]) * (gold_labels[l] != 0)).item()
            # self.total_num += predictions[l].shape[0]
            self.total_num += torch.sum(gold_labels[l] != 0).item()
    
    def reset(self):
        self.matched_num = 0
        self.total_num = 0
    
    def get_metric(self, reset: bool):
        result = self.matched_num / max(self.total_num, 1)
        if reset:
            self.reset()
        return result


# add dynamic threshold for root level (binary classification)
class RootF1Metric(Metric):
    def __init__(self, thres) -> None:
        super().__init__()
        self.reset()
        self._thres = thres

    def get_root_metric(self, thres=0.5):
        tp = torch.sum((self.root_pred >= thres) * (self.root_label == 1)).item()
        fp = torch.sum((self.root_pred >= thres) * (self.root_label == 0)).item()
        fn = torch.sum((self.root_pred < thres) * (self.root_label == 1)).item()
        prec = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * prec * recall / max(prec + recall, 1)
        return prec, recall, f1
    
    def get_best_threshold(self, interval=[0.3, 0.8]):
        # the threshold between [0.3, 0.8) with granularity = 0.01 to get best root f1 score
        best_f1 = 0.0
        best_thres = interval[0]
        for thres in np.arange(interval[0], interval[1], 0.01):
            _, _, f1 = self.get_root_metric(thres)
            if f1 > best_f1: best_thres = thres
        logger.info(f"get thres = {thres} for best f1 score")
        self._thres = best_thres

    def reset(self) -> None:
        self.root_pred = None
        self.root_label = None
        self.validation = False
        self._thres = 0.5
    
    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor]=None):
        self.root_pred = predictions[:, 1] if self.root_pred is None else torch.cat((self.root_pred, predictions[:, 1]))
        self.root_label = gold_labels if self.root_label is None else torch.cat((self.root_label, gold_labels))
    
    def get_metric(self, reset: bool):
        metrics = dict()
        metrics["prec"], metrics["recall"],metrics["f1"] = self.get_root_metric(self._thres)
        if reset:
            if self.validation:
                self.get_best_threshold()
                metrics["prec"], metrics["recall"],metrics["f1"] = self.get_root_metric(self._thres)
                metrics["thres"] = self._thres
            self.reset()
        return metrics