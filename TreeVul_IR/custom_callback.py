from allennlp.training.callbacks.callback import TrainerCallback
from typing import Dict, Any, List
import numpy as np
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@TrainerCallback.register("reload_training_data")
class ReloadDataCallback(TrainerCallback):
    def __init__(self,
                 serialization_dir: str = None) -> None:
        super().__init__(serialization_dir)

    def on_epoch(self, trainer: "GradientDescentTrainer", metrics: Dict[str, Any], epoch: int, is_primary: bool, **kwargs) -> None:
        # data_loader will re-read the dataset before next epoch (all the pos samples and re-sampled negative samples)
        trainer.data_loader._instances = None
        return


@TrainerCallback.register("dynamic_loss_weight")
class DynamicLossWeightCallback(TrainerCallback):
    def __init__(self,
                 epoch_weight_map: Dict[int, List] = None,
                 serialization_dir: str = None) -> None:
        super().__init__(serialization_dir)
        self.epoch_weight_map = epoch_weight_map or Dict[int, Any]
    
    def on_start(self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs) -> None:
        # check the weight size
        self.trainer = trainer
        model = trainer.model
        for weight in self.epoch_weight_map.values():
            if len(weight) != model._level_num:
                raise ValueError("the loss weight must match the level_num of the model")

    def on_epoch(self, trainer: "GradientDescentTrainer", metrics: Dict[str, Any], epoch: int, is_primary: bool, **kwargs) -> None:
        # data_loader will re-read the dataset before next epoch (all the pos samples and re-sampled negative samples)
        if epoch in self.epoch_weight_map.keys():
            weight = self.epoch_weight_map[epoch]
            trainer.model._loss.weight = weight
            logger.info(f"update the loss weight to {weight}")


@TrainerCallback.register("change_teacher_forcing_ratio")
class TeacherForcingCallback(TrainerCallback):
    # adjust the ratio of teacher forcing learning during the training (scheduled sampling)
    def __init__(self, 
                 decay_strategy: str = "linear", 
                 start_learning_ratio: float=1.0, 
                 end_learning_ratio: float = 0, 
                 start_learning_epochs: int = 0, 
                 end_learning_epochs: int = 50, 
                 k=13, 
                 serialization_dir: str = None) -> None:
        super().__init__(serialization_dir)

        if decay_strategy not in ["linear", "inverse_sigmoid"]:
            raise ValueError("decay_strategy must be one of the (linear, inverse_sigmoid)")

        self._decay_strategy = decay_strategy  # choose one strategy for ratio decay
        self._start_learning_ratio = start_learning_ratio  # maximum learning ratio
        self._end_learning_ratio = end_learning_ratio  # minimum learning ratio
        self._start_learning_epochs = start_learning_epochs  # keep start_learning_ratio in the epochs < start_learning_epochs
        self._end_learning_epochs = end_learning_epochs  # keep end_learning_ratio in epochs > end_learning_epochs

        self._k = k  # for inverse sigmoid decay

        # for linear decay
        if decay_strategy == "linear":
            self._k = (start_learning_ratio - end_learning_ratio) / (start_learning_epochs - end_learning_epochs)
    
    def linear_decay(self, i):
        # linear decay for schedule sampling
        # i start from 1
        return self._start_learning_epochs - self._k * (i - self._start_learning_epochs)

    def inverse_sigmoid_decay(self, i):
        # inverse sigmoid decay strategy for schedule sampling
        return self._k / (self._k + np.exp(i / self._k))
    
    def on_start(self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs) -> None:
        self.trainer = trainer

        # set the initial learning ratio
        model = trainer.model
        model._teacher_forcing_ratio = self._start_learning_ratio

    def on_epoch(self, trainer: "GradientDescentTrainer", metrics: Dict[str, Any], epoch: int, is_primary: bool, **kwargs) -> None:
        '''
        called at the last of each epoch, used to set the teacher forcing ratio for next epoch
        the epoch start from 1
        '''
        model = trainer.model
        epoch += 2  # next epoch & epoch start from 0

        if epoch <= self._start_learning_epochs:
            model._teacher_forcing_ratio = self._start_learning_ratio
        elif epoch > self._end_learning_epochs:
            model._teacher_forcing_ratio = self._end_learning_ratio
        else:
            # epoch in (start_learning_epochs, end_learning_epochs]
            decay_method = getattr(self, f"{self._decay_strategy}_decay")
            model._teacher_forcing_ratio = decay_method(epoch)
            
        logger.info(f"change teacher forcing ratio into {model._teacher_forcing_ratio}")
    