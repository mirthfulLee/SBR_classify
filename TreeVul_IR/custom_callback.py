from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from typing import Dict, Any

@TrainerCallback.register("embedding_update_callback")
class CustomValidation(TrainerCallback):
    def __init__(self,
                 serialization_dir: str = None) -> None:
        super().__init__(serialization_dir)
        # for custom validation, this reader is used to read the golden anchor

    def on_epoch(self, trainer: "GradientDescentTrainer", metrics: Dict[str, Any], epoch: int, is_primary: bool, **kwargs) -> None:
        # not in a distributed mode，trainer.model euqals to trainer._pytorch_model
        # update model embeddings in the tree node
        model = trainer.model
        model.update_embeddings()

        # data_loader will re-read the dataset before next epoch (all the pos samples and re-sampled negative samples)
        trainer.data_loader._instances = None
