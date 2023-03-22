import torch
import allennlp
from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.data_loaders.data_loader import DataLoader
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from typing import Dict, Any
from reader_Tree import ReaderTree

@TrainerCallback.register("embedding_update_callback")
class CustomValidation(TrainerCallback):
    def __init__(self,
                 cwe_info_file: str,
                 serialization_dir: str = None) -> None:
        super().__init__(serialization_dir)
        # for custom validation, this reader is used to read the golden anchor
        PTM = "bert-base-uncased"
        reader = ReaderTree(tokenizer = PretrainedTransformerTokenizer(PTM, add_special_tokens=True, max_length=512),
                                          token_indexers = {"tokens": PretrainedTransformerIndexer(PTM, namespace="tags")})
        self._anchors = list(reader.read_(cwe_info_file))

    def on_start(self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs):
        model = trainer.model
        model.get_level_info(self._anchors)
        model.update_embeddings()
        
    def on_epoch(self, trainer: "GradientDescentTrainer", metrics: Dict[str, Any], epoch: int, is_primary: bool, **kwargs) -> None:
        # not in a distributed mode，trainer.model euqals to trainer._pytorch_model
        model = trainer.model
        model.update_embeddings()
