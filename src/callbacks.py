import os
from pathlib import Path
from typing import Union

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.callbacks import ModelCheckpoint as PLModelCheckpoint


class ModelCheckpoint(PLModelCheckpoint):
    """Save checkpoint using HuggingFace save function."""

    # remove file extension because now we are saving a directory
    FILE_EXTENSION = ""

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        # # remove extension since HuggingFace saves a directory
        # filepath = filepath.strip(self.FILE_EXTENSION)
        trainer.lightning_module.backbone.save_pretrained(filepath)


class PredictionWriter(BasePredictionWriter):
    """Writes predictions to parquet.

    ref: https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.callbacks.BasePredictionWriter.html#pytorch_lightning.callbacks.BasePredictionWriter
    """

    def __init__(self, dirpath: Union[str, Path]) -> None:
        super().__init__(write_interval="epoch")
        self.dirpath = Path(dirpath)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        """Save predictions at the end of the epoch.

        This will create N (num processes) files in `dirpath` each containing
        the predictions of it's respective rank
        """

        # create directory if it does not exist
        self.dirpath.mkdir(exist_ok=True, parents=True)

        # flatten prediction list
        predictions = [i for pred in predictions[0] for i in pred]
        pd.DataFrame(predictions, columns=["qid", "target_text", "predicted_text"]).to_parquet(
            self.dirpath / f"predictions_gpu={trainer.global_rank}.parquet", index=False
        )
