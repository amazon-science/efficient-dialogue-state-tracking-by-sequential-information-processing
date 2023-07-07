import os
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import torch
from datasets import Dataset, DatasetDict
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from transformers import PreTrainedTokenizerBase, logging

from src.utilities.logging import get_logger

logging.set_verbosity_error()

log = get_logger("hydra")


class DataModule(LightningDataModule):

    train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset

    def __init__(
        self,
        dataset_dict: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        columns_to_keep: Optional[List[str]] = None,
        max_source_length: int = 128,
        max_target_length: int = 128,
        filter_long_sequences: bool = False,
        debugging: bool = False,
        batch_size: Optional[int] = 32,
        eval_batch_size: Optional[int] = 32,
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = True,
        drop_last: Optional[bool] = False,
        persistent_workers: Optional[bool] = False,
        shuffle: Optional[bool] = True,
        seed_dataloader: Optional[int] = 42,
        replacement: bool = False,
    ) -> None:
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.dataset_dict = dataset_dict
        self.tokenizer = tokenizer
        self.columns_to_keep = columns_to_keep or []

        # data collator
        assert max_source_length >= max_target_length, ValueError(
            "For consistency `max_source_length` >= `max_target_length` must be true."
        )
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.debugging = debugging
        self.filter_long_sequences = filter_long_sequences

        # data loading
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.shuffle = shuffle
        self.seed_dataloader = seed_dataloader
        self.replacement = replacement

        self.save_hyperparameters(ignore=["tokenizer", "dataset_dict"])

    def setup(self, stage: Optional[str] = None) -> None:
        for stage in ("train", "validation", "test"):
            if stage in self.dataset_dict:

                dataset = self.dataset_dict[stage].with_format(
                    columns=self.columns_to_keep + ["input_ids", "attention_mask", "labels"]
                )

                if self.filter_long_sequences:
                    log.critical(
                        f"{stage} dataset: Removing instances that do not match the `max_target_length` constraint"
                    )
                    features = dataset.features
                    df = dataset.to_pandas()
                    old_len = len(df)
                    df = df.loc[df["labels"].map(len) <= self.max_target_length]
                    dataset = Dataset.from_pandas(df, features=features, preserve_index=False)
                    log.critical(f"{stage} dataset: Size changed from {old_len} to {len(df)}")

                setattr(self, f"{stage}_dataset", dataset)

    @property
    def label_pad_token_id(self) -> int:
        return -100

    def train_dataloader(self) -> DataLoader:
        # for why using BatchSampler see https://huggingface.co/docs/datasets/use_with_pytorch
        # fix seed of random
        if not self.shuffle:
            sampler = SequentialSampler(self.train_dataset)
        else:
            g = torch.Generator()
            g.manual_seed(self.seed_dataloader)
            sampler = RandomSampler(self.train_dataset, generator=g, replacement=self.replacement)

        batch_sampler = BatchSampler(sampler, batch_size=self.batch_size, drop_last=self.drop_last)

        return DataLoader(
            self.train_dataset,
            collate_fn=self.data_collator,
            sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def _evaluation_dataloader(self, stage: str) -> Union[DataLoader, None]:
        dataset = getattr(self, f"{stage}_dataset", None)

        if dataset is None:
            return

        batch_sampler = BatchSampler(
            SequentialSampler(dataset), batch_size=self.eval_batch_size, drop_last=self.drop_last
        )
        return DataLoader(
            dataset,
            collate_fn=self.data_collator,
            sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return self._evaluation_dataloader("validation")

    def test_dataloader(self) -> DataLoader:
        return self._evaluation_dataloader("test")

    def predict_dataloader(self, stage: str = "test") -> DataLoader:
        return self._evaluation_dataloader(stage)

    def transfer_batch_to_device(
        self, batch: Dict[str, Union[str, Tensor]], device: torch.device, dataloader_idx: int
    ) -> Dict[str, Union[str, Tensor]]:

        # remove string columns that cannot be transfered on gpu
        columns_on_cpu = batch.pop("on_cpu")

        # transfer the rest on gpu
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)

        # add the columns on cpu to the batch
        batch["on_cpu"] = columns_on_cpu

        return batch

    @property
    def data_collator(self) -> Callable:

        return partial(
            collate_fn,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            columns_to_keep=self.columns_to_keep,
            pad_token_id=self.tokenizer.pad_token_id,
            label_pad_token_id=self.label_pad_token_id,
            pad_fn=_pad if not self.debugging else _debug_pad,
        )


"""
Define as globals otherwise pickle complains when running in multi-gpu
"""


def _pad(inputs: List[int], padding_value: float, max_length: int) -> Tensor:
    # truncate -> convert to tensor -> pad
    return pad_sequence(
        [torch.tensor(t[:max_length]) for t in inputs],
        batch_first=True,
        padding_value=padding_value,
    )


def _debug_pad(inputs: List[int], padding_value: float, max_length: int) -> Tensor:
    # pad the first element in the batch to the max_length
    # to causes the batch to pad to maximum length too
    inputs[0] += [0] * (max_length - len(inputs[0]))
    padded_tensor = _pad(inputs, padding_value, max_length)
    print("DEBUGGING -- tensor_shapes:", padded_tensor.shape, flush=True)
    return padded_tensor


def collate_fn(
    batch: List[Dict[str, Union[List[str], Tensor]]],
    max_source_length: int,
    max_target_length: int,
    columns_to_keep: List[str],
    pad_token_id: int,
    label_pad_token_id: int,
    pad_fn: Callable,
) -> Dict[str, Union[List[str], Tensor]]:
    # NOTE: beacuse of the batch_sampler we already obtain dict of lists
    # however the dataloader will try to create a list, so we have to unpack it
    assert len(batch) == 1, "Look at the data collator"
    batch = batch[0]

    # remove string columns that cannot be transfered on gpu
    columns_on_cpu = {col: batch.pop(col) for col in columns_to_keep}

    labels = batch.pop("labels")

    # input_ids and attention_mask to tensor
    # truncate -> convert to tensor -> pad
    batch = {
        k: pad_fn(
            inputs=batch[k],
            padding_value=pad_token_id,
            max_length=max_source_length,
        )
        for k in ("input_ids", "attention_mask")
    }

    # labels to tensor, pad, and batch together
    # truncate -> convert to tensor -> pad
    batch["labels"] = pad_fn(labels, padding_value=label_pad_token_id, max_length=max_target_length)

    # add things that need to remain on cpu
    batch["on_cpu"] = columns_on_cpu

    return batch
