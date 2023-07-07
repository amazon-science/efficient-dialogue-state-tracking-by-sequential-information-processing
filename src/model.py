from typing import Any, Dict, List, Optional, Type, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.utilities.registries import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY, _Registry

MODEL_REGISTRY = _Registry()


def register_model(model_class: Type) -> Type:
    name = model_class.__name__.replace("Task", "").lower().strip()
    MODEL_REGISTRY[name] = model_class
    return model_class


class BaseSeq2SeqTask(pl.LightningModule):
    """Provides helper functions for optimizers and schedulers and boilerplate code for train/test/validation."""

    optimizers_registry = OPTIMIZER_REGISTRY
    schedulers_registry = SCHEDULER_REGISTRY
    predictions = []

    def __init__(
        self,
        backbone: PreTrainedModel,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self._tokenizer = tokenizer

        # # checks
        # if optimizer_kwargs and optimizer_kwargs.get("name", None) is None or optimizer_kwargs.get("lr", None) is None:
        #     raise MisconfigurationException("`optimizer_kwargs` must include at least 'name' and 'lr'")

        # if scheduler_kwargs and scheduler_kwargs is not None and scheduler_kwargs.get("name", None) is None:
        #     raise MisconfigurationException("`scheduler_kwargs` must include at least 'name'")

        # note: instantiation is deferred and the trainer is in charge of it
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.generation_kwargs = generation_kwargs or {}
        self._special_tokens = None  # use in deconding to decide whether to remove all special or only a subset

        self.configure_metrics()
        self.save_hyperparameters(ignore=["backbone", "tokenizer"])

    def on_train_start(self) -> None:
        """Log effective batch size and gradient accumulation info."""

        accumulate_grad_batches = self.trainer.accumulate_grad_batches
        num_gpus = self.trainer.num_devices
        effective_batch_size = accumulate_grad_batches * self.trainer.datamodule.batch_size * num_gpus

        # log the effective batch size
        self.logger.log_hyperparams(
            {
                "effective_batch_size": int(effective_batch_size),
                "accumulate_grad_batches": accumulate_grad_batches,
                "num_gpus": num_gpus,
            }
        )

    def forward(self, **inputs) -> Seq2SeqLMOutput:
        return self.backbone(**inputs)

    """
    Constructor-related methods
    """

    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        if (
            self._tokenizer is None
            and hasattr(self, "trainer")  # noqa: W503
            and hasattr(self.trainer, "datamodule")  # noqa: W503
            and hasattr(self.trainer.datamodule, "tokenizer")  # noqa: W503
        ):
            self._tokenizer = self.trainer.datamodule.tokenizer
        return self._tokenizer

    """
    Optimization-related methods
    """

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        return self.trainer.estimated_stepping_batches

    def configure_optimizers(self) -> Dict[str, Any]:
        """Handled optimizer and scheduler configuration."""

        # collect optimizer kwargs
        name, no_decay, weight_decay = (
            self.optimizer_kwargs["name"],
            self.optimizer_kwargs.get("no_decay", None),
            self.optimizer_kwargs.get("weight_decay", None),
        )
        optimizer_kwargs = {
            k: v for k, v in self.optimizer_kwargs.items() if k not in ("name", "no_decay", "weight_decay")
        }

        # filter parameters to optimize
        if no_decay is not None and (weight_decay is not None and weight_decay > 0.0):
            params = [
                {
                    "params": [
                        p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            params = filter(lambda p: p.requires_grad, self.parameters())

        # instantiate optimizer
        optimizer = self.optimizers_registry.get(name)(params, **optimizer_kwargs)

        # prepare `out` dict
        out = {"optimizer": optimizer}

        if self.scheduler_kwargs:

            # pop `name` from scheduler_kwargs
            scheduler_kwargs = {k: v for k, v in self.scheduler_kwargs.items() if k not in ("name")}
            scheduler_name = scheduler_kwargs["name"]

            # resolve num_warmup_steps
            num_warmup_steps = scheduler_kwargs.get("num_warmup_steps", None)
            if num_warmup_steps is not None:
                scheduler_kwargs["num_warmup_steps"] = self._compute_warmup(num_warmup_steps)
                rank_zero_info(f"Inferring number of warmup steps: {scheduler_kwargs['num_warmup_steps']}.")

            # resolve num_training_steps
            num_training_steps = scheduler_kwargs.get("num_training_steps", None)
            if num_training_steps is not None and num_training_steps < 0:
                scheduler_kwargs["num_training_steps"] = self.num_training_steps
                rank_zero_info(f"Inferring number of training steps: {scheduler_kwargs['num_training_steps']}.")

            # instantiate scheduler and add to `out`
            scheduler = self.schedulers_registry.get(scheduler_name)(optimizer, **scheduler_kwargs)

            # add scheduler to `out`
            out["lr_scheduler"] = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return out

    def _compute_warmup(self, num_warmup_steps: Union[int, float]) -> int:
        """Resolves type of `num_warmup_steps`."""
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= self.num_training_steps
            rank_zero_info(f"Inferring number of warmup steps from ratio, set to {num_warmup_steps}")
        return num_warmup_steps

    """
    Main methods
    """

    def training_step(self, batch: Any, batch_idx: int = 0) -> Dict[str, Any]:
        return self.common_step(
            batch,
            stage="train",
            log_on_step=True,
            log_on_epoch=False,
            log_prog_bar=False,
            batch_idx=batch_idx,
        )

    def validation_step(self, batch: Any, batch_idx: int = 0, dataloader_idx: Optional[int] = None) -> None:
        _ = self.common_step(
            batch,
            stage="validation",
            log_on_step=False,
            log_on_epoch=True,
            log_prog_bar=False,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def test_step(self, batch: Any, batch_idx: int = 0, dataloader_idx: Optional[int] = None) -> None:
        _ = self.common_step(
            batch,
            stage="test",
            log_on_step=False,
            log_on_epoch=True,
            log_prog_bar=False,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def configure_metrics(self) -> None:
        """Override to configure metrics for train/validation/test.

        This is called on fit start to have access to the datamodule, and initialize any data specific metrics.
        This must set `self.train_metrics`, `self.val_metrics`, `self.test_metrics`. Additionally, for Seq2Seq
        tasks this method must set `self.train_generation_metrics`, `self.val_generation_metrics`,
        `self.test_generation_metrics`.
        """
        pass

    def common_step(
        self,
        batch: Any,
        stage: str,
        log_on_step: bool,
        log_on_epoch: bool,
        log_prog_bar: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    """
    Generation utilities
    """

    @property
    def special_tokens(self) -> List[str]:
        if self._special_tokens is None:
            self._special_tokens = [
                v for k, v in self.tokenizer.special_tokens_map.items() if k != "additional_special_tokens"
            ]
        return self._special_tokens

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: Union[bool, str] = "only_special") -> List[str]:
        """Decodes token_ids, strips and removes pad token from the resulting string."""

        skipping_all_special = True if skip_special_tokens in ("all", True) else False
        skipping_only_special = True if skip_special_tokens in ("only_special") else False

        label_str = self.tokenizer.batch_decode(token_ids, skip_special_tokens=skipping_all_special)
        if skipping_only_special:
            out = []
            for text in label_str:
                for token in self.special_tokens:
                    if token != "<sep>":
                        text = text.replace(token, "")
                out.append(text)
            label_str = out

        return [s.strip() for s in label_str]

    def decode_target(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """Manually change pad token for labels.

        Ref: https://github.com/huggingface/transformers/blob/9adff7a0f49f88a6cc718a1d30088988dc78bb6a/examples/pytorch/translation/run_translation.py#L498-L517  # noqa: E501
        """
        new_token_ids = torch.where(token_ids != -100, token_ids, self.tokenizer.pad_token_id)
        return self.decode(new_token_ids, skip_special_tokens)


@register_model
class DSTTask(BaseSeq2SeqTask):
    def common_step(
        self,
        batch: Dict[str, Union[List[str], Tensor]],
        stage: str,
        log_on_step: bool,
        log_on_epoch: bool,
        log_prog_bar: bool,
        **kwargs,
    ) -> Tensor:

        # remove cpu things from batch
        _ = batch.pop("on_cpu")

        # gather metrics regarding batch size and sequence length
        batch_size, source_seq_length = batch["input_ids"].shape
        target_seq_length = batch["labels"].shape[1]

        # compute the loss
        loss = self(**batch).loss
        logs = {
            "loss": loss,
            "source_seq_length": float(source_seq_length),
            "target_seq_length": float(target_seq_length),
        }

        # log
        self.log_dict(
            {f"{stage}/{k}": v for k, v in logs.items()},
            on_step=log_on_step,
            on_epoch=log_on_epoch,
            prog_bar=log_prog_bar,
            batch_size=float(batch_size),
            sync_dist=True,
        )

        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:

        # remove cpu things from batch
        columns_on_cpu = batch.pop("on_cpu")

        # compute generation metrics
        predicted_tensors = self.backbone.generate(batch["input_ids"], **self.generation_kwargs)
        predicted_text = self.decode(predicted_tensors, skip_special_tokens="only_special")
        predictions = list(zip(columns_on_cpu["qid"], columns_on_cpu["target_text"], predicted_text))

        return predictions
