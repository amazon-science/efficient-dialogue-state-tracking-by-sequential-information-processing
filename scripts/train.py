from pathlib import Path

import hydra
import srsly
from datasets import load_from_disk
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.callbacks import PredictionWriter
from src.data.datamodule import DataModule
from src.model import MODEL_REGISTRY
from src.utilities.logging import (
    get_cwd_info,
    get_logger,
    save_experiment_configuration,
    save_generation_metadata,
    save_metadata,
)

# collect loggers
log = get_logger("hydra")
sep_line = f"{'=' * 70}"


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs training/validation and potentially testing (and stores the predictions).

    It performs the following steps:

    - Read the experiment configuration

    - Load tokenizer and model backbone (pre-trained model)

    - Load data, configurations used to prepare the data, and create a datamodul[e
    """

    # check config
    OmegaConf.resolve(cfg)  # resolve cfg
    log.info(sep_line)
    get_cwd_info(log)

    # load metadata
    metadata = OmegaConf.load(Path(cfg.dataset_path) / "metadata.yaml")

    # complement config from data metadata
    # if cfg.data.max_target_length < metadata.max_target_length:
    #     raise ValueError("max_target_length is less than those in the data")
    # cfg.data.max_target_length = metadata.max_target_length + 5  # add 5 just to be sure
    # if cfg.data.max_source_length > metadata.max_source_length:
    #     cfg.data.max_source_length = metadata.max_source_length + 5  # add 5 just to be sure
    if cfg.model.name_or_path is None:
        cfg.model.name_or_path = metadata.tokenizer_name
    metadata.update(
        {
            "task": cfg.task,
            "model_name": cfg.model.name_or_path,
            "seed": cfg.seed,
            "dataset_path": str(Path(cfg.dataset_path).absolute()),
            "data": OmegaConf.to_container(cfg.data, resolve=True),
        }
    )

    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    if cfg.trainer.fast_dev_run:
        log.critical("\n\n\t !!! DEBUGGING !!! \n\n")

    # save experiment configuration
    save_experiment_configuration(cfg)

    # starting procedure
    log.info(sep_line)
    seed_everything(cfg.seed, workers=True)

    # load backbone and tokenizer
    log.info(f"Loading backbone={cfg.model.name_or_path} and tokenizer={metadata.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        metadata.tokenizer_name, model_max_length=int(1e30), sep_token=metadata.tokenizer_sep_token
    )
    backbone = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.name_or_path)
    backbone.resize_token_embeddings(new_num_tokens=len(tokenizer))
    log.info(f"tokenizer vocab_size: {len(tokenizer)}, Embedding table size: {backbone.resize_token_embeddings()}")

    # load datamodule
    log.info("Loading dataset_dict")
    dataset_dict = load_from_disk(cfg.dataset_path, keep_in_memory=True)
    log.info(f"Input example:\n{dataset_dict['train'][20]['input_text']}")
    log.info(f"Target example:\n{dataset_dict['train'][20]['target_text']}")

    log.info(
        f"Creating datamodule with max_source_length={cfg.data.max_source_length} and max_target_length={cfg.data.max_target_length}"
    )
    datamodule = DataModule(
        dataset_dict=dataset_dict, tokenizer=tokenizer, **cfg.data, debugging=cfg.trainer.fast_dev_run
    )
    log.info(f"Datamodule hyperparameters\n{datamodule.hparams}")

    # load lightning module
    log.info("Creating lightning module")
    model_cls = MODEL_REGISTRY.get(cfg.task)
    model = model_cls(
        backbone=backbone,
        tokenizer=tokenizer,
        optimizer_kwargs=cfg.optimizer,
        scheduler_kwargs=cfg.scheduler,
    )
    log.info(f"LightningModule hyperparameters\n{model.hparams}")

    # instantiate callbacks, loggers, and the trainer
    log.info("Instantiating callbacks, loggers, and the trainer")
    loggers = instantiate(cfg.loggers) or {}
    callbacks = instantiate(cfg.callbacks) or {}
    trainer = Trainer(
        **cfg.trainer,
        callbacks=list(callbacks.values()),
        logger=list(loggers.values()),
        enable_progress_bar=not cfg.trainer.fast_dev_run,
    )

    # fit model
    log.info("Starting training")
    trainer.fit(model, datamodule=datamodule)

    best_model_path = trainer.checkpoint_callback.best_model_path
    metadata["best_model_path"] = Path(best_model_path).name
    log.info(f"Best model path: {best_model_path}")
    log.info(f"Best model score: {trainer.checkpoint_callback.best_model_score}")

    # save metadata
    log.info("Saving metadata")
    srsly.write_yaml(
        path="metadata.yaml",
        data={
            "model_name": cfg.model.name_or_path,
            "seed": cfg.seed,
            "dataset_path": str(Path(cfg.dataset_path).absolute()),
        },
    )
    # save_metadata(metadata=metadata, loggers=trainer.loggers)

    # # reload checkpoint
    # log.info("Loading best model and recreating LightningModule")
    # backbone = AutoModelForSeq2SeqLM.from_pretrained(best_model_path)
    # model = model_cls(backbone=backbone, tokenizer=tokenizer, generation_kwargs=cfg.model.generation_kwargs)

    # # test
    # log.info("Running model on test dataset to get loss")
    # trainer.test(model=model, datamodule=datamodule)

    # # save generation metadata
    # save_generation_metadata(metadata=metadata, **cfg.model.generation_kwargs)

    # # gather predictions
    # log.info("Running model on test dataset to get predictions")
    # log.info(f"Predicting with the following generation_kwargs:\n{OmegaConf.to_yaml(cfg.model.generation_kwargs)}")
    # callbacks = [PredictionWriter(f"./predictions/{metadata['best_model_path']}")]
    # trainer = Trainer(**cfg.trainer, callbacks=callbacks, enable_progress_bar=not cfg.trainer.fast_dev_run, enable_checkpointing=False, logger=False)
    # trainer.predict(model=model, datamodule=datamodule, return_predictions=False)


if __name__ == "__main__":
    main()
