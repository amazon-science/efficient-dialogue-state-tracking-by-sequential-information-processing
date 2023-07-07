import logging
import os
import re
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import rich
import srsly
import transformers
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from rich.syntax import Syntax
from rich.tree import Tree

"""
Hydra/OmegaConf related stuff
"""
# add resolver to remove "/" from tokenizer and model names that would
# otherwise cause the creation of a subfolder
OmegaConf.register_new_resolver("replace_bar", lambda x: x.replace("/", "__") if x is not None else None)


def get_serializer_name(s: str) -> str:
    # remove suffix
    s = s.split(".")[-1].replace("History", "").replace("State", "").replace("Intent", "")

    # camelcase to snakecase
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def get_dataset_config(history_serializer, state_serializer, intent_serializer) -> str:
    hist_name = get_serializer_name(history_serializer._target_)
    state_name = get_serializer_name(state_serializer._target_)
    out = f"his:{hist_name}-state:{state_name}_full:{str(state_serializer.keep_full_state).lower()}"
    if intent_serializer is not None:
        out += f"-intent:{get_serializer_name(intent_serializer._target_)}"
    return out


OmegaConf.register_new_resolver("get_name", get_serializer_name)
OmegaConf.register_new_resolver("lower", lambda x: str(x).lower().strip())
OmegaConf.register_new_resolver("get_dataset_config", get_dataset_config)


def get_serializers_info(preparator_metadata: Dict) -> Dict:
    md = deepcopy(preparator_metadata)
    input_info = md.pop("input_serializer")
    target_info = md.pop("target_serializer")

    out = {}

    hist = input_info.pop("history_serializer")
    if hist is not None:
        hist["name"] = get_serializer_name(hist.pop("_target_"))
    else:
        hist = {"name": None}
    out["history_serializer"] = hist

    state = input_info.pop("state_serializer")
    state["name"] = get_serializer_name(state.pop("_target_"))
    out["state_serializer"] = state

    all_slots = input_info.pop("all_slots_serializer", None)
    if all_slots is not None:
        all_slots["name"] = get_serializer_name(all_slots.pop("_target_"))
    else:
        all_slots = {"name": None}
    out["all_slots_serializer"] = all_slots

    all_intents = input_info.pop("all_intents_serializer", None)
    if all_intents is not None:
        all_intents["name"] = get_serializer_name(all_intents.pop("_target_"))
    else:
        all_intents = {"name": None}
    out["all_intents_serializer"] = all_intents

    out["input_template"] = input_info.pop("template")
    out["input_sep"] = input_info.pop("sep")

    intent = target_info.pop("intent_serializer", None)
    if intent is not None:
        intent["name"] = get_serializer_name(intent.pop("_target_"))
    else:
        intent = {"name": None}
    out["intent_serializer"] = intent

    out["target_template"] = target_info.pop("template")
    out["target_sep"] = target_info.pop("sep")

    out = {k: OmegaConf.to_container(v) if isinstance(v, DictConfig) else v for k, v in out.items()}

    return out


"""
Training script logging
"""


@rank_zero_only
def save_experiment_configuration(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    srsly.write_yaml("./experiment_config.yaml", cfg_dict)


@rank_zero_only
def save_metadata(metadata: DictConfig, loggers: List) -> None:
    """Saves metadata to disk and possibly to WandB experiment config.

    Args:
        metadata (Dict): Metadata to save to file and possibly to complement on WandB
        loggers (List): List of loggers from the trainer. If WandB is present, metadata
            will be added to the `config`.
    """

    serializers_info = get_serializers_info(OmegaConf.to_container(metadata["preparator"]))
    serializers_info.update({"seed": metadata["seed"], "model_name": metadata["model_name"]})

    # add info to wandb if wandb logger is instantiated
    wandb_logger = get_wandb_logger(loggers)
    if wandb_logger is not None:
        wandb_logger.experiment.config.update(serializers_info)
        # add wandb run_id
        metadata["wandb_run_path"] = wandb_logger.experiment.path
    else:
        metadata["wandb_run_path"] = None

    # add metadata from data config
    OmegaConf.save(metadata, "metadata.yaml")


@rank_zero_only
def save_generation_metadata(metadata: Dict, path: Union[str, Path] = "./", **generation_kwargs) -> None:

    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    # update metadata in WandB
    if metadata["wandb_run_path"] is not None:
        run = wandb.Api().run(metadata["wandb_run_path"])
        for k, v in generation_kwargs.items():
            run.config[k] = v
        run.update()

    srsly.write_yaml(path / "generation_metadata.yaml", generation_kwargs)


@rank_zero_only
def get_wandb_logger(loggers: List) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""
    if loggers is None or len(loggers) < 1:
        return

    for logger in loggers:
        if isinstance(logger, WandbLogger):
            return logger


@rank_zero_only
def get_cwd_info(logger=None) -> None:
    print_fn = logger.info if logger else print
    print_fn(f"Is Hydra changing the working directory: {HydraConfig().get().job.chdir}")
    print_fn(f"Original working directory: {get_original_cwd()}")
    print_fn(f"Current working directory: {os.getcwd()}")
    print_fn(f"Hydra run directory: {HydraConfig().get().runtime.output_dir}")


def remove_non_prediction_config(cfg: DictConfig) -> DictConfig:
    cfg_dict = OmegaConf.to_container(cfg, resolve=False)

    for k in ("optimizer", "scheduler", "task"):
        cfg_dict.pop(k)

    if cfg_dict.get("callbacks", None) is not None:
        to_remove = [c for c in cfg_dict["callbacks"] if c != "prediction_writer"]
        for k in to_remove:
            cfg_dict["callbacks"].pop(k)

    return DictConfig(cfg_dict)


def set_ignore_warnings():
    warnings.simplefilter("ignore")
    # set os environ variable for multiprocesses
    os.environ["PYTHONWARNINGS"] = "ignore"
    transformers.logging.set_verbosity_error()


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""
    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


@rank_zero_only
def print_config(cfg: Dict[str, Any]) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    # resolve and cast to dict
    cfg = OmegaConf.structured(cfg)

    style = "white"
    guide_style = "white"
    tree = Tree("\n[underline][bold]Experiment configurations", style=style, guide_style=guide_style)

    quee = []

    for field in cfg:
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)

        cfg_group = cfg[field]
        if isinstance(cfg_group, DictConfig):
            branch_content = OmegaConf.to_yaml(cfg_group)
        else:
            branch_content = str(cfg_group)

        branch.add(Syntax(code=branch_content, lexer="yaml", tab_size=2, theme="native"))

    rich.print(tree)

    with open("cfg_tree.log", "w") as file:
        rich.print(tree, file=file)
