import time
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import pandas as pd
import srsly
import torch
from datasets import load_from_disk
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data.serializers import BaseSerializer
from src.model import DSTTask
from src.utilities.logging import get_logger

pd.set_option("display.max_colwidth", None)
import torch

from src.data.utilities import remove_empty_slots

log = get_logger("hydra")
sep_line = f"{'=' * 70}"


class PreviousStateRecorder:
    def __init__(self):
        self.states = {}
        self.predictions = {}

    def record(self, dialogue_id: str, turn_id: int, state: Dict[str, str], prediction: str) -> None:
        self.exists(dialogue_id)
        self.states[dialogue_id][turn_id] = state
        self.predictions[dialogue_id][turn_id] = prediction

    def get_state(self, dialogue_id: str, turn_id: int) -> Dict[str, str]:
        return self.states[dialogue_id].get(turn_id, None)

    def last_turn_id(self, dialogue_id):
        turn_ids = self.states[dialogue_id].keys()
        if len(turn_ids) > 0:
            return max(turn_ids)
        return -1

    def exists(self, dialogue_id):
        if dialogue_id not in self.states:
            self.states[dialogue_id] = {}
            self.predictions[dialogue_id] = {}

    def to_pandas(self):
        states = pd.concat(
            [
                pd.DataFrame(zip(v.keys(), v.values()), columns=["turn_id", "states"]).assign(dialogue_id=k)
                for k, v in self.states.items()
            ]
        )[["dialogue_id", "turn_id", "states"]]

        predictions = pd.concat(
            [
                pd.DataFrame(zip(v.keys(), v.values()), columns=["turn_id", "predictions"]).assign(dialogue_id=k)
                for k, v in self.predictions.items()
            ]
        )[["dialogue_id", "turn_id", "predictions"]]

        out = pd.merge(states, predictions, on=["dialogue_id", "turn_id"], how="inner")

        assert len(out) == len(states) == len(predictions)

        return out


def generate(batch: List[str], generation_kwargs: Dict, task: DSTTask) -> Tuple:
    """Runs generation and keeps track of time.

    NOTE: No truncation is applied because we are using T5 that uses positional encoding
    and can handle longer sequences than the ones seen during training (>512). This allows
    to pass the full history to models, for example.
    """
    input_ids = task.tokenizer(batch, return_tensors="pt", padding=True, truncation=False)["input_ids"].to(task.device)

    start_time = time.perf_counter()
    pred_ids = task.backbone.generate(input_ids, **generation_kwargs)
    end_time = time.perf_counter()

    return task.decode(pred_ids), end_time - start_time, input_ids.shape[0], input_ids.shape[1]


def resolve_version(version: str) -> Path:
    """Create folders to store predictions.

    Creates the prediction folder if does not exist and resolves the version
    (versions are used when you infer the same data with different generation kwargs)
    """
    path = Path("predictions")
    path.mkdir(exist_ok=True, parents=True)

    if version is None:
        version = len(list(path.glob("v*")))

    version_path = path / f"v{version}"
    version_path.mkdir(exist_ok=True, parents=True)

    return version_path


def infer(
    data: Dict,
    task: DSTTask,
    input_serializer: BaseSerializer,
    target_serializer: BaseSerializer,
    schema: List[str],
    batch_size: int,
    use_gold: bool,
    generation_kwargs: Dict,
    padded_state: bool,
    remove_wrong_slots: bool,
) -> pd.DataFrame:

    recorder = PreviousStateRecorder()
    batches = []
    generation_info = []
    null_pad = (
        {k.lower().strip(): input_serializer.state_serializer.null_value for k in schema} if padded_state else None
    )

    dial_ids = list(data.keys())
    pbar = tqdm(total=len(dial_ids), desc="Batches", position=2)
    while len(dial_ids) > 0:

        # select `batch_size` dialogues to create a batch
        # we process the last turn from each dialogue
        cur_dial_ids = dial_ids[:batch_size]

        # build batch across dialogues and process only one turn per dialogue
        batch = []
        for cur_dial_id in cur_dial_ids:
            recorder.exists(cur_dial_id)
            cur_turn_id = recorder.last_turn_id(cur_dial_id) + 1

            # if dialogue is complete (ie, does not have other turns)
            if cur_turn_id >= len(data[cur_dial_id]["turn_id"]):
                dial_ids.remove(cur_dial_id)
                pbar.update(1)
                continue

            # get current dialogue and build the input text dynamically
            # this is needed because at each step we need to input the
            # state from the previous turn
            dial = data[cur_dial_id]

            if use_gold:
                previous_state = dial["previous_states"][cur_turn_id]
            else:
                previous_state = recorder.get_state(cur_dial_id, cur_turn_id - 1) or null_pad

            input_text = input_serializer.serialize(
                sys_utt=dial["sys_utt"][cur_turn_id],
                usr_utt=dial["usr_utt"][cur_turn_id],
                previous_state=previous_state,
                sys_history=dial["sys_history"][cur_turn_id],
                usr_history=dial["usr_history"][cur_turn_id],
                schema=schema,
                all_intents=None,
            )
            batch.append((cur_dial_id, cur_turn_id, input_text, previous_state))

        # do something with batch
        if len(batch) > 0:
            batches.append(batch)

            dialogue_ids, turn_ids, input_texts, prev_states = zip(*batch)

            # generate
            states_str, runtime, b_size, seq_len = generate(
                batch=list(input_texts), generation_kwargs=generation_kwargs, task=task
            )
            generation_info.append((runtime, b_size, seq_len))

            # deserialize and record
            for dialogue_id, turn_id, state_str, prev_state in zip(dialogue_ids, turn_ids, states_str, prev_states):

                _, new_state = target_serializer.deserialize(state_str, prev_state)

                if remove_wrong_slots:
                    new_state = {k: v for k, v in new_state.items() if k in schema}

                # record state
                recorder.record(dialogue_id=dialogue_id, turn_id=turn_id, state=new_state, prediction=state_str)

    preds = recorder.to_pandas()
    batches_created = pd.DataFrame(
        data=[(*i, *info) for batch, info in zip(batches, generation_info) for i in batch],
        columns=["dialogue_id", "turn_id", "input_text", "previous_states", "runtime", "batch_size", "seq_len"],
    )
    pred_df = pd.merge(preds, batches_created, on=["dialogue_id", "turn_id"], how="inner")
    assert len(pred_df) == len(preds) == len(batches_created)

    return pred_df


@hydra.main(version_base=None, config_path="../conf", config_name="infer_sequential")
def main(cfg: DictConfig) -> None:
    """Run inference for all checkpoints and test/validation splits.

    Saves predictions into the following folder structure

    ```
    {the experiment folder path}
    |   ├── predictions
    │   ├── v{version number}
    │   │   ├── metadata.yaml
    │   │   ├── epoch={epoch number}
    │   │   │   ├── test
    │   │   │   │   └── preds.parquet
    │   │   │   └── validation
    │   │   │       └── preds.parquet
    │   │   ├── epoch=...
    │   ├── v{...}
    ```

    NOTE: working directory is automatically set to the path of the experiment
    therefore all paths are relative.
    """

    log.info(f"Use gold: {cfg.use_gold}")

    # create torch device
    device = torch.device(f"cuda:{cfg.device}")

    # load experiment metadata
    experiment_metadata = OmegaConf.load("metadata.yaml")

    # initialize serializers
    metadata = OmegaConf.load(Path(experiment_metadata.dataset_path) / "metadata.yaml")
    input_serializer = instantiate(metadata.input_serializer)
    target_serializer = instantiate(metadata.target_serializer)
    schema = srsly.read_yaml(Path(metadata.processed_data_path) / "schema.yaml")

    # load gold data and remove empty slots and remove empty slots
    # assumes we have "test" and "validation" keys
    dataset_dict = load_from_disk(experiment_metadata.dataset_path)
    log.info(f"Data loaded from {experiment_metadata.dataset_path}")

    # model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(metadata.tokenizer_name, sep_token=metadata.tokenizer_sep_token)

    # setup folder structure to store predictions
    version_path = resolve_version(cfg.version)

    # save metadata
    srsly.write_yaml(
        version_path / "metadata.yaml",
        {
            "generation_kwargs": OmegaConf.to_container(cfg.generation_kwargs, resolve=True),
            "use_gold": cfg.use_gold,
        },
    )

    for split in ("test", "validation"):

        # select data split
        df = dataset_dict[split].to_pandas()
        df["previous_states"] = df["previous_states"].map(remove_empty_slots)
        df["states"] = df["states"].map(remove_empty_slots)

        # prepare data into a dictionary
        data = (
            df
            # .loc[df["dialogue_id"].isin(df["dialogue_id"].unique().tolist()[:10])]  # <- HERE
            .sort_values(["dialogue_id", "turn_id"])
            .groupby("dialogue_id")
            .agg(lambda ex: list(ex))
            .to_dict(orient="index")
        )

        for epoch_path in list(Path("checkpoints").glob("epoch*")):

            log.info(sep_line)
            log.info(f"Processing epoch {epoch_path.name}")

            # setup split path
            prediction_path = version_path / epoch_path.name / split

            # when folder exists skip to the next checkpoint
            if prediction_path.exists():
                log.info(f"{prediction_path} already computed - not processing checkpoint {epoch_path.name}")
                continue

            log.info(f"Creating folder {prediction_path}")
            prediction_path.mkdir(parents=True, exist_ok=True)

            # load checkpoint and create task
            backbone = AutoModelForSeq2SeqLM.from_pretrained(epoch_path)
            log.info(
                f"tokenizer vocab_size: {len(tokenizer)}, Embedding table size: {backbone.resize_token_embeddings()}"
            )
            task = DSTTask(backbone, tokenizer)

            # move model to gpu
            log.info(f"Moving task to device: {device}")
            _ = task.to(device)

            # run inference
            pred_df = infer(
                data=data,
                task=task,
                input_serializer=input_serializer,
                target_serializer=target_serializer,
                schema=schema,
                batch_size=cfg.batch_size,
                use_gold=cfg.use_gold,
                generation_kwargs=cfg.generation_kwargs,
                padded_state=metadata.get("padded_state", False),
                remove_wrong_slots=cfg.remove_wrong_slots,
            )

            print(pred_df)

            # save predictions
            pred_df.to_parquet(prediction_path / "preds.parquet", index=False)


if __name__ == "__main__":
    main()
