import pprint
from pathlib import Path
from typing import Dict, List, Union

import hydra
import numpy as np
import pandas as pd
import srsly
import swifter
from datasets import Dataset, DatasetDict, load_from_disk
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.data.serializers import InputSerializer, TargetSerializer
from src.data.utilities import remove_empty_slots

pp = pprint.PrettyPrinter(indent=4)


def select_values(state: Union[Dict[str, List[str]], None], utterance: str) -> Dict[str, str]:
    if state is None:
        return None

    new_state = {}
    for k, v_list in state.items():

        # check if any value is present
        value = [v for v in v_list if v in utterance]

        # if none is present, try lower-casing
        if len(value) < 1:
            for v in v_list:
                if v.lower().strip() in utterance.lower():
                    value.append(v)

        # if still nothing or there are multiple values, then take the longest
        if len(value) < 1:
            value = sorted(v_list, key=len)[-1]

        elif len(value) > 1:
            value = sorted(value, key=len)[-1]

        else:
            value = value[0]

        new_state[k] = value

    return new_state


def unify_similar_services(states: Union[Dict[str, str], None]) -> Dict[str, str]:
    if states is None:
        return None
    new_states = {}
    for k, v in states.items():
        domain, slot = k.split("-")
        domain = domain.split("_")[0]

        new_states[f"{domain}-{slot}"] = v

    return new_states


def accumulate_history(row):
    all_conversation = row.tolist()
    return [all_conversation[: i + 1] for i in range(len(all_conversation))]


def convert_to_pandas(dataset_dict: DatasetDict) -> pd.DataFrame:
    # cast to pandas
    cols = ["dialogue_id", "turn_id", "sys_utt", "usr_utt", "states", "split"]

    df = pd.concat(
        [dataset_dict[k].to_pandas().assign(split=k).loc[:, cols] for k in dataset_dict], ignore_index=False, axis=0
    )

    df = (
        df.reset_index(drop=True)
        # remove k: None from the state dictionary that is introduced by arrow when it saves to disk
        .assign(states=lambda df_: df_["states"].map(remove_empty_slots))
    )

    l, u = len(df), df["dialogue_id"].nunique()
    print(f"# Dialogues: {u}\n# Turns: {l}")

    return df


def convert_to_dataset_dict(df: pd.DataFrame) -> DatasetDict:
    return DatasetDict(
        {
            split: Dataset.from_pandas(df.loc[df["split"] == split], preserve_index=False)
            for split in df["split"].unique()
        }
    )


def featurize(
    df: pd.DataFrame,
    schema: Dict[str, List[str]],
    input_serializer: InputSerializer,
    target_serializer: TargetSerializer,
    unify: bool,
) -> pd.DataFrame:

    # utils
    def accumulate_history(row: pd.Series) -> List[List[str]]:
        all_conversation = row.tolist()
        return [all_conversation[:i] for i in range(len(all_conversation))]

    def accumulate_history_including_current(row: pd.Series) -> List[List[str]]:
        all_conversation = row.tolist()
        return [all_conversation[: i + 1] for i in range(len(all_conversation))]

    def compute_previous_state(df_):
        df_["previous_states"] = df_["states"].shift()
        df_["previous_id"] = df_["dialogue_id"].shift()
        df_.loc[df_["dialogue_id"] != df_["previous_id"], "previous_states"] = None
        return df_["previous_states"]

    # clean states
    df = df.assign(
        # among all the possible values in the states, take the one present in the dialogue history or the longest
        utterance=lambda df_: df_["sys_utt"] + " | " + df_["usr_utt"],
        utterance_history=lambda df_: (
            df_.groupby(["dialogue_id", "split"])["utterance"]
            .transform(accumulate_history_including_current)
            .str.join(" || ")
        ),
        states=lambda df_: df_.apply(lambda row: select_values(row["states"], row["utterance_history"]), axis=1),
    ).drop(columns=["utterance", "utterance_history"])

    if unify:
        df["states"] = df["states"].map(unify_similar_services)

    df = df.assign(
        # create previous state
        previous_states=lambda df_: compute_previous_state(df_),
        # accumulate system and user histories
        sys_history=lambda df_: df_.groupby(["dialogue_id", "split"])["sys_utt"].transform(accumulate_history),
        usr_history=lambda df_: df_.groupby(["dialogue_id", "split"])["usr_utt"].transform(accumulate_history),
        schema=lambda df_: df_["split"].map(schema),
        # serialize input and outputs
        input_text=lambda df_: df_.swifter.apply(
            lambda ex: input_serializer.serialize(
                sys_utt=ex["sys_utt"],
                usr_utt=ex["usr_utt"],
                sys_history=ex["sys_history"],
                usr_history=ex["usr_history"],
                previous_state=ex["previous_states"],
                schema=ex["schema"],
            ),
            axis=1,
        ),
        target_text=lambda df_: df_.swifter.apply(
            lambda ex: target_serializer.serialize(
                state=ex["states"],
                previous_state=ex["previous_states"],
            ),
            axis=1,
        ),
    )

    print(df["schema"])

    return df


def tokenize(ds: Union[Dataset, DatasetDict], tokenizer: PreTrainedTokenizerBase) -> Union[Dataset, DatasetDict]:
    def convert_to_features(examples: Dict[str, List[str]]) -> Dict[str, Union[List[str], List[int], List[List[int]]]]:

        # prompt inputs
        # input_text = self.format_input_text(examples)
        input_text = examples["input_text"]

        # prompt targets
        # target_text = self.format_target_text(examples)
        target_text = examples["target_text"]

        # tokenizer prompted inputs
        model_inputs = tokenizer(
            input_text,
            add_special_tokens=True,
            max_length=None,
            truncation=False,  # do it dynamically
            padding="do_not_pad",  # do it dynamically
        )

        # tokenizer prompted targets
        targets = tokenizer(
            text_target=target_text,
            add_special_tokens=True,
            max_length=None,
            truncation=False,  # do it dynamically
            padding="do_not_pad",  # do it dynamically
        )
        model_inputs["labels"] = targets["input_ids"]

        return model_inputs

    return ds.map(convert_to_features, batched=True)


def build_metadata(
    tokenizer: PreTrainedTokenizerBase,
    input_serializer: InputSerializer,
    target_serializer: TargetSerializer,
    data_path: Union[str, Path],
) -> Dict:
    return {
        "tokenizer_name": tokenizer.name_or_path,
        "tokenizer_sep_token": tokenizer.sep_token,
        "input_serializer": input_serializer.metadata,
        "target_serializer": target_serializer.metadata,
        "processed_data_path": str(Path(data_path).absolute()),
    }


def show_example(df: pd.DataFrame) -> None:
    LINE = "=" * 70
    # select two dialogues
    ids = df["dialogue_id"].sample(2)
    # take the first 3 turns from each of them
    t = df.loc[(df["dialogue_id"].isin(ids)) & (df["turn_id"] < 3)]

    print("Example of preprocessed data:\n")
    for i in t[["dialogue_id", "turn_id", "input_text", "target_text"]].to_dict(orient="records"):
        pp.pprint(i)
        print(LINE)


def create_splits(df: pd.DataFrame, seed: int) -> DatasetDict:
    # divide ids in train/validation/test
    seed_everything(seed)
    splits = {}
    available_ids = list(df["dialogue_id"].unique())
    for split in ("test", "validation"):
        splits[split] = np.random.choice(available_ids, size=2_000, replace=False)
        available_ids = [i for i in available_ids if i not in splits[split]]
    splits["train"] = available_ids

    # create dataset_dict
    dataset_dict = {}
    for split, ids in splits.items():
        dataset_dict[split] = Dataset.from_pandas(df.loc[df["dialogue_id"].isin(ids)], preserve_index=False)

    dataset_dict = DatasetDict(dataset_dict)

    return dataset_dict


@hydra.main(version_base=None, config_path="../conf/data_conf", config_name="config")
def main(cfg: DictConfig) -> None:

    seed_everything(42)  # no randomness here, but just in case

    # load data
    data_path = Path(cfg.input_dir)
    dataset_dict = load_from_disk(data_path)
    schema = srsly.read_yaml(data_path / "schema.yaml")

    # instantiate serializers
    history_serializer = instantiate(cfg.history_serializer)
    state_serializer = instantiate(cfg.state_serializer)
    schema_serializer = instantiate(cfg.schema_serializer)
    input_serializer = InputSerializer(
        template=cfg.input_template,
        history_serializer=history_serializer,
        schema_serializer=schema_serializer,
        state_serializer=state_serializer,
        sep=cfg.sep_token,
    )
    target_serializer = TargetSerializer(state_serializer=state_serializer, template=cfg.target_template)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer, model_max_length=int(1e30), sep_token=cfg.sep_token)

    # convert to pandas
    df = convert_to_pandas(dataset_dict)

    # remove turns in which the user does not speak
    df = df.loc[df["usr_utt"] != "none"]

    # featurize
    df = featurize(
        df, schema=schema, input_serializer=input_serializer, target_serializer=target_serializer, unify=cfg.unify
    )
    show_example(df)

    # build metadata
    metadata = build_metadata(
        tokenizer=tokenizer,
        input_serializer=input_serializer,
        target_serializer=target_serializer,
        data_path=cfg.input_dir,
    )

    # use training set to create test and validation splits
    if cfg.format == "mwoz":

        # tokenize once before splitting
        ds = Dataset.from_pandas(df.loc[df["split"] == "train"], preserve_index=False)
        ds = tokenize(ds, tokenizer)
        df = ds.to_pandas()

        for seed in range(cfg.n_splits):
            # create splits from training set
            dataset_dict = create_splits(df, seed=seed)

            # save data and metadata
            data_path = Path(cfg.output_dir + f"_train_{seed}")
            dataset_dict.save_to_disk(data_path)
            srsly.write_yaml(data_path / "metadata.yaml", metadata)

        return

    # convert to dataset_dict
    dataset_dict = convert_to_dataset_dict(df)

    # tokenize
    dataset_dict = tokenize(dataset_dict, tokenizer)

    # save data and metadata
    data_path = Path(cfg.output_dir)
    dataset_dict.save_to_disk(data_path)
    srsly.write_yaml(data_path / "metadata.yaml", metadata)


if __name__ == "__main__":
    main()
