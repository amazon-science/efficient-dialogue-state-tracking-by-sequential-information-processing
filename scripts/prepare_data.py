import pprint
from pathlib import Path
from typing import Dict, List, Union

import hydra
import pandas as pd
import srsly
import swifter
from datasets import Dataset, DatasetDict, load_from_disk
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.data.serializers import InputSerializer, TargetSerializer
from src.data.utilities import prepare_states_train, remove_empty_slots

pp = pprint.PrettyPrinter(indent=4)


def convert_to_pandas(dataset_dict: DatasetDict) -> pd.DataFrame:
    # cast to pandas
    cols = ["dialogue_id", "turn_id", "sys_utt", "usr_utt", "states", "split"]

    df = pd.concat(
        [dataset_dict[k].to_pandas().assign(split=k).loc[:, cols] for k in dataset_dict], ignore_index=False, axis=0
    ).reset_index(drop=True)

    df["states"] = df["states"].map(remove_empty_slots)

    # remove last dialogue turn in which the user does not speak
    df.loc[df["usr_utt"] != "none"]

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
    schema: List[str],
    input_serializer: InputSerializer,
    target_serializer: TargetSerializer,
    padded_state: bool = False,
    lowercase: bool = False,
) -> pd.DataFrame:

    # utils
    def accumulate_history(row: pd.Series) -> List[List[str]]:
        all_conversation = row.tolist()
        return [all_conversation[:i] for i in range(len(all_conversation))]

    def compute_previous_state(df_: pd.DataFrame, null_pad: Union[Dict, None]) -> pd.Series:
        dd = df_.copy()
        dd["previous_states"] = dd["states"].shift()
        dd["previous_id"] = dd["dialogue_id"].shift()
        dd["previous_states"] = dd["previous_states"].where(dd["dialogue_id"] == dd["previous_id"], null_pad)
        return dd["previous_states"]

    def prepare_states_train(
        state: Union[Dict, None], pad: str, schema: List[str], padded_state: bool, null_pad
    ) -> Union[Dict, None]:
        if state is None:
            return null_pad
        new_state = {k.lower().strip(): sorted(v, key=len)[-1] for k, v in state.items()}

        if padded_state:
            for k in schema:
                if k not in new_state:
                    new_state[k] = pad

        return new_state

    # maybe lowercase
    if lowercase:
        df["sys_utt"] = df["sys_utt"].str.lower()
        df["usr_utt"] = df["usr_utt"].str.lower()

    # schema always lowercase
    schema = [slot.lower().strip() for slot in schema]

    # among all the possible values in the states, take the longest
    # and maybe pad
    null_pad = {k: input_serializer.state_serializer.null_value for k in schema} if padded_state else None
    df["states"] = df["states"].map(
        lambda ex: prepare_states_train(
            ex,
            pad=input_serializer.state_serializer.null_value,
            schema=schema,
            padded_state=padded_state,
            null_pad=null_pad,
        )
    )

    # featurize
    df = df.assign(
        # create previous states
        previous_states=lambda df_: compute_previous_state(df_, null_pad),
        # accumulate system and user histories
        sys_history=lambda df_: df_.groupby("dialogue_id")["sys_utt"].transform(accumulate_history),
        usr_history=lambda df_: df_.groupby("dialogue_id")["usr_utt"].transform(accumulate_history),
        input_text=lambda df_: df_.swifter.apply(
            lambda ex: input_serializer.serialize(
                sys_utt=ex["sys_utt"],
                usr_utt=ex["usr_utt"],
                sys_history=ex["sys_history"],
                usr_history=ex["usr_history"],
                previous_state=ex["previous_states"],
                schema=schema,
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

    # featurize
    df = featurize(
        df,
        schema=schema,
        input_serializer=input_serializer,
        target_serializer=target_serializer,
        padded_state=cfg.padded_state,
        lowercase=cfg.lowercase,
    )
    show_example(df)

    # convert to dataset_dict
    dataset_dict = convert_to_dataset_dict(df)

    # tokenize
    dataset_dict = tokenize(dataset_dict, tokenizer)

    # save data and metadata
    data_path = Path(cfg.output_dir)
    dataset_dict.save_to_disk(data_path)
    metadata = build_metadata(
        tokenizer=tokenizer,
        input_serializer=input_serializer,
        target_serializer=target_serializer,
        data_path=cfg.input_dir,
    )
    srsly.write_yaml(data_path / "metadata.yaml", metadata)


if __name__ == "__main__":
    main()
