from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import srsly
from datasets import Dataset, DatasetDict
from pandas import DataFrame
from tqdm.auto import tqdm

DOMAINS_TO_IGNORE = ("bus", "police", "hospital")
VALUES_TO_IGNORE = ("not mentioned", "none", "unknown")
SLOTS_TO_IGNORE = ("booked",)
INVALID_DIALOGUES = ("SNG01862.json",)


@dataclass
class Dialogue:
    dialogue_id: List[int]
    turn_id: List[int]
    sys_utt: List[str]
    usr_utt: List[str]
    states: List[Union[Dict[str, str], None]]
    turn_services: List[List[str]]
    split: str


class BaseDataset:
    DOMAINS_TO_IGNORE = []
    VALUES_TO_IGNORE = []
    _data = None
    _dialogues = None
    _ontology = None

    @property
    def data(self) -> Optional[Dict[str, Dict]]:
        return self._data

    @property
    def dialogues(self) -> Optional[List[Dialogue]]:
        return self._dialogues

    @property
    def ontology(self) -> Optional[Dict[str, Dict[str, List[str]]]]:
        return self._ontology

    @property
    def schema(self) -> Optional[Dict[str, List[str]]]:
        if self._ontology is None:
            return

        return {split: sorted(list(ont.keys())) for split, ont in self._ontology.items()}

    def setup(self) -> None:
        self.load()
        self.process()

    def load(self) -> None:
        raise NotImplementedError

    def process(self) -> None:
        raise NotImplementedError

    def to_pandas(self) -> DataFrame:
        """Parses dialogues into a DataFrame.

        It sorts the data based on the `dialogue_id` and the `turn_id`.
        """

        if self.dialogues is None:
            print("You need to setup the data first")
            return

        df = DataFrame(self.dialogues)

        # explode all columns except the first which is the dialogue_id
        return df.explode(df.columns[1:-1].tolist()).sort_values(["dialogue_id", "turn_id"]).reset_index(drop=True)

    def save_to_disk(self, path: Union[str, Path]) -> None:
        """Saves dialogues as HuggingFace Dataset or DatasetDict (if splits are added)."""
        path = Path(path)
        df = self.to_pandas()

        if df["split"].isna().sum() > 0:
            df = df.drop(columns=["split"])
            ds = Dataset.from_pandas(df, preserve_index=False)
            ds.save_to_disk(path)
            return

        dataset_dict = {}
        for split in df["split"].unique():
            dataset_dict[split] = Dataset.from_pandas(df.loc[df["split"] == split].copy(), preserve_index=False)

        dataset_dict = DatasetDict(dataset_dict)
        dataset_dict.save_to_disk(path)

        if self._ontology is not None:
            srsly.write_yaml(path / "schema.yaml", self.schema)
            srsly.write_yaml(path / "ontology.yaml", self.ontology)

    def build_ontology_and_schema(self) -> None:
        if self.dialogues is None:
            print("You need to setup the data first")
            return

        ontology = {}
        for dialogue in tqdm(deepcopy(self.dialogues), desc="Building ontology"):
            # NOTE: without the deepcopy this modifies the self.dialogue object somehow: investigate

            if dialogue.split not in ontology:
                ontology[dialogue.split] = {}

            for state in dialogue.states:
                if state is None:
                    continue
                for k, v in state.items():
                    if k in ontology[dialogue.split]:
                        ontology[dialogue.split][k] += v
                    else:
                        ontology[dialogue.split][k] = v

        self._ontology = {split: {k: list(set(v)) for k, v in ont.items()} for split, ont in ontology.items()}


class MWozDataset(BaseDataset):
    """Class to extract the data from the MultiWoz `data.json` file.

    This works with the default MultiWoz format, not the SGD-like format
    like the one used for MultiWoz v2.2.

    Usage is as simple as

        ```python
        dataset = MWozDataset(path_to_data=...)
        dataset.setup()
        dataset.to_pandas()

        # or
        dataset = MWozDataset(path_to_data=...)
        dataset.setup()
        dataset.add_splits(validation_path=..., test_path=...)
        dataset.save_to_disk(path=...)
        ```
    """

    DOMAINS_TO_IGNORE = DOMAINS_TO_IGNORE
    VALUES_TO_IGNORE = VALUES_TO_IGNORE
    SLOTS_TO_IGNORE = SLOTS_TO_IGNORE
    INVALID_DIALOGUES = INVALID_DIALOGUES
    _splits = None

    def __init__(self, path_to_data: Union[str, Path]) -> None:
        super().__init__()
        self.path_to_data = Path(path_to_data)

    @property
    def schema(self) -> Optional[List[str]]:
        """For MultiWoz the schema is the same in each split."""
        if self._ontology is None:
            return

        return sorted(list(set([k for ont in self._ontology.values() for k in ont])))

    def load(self) -> None:
        self._data = srsly.read_json(self.path_to_data)
        print(f"Data successfully loaded from {self.path_to_data}")

    def process(self) -> None:
        """Processes MultiWoz datasets (expects the standard MultiWoz format).

        Format description

            - data: A dictionary with all the dialogues, the keys are the
                dialogue ids and the values are the actual dialogues.

            - dialogue: An individual dialogue is dictionary with two keys, "goal" and
                "log"; ignore "goal".

            - log: A log is a list of turns.

            - turn: A turn is a dictionary with two keys, "metadata" and "text" containing
                the states and the turn utterance, respectively.

            - text: The value corresponding to the "text" keys is simply a string with the
                turn utterance

            - metadata: The value corresponding to the "metadata" key is a dictionary with 7 keys,
                one for each domain, i.e. taxi, police, restaurant, hospital, hotel, attraction, train.
                The values are the slot annotations

            - slot annotations: The value corresponding to an individual domain is a dictionary with
                two keys: "book" and "semi".

            - "book" and "semi": Both are dictionaries where the keys are the slot names and the values
                are the slot values. Note that in the "book" dictionary the "booked" key is ignored since
                it includes information that the system gives to the user with respect to an individual
                booking, e.g.

                    ```
                    "agent": (book a hotel and asks for reference number)
                    "system": "Booking was successful.\nReference number is : 7GAWK763. Anything else I can do for you?"
                    ```

                state:

                    ```
                    "hotel": {
                        "book": {
                            "booked": [
                                {
                                    "name": "the cambridge belfry",
                                    "reference": "7GAWK763"
                                }
                            ],
                            "stay": "2",
                            "day": "tuesday",
                            "people": "6"
                        },
                        "semi": { ... }
                    }

                    ```
        """
        dialogues = []
        for dialogue_id, dialogue in tqdm(self.data.items(), desc="Processing data"):

            # ignore invalid dialogue
            if dialogue_id in self.INVALID_DIALOGUES:
                print(f"Skipping invalid dialogue {dialogue_id}")
                continue

            diag = self._extract_dialogue(dialogue["log"])
            diag.dialogue_id = dialogue_id

            # ignore dialogue that is all about domains to ignore
            if all(
                domain in self.DOMAINS_TO_IGNORE
                for domain in list(set(s for turn_s in diag.turn_services for s in turn_s))
            ):
                print(f"Skipping dialogue {diag.dialogue_id} because is only about domains to ignore")
                continue

            dialogues.append(diag)

        self._dialogues = dialogues

    def _extract_dialogue(self, log: Dict[str, Union[str, Dict]]) -> Dialogue:
        """Retrieves information and creates an individual dialogue object."""

        states = []
        services = []
        sys_utterances = ["none"]
        usr_utterances = []

        for turn_idx, turn in enumerate(log):

            # user turn
            if turn_idx % 2 == 0:
                usr_utterances.append(turn["text"])
                continue

            # system turn
            else:
                sys_utterances.append(turn["text"])

                # get turn states
                state, service = self._extract_states(turn["metadata"])
                states.append(state)
                services.append(service)

        # append "none" to user utterances to make it the same length as system
        # note that system always speaks first with a "none" utterance
        usr_utterances.append("none")

        # pad the last dialogue turn with the state from the previous turn
        # in theory here you do not have states because it is greeting etc
        states.append(states[-1])
        services.append(services[-1])

        return Dialogue(
            dialogue_id=None,
            turn_id=list(range(len(sys_utterances))),
            sys_utt=sys_utterances,
            usr_utt=usr_utterances,
            states=states,
            turn_services=services,
            split=None,
        )

    def _extract_states(self, metadata: Dict[str, Dict]) -> Tuple[Union[Dict[str, str], None], List[str]]:
        f"""Process the turn "metadata" field.

        1. It ignores the domain in {self.DOMAINS_TO_IGNORE}

        1. It ignores the slots in {self.SLOTS_TO_IGNORE}

        1. It ignores the values in {self.VALUES_TO_IGNORE}

        1. It creates states where the key is formed as "[domain]-[slot]"
        """
        turn_services = []
        turn_states = {}
        for domain_name, book_semi_dict in metadata.items():

            # ignore domains
            if domain_name in self.DOMAINS_TO_IGNORE:
                continue

            for _, states in book_semi_dict.items():
                for slot, value in states.items():

                    # minimal text normalization (just in case)
                    slot = slot.strip()

                    # ignores the "booked" key in the "book" dictionary
                    # ignores empty values
                    # ignores values
                    if slot in self.SLOTS_TO_IGNORE or len(value) < 1 or value in self.VALUES_TO_IGNORE:
                        continue

                    # create slot name accoring to the format `[domain]-[slot]`
                    slot_name = f"{domain_name}-{slot}".lower()

                    # do it here because if key is "booked", value can be a list
                    # the original MultiWoz format has multiple options divided
                    # by a pipe ("|"). If the pipe is not present, the `.split`
                    # method creates a list anyway and it is good because
                    # value is a list in MultiWoz 2.2 and this makes it compatible
                    value = value.strip().split("|")

                    # in MultiWoz when the user says "I prefer this but that is good"
                    # the annotation is "this>that". Since we can retrieve only one
                    # value, we choose the preferred one, that is the one on the left
                    # of the `>` sign
                    # of course there are annotation errors and in a few cases the `>` is `<`
                    value = [v.split(">")[0].split("<")[0] for v in value]

                    if slot_name in turn_states:
                        print(f"Overwriting: {slot_name} in {turn_states}")

                    turn_states.update({slot_name: value})
                    turn_services.append(domain_name)

        turn_states = turn_states if len(turn_states) > 0 else None

        return turn_states, list(set(turn_services))

    def add_splits(self, validation_path, test_path) -> None:
        """Adds split identified to each dialogue."""
        # load splits
        splits = {}
        with open(validation_path) as fl:
            splits["validation"] = [i for i in fl.read().split("\n")]
        with open(test_path) as fl:
            splits["test"] = [i for i in fl.read().split("\n")]

        self.splits = splits

        # add them to dialogues
        for diag in tqdm(self.dialogues):
            if diag.dialogue_id in self.splits["validation"]:
                diag.split = "validation"
            elif diag.dialogue_id in self.splits["test"]:
                diag.split = "test"
            else:
                diag.split = "train"


class SchemaGuidedDialogueDataset(BaseDataset):
    def __init__(self, path_to_folder: Union[str, Path]) -> None:
        super().__init__()
        self.path_to_folder = Path(path_to_folder)

    def load(self) -> None:
        data = {}
        for split in ("train", "dev", "test"):
            data[split if split != "dev" else "validation"] = [
                dialogue
                for path in tqdm(list((self.path_to_folder / split).glob("*dial*.json")), desc=f"Loading {split} split")
                for dialogue in srsly.read_json(path)
            ]

        self._data = data

    def process(self) -> None:
        """Processes MultiWoz datasets (expects the standard MultiWoz format).

        Format description

            - data: A list with all the dialogues.

            - dialogue: An individual dialogue is dictionary with three keys, "dialogue_id",
                "services" (all the services discussed in the entire dialogue), and "turns"

            - turns: A list of turns in which each individual turn is a dictionary with four keys,
                "frames", "speaker" (can be "USER" or "SYSTEM"), "turn_id", and "utterance".

            - frames: A list of frames in which each frame is a dictionary with four keys, "actions",
                "service", "slots", "state". Ignore "actions". If a turn has multiple active domains/services
                more than one frame will have non-empty entries.

            - state: A dictionary with three keys, "active_intent", "requested_slots", "slot_values".
                Consider only "slot_values"

                - slot_values: The actual state in the form of a Dict[str, List[str]]

            - service: The service referred in the state

            - slots: A list of dictionaries in which each individual dictionary has four keys, "slot",
                "start", "exclusive_end", containing the name of the slot and the start and
                end index of the characters of the value in the utterance, respectively.
        """
        dialogues = []
        for split, dialogue_list in self.data.items():
            for dialogue in tqdm(dialogue_list, desc=f"Processing {split} split"):
                # ignore domains at the dialogue level if a dialogue is only about that
                # for SGD there are no domains to ignore but we reuse this function for MultiWoz 2.2
                if all(domain in self.DOMAINS_TO_IGNORE for domain in dialogue["services"]):
                    print(f"Skipping dialogue {dialogue['dialogue_id']} because is only about domains to ignore")
                    continue
                diag = self._extract_dialogue(dialogue)
                diag.split = split
                dialogues.append(diag)

        self._dialogues = dialogues

    def _extract_dialogue(self, dialogue: Dict[str, Union[str, Dict]]) -> Dialogue:
        """Retrieves information and creates an individual dialogue object."""

        cumulative_state = {}
        states = []
        services = []
        sys_utterances = ["none"]
        usr_utterances = []

        for turn in dialogue["turns"]:

            utt = turn["utterance"]

            # user turn
            if turn["speaker"] == "USER":
                usr_utterances.append(utt)
                state, service = self._extract_states(
                    frames=turn["frames"], utterance=utt, cumulative_state=cumulative_state
                )
                states.append(state)
                if state is not None:
                    cumulative_state = {**cumulative_state, **state}
                services.append(service)

            # system turn
            else:
                sys_utterances.append(utt)

        # append "none" to user utterances to make it the same length as system
        # note that system always speaks first with a "none" utterance
        usr_utterances.append("none")

        # pad the last dialogue turn with the state from the previous turn
        # in theory here you do not have states because it is greeting etc
        states.append(states[-1])
        services.append(services[-1])

        return Dialogue(
            dialogue_id=dialogue["dialogue_id"],
            turn_id=list(range(len(sys_utterances))),
            sys_utt=sys_utterances,
            usr_utt=usr_utterances,
            states=states,
            turn_services=services,
            split=None,
        )

    def _extract_states(
        self, frames: List[Dict[str, Any]], utterance: str, cumulative_state: Dict
    ) -> Tuple[Union[Dict[str, str], None], List[str]]:
        """Extract states from frames of the current turn."""
        turn_services = []
        turn_states = deepcopy(cumulative_state)  # make a copy

        for frame in frames:
            state = frame["state"]["slot_values"]
            domain = frame["service"].strip().lower()  # just in case

            # ignore if state is empty
            # ignore individual domains at the slot level
            if domain in self.DOMAINS_TO_IGNORE or len(state) < 1:
                continue

            # get state and update with the latest available info (overwrite previous present state)
            for k, v in state.items():

                # ignore wrong values
                v = [i for i in v if i not in self.VALUES_TO_IGNORE]
                if len(v) < 1:
                    continue

                # slot name for SGD need to add the domain
                slot_name = (f"{domain}-{k}" if f"{domain}-" not in k else k).lower()
                turn_states[slot_name] = v

            # get domain
            turn_services.append(domain)

        # add span annotation later
        for frame in frames:
            # get span annotation if present
            span_annotations = frame["slots"]
            if len(span_annotations) > 0:
                self._add_span_annotation(
                    span_annotations=span_annotations,
                    current_state=turn_states,
                    utterance=utterance,
                    domain=frame["service"].strip().lower(),
                )

        if not len(turn_states) > 0:
            turn_states = None

        return turn_states, list(set(turn_services))

    def _add_span_annotation(
        self, span_annotations: List[Dict], current_state: Dict, utterance: str, domain: str
    ) -> None:
        for annotation in span_annotations:

            # get the value from the utterance
            if "start" in annotation:
                span = utterance[annotation["start"] : annotation["exclusive_end"]]
                span = [span]  # for compatibility with the case below

            slot = f"{domain}-{annotation['slot']}"
            if slot not in current_state:
                current_state[slot] = span

            elif slot in current_state:
                current_state[slot] = list(set(span + current_state[slot]))


class MWoz22Dataset(SchemaGuidedDialogueDataset):

    DOMAINS_TO_IGNORE = DOMAINS_TO_IGNORE
    VALUES_TO_IGNORE = VALUES_TO_IGNORE

    def _extract_states(
        self, frames: List[Dict[str, Any]], utterance: str, cumulative_state: Dict
    ) -> Tuple[Union[Dict[str, str], None], List[str]]:
        """Extract states from frames of the current turn."""

        turn_states, turn_services = super()._extract_states(
            frames=frames, utterance=utterance, cumulative_state=cumulative_state
        )

        if turn_states is not None:
            # remove "book" from slot name for compatibility with MultiWoz 2.1, 2.3, 2.4
            turn_states = {k.replace("book", ""): v for k, v in turn_states.items()}

        return turn_states, turn_services

    def _add_span_annotation(
        self, span_annotations: List[Dict], current_state: Dict, utterance: str, domain: str
    ) -> None:

        for annotation in span_annotations:

            # check that domain is included in the slot name
            assert domain == annotation["slot"].split("-")[0].strip()

            # ignore domains at the slot level
            # can do this outside the loop but in this way I can check the domain correctness
            # bacause I am not sure I can trust the service `annotation`
            if domain in self.DOMAINS_TO_IGNORE:
                continue

            # get the value from the utterance
            if "start" in annotation:
                span = utterance[annotation["start"] : annotation["exclusive_end"]]
                if span != annotation["value"]:
                    print(
                        f"Span extraction from utterance did not work. Using the value provided -- "
                        f'extracted: {span} != provided: {annotation["value"]}'
                    )
                span = annotation["value"]
                span = [span]  # for compatibility with the case below

            # get the value from another slot already in the state
            # ---
            # for example, a user utterance can be "I also need a taxi from the restaurant to the hotel.",
            # in which the state values of "taxi-departure" and "taxi-destination" are respectively carried
            # over from that of "restaurant-name" and "hotel-name".
            # ---
            elif "copy_from" in annotation:
                if (
                    annotation["copy_from"] in current_state
                    and annotation["slot"] in current_state
                    and (
                        all(i in current_state[annotation["slot"]] for i in annotation["value"])
                        or all(i in current_state[annotation["copy_from"]] for i in annotation["value"])
                    )
                ):
                    # print(f"\nCopy from and Slot are already in state. Skipping.")
                    continue

                elif annotation["copy_from"] not in current_state and annotation["slot"] not in current_state:
                    print("Either `slot` and `copy_from` are not in the span annotation")
                    continue

                elif annotation["copy_from"] not in current_state and annotation["slot"] in current_state:
                    print(
                        "\nAnnotation error in the dataset. The keys `copy_from` and `slot` are switched ",
                        f"\n  utterance: {utterance}\n  annotation: {annotation}\n  current_state: {current_state}\n",
                    )
                    # switch keys and correct error
                    copy_from = annotation["slot"]
                    annotation["slot"] = annotation["copy_from"]
                    annotation["copy_from"] = copy_from

                span = current_state[annotation["copy_from"]]  # this is a list
                if set(span) != set(annotation["value"]):
                    print(
                        f"\nCopy from other slot did not work. Using the value provided -- "
                        f'extracted: {span} != provided: {annotation["value"]}'
                        f"\n  utterance: {utterance}\n  annotation: {annotation}\n  current_state: {current_state}\n"
                    )

            slot = annotation["slot"]
            if slot not in current_state:
                # print(
                #     f"\nAdding new slot\n  `{slot}: {span}`\n  current_state `{current_state}`\n"
                #     f"\n  utterance: {utterance}\n"
                # )
                current_state[slot] = span

            elif slot in current_state:
                # print(f"Adding new value {span} for slot {slot} in current_state {current_state}")
                current_state[slot] = list(set(span + current_state[slot]))


if __name__ == "__main__":
    dataset = SchemaGuidedDialogueDataset("/home/lescipi/t2t-dst/data/raw/schema_guided_dialogue")
    dataset.load()
    dataset.process()
