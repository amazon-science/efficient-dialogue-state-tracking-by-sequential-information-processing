import random
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

DOMAINS_TO_REMOVE = ("bus", "police", "hospital")


def process_states_mwoz22(ex: Union[Dict[str, List[str]], None]) -> Union[Dict[str, List[str]], None]:
    """Clean the dialogue state.

    A dialogue state can either be None or a Dict[str, List[str]].

    It performs the following operations:

        - remove None

        - make slot name lower-case

        - remove "book" from slot names

        - remove slots pertaining "bus", "police", and "hospital" domain
    """
    if ex is None:
        return None

    states = {
        k.lower().replace("book", "").strip(): [i.strip() for i in v]
        for k, v in ex.items()
        if all(i not in k for i in DOMAINS_TO_REMOVE) and v is not None
    }

    if len(states) < 1:
        return None

    return states


def extract_slots(ex: Union[Dict[str, Any], None]) -> List:
    if ex is None:
        return []
    return sorted(list(ex.keys()))


def extract_values(ex: Union[Dict[str, Union[str, List[str]]], None]) -> List:
    if ex is None:
        return []

    values = []
    for v in ex.values():
        if isinstance(v, str):
            v = [v]
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        values += v

    return sorted(values)


def extract_slot_values(ex: Union[Dict[str, Any], None]) -> List:
    if ex is None:
        return []

    return sorted(list((k, v) for k, v_list in ex.items() for v in v_list))


@dataclass
class Dialogue:
    dialogue_id: str
    turn_id: List[int]
    sys_utt: List[str]
    usr_utt: List[str]
    states: List[Dict[str, List]]


def extract_dialogue(dialogue_id: str, log: Dict[str, Any]) -> Dialogue:
    """Extracts dialogue from MultiWoz (2.1 format)."""
    states = []
    sys_utterances = ["none"]
    usr_utterances = []

    for turn_idx, turn in enumerate(log):
        if turn_idx % 2 == 0:
            # user turn
            usr_utterances.append(turn["text"])
            continue
        else:
            sys_utterances.append(turn["text"])

            turn_states = {}
            for dom, dom_ds in turn["metadata"].items():
                for _, slot_vals in dom_ds.items():
                    for slot, vals in slot_vals.items():
                        if slot != "booked" and len(vals) > 0:
                            if f"{dom}-{slot}" in turn_states:
                                print(
                                    "overwriting",
                                    f"{dom}-{slot}",
                                    turn_states,
                                )
                            turn_states.update({f"{dom}-{slot}": vals})

            states.append(turn_states)

    usr_utterances.append("none")
    states.append(states[-1])

    return Dialogue(
        dialogue_id=dialogue_id,
        turn_id=list(range(len(sys_utterances))),
        sys_utt=sys_utterances,
        usr_utt=usr_utterances,
        states=states,
    )


def parse_dialogues(data: Dict[str, Any]) -> pd.DataFrame:
    """Parses dialogues into a DataFrame.

    Args:
        data (Dict[str, Any]): The data object resulting from reading the data.json file in
            MultiWoz v2.1, v2.3, v2.4.

    Returns:
        pd.DataFrame: Dialogues in a DataFrame
    """
    df = pd.DataFrame([extract_dialogue(dialogue_id, dialogue["log"]) for dialogue_id, dialogue in tqdm(data.items())])

    # explode all columns except the first which is the dialogue_id
    return df.explode(df.columns[1:].tolist()).reset_index(drop=True)


def process_states_mwoz(ex: Union[Dict[str, str], None]) -> Union[Dict[str, List[str]], None]:
    """Clean the dialogue state.

    It performs the following operations:

        - remove "not mentioned" and None

        - make slot name lower-case

        - split "|"-separated slot values and take the first

        - remove slots pertaining "bus", "police", and "hospital" domain
    """
    if ex is None:
        return None

    states = {
        k.lower().strip(): [i.strip() for i in v.split("|")]
        for k, v in ex.items()
        if all(i not in k for i in DOMAINS_TO_REMOVE) and v not in ("not mentioned", "none") and v is not None
    }

    if len(states) < 1:
        return None

    return states


def remove_empty_slots(state: Union[Dict[str, Union[List[str], None]], None]) -> Union[Dict[str, List[str]], None]:
    """Removes empty slots from state.

    To be able to save data as a parquet or arrow file you need to "pad" the states to all have
    the same keys. To do this the None value is introduced. Therefore, when you load it back
    it is convenient to have a way to remove the "padding".

    Args:
        state (Union[Dict[str, Union[List[str], None]], None]): The dialogue state.

    Returns:
        Union[Dict[str, List[str]], None]: The cleaned dialogue state.
    """

    if state is None:
        return None

    new_state = {k: remove_empty_slots(v) if isinstance(v, dict) else v for k, v in state.items() if v is not None}

    if len(new_state) < 1:
        return None

    return new_state


def diff(left: Union[Dict[str, List[str]], None], right: Union[Dict[str, List[str]], None]) -> Dict:
    """Computes the Joint Goal Accuracy for the current turn."""

    # both `None`
    if left is None and right is None:
        return {}

    # one of them is `None` and the other is not
    elif left is not None and right is None:
        return {"left-right": left}

    elif left is None and right is not None:
        return {"right-left": right}

    # check wrong values
    diff_a = {}
    for k, v in left.items():
        if k not in right:
            diff_a[k] = v

        elif k in right and set(v) not in set(right[k]):
            d = list(set.symmetric_difference(set(v), set(right[k])))
            if len(d) > 0:
                diff_a[k] = d

    diff_b = {}
    for k, v in right.items():
        if k not in left:
            diff_b[k] = v

        elif k in left and set(v) not in set(left[k]):
            d = list(set.symmetric_difference(set(v), set(left[k])))
            if len(d) > 0:
                diff_b[k] = d

    diffs = {}

    if len(diff_a) > 0:
        diffs["left-right"] = diff_a

    if len(diff_b) > 0:
        diffs["right-left"] = diff_b

    return diffs


def diff_values(left: Union[Dict[str, List[str]], None], right: Union[Dict[str, List[str]], None]) -> Dict:
    """Computes the Joint Goal Accuracy for the current turn."""

    # some of them is `None`
    if left is None or right is None:
        return None

    # check wrong values
    diffs = {}
    for k, v in left.items():
        if k in right and set(v) not in set(right[k]):
            a = list(set.difference(set(v), set(right[k])))
            b = list(set.difference(set(right[k]), set(v)))
            if len(a) > 0 or len(b) > 0:
                diffs[k] = (a, b)

    return diffs


def prepare_states_train(state: Union[Dict[str, Union[List[str], None]], None]) -> Union[Dict[str, str], None]:
    """Prepares the states for training.

    It performs the following operations:

    - If the state is None it returns None

    - It lower-cases and strips the slot name and selects the longest

    Args:
        state (Union[Dict[str, List[str]], None]): The state.

    Returns:
        Union[Dict[str, str], None]: The cleaned state.
    """

    if state is None:
        return None

    return {k.lower().strip(): sorted(v, key=len)[-1] if v is not None else None for k, v in state.items()}


def check_dict_equal(pred: Union[Dict[str, str], None], true: Union[Dict[str, str], None]) -> bool:
    """Computes the Joint Goal Accuracy for the current turn."""

    # both `None`
    if pred is None and true is None:
        return True

    # one of them is `None` and the other is not
    elif (pred is not None and true is None) or (pred is None and true is not None):
        return False

    # check all slots are equal
    if not set(pred.keys()) == set(true.keys()):
        return False

    # once you know all slots are equal, check all values are equal
    for k, v in pred.items():
        if v != true[k]:  # note: here you might consider using string distance metrics
            return False

    return True


def add_wrong_slot(
    states: Union[Dict[str, str], None], neg_list: List[Tuple[str, str]], p: float = 0.1, how_many: int = 1
) -> Union[Dict[str, str], None]:

    states = {} if states is None else {**states}

    if random.random() <= p:
        for _ in range(how_many):
            k, v = random.choice(neg_list)
            states[k] = v

    if len(states) < 1:
        return None

    return states


def diff_train(left: Union[Dict[str, str], None], right: Union[Dict[str, str], None]) -> Dict:
    """Computes the Joint Goal Accuracy for the current turn."""

    # both `None`
    if left is None and right is None:
        return {}

    # one of them is `None` and the other is not
    elif left is not None and right is None:
        return {"left-right": left}

    elif left is None and right is not None:
        return {"right-left": right}

    # check wrong values
    diff_a = {}
    for k, v in left.items():
        if k not in right:
            diff_a[k] = v

        elif k in right and v != right[k]:
            diff_a[k] = v

    diff_b = {}
    for k, v in right.items():
        if k not in left:
            diff_b[k] = v

        elif k in left and v != left[k]:
            diff_b[k] = v

    diffs = {}

    if len(diff_a) > 0:
        diffs["left-right"] = diff_a

    if len(diff_b) > 0:
        diffs["right-left"] = diff_b

    return diffs


def extract_domains(states: Union[Dict, None]) -> List[str]:
    if states is None:
        return []

    return list(set(k.split("-")[0] for k in states))


def complement_labels(
    ref: Union[Dict[str, List[str]], None], other: Union[Dict[str, List[str]], None]
) -> Union[Dict[str, List[str]], None]:

    skip = ("attraction-name", "restaurant-name", "hotel-name")

    # do not change other
    if ref is None or other is None:
        return other

    new_other = deepcopy(other)
    for k, v in ref.items():
        if k not in new_other:
            continue
        for i in v:
            if i not in new_other[k]:
                new_other[k].append(i)

    return new_other


"""
Text cleaning
"""


def clean_slot_values(
    state: Union[Dict[str, Union[str, List[str]]], None],
    regexs: Tuple[str, str],
) -> Union[Dict[str, Union[str, List[str]]], None]:
    if state is None:
        return state

    new_states = {k: [clean_time(fix_typos(i, regexs)) for i in v] for k, v in state.items()}
    # new_states = fix_names(new_states)

    return new_states


def fix_typos(value, regexs):
    for regex, sub in regexs:
        value = regex.sub(lambda x: replace_with_case(x, sub), value)
    value = value.replace(r'"', "")
    return value.strip()


def fix_names(states: Dict[str, List[str]]) -> Dict[str, List[str]]:

    for slot in ("attraction", "hotel"):

        if not (f"{slot}-name" in states and f"{slot}-type" in states):
            continue

        new_names = []
        for name in states[f"{slot}-name"]:
            if (attr_type := states[f"{slot}-type"][0]) not in name:
                new_names.append(f"{name} {attr_type}")

        if len(new_names) > 0:
            states[f"{slot}-name"] = list(set(states[f"{slot}-name"] + new_names))

    return states


def clean_time(value: str) -> str:
    for regex, sub in RE_TIME_EXPRESSIONS:
        value = regex.sub(sub, value)
    return value.replace("hrs", "").strip()


def replace_with_case(match, replacement):
    g = match.group()
    if g.islower():
        return replacement.lower()
    if g.istitle():
        return replacement.title()
    if g.isupper():
        return replacement.upper()
    return replacement


RE_TIME_EXPRESSIONS = [
    (re.compile(r"(eleven o'clock) ?(a\.?m\.?|p\.?m\.?)", flags=re.IGNORECASE), r"11:00"),
    (re.compile(r"(quarter to 2)", flags=re.IGNORECASE), r"13:45"),
    (re.compile(r"(after noon)", flags=re.IGNORECASE), r"after 12:00"),
    (re.compile(r"(\d{1})(a\.?m\.?|p\.?m\.?)", flags=re.IGNORECASE), r"\1 \2"),  # am/pm without space
    (re.compile(r"(^| )(\d{1,2}) (a\.?m\.?|p\.?m\.?)", flags=re.IGNORECASE), r"\1\2:00 \3"),  # am/pm short to long form
    (re.compile(r"(^| )(\d{2})[;., /\|l]?(\d{2})\b", flags=re.IGNORECASE), r"\1\2:\3"),  # Wrong separator
    (
        re.compile(r"(^| )(at|from|by|until|after|around) ?(\d{1,2})[\.:;](\d{2})\b([^0-9]|$)", flags=re.IGNORECASE),
        r"\1\2 \3:\4\5",
    ),
    (
        re.compile(r"(^| )(at|from|by|until|after|around) ?([012]\d{1}|\d{1})([;.,? ]|$)", flags=re.IGNORECASE),
        r"\1\2 \3:00\4",
    ),
    (re.compile(r"(^| )(\d{1,2}) ?(\d{2}) ?(hrs)?\b", flags=re.IGNORECASE), r"\1\2:\3"),
    (re.compile(r"(^| )(\d{1}:\d{2})"), r"\g<1>0\2"),  # Add missing leading 0
    (
        re.compile(r"(\d{2})(:\d{2}) ?p\.?m\.?", flags=re.IGNORECASE),
        lambda x: str(int(x.groups()[0]) + 12 if int(x.groups()[0]) < 12 else int(x.groups()[0])) + x.groups()[1],
    ),  # Map 12 hour times to 24 hour times
    (re.compile(r"(^| )24:(\d{2})", flags=re.IGNORECASE), r"\g<1>00:\2"),  # Correct times that use 24 as hour
]


def remove_values(
    states: Union[Dict[str, Union[str, List[str]]], None], to_remove: Dict[str, List[str]]
) -> Union[Dict[str, Union[str, List[str]]], None]:
    if states is None:
        return states

    new_states = {}
    for k, v_list in states.items():
        if k not in to_remove:
            new_states[k] = v_list
        else:
            new_v_list = []
            for v in v_list:
                if v not in to_remove[k]:
                    new_v_list.append(v)

            # assert len(new_v_list) > 0, ValueError(f"While removing {k} {v} in {to_remove[k]} you discarded the entire field")
            if len(new_v_list) > 0:
                new_states[k] = new_v_list
            else:
                print(f"While removing {k} {v} you discarded the entire field")

    return new_states


def convert_to_morning(ex):
    a = ex.split(":")
    if len(a) != 2:
        return ex
    a, b = a
    a = int(a)

    if a == 12:
        return f"{a}:{b}"

    a = f"0{a - 12}" if a > 12 else str(a + 12)
    return f"{a}:{b}"


def remove_15_mins(ex):
    if len(ex.split(":")) != 2:
        return ex
    a, b = ex.split(":")
    a, b = int(a), int(b)

    if b - 15 < 0:
        b += 45
        a -= 1
    else:
        b -= 15

    a = f"0{a}" if a < 10 else str(a)
    b = f"0{b}" if b < 10 else str(b)

    return f"{a}:{b}"


def fix_hours_with_one(ex):
    if len(ex.split(":")) != 2:
        return ex

    a, b = ex.split(":")
    a = int(a)

    if a > 10:
        return ex

    a = f"1{a}" if a < 10 else str(a)

    return f"{a}:{b}"
