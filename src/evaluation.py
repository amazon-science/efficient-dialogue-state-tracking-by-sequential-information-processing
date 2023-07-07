import re
from copy import deepcopy
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd


def states_to_set(states: Union[Dict, str, None], null_value: str) -> Set:
    if states is None or states == null_value:
        return set(null_value)

    states_clean = {
        k.lower(): v[0].lower() if isinstance(v, (list, np.ndarray)) else v.lower()
        for k, v in states.items()
        if v is not None
    }

    if len(states_clean) < 1:
        return set(null_value)

    states_set = set((k, v) for k, v in states_clean.items())

    return states_set


def jga(pred: Union[Dict[str, str], None], true: Union[Dict[str, List[str]], None]) -> bool:
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
        if v not in true[k]:  # note: here you might consider using string distance metrics
            return False

    return True


def diff(pred: Union[Dict[str, str], None], true: Union[Dict[str, List[str]], None]) -> Dict:
    """Computes the Joint Goal Accuracy for the current turn."""

    # both `None`
    if pred is None and true is None:
        return {}

    # one of them is `None` and the other is not
    elif pred is not None and true is None:
        return {"pred-true": pred}

    elif pred is None and true is not None:
        return {"true-pred": true}

    # check wrong values
    diff_a = {}
    for k, v in pred.items():
        if (k not in true) or (k in true and v not in true[k]):
            diff_a[k] = v

    diff_b = {}
    for k, v in true.items():
        if k not in pred:
            diff_b[k] = v
        elif k in pred and pred[k] not in v:
            diff_b[k] = list(set.difference(set(v), set([pred[k]])))

    diffs = {}

    if len(diff_a) > 0:
        diffs["pred-true"] = diff_a

    if len(diff_b) > 0:
        diffs["true-pred"] = diff_b

    return diffs


def prepare_states_eval(
    state: Union[Dict[str, Union[str, List[str], np.ndarray]], str, None]
) -> Union[Dict[str, Union[str, List[str]]], None]:
    """Prepares the predicted states for evaluation.

    It performs the following operations:

    - If the state is a string it returns None

    - It lower-cases and strips both the slot name and the slot value

    Args:
        state (Union[Dict[str, str], str]): The state predicted by the model.

    Returns:
        Union[Dict[str, str], None]: The cleaned state.
    """

    if isinstance(state, str) or state is None:
        return None

    return {
        k.lower().strip(): [i.lower().strip() for i in v] if isinstance(v, (list, np.ndarray)) else v.lower().strip()
        for k, v in state.items()
    }


def slot_metrics(pred: List[str], true: List[str]) -> List[float]:
    """Eq. 2 in https://aclanthology.org/2022.acl-short.35.pdf"""
    pred, true = set(pred), set(true)

    if len(pred) == 0 or len(true) == 0:
        return 1 if len(pred) == len(true) else 0

    FN = len(true.difference(pred))
    FP = len(pred.difference(true))
    TP = len(set.intersection(pred, true))
    # TN = N - (FP + TP + FN)

    # accuracy =  (TP + TN) / N if N > 0 else 1
    precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
    recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0

    return [precision, recall, f1]


def compute_prf(
    pred: Union[Dict[str, str], None], gold: Union[Dict[str, List[str]], None], num_slots: int
) -> Tuple[float, float, float, float, float]:
    TP, FP, FN = 0, 0, 0
    missed_slots = []
    missed_gold = 0
    wrong_preds = 0

    if pred is None and gold is None:
        return 1.0, 1.0, 1.0, 1.0, 1.0

    elif (pred is None and gold is not None) or (pred is not None and gold is None):
        return 0.0, 0.0, 0.0, 0.0, 0.0

    else:

        for g in gold:
            if g in pred:
                TP += 1

                if pred[g] not in gold[g]:
                    missed_gold += 1
                    missed_slots.append(g)

            else:
                FN += 1
                missed_slots.append(g)
                missed_gold += 1

        for p in pred:
            if p not in gold:
                FP += 1
                if p not in missed_slots:
                    wrong_preds += 1

        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        accuracy = (num_slots - missed_gold - wrong_preds) / float(num_slots)

        all_slots = len(set(list(pred.keys()) + list(gold.keys())))
        relative_accuracy = (all_slots - missed_gold - wrong_preds) / float(all_slots)

    return F1, recall, precision, accuracy, relative_accuracy
