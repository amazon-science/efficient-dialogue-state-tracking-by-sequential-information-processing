import re
from random import shuffle
from string import Template
from typing import Dict, List, Optional, Tuple, Union

from pytorch_lightning.core.mixins.hparams_mixin import HyperparametersMixin


class BaseSerializer(HyperparametersMixin):
    def __init__(self) -> None:
        super().__init__()

    @property
    def metadata(self) -> Dict:
        metadata = {}
        for k, v in self.hparams.items():
            if isinstance(v, BaseSerializer):
                metadata[k] = dict(v.hparams)
                metadata[k].pop("__target__", None)
                metadata[k]["_target_"] = f"{v.__class__.__module__}.{v.__class__.__name__}"
            else:
                metadata[k] = v

        metadata["_target_"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return metadata

    def serialize(self, state: Union[Dict, None], *args, **kwargs) -> str:
        raise NotImplementedError


"""
Intent serializers
"""


class SimpleIntent(BaseSerializer):
    def __init__(self, intent_sep: str, replace_punct: bool = True) -> None:
        super().__init__()
        self.intent_sep = intent_sep
        self.replace_punct = replace_punct
        self.save_hyperparameters()

    def serialize(self, intents: List[str]) -> str:
        if self.replace_punct:
            intents = [i.replace("_", " ") for i in intents]
        return f"{self.intent_sep} ".join(intents)

    def deserialize(self, intents: str) -> List[str]:
        return [s.strip() for s in intents.split(self.intent_sep)]


"""
History serializers
"""


class ZippedHistory(BaseSerializer):
    def __init__(self, template: str, turn_sep: str, lag: int = -1, null_value: str = "none") -> None:
        super().__init__()
        self.template = Template(template)
        self.lag = lag if lag >= 0 else 1e30
        self.turn_sep = turn_sep
        self.null_value = null_value
        self.save_hyperparameters()

    def serialize(self, sys_history: List[str], user_history: List[str]) -> str:
        assert len(sys_history) == len(user_history)

        history = []
        lag = 0
        for sys, usr in reversed(list(zip(sys_history, user_history))):
            if lag >= self.lag:
                break
            history_str = self.template.substitute(sys_utt=sys, usr_utt=usr)
            history.append(history_str)
            lag += 1

        if len(history) < 1:
            return self.null_value

        return f"{self.turn_sep} ".join(history)


"""
State serializers
"""


class NaturalLanguageState(BaseSerializer):
    """Format states as natural language.

    Args:
        template (str): A template as a natural language sentence with `$slot` and `$value` fields.
        state_sep (str): Token that separates the states.
        null_value (str, optional): The value to use when the state is None. Defaults to "none".
    """

    def __init__(
        self,
        state_sep: str,
        template: str = "$slot: $value",
        null_value: str = "none",
        keep_full_state: bool = False,
        replace_slot_punct: bool = True,
    ) -> None:
        super().__init__()
        self.template = Template(template)
        self.state_sep = state_sep
        self.null_value = null_value
        self.keep_full_state = keep_full_state
        self.replace_slot_punct = replace_slot_punct
        reverse_template = re.sub(r"\\\$(\w+)", r"(?P<\1>.+)", re.escape(template))
        self.reverse_template = re.compile(r"\s{0,}" + reverse_template + r"\s{0,}")
        self.save_hyperparameters()

    def serialize(self, state: Dict[str, str], *args, **kwargs) -> str:

        if state is None or len(state) < 1:
            return self.null_value

        if self.keep_full_state:
            state = {
                k.replace("-", " ") if self.replace_slot_punct else k: v[0] if v is not None else self.null_value
                for k, v in state.items()
            }
        else:
            state = {
                k.replace("-", " ") if self.replace_slot_punct else k: v[0] for k, v in state.items() if v is not None
            }

        if len(state) < 1:
            return self.null_value

        return f"{self.state_sep} ".join(self.template.safe_substitute(slot=k, value=v) for k, v in state.items())

    def deserialize(self, state: str, *args, **kwargs) -> Union[Dict[str, str], str]:
        if state == self.null_value:
            return state
        states = [self.reverse_template.match(s) for s in state.split(self.state_sep)]
        states = [s.groupdict() for s in states if s]
        states = {s["slot"].strip().replace(" ", "-"): s["value"].strip() for s in states if s}

        if len(states) == 0:
            return self.null_value

        return states


class KeyValueState(BaseSerializer):
    """Format states as key-value pairs.

    Args:
        template (str): A template as a natural language sentence with `$slot` and `$value` fields.
        state_sep (str): Token that separates the states.
        null_value (str, optional): The value to use when the state is None. Defaults to "none".
    """

    def __init__(
        self,
        state_sep: str,
        template: str = "$slot = $value",
        shuffle: bool = True,
        null_value: str = "none",
    ) -> None:
        super().__init__()
        self.state_sep = state_sep
        self.template = Template(template)
        self.shuffle = shuffle
        self.null_value = null_value
        reverse_template = re.sub(r"\\\$(\w+)", r"(?P<\1>.+)", re.escape(template))
        self.reverse_template = re.compile(r"\s{0,}" + reverse_template + r"\s{0,}")
        self.save_hyperparameters()

    def flatten_states(self, states: Union[Dict, None]) -> List[Tuple[str, str]]:
        if states is None:
            return self.null_value

        states_list = list(states.items())
        if self.shuffle:
            shuffle(states_list)

        return states_list

    def serialize(self, state: Dict[str, str], *args, **kwargs) -> str:

        if state is None or len(state) < 1:
            return self.null_value

        states_flat = self.flatten_states(state)

        return f"{self.state_sep} ".join(self.template.safe_substitute(slot=k, value=v) for k, v in states_flat)

    def deserialize(self, state: str, *args, **kwargs) -> Union[Dict[str, str], None]:
        if state == self.null_value:
            return None

        states = [self.reverse_template.match(s) for s in state.split(self.state_sep)]
        states = [s.groupdict() for s in states if s]
        states = {s["slot"].strip().replace(" ", "-"): s["value"].strip() for s in states if s}

        # ignore `k: none`
        # states = {k: v for k, v in states.items() if v != self.null_value}

        if len(states) == 0:
            return None

        return states


class KeyStateChanges(BaseSerializer):
    """Serializes states as the slot names (without the values).

    Note that this captures situations in which the slot is the same but
    the value changes. The rationale is that the slot, in this case, needs
    to be acted upon.
    """

    def __init__(self, state_sep: str, shuffle: bool = True, null_value: str = "none") -> None:
        super().__init__()
        self.state_sep = state_sep
        self.shuffle = shuffle
        self.null_value = null_value
        self.save_hyperparameters()

    def flatten_states(self, states: Union[List, Dict, None]) -> str:
        if states is None:
            return self.null_value

        # NOTE: for this serializer, we also expect List[str] as states
        # because when the model predicts the slots, at the following turn
        # this serializer deserializes it as a string
        if isinstance(states, dict):
            states = list(states.keys())

        if self.shuffle:
            shuffle(states)

        return f"{self.state_sep} ".join(states)

    def states_to_diffs(self, cur: Union[Dict, None], prev: Union[Dict, None]) -> str:
        if cur is None:
            return self.null_value

        insert = []
        delete = []
        if prev is None:
            insert = list(cur.keys())

        elif prev == cur:
            return self.null_value

        else:
            for k, v in cur.items():
                if (k not in prev) or (k in prev and v != prev[k]):
                    insert.append(k)

            for k, v in prev.items():
                if k not in cur:
                    delete.append(k)

        out = list(set(insert + delete))
        if self.shuffle:
            shuffle(out)

        return f"{self.state_sep} ".join(out)

    def serialize(
        self,
        state: Union[List[str], Dict[str, str], None],
        previous_state: Union[Dict, None] = None,
        is_input: bool = True,
    ) -> str:

        if is_input:
            return self.flatten_states(state)

        return self.states_to_diffs(cur=state, prev=previous_state)

    def deserialize(self, states: str, *args, **kwargs) -> Union[List[str], None]:
        if states.strip() == self.null_value:
            return None

        return [i.strip() for i in states.split(self.state_sep)]


class KeyValueStateChanges(BaseSerializer):
    def __init__(
        self,
        state_sep: str,
        cmd_template: str = "$cmd $state",
        state_template: str = "$slot = $value",
        insert_cmd: str = "INSERT",
        delete_cmd: str = "DELETE",
        shuffle: bool = True,
        null_value: str = "none",
    ) -> None:
        super().__init__()
        self.cmd_template = Template(cmd_template)
        self.state_template = Template(state_template)
        self.insert_cmd = insert_cmd
        self.delete_cmd = delete_cmd
        self.shuffle = shuffle
        self.null_value = null_value
        self.state_sep = state_sep

        self.remove_regex = re.compile(f"{self.insert_cmd}|{self.delete_cmd}")
        reverse_template = re.sub(r"\\\$(\w+)", r"(?P<\1>.+)", re.escape(state_template))
        self.reverse_template = re.compile(r"\s{0,}" + reverse_template + r"\s{0,}")

        self.save_hyperparameters()

    def states_to_commands(self, cur: Union[Dict, None], prev: Union[Dict, None]) -> str:
        insert = {}
        delete = {}

        if cur is None and prev is None:
            return self.null_value

        elif prev is None and cur is not None:
            insert = cur

        elif cur is None and prev is not None:
            delete = prev

        elif prev == cur:
            return self.null_value

        else:
            for k, v in cur.items():
                if (k not in prev) or (k in prev and v != prev[k]):
                    insert[k] = v

            for k, v in prev.items():
                if k not in cur:
                    delete[k] = v

        out = []
        if len(insert) > 0:
            out += [
                self.cmd_template.substitute(
                    cmd=self.insert_cmd,
                    state=self.state_template.substitute(slot=k, value=v),
                )
                for k, v in insert.items()
            ]
        if len(delete) > 0:
            out += [
                self.cmd_template.substitute(
                    cmd=self.delete_cmd,
                    state=k,
                )
                for k, v in delete.items()
            ]

        if self.shuffle:
            shuffle(out)

        return f" {self.state_sep} ".join(out)

    def commands_to_states(self, commands: str, prev: Union[Dict, None]) -> Union[Dict, None]:
        new_states = {k: v[0] if isinstance(v, list) else v for k, v in prev.items()} if prev is not None else {}
        if commands == self.null_value:
            return prev

        for s in commands.split(self.state_sep):

            if self.insert_cmd in s:
                c = s.replace(self.insert_cmd, "").strip().split("=")
                # if this is not true it means the command is wrongly formatted
                if len(c) == 2:
                    k, v = c
                    new_states.update({k.strip(): v.strip()})
                else:
                    print(f"\n\n\nWRONG COMMAND: {s}\nCOMMANDS: {commands}\n\n\n")

            elif self.delete_cmd in s:
                k = self.remove_regex.sub("", s).strip()
                new_states.pop(k, None)

        if len(new_states) == 0:
            return None

        return new_states

    def flatten_states(self, states: Union[Dict[str, Union[str, None]], None]) -> str:
        if states is None:
            return self.null_value

        states_list = list(states.items())
        if self.shuffle:
            shuffle(states_list)
        return f"{self.state_sep} ".join(self.state_template.safe_substitute(slot=k, value=v) for (k, v) in states_list)

    def serialize(
        self, state: Union[Dict[str, str], None], previous_state: Union[Dict, None] = None, is_input: bool = True
    ) -> str:

        if is_input:
            return self.flatten_states(state)

        return self.states_to_commands(cur=state, prev=previous_state)

    def deserialize(self, commands: str, previous_state: Union[Dict, None]) -> Union[Dict, None]:
        return self.commands_to_states(commands, previous_state)


"""
Intent and Slot serializers
"""


class SchemaSerializer(BaseSerializer):
    def __init__(self, sep: str, replace_slot_punct: bool = False, shuffle: bool = True) -> None:
        super().__init__()
        self.replace_slot_punct = replace_slot_punct
        self.sep = sep
        self.shuffle = shuffle
        self.save_hyperparameters()

    def serialize(self, schema: List[str]) -> str:
        slots = [slot.replace("-", " ") if self.replace_slot_punct else slot for slot in schema]
        if self.shuffle:
            shuffle(slots)

        return f"{self.sep} ".join(slots)


class IntentSerializer(BaseSerializer):
    def __init__(self, sep: str, replace_intent_punct: bool = True, shuffle: bool = False) -> None:
        super().__init__()
        self.replace_intent_punct = replace_intent_punct
        self.sep = sep
        self.shuffle = shuffle
        self.save_hyperparameters()

    def serialize(self, all_intents: List[str]) -> str:
        intents = [intent.replace("_", " ") if self.replace_intent_punct else intent for intent in all_intents]
        if self.shuffle:
            shuffle(intents)

        return f"{self.sep} ".join(intents)


"""
Input and Output serializers
"""


class TargetSerializer(BaseSerializer):
    """
    NOTE: for now intents can only go first.
    """

    def __init__(
        self,
        state_serializer: BaseSerializer,
        template: str = "$intents $sep $states",
        sep: Optional[str] = None,
        intent_serializer: Optional[BaseSerializer] = None,
    ) -> None:
        super().__init__()
        self.state_serializer = state_serializer
        self.template = Template(template)
        self.sep = sep
        self.intent_serializer = intent_serializer
        if self.intent_serializer is not None:
            assert self.sep is not None

        self.save_hyperparameters()

    def serialize(
        self,
        state: Dict[str, str],
        previous_state: Dict[str, str],
        intents: Optional[List[str]] = None,
    ) -> str:

        states_repr = self.state_serializer.serialize(state=state, previous_state=previous_state, is_input=False)
        if states_repr == self.state_serializer.null_value:
            return self.state_serializer.null_value

        intents_repr, sep = "", ""
        if self.intent_serializer is not None and intents is not None:
            intents_repr = self.intent_serializer.serialize(intents=intents)
            sep = self.sep

        return self.template.safe_substitute(intents=intents_repr, sep=sep, states=states_repr).strip()

    def deserialize(
        self, target_text: str, *args, **kwargs
    ) -> Tuple[Union[List[str], None], Union[Dict[str, str], None]]:

        if self.intent_serializer is None:
            states = self.state_serializer.deserialize(target_text, *args, **kwargs)
            intents = None
            return intents, states

        intents, states = target_text.split(self.sep)
        intents_list = self.intent_serializer.deserialize(intents)
        states_dict = self.state_serializer.deserialize(states)

        return intents_list, states_dict


class InputSerializer(BaseSerializer):
    def __init__(
        self,
        history_serializer: BaseSerializer,
        sep: str,
        template: str = "system: $sys_utt user: $usr_utt $sep $history",
        state_serializer: Optional[BaseSerializer] = None,
        schema_serializer: Optional[BaseSerializer] = None,
        intents_serializer: Optional[BaseSerializer] = None,
    ) -> None:
        super().__init__()
        self.history_serializer = history_serializer
        self.template = Template(template)
        self.sep = sep
        self.state_serializer = state_serializer
        self.schema_serializer = schema_serializer
        self.intents_serializer = intents_serializer
        if self.state_serializer is not None:
            assert self.sep is not None

        self.save_hyperparameters()

    def serialize(
        self,
        sys_utt: str,
        usr_utt: str,
        sys_history: List[str],
        usr_history: List[str],
        previous_state: Optional[Dict[str, str]] = None,
        schema: Optional[List[str]] = None,
        all_intents: Optional[List[str]] = None,
    ) -> str:

        history_repr = (
            self.history_serializer.serialize(sys_history=sys_history, user_history=usr_history)
            if self.history_serializer is not None
            else ""
        )
        previous_state_repr = (
            self.state_serializer.serialize(state=previous_state, previous_state=None, is_input=True)
            if self.state_serializer is not None
            else ""
        )
        schema_repr = self.schema_serializer.serialize(schema) if self.schema_serializer is not None else ""
        all_intents_repr = self.intents_serializer.serialize(all_intents) if self.intents_serializer is not None else ""

        return self.template.safe_substitute(
            sys_utt=sys_utt,
            usr_utt=usr_utt,
            history=history_repr,
            sep=self.sep,
            previous_states=previous_state_repr,
            schema=schema_repr,
            all_intents=all_intents_repr,
        ).strip()


"""
GRAVEYARD
"""


class TableState:
    """Format states as a table.

    For example, `{hotel-name: HCC Taber}` becomes

        | domain | slot |   value   |
        | hotel  | name | HCC Taber |

    Args:
        col_sep (str): Token that separates columns (e.g., "|").
        row_sep (str): Token that separates rows (e.g., "<NEWLINE>").
        null_value (str, optional): The value to use when the state is None. Defaults to "none".
    """

    def __init__(self, col_sep: str, row_sep: str, null_value: str = "none", keep_full_state: bool = False) -> None:
        self.col_sep = col_sep
        self.row_sep = row_sep
        self.null_value = null_value
        self.keep_full_state = keep_full_state

    def serialize(self, state: Dict[str, str], *args, **kwargs) -> str:
        if state is None:
            return self.null_value

        if self.keep_full_state:
            state = {k: v[0] if v is not None else self.null_value for k, v in state.items()}
        else:
            state = {k: v[0] for k, v in state.items() if v is not None}

        table = self.col_sep + self.col_sep.join(["domain", "slot", "value"]) + self.col_sep + self.row_sep
        for k, v in state.items():
            domain, slot = k.split("-")
            table += self.col_sep + self.col_sep.join([domain, slot, v]) + self.col_sep + self.row_sep

        return table

    def deserialize(self, state: str) -> Dict[str, str]:
        if state == self.null_value:
            return self.null_value

        state_dict = {}
        for idx, row in enumerate(state.split(self.row_sep)):
            # ignore header
            if idx == 0:
                continue

            if len(row) > 0:
                cols = [cell for cell in row.split("<extra_id_50>") if len(cell) > 0]
                state_dict[f"{cols[0].strip()}-{cols[1].strip()}"] = cols[2].strip()

        return state_dict


class FormalLanguageState:
    """Format states as table operations.

    Args:
        template (str): A template as a formal language command with `$command`, `$slot`, `$value` fields.
        insert_cmd (str): The token to identify an insertion command.
        update_cmd (str): The token to identify an update command.
        delete_cmd (str): The token to identify a deletion command.
        state_sep (str): Token that separates the states.
        null_value (str, optional): The value to use when the state is None. Defaults to "none".
    """

    def __init__(
        self,
        template: str,
        insert_cmd: str,
        update_cmd: str,
        delete_cmd: str,
        state_sep: str,
        null_value: str = "none",
        replace_slot_punct: bool = True,
    ) -> None:

        self.template = Template(template)
        self.insert_cmd = insert_cmd
        self.update_cmd = update_cmd
        self.delete_cmd = delete_cmd
        self.state_sep = state_sep
        self.null_value = null_value
        self.replace_slot_punct = replace_slot_punct
        reverse_template = re.sub(r"\\\$(\w+)", r"(?P<\1>.+)", re.escape(template))
        self.reverse_template = re.compile(r"\s{0,}" + reverse_template + r"\s{0,}")

    def serialize(self, state: Dict[str, str], previous_state: Union[Dict[str, str], None]) -> str:
        if state is None:
            return self.null_value
        state_dict = {
            k.replace("-", " ") if self.replace_slot_punct else k: v[0] for k, v in state.items() if v is not None
        }
        previous_state_dict = (
            {
                k.replace("-", " ") if self.replace_slot_punct else k: v[0]
                for k, v in previous_state.items()
                if v is not None
            }
            if previous_state is not None
            else {}
        )

        states = []
        command = None
        for k, v in state_dict.items():
            if k not in previous_state_dict:
                command = self.insert_cmd
            elif k in previous_state_dict and (previous_state_dict[k] != v):
                command = self.update_cmd
            else:
                continue
            states.append(self.template.safe_substitute(command=command, slot=k, value=v))

        for k, v in previous_state_dict.items():
            if k not in state_dict:
                states.append(self.template.safe_substitute(command=self.delete_cmd, slot=k, value=v))

        # when the states do not change
        if command is None:
            return self.null_value

        return f"{self.state_sep} ".join(states)

    def deserialize(self, state: str) -> Dict[str, str]:
        if state == self.null_value:
            return self.null_value
        states = [self.reverse_template.match(s) for s in state.split(self.state_sep)]
        states = [s.groupdict() for s in states if s]
        states = {s["slot"].strip().replace(" ", "-"): s["value"].strip() for s in states if s}

        if len(states) == 0:
            return self.null_value

        return states


class NaturalLanguageStateChanges(NaturalLanguageState):
    def __init__(
        self,
        cmd_template: str = "$cmd $state",
        insert_cmd: str = "INSERT",
        update_cmd: str = "UPDATE",
        delete_cmd: str = "DELETE",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cmd_template = Template(cmd_template)
        self.insert_cmd = insert_cmd
        self.update_cmd = update_cmd
        self.delete_cmd = delete_cmd
        self.remove_regex = re.compile(f"{self.insert_cmd}|{self.update_cmd}|{self.delete_cmd}")

    def serialize(
        self, state: Dict[str, str], previous_state: Union[Dict[str, str], None] = None, is_input: bool = True
    ) -> str:

        if is_input:
            return super().serialize(state=state)

        if state is None:
            return self.null_value

        state_dict = {
            k.replace("-", " ") if self.replace_slot_punct else k: v[0] for k, v in state.items() if v is not None
        }
        if len(state_dict) < 1:
            return self.null_value

        previous_state_dict = (
            {
                k.replace("-", " ") if self.replace_slot_punct else k: v[0]
                for k, v in previous_state.items()
                if v is not None
            }
            if previous_state is not None
            else {}
        )

        states = []
        command = None
        for k, v in state_dict.items():
            if k not in previous_state_dict:
                command = self.insert_cmd
            elif k in previous_state_dict and (previous_state_dict[k] != v):
                command = self.update_cmd
            else:
                continue
            state_repr = self.template.safe_substitute(slot=k, value=v)
            states.append(self.cmd_template.safe_substitute(cmd=command, state=state_repr))

        for k, v in previous_state_dict.items():
            if k not in state_dict:
                state_repr = self.template.safe_substitute(slot=k, value=v)
                states.append(self.cmd_template.safe_substitute(cmd=self.delete_cmd, state=state_repr))

        # when the states do not change
        if len(states) < 1:
            return self.null_value

        return f"{self.state_sep} ".join(states)

    def deflatten_states(self, states: str) -> Union[Dict, None]:
        if states == self.null_value:
            return None
        states = [self.reverse_template.match(s) for s in states.split(self.state_sep)]
        states = [s.groupdict() for s in states if s]
        states = {s["slot"].strip(): s["value"].strip() for s in states if s}

        if len(states) == 0:
            return None

        if self.replace_slot_punct:
            states = {k.replace(" ", "-"): v for k, v in states.items()}

        return states

    def deserialize(
        self, state: str, previous_state: Union[Dict[str, Union[List[str], str]], None]
    ) -> Union[Dict[str, str], None]:
        new_states = (
            {k: v[0] if isinstance(v, list) else v for k, v in previous_state.items()}
            if previous_state is not None
            else {}
        )

        for s in state.split(self.state_sep):
            if self.insert_cmd in s or self.update_cmd in s:
                new_state = self.deflatten_states(self.remove_regex.sub("", s).strip())
                if new_state is not None:
                    new_states.update(new_state)
            elif self.delete_cmd in s:
                new_state = self.deflatten_states(self.remove_regex.sub("", s).strip())
                if new_state is not None:
                    for k in new_state:
                        new_states.pop(k, None)

        if len(new_states) == 0:
            return None

        return new_states
