# ============
# Prepare data
# ============

set -e


# ============
# MultiWoz 2.2
# ============


echo 'Preparing MultiWoz v2.2'


# state operations
echo 'Previous state -> State operations'
poetry run python ./scripts/prepare_data.py \
    history_serializer.lag=0 \
    state_serializer=key_value_state_changes \
    input_template='generate update operations: dialogue: system: $sys_utt user: $usr_utt $sep previous dialogue states: $previous_states $sep operations:' \
    target_template='$states' \
    tokenizer='google/t5-v1_1-base' \
    input_dir='./data/processed/multiwoz_22' \
    output_dir='./data/prepared/base/mwoz22_ops_nohist+prev'


echo 'Full-history -> State operations'
poetry run python ./scripts/prepare_data.py \
    state_serializer=key_value_state_changes \
    input_template='generate update operations: dialogue: system: $sys_utt user: $usr_utt $sep history: $history $sep operations:' \
    target_template='$states' \
    tokenizer='google/t5-v1_1-base' \
    input_dir='./data/processed/multiwoz_22' \
    output_dir='./data/prepared/base/mwoz22_ops_partialhist+prev'


echo '4 Turns + Previous state -> State operations'
poetry run python ./scripts/prepare_data.py \
    history_serializer.lag=4 \
    state_serializer=key_value_state_changes \
    input_template='generate update operations: dialogue: system: $sys_utt user: $usr_utt $sep previous dialogue states: $previous_states $sep history: $history $sep operations:' \
    target_template='$states' \
    tokenizer='google/t5-v1_1-base' \
    input_dir='./data/processed/multiwoz_22' \
    output_dir='./data/prepared/base/mwoz22_ops_partialhist+prev'


echo 'Full-history + Previous state -> State operations'
poetry run python ./scripts/prepare_data.py \
    state_serializer=key_value_state_changes \
    input_template='generate update operations: dialogue: system: $sys_utt user: $usr_utt $sep previous dialogue states: $previous_states $sep history: $history $sep operations:' \
    target_template='$states' \
    tokenizer='google/t5-v1_1-base' \
    input_dir='./data/processed/multiwoz_22' \
    output_dir='./data/prepared/base/mwoz22_ops_partialhist+prev'


# cumulative state
echo 'Previous state -> Cumulative state'
poetry run python ./scripts/prepare_data.py \
    history_serializer.lag=0 \
    state_serializer=key_value_state \
    input_template='generate full state: dialogue: system: $sys_utt user: $usr_utt $sep previous dialogue states: $previous_states $sep state:' \
    target_template='$states' \
    tokenizer='google/t5-v1_1-base' \
    input_dir='./data/processed/multiwoz_22' \
    output_dir='./data/prepared/base/mwoz22_cum_prevstate'


echo 'Full-history -> Cumulative state'
poetry run python ./scripts/prepare_data.py \
    state_serializer=key_value_state \
    input_template='generate full state: dialogue: system: $sys_utt user: $usr_utt $sep history: $history $sep state:' \
    target_template='$states' \
    tokenizer='google/t5-v1_1-base' \
    input_dir='./data/processed/multiwoz_22' \
    output_dir='./data/prepared/base/mwoz22_cum_fullhist+nostate'


echo '4 turns + Previous state -> Cumulative state'
poetry run python ./scripts/prepare_data.py \
    history_serializer.lag=4 \
    state_serializer=key_value_state \
    input_template='generate full state: dialogue: system: $sys_utt user: $usr_utt $sep previous dialogue states: $previous_states $sep history: $history $sep state:' \
    target_template='$states' \
    tokenizer='google/t5-v1_1-base' \
    input_dir='./data/processed/multiwoz_22' \
    output_dir='./data/prepared/base/mwoz22_cum_fullhist+nostate'


echo 'Full-history + Previous state -> Cumulative state'
poetry run python ./scripts/prepare_data.py \
    state_serializer=key_value_state \
    input_template='generate full state: dialogue: system: $sys_utt user: $usr_utt $sep previous dialogue states: $previous_states $sep history: $history $sep state:' \
    target_template='$states' \
    tokenizer='google/t5-v1_1-base' \
    input_dir='./data/processed/multiwoz_22' \
    output_dir='./data/prepared/base/mwoz22_cum_fullhist+nostate'


# ============
# MultiWoz 2.1
# ============


echo 'Preparing MultiWoz v2.1'


# state operations
echo 'Previous state -> State operations'
poetry run python ./scripts/prepare_data.py \
    history_serializer.lag=0 \
    state_serializer=key_value_state_changes \
    input_template='generate update operations: dialogue: system: $sys_utt user: $usr_utt $sep previous dialogue states: $previous_states $sep operations:' \
    target_template='$states' \
    tokenizer='google/t5-v1_1-base' \
    input_dir='./data/processed/multiwoz_21' \
    output_dir='./data/prepared/base/mwoz21_ops_nohist+prev'


echo '4 Turns + Previous state -> State operations'
poetry run python ./scripts/prepare_data.py \
    history_serializer.lag=4 \
    state_serializer=key_value_state_changes \
    input_template='generate update operations: dialogue: system: $sys_utt user: $usr_utt $sep previous dialogue states: $previous_states $sep history: $history $sep operations:' \
    target_template='$states' \
    tokenizer='google/t5-v1_1-base' \
    input_dir='./data/processed/multiwoz_21' \
    output_dir='./data/prepared/base/mwoz21_ops_partialhist+prev'


# cumulative state
echo 'Full-history -> Cumulative state'
poetry run python ./scripts/prepare_data.py \
    state_serializer=key_value_state \
    input_template='generate full state: dialogue: system: $sys_utt user: $usr_utt $sep history: $history $sep state:' \
    target_template='$states' \
    tokenizer='google/t5-v1_1-base' \
    input_dir='./data/processed/multiwoz_21' \
    output_dir='./data/prepared/base/mwoz21_cum_fullhist+nostate'