## Introduction
This repository contains the code used for the paper titled [Diable: Efficient Dialogue State Tracking as Operations on Tables](https://aclanthology.org/2023.findings-acl.615/).

If you use this code, please cite our paper:
```
@inproceedings{lesci-etal-2023-diable,
    title = "Diable: Efficient Dialogue State Tracking as Operations on Tables",
    author = "Lesci, Pietro  and
      Fujinuma, Yoshinari  and
      Hardalov, Momchil  and
      Shang, Chao  and
      Marquez, Lluis",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.615",
    pages = "9697--9719",
    abstract = "Sequence-to-sequence state-of-the-art systems for dialogue state tracking (DST) use the full dialogue history as input, represent the current state as a list with all the slots, and generate the entire state from scratch at each dialogue turn. This approach is inefficient, especially when the number of slots is large and the conversation is long. In this paper, we propose {\_}Diable{\_}, a new task formalisation that simplifies the design and implementation of efficient DST systems and allows one to easily plug and play large language models. We represent the dialogue state as a table and formalise DST as a table manipulation task. At each turn, the system updates the previous state by generating table operations based on the dialogue context. Extensive experimentation on the MultiWoz datasets demonstrates that {\_}Diable{\_} {\_}(i){\_} outperforms strong efficient DST baselines, {\_}(ii){\_} is 2.4x more time efficient than current state-of-the-art methods while retaining competitive Joint Goal Accuracy, and {\_}(iii){\_} is robust to noisy data annotations due to the table operations approach.",
}
```

## How to Install

### Install Oh-my-zsh (optional)

[source](https://ohmyz.sh/#install)

```bash
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

### Install conda

[source](https://educe-ubc.github.io/conda.html)
[source](https://developers.google.com/earth-engine/guides/python_install-conda)

```bash
curl -sL \
  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
  "Miniconda3.sh"

bash Miniconda3.sh -b -p

rm Miniconda3.sh

source $HOME/miniconda3/bin/activate

conda init zsh
```

### Install poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then add this to your ~/.zshrc

```bash
export PATH="/home/lescipi/.local/bin:$PATH"

# add convenience tmux shortcuts to ~/.zshrc (optional)
alias tn="tmux new-session -s"
alias ta="tmux attach-session -t"
alias tls="tmux list-sessions"
alias tk="tmux kill-session -t"
```

### Install CUDA

[source](https://support.huaweicloud.com/intl/en-us/usermanual-ecs/ecs_03_0174.html)
[source](https://www.nvidia.com/Download/driverResults.aspx/191339/en-us/)

```bash
BASE_URL=https://us.download.nvidia.com/tesla
DRIVER_VERSION=510.85.02
curl -fSsl -O $BASE_URL/$DRIVER_VERSION/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run

chmod +x NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run
sudo ./NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run

rm NVIDIA-Linux-x86_64-$DRIVER_VERSION.run
```

### Install dependencies
Use poetry to install the environment (if you don't have poetry run )

```bash
conda create -n skg python=3.9 -y
conda activate skg
poetry install --sync
```

### Install AWS CLI

[source](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf ./aws
rm ./awscliv2.zip
```

### Add git user and email

```bash
git config --global user.name "<user>"
git config --global user.email "<email>"
```


---


## Download data

The following script downloads MultiWoz 2.1-2.4

- 2.1 and 2.2 from [https://github.com/budzianowski/multiwoz.git](https://github.com/budzianowski/multiwoz.git)

- 2.3 from [https://github.com/lexmen318/MultiWOZ-coref/raw/main/MultiWOZ2_3.zip](https://github.com/lexmen318/MultiWOZ-coref/raw/main/MultiWOZ2_3.zip)

- 2.4 from [https://github.com/smartyfh/MultiWOZ2.4/raw/main/data/MULTIWOZ2.4.zip](https://github.com/smartyfh/MultiWOZ2.4/raw/main/data/MULTIWOZ2.4.zip)

```bash
./bin/download_data.sh
```

The resulting data are available under the `./data/raw` folder.


## Process data

The following script processed the downloaded from the `./data/raw` folder, and
saves them as HuggingFace `DatasetDict`s into the `./data/processed` folder. 

```bash
./bin/process_data.sh
```


It performs the following processing:

- Extracts the data into a standardized data type

```python
@dataclass
class Dialogue:
    dialogue_id: List[int]
    turn_id: List[int]
    sys_utt: List[str]
    usr_utt: List[str]
    states: List[Union[Dict[str, str], None]]
    turn_services: List[List[str]]
    split: str
```

- Generates the `ontology.yaml` and `schema.yaml` files. The ontology is dependent on the data split (i.e., train/validation/test have a different ontologies). For SGD the schema follows the same structure since it changes across data splits. However, for MultiWoz we only save one schema since it is constant across splits

- Specifically for MultiWoz

    - Removes slots in the `("bus", "police", "hospital")` domains

    - Removes entire dialogues in case the only discussed domains are `("bus", "police", "hospital")`

    - Removes the values `("not mentioned", "none")`

    - Removes invalid dialogues `("SNG01862.json",)`

    - For v2.2 removes `"booked"` from the slot name in order to be compatible with the other versions



### Prepare data

```bash

# for mwoz22 no intents
DATASET=mwoz22
MODEL='google/t5-v1_1-base'

# natural language, full history (v0)
poetry run python scripts/prepare_data.py \
    tokenizer=$MODEL \
    preparator=dst \
    dataset_name=$DATASET \
    +serializers/state_serializer=natural_language \
    +serializers/history_serializer=zipped

# key-value pairs, full history (v1)
poetry run python scripts/prepare_data.py \
    tokenizer=$MODEL \
    preparator=dst \
    dataset_name=$DATASET \
    +serializers/state_serializer=natural_language \
    serializers.state_serializer.template='$slot : $value' \
    +serializers/history_serializer=zipped

# key-value pairs, full history, schema (v2)
poetry run python scripts/prepare_data.py \
    tokenizer=$MODEL \
    preparator=dst \
    dataset_name=$DATASET \
    +serializers/state_serializer=natural_language \
    serializers.state_serializer.template='$slot : $value' \
    +serializers/history_serializer=zipped \
    +serializers/all_slots_serializer=simple \
    preparator.input_serializer.template='schema: $all_slots $sep system: $sys_utt user: $usr_utt $sep history: $history'

# natural language with state changes, no history, previous state (v3)
poetry run python scripts/prepare_data.py \
    tokenizer=$MODEL \
    preparator=dst \
    dataset_name=$DATASET \
    +serializers/state_serializer=natural_language_state_changes \
    preparator.input_serializer.template='system: $sys_utt user: $usr_utt $sep previous state: $previous_states'

# natural language with state changes, no history, previous state, schema (v4)
poetry run python scripts/prepare_data.py \
    tokenizer=$MODEL \
    preparator=dst \
    dataset_name=$DATASET \
    +serializers/all_slots_serializer=simple \
    +serializers/state_serializer=natural_language_state_changes \
    preparator.input_serializer.template='schema: $all_slots system: $sys_utt user: $usr_utt $sep previous state: $previous_states'

MODEL='t5-base'
DATASET='mwoz22'
poetry run python scripts/prepare_data.py \
    tokenizer=$MODEL \
    preparator=dst \
    dataset_name=$DATASET \
    +serializers/all_slots_serializer=simple \
    +serializers/state_serializer=natural_language_state_changes \
    preparator.input_serializer.template='[schema] $all_slots [system] $sys_utt [user] $usr_utt [context] $previous_states'


# ========================

DATASET=mwoz22_active_only
MODEL='google/t5-v1_1-base'

# key-value pairs, full history (v0)
poetry run python scripts/prepare_data.py \
    tokenizer=$MODEL \
    preparator=dst \
    dataset_name=$DATASET \
    +serializers/state_serializer=natural_language \
    serializers.state_serializer.template='$slot : $value' \
    +serializers/history_serializer=zipped

# key-value pairs, full history, intents (v1)
poetry run python scripts/prepare_data.py \
    tokenizer=$MODEL \
    preparator=dst \
    dataset_name=$DATASET \
    +serializers/state_serializer=natural_language \
    serializers.state_serializer.template='$slot : $value' \
    +serializers/history_serializer=zipped \
    +serializers/intent_serializer=simple
```


## Train

```bash
DATASET='large/mwoz22_ops_partialhist+prev'

poetry run python scripts/train.py \
    experiment_group=mwoz22 \
    dataset_name=$DATASET \
    data.num_workers=64 \
    data.batch_size=1 \
    data.eval_batch_size=2 \
    optimizer.lr=1e-4 \
    trainer.devices=8 \
    trainer.accumulate_grad_batches=4 \
    trainer.strategy=ddp_find_unused_parameters_false \
    trainer.max_epochs=20 \
    +callbacks=model_checkpoint
```


### Train with multiple seeds

```bash
DATASET='large/mwoz22_ops_partialhist+prev_2022-12-09T15-27-03'

for seed in 0 1994 23 6006
do
    echo $seed

    poetry run python scripts/train.py \
        experiment_group=mwoz22 \
        dataset_name=$DATASET \
        data.batch_size=8 \
        data.eval_batch_size=16 \
        seed=$seed \
        optimizer.lr=1e-4 \
        trainer.devices='[6,7]' \
        trainer.accumulate_grad_batches=2 \
        trainer.strategy=ddp_find_unused_parameters_false \
        trainer.max_epochs=20 \
        +callbacks=model_checkpoint
done
```




## Infer

```bash
for split in "test" "validation"
do
    poetry run python ./scripts/infer.py \
        experiment_path='outputs/mwoz22_ops/mwoz22_ops_fullhist_2022-11-16T14-35-44' \
        device='0' \
        data_split=$split
done
```

poetry run python ./scripts/infer.py \
    experiment_path='/home/lescipi/t2t-dst/outputs/mwoz22_ops/mwoz22_ops_2022-11-17T13-58-22' \
    device='7' \
    batch_size=16 \
    version=1


## Utils

Copy only predictions files from S3

```bash
aws s3 cp --recursive s3://aws-comprehend-intern-data-us-east-1/lescipi/outputs/mwoz22/ ./preds --exclude '*' --include '*.parquet'

aws s3 cp --recursive s3://aws-comprehend-intern-data-us-east-1/lescipi/outputs/mwoz22/ ./preds --exclude '*' --include '*metadata.yaml'
```
