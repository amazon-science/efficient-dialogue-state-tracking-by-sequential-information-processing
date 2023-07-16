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
    abstract = "Sequence-to-sequence state-of-the-art systems for dialogue state tracking (DST) use the full dialogue history as input, represent the current state as a list with all the slots, and generate the entire state from scratch at each dialogue turn. This approach is inefficient, especially when the number of slots is large and the conversation is long. In this paper, we propose Diable, a new task formalisation that simplifies the design and implementation of efficient DST systems and allows one to easily plug and play large language models. We represent the dialogue state as a table and formalise DST as a table manipulation task. At each turn, the system updates the previous state by generating table operations based on the dialogue context. Extensive experimentation on the MultiWoz datasets demonstrates that Diable (i) outperforms strong efficient DST baselines, (ii) is 2.4x more time efficient than current state-of-the-art methods while retaining competitive Joint Goal Accuracy, and (iii) is robust to noisy data annotations due to the table operations approach.",
}
```

## Get started

We use conda and poetry for environment management. Then, we download, process, and prepare data for training.

**Structure:** In `./bin` we provide shell scripts that call the respective python scripts from the `./scripts` folder. Python scripts depend on `./src`.

### Setup the environment

Install miniconda

```bash
curl -sL \
  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
  "Miniconda3.sh"

bash Miniconda3.sh -b -p

rm Miniconda3.sh

source $HOME/miniconda3/bin/activate

conda init zsh
```

and poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

poetry might ask you to add this line to your `~/.bashrc` file

```bash
export PATH="/home/<username>/.local/bin:$PATH"
```

### Install dependencies
Use poetry to install the environment (if you don't have poetry run )

```bash
conda create -n diable python=3.9 -y
conda activate diable
poetry install --sync
```

## Data preparation



### Download data

The following script downloads MultiWoz 2.1-2.4

- 2.1 and 2.2 from [https://github.com/budzianowski/multiwoz.git](https://github.com/budzianowski/multiwoz.git)

- 2.3 from [https://github.com/lexmen318/MultiWOZ-coref/raw/main/MultiWOZ2_3.zip](https://github.com/lexmen318/MultiWOZ-coref/raw/main/MultiWOZ2_3.zip)

- 2.4 from [https://github.com/smartyfh/MultiWOZ2.4/raw/main/data/MULTIWOZ2.4.zip](https://github.com/smartyfh/MultiWOZ2.4/raw/main/data/MULTIWOZ2.4.zip)

```bash
./bin/download_data.sh
```

The resulting data are available under the `./data/raw` folder.


### Process data

The following script processed the downloaded from the `./data/raw` folder, and
saves them as HuggingFace `DatasetDict`s into the `./data/processed` folder. 

```bash
./bin/process_data.sh
```

It performs the following processing:

1. Extracts the data into a standardized data type
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

1. Generates the `ontology.yaml` and `schema.yaml` files. The ontology is dependent on the data split (i.e., train/validation/test have a different ontologies). For SGD the schema follows the same structure since it changes across data splits. However, for MultiWoz we only save one schema since it is constant across splits

1. Specifically for MultiWoz

    - Removes slots in the `("bus", "police", "hospital")` domains

    - Removes entire dialogues in case the only discussed domains are `("bus", "police", "hospital")`

    - Removes the values `("not mentioned", "none")`

    - Removes invalid dialogues `("SNG01862.json",)`

    - For v2.2 removes `"booked"` from the slot name in order to be compatible with the other versions


### Prepare data

In this step, we prepare the data for training. In this step we basically apply the different templates/prompts and tokenize the data. The resulting prepared datasets are saved in the `./data/prepared` folder. For this project, the templates are called "serializers".

**NOTE:** we use [hydra](https://hydra.cc/) for configuring data preparation and experiments.

For example, let's say we want to train a full history model that uses only the previous state to predict the cumulative state on MultiWoz v2.2, then we run

```bash
echo 'Previous state -> State operations'

poetry run python ./scripts/prepare_data.py \
    history_serializer.lag=0 \
    state_serializer=key_value_state_changes \
    input_template='generate update operations: dialogue: system: $sys_utt user: $usr_utt $sep previous dialogue states: $previous_states $sep operations:' \
    target_template='$states' \
    tokenizer='google/t5-v1_1-base' \
    input_dir='./data/processed/multiwoz_22' \
    output_dir='./data/prepared/base/mwoz22_ops_nohist+prev'

```

More examples are contained in the `./bin/prepare_data.sh` file.


## Run experiments

### Train

Once you prepare the data, you can run an experiment by simply passing in the name of the prepared dataset. You can also edit any training setting. We use the [Pytorch-Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html), so refer to their documentation for more info about the settings.

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

In `./bin/train.sh` we provide some examples of training commands. The trained model and the logs will be saved into the `./outputs` folder.


### Infer

Once you have a trained model, you can use it for text generation by simply pointing to the experiment output folder. For example, to run inference with a specific checkpoint, you can run

```bash
for split in "test" "validation"
do
    poetry run python ./scripts/infer.py \
        experiment_path='outputs/mwoz22_ops/mwoz22_ops_fullhist_2022-11-16T14-35-44' \
        device='0' \
        data_split=$split
done
```

and it wil automatically pick up the right dataset to use from the experiment logs. 

**NOTE:** the code supports inference only on a single GPU.