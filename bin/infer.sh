DEVICE=$1
BATCH_SIZE=$2
MAX_LENGTH=$3

for run in /home/lescipi/t2t-dst/outputs/*
do
    echo Experiment: $run -- Device: $DEVICE, Batch size: $BATCH_SIZE, max_length: $MAX_LENGTH
    poetry run python ./scripts/infer.py \
        experiment_path=$run \
        device=$DEVICE \
        batch_size=$BATCH_SIZE \
        generation_kwargs.max_new_tokens=$MAX_LENGTH
done