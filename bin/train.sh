# Experiment 1: MultiWoz v2.2
for dataset in 'base/mwoz22_cum_fullhist+nostate' 'base/mwoz22_cum_fullhist+prev' 'base/mwoz22_cum_nohist+prev' 'base/mwoz22_cum_partialhist+prev' 'base/mwoz22_ops_fullhist+nostate' 'base/mwoz22_ops_fullhist+prev' 'base/mwoz22_ops_nohist+prev' 'base/mwoz22_ops_partialhist+prev'; do
    for seed in 42 0 1994 23 6006; do
        echo Running experiment $dataset with seed=$seed

        poetry run python scripts/train.py \
            experiment_group=$EXPERIMENT_GROUP \
            dataset_name=$dataset \
            data.batch_size=8 \
            data.eval_batch_size=16 \
            task=dst \
            optimizer=adafactor \
            optimizer.lr=1e-4 \
            trainer.devices=4 \
            trainer.accumulate_grad_batches=1 \
            trainer.max_epochs=20 \
            trainer.strategy=ddp_find_unused_parameters_false \
            +callbacks=model_checkpoint \
            +loggers=wandb
    done
done


# Experiment 2: MultiWoz v2.1
for dataset in 'base/mwoz21_ops_nohist+prev' 'base/mwoz21_ops_partialhist+prev' 'base/mwoz21_cum_fullhist+nostate'; do
    for seed in 42 0 1994 23 6006; do
        echo Running experiment $dataset with seed=$seed

        poetry run python scripts/train.py \
            experiment_group=experiment_2 \
            dataset_name=$dataset \
            data.batch_size=8 \
            data.eval_batch_size=16 \
            seed=$seed \
            optimizer.lr=1e-4 \
            trainer.devices=4 \
            trainer.accumulate_grad_batches=1 \
            trainer.strategy=ddp_find_unused_parameters_false \
            trainer.max_epochs=20 \
            +callbacks=model_checkpoint \
            +loggers=wandb
    done
done