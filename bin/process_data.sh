# ============
# Process data
# ============


set -e


# MultiWoz 2.1
echo 'Processing MultiWoz v2.1'
poetry run python ./scripts/process_data.py \
    --input_filepath './data/raw/multiwoz_21/data.json' \
    --output_dir './data/processed/multiwoz_21' \
    --val_split_filepath './data/raw/multiwoz_21/valListFile.txt' \
    --test_split_filepath './data/raw/multiwoz_21/testListFile.txt' \
    --format '2.1'


# MultiWoz 2.2
echo 'Processing MultiWoz v2.2'
poetry run python ./scripts/process_data.py \
    --input_filepath './data/raw/multiwoz_22' \
    --output_dir './data/processed/multiwoz_22' \
    --format '2.2'


# MultiWoz 2.3 (does not have split files)
echo 'Processing MultiWoz v2.3'
poetry run python ./scripts/process_data.py \
    --input_filepath './data/raw/multiwoz_23/data.json' \
    --output_dir './data/processed/multiwoz_23' \
    --val_split_filepath './data/raw/multiwoz_21/valListFile.txt' \
    --test_split_filepath './data/raw/multiwoz_21/testListFile.txt' \
    --format '2.1'


# MultiWoz 2.4
echo 'Processing MultiWoz v2.4'
poetry run python ./scripts/process_data.py \
    --input_filepath './data/raw/multiwoz_24/data.json' \
    --output_dir './data/processed/multiwoz_24' \
    --val_split_filepath './data/raw/multiwoz_24/valListFile.json' \
    --test_split_filepath './data/raw/multiwoz_24/testListFile.json' \
    --format '2.1'