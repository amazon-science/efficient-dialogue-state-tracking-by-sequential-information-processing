import argparse

from src.data.dataset import MWoz22Dataset, MWozDataset, SchemaGuidedDialogueDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filepath", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--val_split_filepath", type=str, default=None)
    parser.add_argument("--test_split_filepath", type=str, default=None)
    parser.add_argument("--format", type=str, default=None)
    args = parser.parse_args()

    if args.format == "SGD":
        dataset = SchemaGuidedDialogueDataset(args.input_filepath)
        dataset.setup()

    elif args.format == "2.2":
        dataset = MWoz22Dataset(args.input_filepath)
        dataset.setup()

    else:
        dataset = MWozDataset(args.input_filepath)
        dataset.setup()
        dataset.add_splits(
            validation_path=args.val_split_filepath,
            test_path=args.test_split_filepath,
        )

    dataset.build_ontology_and_schema()
    dataset.save_to_disk(args.output_dir)

    line = "=" * 70
    print(f"\n{line}\nDataset {args.output_dir} has {len(dataset.dialogues)} dialogues\n{line}\n")
