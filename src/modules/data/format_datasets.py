# this file is responsible for reformatting the datasets into a unified format
from src.modules.data.format_utils import reformat_dialogue_with_template, select_binary_balanced_dataset, \
    partition_dataset, preprocess_conversation, tokenize_input_output_pair

from src.modules.data.load import read_dataset_to_hf
from src.modules.templates import *


def reformat_realtoxicity_prompts_for_inferencing(dataset):
    """
    Dataset({
        features: ['filename', 'begin', 'end', 'challenging', 'prompt', 'continuation'],
        num_rows: 1000
    })
    """

    def reformat_row(row):
        return {"prompt": row["prompt"]["text"]}

    return dataset.map(reformat_row, batched=False)


def load_and_reformat_dataset(dataset_name, dataset_file, splits, seed, num_proc=1, tokenizer=None, max_seq_len=None, use_loss_mask=False, **kwargs):
    """Load and reformat a dataset. If training or evaluation dataset, we also do tokenization. Else we just load and reformat
    params:
    dataset_name: str, the name of the dataset
    dataset_file: str, the path to the dataset file
    splits: dict, a dictionary of splits
    seed: int, seed for shuffling
    tokenizer: tokenizer, the tokenizer to use
    kwargs: dict, additional arguments"""

    if (dataset_name == "real-toxicity-prompts"):
        # This is a generation dataset, so we select num_generate_examples examples without much reformatting
        if "generation" not in splits:
            raise Exception("real toxicity prompts currently only supports generation")

        generation_dataset = read_dataset_to_hf(dataset_file)["train"].shuffle(seed=seed)

        generation_dataset = generation_dataset.select(range(splits["generation"]))
        return {"generation": reformat_realtoxicity_prompts_for_inferencing(generation_dataset)}

    else:
        raise ValueError(f"Unknown dataset: {dataset_file}")

