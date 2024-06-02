from tqdm import tqdm

from src.modules.templates import *
from datasets import concatenate_datasets, Dataset


# note that we assume pretraining dataset has no loss mask
# def load_generator(path):
#     with open(path, "r") as f:
#         for line in f:
#             yield {"input_ids": json.loads(line)}
#
# gen_kwargs = {"path": configs.data.input_data_fn}

# dataset = read_dataset_to_hf(load_generator, gen_kwargs=gen_kwargs).shuffle(seed=configs.seed).select(range(configs.data.num_data_examples))

def select_binary_balanced_dataset(hf_dataset, binary_eval_func, seed, num_examples_per_class):
    """ returns a set of examples that are balanced according to the eval_func"""
    false_dataset = hf_dataset.filter(lambda x: not binary_eval_func(x)).shuffle(seed=seed).select(range(num_examples_per_class))
    true_dataset = hf_dataset.filter(binary_eval_func).shuffle(seed=seed).select(range(num_examples_per_class))
    return concatenate_datasets([false_dataset, true_dataset]).shuffle(seed=seed)

def partition_dataset(hf_dataset, partitions):
    """
    partitions the dataset into the given splits
    :param hf_dataset: the dataset
    :param partitions: a dictionary of the form {"split_name": num_examples}
    """
    if (sum(partitions.values()) > len(hf_dataset)):
        raise Exception(f"sum of partitions {sum(partitions.values())} is greater than dataset size {len(hf_dataset)}")
    dataset_splits = {}
    current_total = 0
    for key, value in partitions.items():
        dataset_splits[key] = hf_dataset.select(range(current_total, current_total + value))
        current_total += value
    return dataset_splits


# performs concatenation of each line in dataset similar to pretraining
def format_to_pretraining(hf_dataset, tokenizer, max_seq_len):
    """
    Assumes the dataset is already tokenized. Assumes there is an "input_ids" field
    :param hf_dataset: the dataset. Assumes "input_ids" to be in the dataset.
    :param tokenizer: the tokenizer for the eos token
    :param max_seq_len: each sequence is concatenated until this length
    :return:
    """
    assert(tokenizer.eos_token_id is not None)

    def format_generator(dataset, tqdm_object):
        current_dict = {k: [] for k in dataset.column_names}
        for example in dataset:
            tqdm_object.update(1)
            for k in dataset.column_names:
                current_dict[k] += example[k]
                if (k == "input_ids"):
                    current_dict[k] += [tokenizer.eos_token_id]
                elif (k == "loss_mask" or k == "attention_mask"):
                    current_dict[k] += [1]
                else:
                    raise Exception(f"Unknown column name {k}")

            if (len(current_dict["input_ids"]) >= max_seq_len):
                for k, v in current_dict.items():
                    if len(v) > max_seq_len:
                        current_dict[k] = v[:max_seq_len]

                yield current_dict
                current_dict = {k: [] for k in dataset.column_names}

    tqdm_object = tqdm(total=len(hf_dataset))
    processed_dataset = [x for x in format_generator(hf_dataset, tqdm_object)]

    formatted_dataset = Dataset.from_list(processed_dataset)

    return formatted_dataset



def tokenize_input_output_pair(tokenizer, input, output):
    """
    Tokenizes the input and output pair with the correct spacing.
    :param tokenizer: a hf tokenizer
    :param input: str
    :param output: str
    :return: tokenized input and output
    """
    if "olmo" in tokenizer.name_or_path.lower():
        if input[-1] != " ":
            raise Exception("Input expected to end with a space, got " + input[-1] + " instead")
        if output[0] == " ":
            raise Exception("Output expected to NOT start with a space, got space instead")
        input = input[:-1]
        output = " " + output
        input_tokens = tokenizer.encode(input, add_special_tokens=False)
        output_tokens = tokenizer.encode(output, add_special_tokens=False)
        return input_tokens, output_tokens
    else:
        raise Exception(f"Tokenizer {tokenizer.name_or_path} not supported")



