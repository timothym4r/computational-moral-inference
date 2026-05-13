import torch
import numpy as np


def add_special_tokens_to_tokenizer(tokenizer, special_tokens):
    to_add = [t for t in special_tokens if t not in tokenizer.get_vocab()]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    return len(to_add)


def get_type_token_suffix(add_type_tokens):
    return "typed" if add_type_tokens else "untyped"


def build_processed_data_stem(pooling_method, threshold, sentence_mask_type=None, model_name="bert-base-uncased", add_type_tokens=True):
    stem = f"{pooling_method}_{threshold}_{get_type_token_suffix(add_type_tokens)}"
    if sentence_mask_type is not None:
        stem = f"{stem}_{sentence_mask_type}"
    return stem


def build_processed_data_paths(output_dir, pooling_method, threshold, sentence_mask_type=None, model_name="bert-base-uncased", add_type_tokens=True):
    stem = build_processed_data_stem(
        pooling_method=pooling_method,
        threshold=threshold,
        sentence_mask_type=sentence_mask_type,
        model_name=model_name,
        add_type_tokens=add_type_tokens
    )
    train_path = f"{output_dir}/train_data_{stem}.json"
    test_path = f"{output_dir}/test_data_{stem}.json"
    return train_path, test_path


def build_char_cache_dir(output_dir, pooling_method, model_name="bert-base-uncased", add_type_tokens=True):
    return f"{output_dir}/char_cache_{pooling_method}_{get_type_token_suffix(add_type_tokens)}"

def normalize_mask_token(data, tokenizer):
    """
    Replace [MASK] with the tokenizer's mask token in the masked sentences.

    Currently, it is assumed that the masked token in the data is represented as [MASK].
    """

    ms = tokenizer.mask_token
    for row in data:
        row["masked_sentence"] = (
            row["masked_sentence"]
            .replace("[MASK]", ms)
        )
    return data


def exponential_smoothing(data, alpha=0.3):
    """
    Apply exponential smoothing to a 2D numpy array.
    """
    smoothed_data = np.zeros_like(data)
    smoothed_data[0] = data[0]
    
    for t in range(1, len(data)):
        smoothed_data[t] = alpha * data[t] + (1 - alpha) * smoothed_data[t - 1]
    
    return smoothed_data

def moving_average(data, window_size=5):
    pass
