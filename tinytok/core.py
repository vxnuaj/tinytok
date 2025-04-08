from typing import List, Union, Tuple
import pandas as pd
import pyarrow.parquet as pq
from multiprocessing import Pool
from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers
import torch

def stream_parquet_texts(file_paths: List[str]):
    """
    Generator function that streams individual text strings from Parquet files.

    This function reads Parquet files from a list of file paths, processes them in batches of 10,000 rows,
    extracts the 'text' column, and yields each text string individually. It is designed for memory-efficient
    handling of large datasets by streaming data one text at a time.

    Args:
        file_paths (List[str]): A list of file paths to Parquet files containing a 'text' column.

    Yields:
        str: Individual text strings extracted from the 'text' column of the Parquet files.
    """
    for path in file_paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["text"], batch_size=10000):
            df = batch.to_pandas()
            yield from df['text'].tolist()

def data_process(files: list, eos_str: str = None, return_single_str: bool = False, return_list_str: bool = False, processes: int = 0) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str], List[str]]:
    """
    Processes a list of Parquet files and returns the data in various formats.

    Reads Parquet files into pandas DataFrames, optionally appends an end-of-sequence string to each text,
    and returns the data as a DataFrame, a tuple of DataFrame and concatenated string, or a list of strings,
    depending on the parameters. Supports parallel processing for faster reading.

    Args:
        files (list): A list of file paths to Parquet files.
        eos_str (str, optional): String to append to each text sequence. Defaults to None.
        return_single_str (bool, optional): If True, returns a tuple of DataFrame and a single concatenated string. Defaults to False.
        return_list_str (bool, optional): If True, returns a list of text strings. Defaults to False.
        processes (int, optional): Number of processes for parallel reading. If 0, reads sequentially. Defaults to 0.

    Returns:
        Union[pd.DataFrame, Tuple[pd.DataFrame, str], List[str]]: Processed data in the specified format:
            - pd.DataFrame: If no return flags are set.
            - Tuple[pd.DataFrame, str]: If return_single_str is True.
            - List[str]: If return_list_str is True.
    """
    tqdm.pandas()
    if processes > 0:
        with Pool(processes=processes) as pool:
            dfs = list(tqdm(pool.map(pd.read_parquet, files), total=len(files), desc="Reading Files"))
    else:
        dfs = [pd.read_parquet(f_path) for f_path in tqdm(files, desc="Reading Files")]
    data = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    if eos_str:
        print("Adding EOS string to every sequence")
        data['text'] = data['text'] + eos_str
    if return_list_str:
        print(f"Returning list of {len(data)} strings")
        return data['text'].tolist()
    if return_single_str:
        print(f"Concatenating {len(data)} sequences into a single string and returning")
        return data, "".join(data['text'])
    return data

def train_new_tokenizer_bpe(data: List[str], vocab_size, special_tokens, save_path=None) -> Tokenizer:
    """
    Trains a new Byte Pair Encoding (BPE) tokenizer.

    Initializes and trains a BPE tokenizer using the Hugging Face Tokenizers library with the provided dataset,
    vocabulary size, and special tokens. Configures the tokenizer with ByteLevel pre-tokenizer, decoder, and
    post-processor. Optionally saves the trained tokenizer to a file.

    Args:
        data (List[str]): List of text strings to train the tokenizer on.
        vocab_size (int): Desired size of the tokenizer's vocabulary.
        special_tokens (list): List of special tokens to include in the vocabulary.
        save_path (str, optional): Path to save the trained tokenizer. If None, the tokenizer is not saved. Defaults to None.

    Returns:
        Tokenizer: The trained BPE tokenizer object.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, show_progress=True)
    tokenizer.train_from_iterator(data, trainer=trainer)
    if save_path:
        tokenizer.save(save_path)
    return tokenizer

def tokenize(data: pd.DataFrame, tokenizer, flat_tensor: bool = True, batch_size: int = 1000) -> torch.Tensor | List[torch.Tensor]:
    """
    Tokenizes text data from a DataFrame using a provided tokenizer.

    Processes the text data in batches to manage large datasets efficiently. Returns either a single flattened
    tensor of all token IDs or a list of tensors, each representing a tokenized sequence, based on the flat_tensor parameter.

    Args:
        data (pd.DataFrame): DataFrame with a 'text' column containing the text to tokenize.
        tokenizer (Tokenizer): Trained tokenizer object to encode the text.
        flat_tensor (bool, optional): If True, returns a single flattened tensor of token IDs. If False, returns a list of tensors. Defaults to True.
        batch_size (int, optional): Number of text sequences to process per batch. Defaults to 1000.

    Returns:
        torch.Tensor | List[torch.Tensor]: Tokenized data:
            - torch.Tensor: A flattened tensor of all token IDs if flat_tensor is True.
            - List[torch.Tensor]: A list of tensors, each representing a sequence, if flat_tensor is False.
    """
    print(f"Tokenizing {len(data)} strings")
    token_ids = []
    for i in tqdm(range(0, len(data), batch_size), desc="Tokenizing"):
        batch = data['text'].iloc[i:i+batch_size].tolist()
        encodings = tokenizer.encode_batch(batch)
        token_ids.extend(enc.ids for enc in encodings)
    if flat_tensor:
        total_tokens = sum(len(ids) for ids in token_ids)
        data_flat = torch.zeros(total_tokens, dtype=torch.long)
        offset = 0
        for ids in tqdm(token_ids, desc='Processing Tensors'):
            data_flat[offset:offset+len(ids)] = torch.tensor(ids, dtype=torch.long)
            offset += len(ids)
        return data_flat
    return [torch.tensor(ids) for ids in token_ids]

def create_sequences_gen(data: torch.Tensor, context_len: int, seq_tensor_size: int = None, max_toks: int = None):
    """
    Generates input and target sequences for language model training.

    Creates sequences of a specified context length from a tensor of token IDs. Can either yield batches of sequences
    of size seq_tensor_size or return all sequences at once if seq_tensor_size is None. The number of sequences is
    determined by max_toks and context_len, with sequences spaced evenly across the data.

    Args:
        data (torch.Tensor): 1D tensor of token IDs.
        context_len (int): Length of the context window for each sequence.
        seq_tensor_size (int, optional): Number of sequences per yielded tensor. If None, returns all sequences at once. Defaults to None.
        max_toks (int, optional): Maximum number of tokens to use for sequences. If None, uses all data. Defaults to None.

    Yields or Returns:
        Tuple[torch.Tensor, torch.Tensor]: Depending on seq_tensor_size:
            - Yields tuples of (X, y) tensors, where X is input sequences and y is target sequences, if seq_tensor_size is an int.
            - Returns a single tuple of (X, y) tensors containing all sequences if seq_tensor_size is None.
    """
    max_toks = min(max_toks, len(data)) if max_toks is not None else len(data)
    num_sequences = max_toks // context_len
    step_size = (len(data) - context_len) // num_sequences
    print(f"Creating {num_sequences} sequences with a step size of {step_size}")
    X_tnsr, y_tnsr = [], []
    if isinstance(seq_tensor_size, int):
        for i in tqdm(range(num_sequences), desc=f"Creating {num_sequences} sequences"):
            start_idx = i * step_size
            if start_idx + context_len + 1 > len(data):
                break
            X_tnsr.append(data[start_idx:start_idx + context_len])
            y_tnsr.append(data[start_idx + 1:start_idx + context_len + 1])
            if len(X_tnsr) == seq_tensor_size:
                yield torch.stack(X_tnsr, dim=0), torch.stack(y_tnsr, dim=0)
                X_tnsr, y_tnsr = [], []
        if X_tnsr:
            yield torch.stack(X_tnsr, dim=0), torch.stack(y_tnsr, dim=0)
    else:
        for i in tqdm(range(num_sequences), desc=f"Creating {num_sequences} sequences"):
            start_idx = i * step_size
            if start_idx + context_len + 1 > len(data):
                break
            X_tnsr.append(data[start_idx:start_idx+context_len])
            y_tnsr.append(data[start_idx+1:start_idx+context_len+1])
        return torch.stack(X_tnsr, dim=0), torch.stack(y_tnsr, dim=0)