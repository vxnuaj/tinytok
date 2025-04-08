import torch
import pandas as pd
import pyarrow as pq

from torch.nn.utils.rnn import pad_sequence
from multiprocessing import Pool
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors
from typing import List, Union, Tuple
from tqdm import tqdm

def stream_parquet_texts(file_paths: List[str]):
    """
    Streams text data from multiple Parquet files. The function reads the "text" column from each file in batches.

    Parameters:
    -----------
    file_paths : List[str]
        A list of file paths to Parquet files containing a "text" column.

    Yields:
    -------
    str
        A text string from each row in the "text" column of the Parquet files.
    """
    for path in file_paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["text"], batch_size=10000):
            df = batch.to_pandas()
            yield from df['text'].tolist()
            
def data_process(files: list, eos_str: str = None, return_single_str: bool = False, return_list_str: bool = False, processes: int = 0) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str], List[str]]:
    """
    Processes the text data from a list of Parquet files, optionally adding an end-of-sequence string.

    Parameters:
    -----------
    files : list
        A list of file paths to the Parquet files.
    eos_str : str, optional (default=None)
        An optional string to append at the end of each sequence in the "text" column.
    return_single_str : bool, optional (default=False)
        If True, concatenates all sequences into a single string and returns it along with the DataFrame.
    return_list_str : bool, optional (default=False)
        If True, returns the "text" column as a list of strings.
    processes : int, optional (default=0)
        Number of processes to use for reading files. If 0, files are read in the main process.

    Returns:
    --------
    Union[pd.DataFrame, Tuple[pd.DataFrame, str], List[str]]
        A DataFrame with the processed data, or a tuple with the DataFrame and a single concatenated string, or a list of strings.
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
        
def train_new_tokenizer_bpe(data: List[str], vocab_size, special_tokens: list, save_path=None) -> Tokenizer:
    """
    Trains a new BPE tokenizer on the provided data.

    Parameters:
    -----------
    data : List[str]
        A list of string sequences to train the tokenizer.
    vocab_size : int
        The desired vocabulary size for the tokenizer.
    special_tokens : list
        A list of special tokens to be added to the tokenizer.
    save_path : str, optional (default=None)
        The path to save the trained tokenizer. If None, the tokenizer is not saved.

    Returns:
    --------
    Tokenizer
        The trained tokenizer.
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
    Tokenizes the text data in the DataFrame using the provided tokenizer.

    Parameters:
    -----------
    data : pd.DataFrame
        A DataFrame containing a "text" column to be tokenized.
    tokenizer : Tokenizer
        The tokenizer to use for tokenization.
    flat_tensor : bool, optional (default=True)
        If True, returns a flattened tensor. If False, returns a list of tensors.
    batch_size : int, optional (default=1000)
        The batch size for tokenizing the data.

    Returns:
    --------
    torch.Tensor | List[torch.Tensor]
        A flattened tensor of tokenized data (if `flat_tensor=True`), or a list of tensors (if `flat_tensor=False`).
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

def create_train_sequences_gen(data: torch.Tensor, context_len: int, seq_tensor_size: int = None, max_toks: int = None):
    """
    Generates sequences of tokens from the data for training. Each sequence is of length `context_len`.

    Parameters:
    -----------
    data : torch.Tensor
        A 1D tensor containing the tokenized data.
    context_len : int
        The length of the context window for each sequence.
    seq_tensor_size : int, optional (default=None)
        The number of sequences to yield in each batch.
    max_toks : int, optional (default=None)
        The maximum total number of tokens to use from the dataset.

    Yields:
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A batch of input and target tensors representing sequences.
    """
    assert max_toks is not None, 'max_toks must be assigned to an integer value, as the total target number of tokens for the entire dataset.'
    
    max_toks = min(max_toks, len(data))
    num_sequences = max_toks // context_len  # total number of sequences to be created, prior to step_size, assuming step_size = 1, given a max_toks
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
   
def create_val_sequences(data: List[torch.Tensor], batch_first: bool = True, padding_value: int = 0) -> torch.Tensor:
    """
    Stacks a list of variable-length tensors into a single padded 2D tensor for validation.

    Parameters:
    -----------
    data : List[torch.Tensor]
        A list of 1D tensors (sequences) to be padded and stacked.
    batch_first : bool, optional (default=True)
        If True, the resulting tensor will have shape (batch_size, max_seq_length).
        If False, it will have shape (max_seq_length, batch_size).
    padding_value : int, optional (default=0)
        The value to pad shorter sequences with.

    Returns:
    --------
    torch.Tensor
        A 2D tensor of padded sequences.
    """
    X = [seq[:-1] for seq in data]
    y = [seq[1:] for seq in data]
   
    X_padded = pad_sequence(
        sequences = X,
        batch_first = batch_first,
        padding_value = float(padding_value)
    )
    
    y_padded = pad_sequence(
        sequences = y,
        batch_first=batch_first,
        padding_value = float(padding_value)
    )

    return X_padded, y_padded
