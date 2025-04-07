import torch
import pandas as pd

from multiprocessing import Pool
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors
from typing import List, Union, Tuple
from tqdm import tqdm

def data_process(files: list, eos_str: str = None, return_single_str: bool = False, return_list_str: bool = False, processes: int = 0) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str], List[str]]:
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
    '''
    Trains a new BPE Tokenizer
    data: entire dataset as a list of string sequences
    vocab_size: the vocabulary size
    special_tokens: a list of special tokens
    save_path: the path to save the tokenizer, if None the tokenizer will not save
    '''
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, show_progress=True)
    tokenizer.train_from_iterator(data, trainer=trainer)
    if save_path:
        tokenizer.save(save_path)
    return tokenizer

def tokenize(data: pd.DataFrame, tokenizer, flat_tensor: bool = True) -> torch.Tensor | List[torch.Tensor]:
    '''
    flat_tensor: if True, returns a flattened tensor.
    ''' 
    print(f"Tokenizing {len(data)} strings")
    token_lists = tokenizer.encode_batch(data['text'].tolist())
    token_ids = [enc.input_ids for enc in token_lists]
    if flat_tensor:
        total_tokens = sum(len(ids) for ids in token_ids)
        data_flat = torch.zeros(total_tokens, dtype=torch.long)
        offset = 0
        for ids in tqdm(token_ids, desc='Processing Tensors'):
            data_flat[offset:offset+len(ids)] = torch.tensor(ids)
            offset += len(ids)
        return data_flat
    return [torch.tensor(ids) for ids in token_ids]

def create_sequences(data:torch.Tensor, context_len:int, chunk_size:int):
    '''
    data: a 1D torch.Tensor of the tokenized dataset
    '''
    num_sequences = len(data) - context_len
    X_chunks, y_chunks = [], [] 
    for start in tqdm(range(start = 0, stop = num_sequences, step = chunk_size), desc = f"Creating {num_sequences} sequences"):
        end = min(start + chunk_size, num_sequences)
        indices = torch.arange(start, end)
        X_chunks.append(data[indices[:] + torch.arange(context_len)])
        y_chunks.append(data[indices[:] + torch.arange(1, context_len+1)]) 
    X = torch.cat(X_chunks)
    y = torch.cat(y_chunks) 
    return X, y