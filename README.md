# tinytok

> DISCLAIMER: This README.md was written by GPT

Simple utility funcs to process TinyStories by Eldan & Li, train a Byte-Pair Encoding (BPE) tokenizer, and create tokenized sequences for transformer models.

Primarily made for personal use.

## Features

- Read and concatenate `.parquet` text datasets
- Optionally append EOS tokens and return raw text
- Train a new BPE tokenizer with `tokenizers` library
- Tokenize using the trained tokenizer into PyTorch tensors
- Generate sequences for transformer model training

## Installation

```bash
pip install tinytok
```

## Example Usage

```python
import torch
from utils import data_process, tokenize, train_new_tokenizer_bpe, create_sequences

model_tokenizer_name = 'EleutherAI/gpt-neo-1.3B'

file_1 = 'data/train1.parquet'
file_2 = 'data/train2.parquet'
file_3 = 'data/train3.parquet'
file_4 = 'data/train4.parquet'
file_val = 'data/validation.parquet'

#files = [file_1, file_2, file_3, file_4]
files = [file_1]
file_val = [file_val]

# PARAMS -----------------
 
return_single_str = True
vocab_size = 10000
special_tokens = ['<|endoftext|>']
save_path = 'data/tokenizer.json'
return_freqs = False
return_flat_tnsr = True
create_train_test = True
context_len = 512
processes = 4

if __name__ == "__main__":

    data, data_str = data_process(
        files, 
        eos_str = special_tokens[0],
        return_single_str = return_single_str,
        processes = processes
        ) # data.shape -> (2119719, 1)

    tokenizer = train_new_tokenizer_bpe(
        data = data_str,
        vocab_size = vocab_size,
        special_tokens = special_tokens,
        save_path = save_path
    ) # tokenizer object

    data_tensor = tokenize(
        data = data,
        tokenizer = tokenizer,
        flat_tensor = True
    ) # List[torch.Tensor]

    X_train, y_train = create_sequences(
        data_tensor = data_tensor, 
        context_len = context_len,
        create_train_test = create_train_test,
        )

    torch.save(X_train, f = 'data/tensors/X_train')
    torch.save(y_train, f = 'data/tensors/y_train')

    data, data_str = data_process(
        files, 
        eos_str = '<|endoftext|>',
        return_single_str = return_single_str
        )  

    data_tensor = tokenize(
        data = data,
        tokenizer = tokenizer
    )

    X_val, y_val = create_sequences(
        data_tensor = data_tensor, 
        context_len = context_len,
        create_train_test = create_train_test,
        )

    torch.save(X_val, f = 'data/tensors/X_val')
    torch.save(y_val, f = 'data/tensors/y_val')
```

## Requirements

- torch
- pandas
- tqdm
- tokenizers