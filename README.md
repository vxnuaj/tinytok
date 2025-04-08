# tinytok

> DISCLAIMER: This README.md was written by ~~GPT~~ Grok | The docstrings for the functions were written by ~~GPT~~ Grok.

Simple utility funcs to process TinyStories by Eldan & Li, train a Byte-Pair Encoding (BPE) tokenizer, and create tokenized sequences to train tiny transformer models.

Primarily made for personal use.

## Features

- Read and concatenate `.parquet` text datasets
- Optionally append EOS tokens and return raw text
- Train a new BPE tokenizer with `tokenizers` library
- Tokenize using the trained tokenizer into PyTorch tensors
- Generate sequences for transformer model training

## Installation

```bash
pip install tinytok==0.1.0
```

## Example Usage

```python
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tinytok import data_process, tokenize, train_new_tokenizer_bpe, create_val_sequences, create_train_sequences_gen

file_1 = 'data/train1.parquet'
file_2 = 'data/train2.parquet'
file_3 = 'data/train3.parquet'
file_4 = 'data/train4.parquet'
file_val = 'data/validation.parquet'

file_train = [file_1, file_2, file_3, file_4]
file_val = [file_val]

# PARAMS -----------------
return_single_str = False
vocab_size = 10_000
special_tokens = {'eos': '<|endoftext|>', 'pad': ' '}
save_tokenizer_path = 'data/tokenizer.json'
context_len = 512
processes = 4
flat_tensor = True
flat_tensor_val = False
seq_tensor_size = 25_000
val_seq_tensor_size = None
max_toks = 350_000_000  
val_max_toks = None
batch_first = True

X_train_pth = 'data/tensors/train/X'
y_train_pth = 'data/tensors/train/y'
val_pth = 'data/tensors/val'

if __name__ == "__main__":
    os.makedirs(X_train_pth, exist_ok=True)
    os.makedirs(y_train_pth, exist_ok=True)
    os.makedirs(val_pth, exist_ok=True)
    
    data = data_process(
        files=file_train,
        eos_str=special_tokens['eos'],
        return_single_str=return_single_str,
        processes=processes
    )

    tokenizer = train_new_tokenizer_bpe(
        data=data['text'].tolist(),
        vocab_size=vocab_size,
        special_tokens=list(special_tokens.values()),
        save_path=save_tokenizer_path
    )
    
    data_tensor = tokenize(
        data=data,
        tokenizer=tokenizer,
        flat_tensor=flat_tensor,
        processes=processes
    )

    if isinstance(seq_tensor_size, int):
        sequence_generator = create_train_sequences_gen(
            data=data_tensor,
            context_len=context_len,
            seq_tensor_size=seq_tensor_size,
            max_toks=max_toks,
            processes=processes
        )
        for i, (X, y) in enumerate(sequence_generator):
            torch.save(X, os.path.join(X_train_pth, f'X_train_{i}.pt'))
            torch.save(y, os.path.join(y_train_pth, f'y_train_{i}.pt'))
            # if i == 10:
            #     sys.exit(0)
    elif not seq_tensor_size:
        X_train, y_train = create_train_sequences_gen(
            data=data_tensor,
            context_len=context_len,
            seq_tensor_size=seq_tensor_size,
            max_toks=max_toks,
            processes=processes
        )
        torch.save(X_train, os.path.join(X_train_pth, "X_train.pt"))
        torch.save(y_train, os.path.join(y_train_pth, "y_train.pt"))
        del X_train, y_train

    # Validation Data
    data = data_process(
        files=file_val,
        eos_str=special_tokens['eos'],
        return_single_str=return_single_str,
        processes=processes
    )

    data_tensor = tokenize(
        data=data,
        tokenizer=tokenizer,
        flat_tensor=flat_tensor_val,
        processes=processes
    )

    X_val, Y_val = create_val_sequences(
        data=data_tensor,
        batch_first=batch_first,
        padding_value=tokenizer.encode(special_tokens['pad']).ids[0]
    )
    
    torch.save(X_val, os.path.join(val_pth, 'X_val.pt'))
    torch.save(Y_val, os.path.join(val_pth, 'Y_val.pt'))
```

## Requirements

- torch
- pandas
- tqdm
- tokenizers