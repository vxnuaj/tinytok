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
import sys

from tinytok import data_process, tokenize, train_new_tokenizer_bpe, create_sequences_gen

file_1 = 'data/train1.parquet'
file_2 = 'data/train2.parquet'
file_3 = 'data/train3.parquet'
file_4 = 'data/train4.parquet'
file_val = 'data/validation.parquet'

file_train = [file_1, file_2, file_3,  file_4]
file_val = [file_val]

# PARAMS -----------------
 
return_single_str = False
vocab_size = 10_000
special_tokens = ['<|endoftext|>']
save_path = 'data/tokenizer.json'
context_len = 512
processes = 4
flat_tensor = True
seq_tensor_size = 25_000 # NOTE -- fiddle with this if u are RAM poor, lol.
max_toks = 350_000_000 # NOTE -- modify per the size of the model, following chincilla scaling laws, where num_params * 20 = ideal_training_tok_size

X_train_pth = 'data/tensors/train/X'
y_train_pth = 'data/tensors/train/y'
X_val_pth = 'data/tensors/val/X'
y_val_pth = 'data/tensors/val/y'

if __name__ == "__main__":
   
    os.makedirs(X_train_pth, exist_ok=True)
    os.makedirs(y_train_pth, exist_ok=True)
    os.makedirs(X_val_pth, exist_ok=True)
    os.makedirs(y_val_pth, exist_ok=True)
    
    data = data_process(
        files = file_train,
        eos_str=special_tokens[0],
        return_single_str=return_single_str,
        processes=processes
    )

    tokenizer = train_new_tokenizer_bpe(
        data=data['text'].tolist(),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        save_path=save_path
    )
    
    data_tensor = tokenize(
        data=data,
        tokenizer=tokenizer,
        flat_tensor=flat_tensor
    ) # returns a 1D tensor

    if isinstance(seq_tensor_size, int):
        sequence_generator = create_sequences_gen(
                data = data_tensor,
                context_len = context_len,
                seq_tensor_size = seq_tensor_size,
                max_toks = max_toks
            )       
        for i, (X, y) in enumerate(sequence_generator):
            torch.save(X, f = os.path.join(X_train_pth, f'X_train_{i}.pt'))
            torch.save(y, f = os.path.join(y_train_pth, f'y_train_{i}.pt') )
            
            if i == 10:
                sys.exit(0) 
            
    elif not seq_tensor_size:
        X_tnsr,  y_tnsr = create_sequences_gen(
            data = data_tensor,
            context_len = context_len,
            seq_tensor_size = seq_tensor_size,
            max_toks = max_toks
        )
        
        torch.save(X_tnsr, f = os.path.join(X_train_pth, f"X_train.pt"))
        torch.save(y_tnsr, f = os.path.join(y_train_pth, f"y_train.pt")) 
        del X_tnsr, y_tnsr 
        
    # ------ VALIDATION DATA

    data = data_process(
        files = file_val, 
        eos_str='<|endoftext|>',
        return_single_str=return_single_str
    )

    data_tensor = tokenize(
        data=data,
        tokenizer=tokenizer,
        flat_tensor = flat_tensor
    )

    if isinstance(seq_tensor_size, int):
        sequence_generator = create_sequences_gen(
                data = data_tensor,
                context_len = context_len,
                seq_tensor_size = seq_tensor_size
            ) 
        for i, (X, y) in enumerate(sequence_generator):
            torch.save(X, f = os.path.join(X_val_pth, f'X_val_{i}.pt'))
            torch.save(y, f = os.path.join(y_val_pth, f'y_val_{i}.pt') )
    elif not seq_tensor_size:
        X_tnsr,  y_tnsr = create_sequences_gen(
            data = data_tensor,
            context_len = context_len,
            seq_tensor_size = seq_tensor_size
        )
        
        torch.save(X_tnsr, f = os.path.join(X_val_pth, f"X_val.pt"))
        torch.save(y_tnsr, f = os.path.join(y_train_pth, f"y_val.pt")) 
        
        del X_tnsr, y_tnsr 
```

## Requirements

- torch
- pandas
- tqdm
- tokenizers