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

## Usage

```python
from tokenizer_utils import data_process, train_new_tokenizer_bpe, tokenize, create_sequences

# Process your dataset
df = data_process(['data1.parquet', 'data2.parquet'], eos_str='<|endoftext|>')

# Train a tokenizer
tokenizer = train_new_tokenizer_bpe(df['text'].tolist(), vocab_size=32000, special_tokens=["<|endoftext|>"])

# Tokenize and get tensor
data_tensor = tokenize(df, tokenizer)

# Create training sequences
X, y = create_sequences(data_tensor, context_len=128, chunk_size=2048)
```

## Requirements

- torch
- pandas
- tqdm
- tokenizers