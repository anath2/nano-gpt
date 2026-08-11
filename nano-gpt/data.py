import os
import time

import torch

from tokenizer import BPETokenizer


HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATASET_PATH = os.path.join(HERE, '..', 'data', 'wikitext103.train.txt')
VAL_DATASET_PATH = os.path.join(HERE, '..', 'data', 'wikitext103.valid.txt')
VOCAB_SIZE = 256
MERGES_PATH = os.path.join(HERE, f'bpe_merges_{VOCAB_SIZE}.txt')
BATCH_SIZE = 64


def load_dataset(path):
    with open(path) as rt:
        return rt.read()


def build_bpe_tokenizer(dataset, merges_path=MERGES_PATH, vocab_size=VOCAB_SIZE):
    if os.path.exists(merges_path):
        tok = BPETokenizer.load(merges_path)
        # a stale merge table would silently mis-size the embedding table
        assert tok.get_vocab_size() == vocab_size, (
            f'{merges_path} holds vocab {tok.get_vocab_size()}, expected {vocab_size}')
        print(f'Tokenizer: loaded {merges_path} (vocab {tok.get_vocab_size()})')
    else:
        print(f'Tokenizer: no {merges_path}, training to vocab {vocab_size}...')
        t0 = time.time()
        tok = BPETokenizer(dataset, vocab_size=vocab_size)
        tok.train()
        tok.save(merges_path)
        print(f'Tokenizer: trained vocab {tok.get_vocab_size()} in '
              f'{(time.time() - t0) / 60:.1f} min -> {merges_path}')
    return tok


def get_batch(split, dtrain, dval, context_len, device):
    data = dtrain if split == 'train' else dval
    ix = torch.randint(len(data) - context_len, (BATCH_SIZE,))
    x = torch.stack([data[i: i + context_len] for i in ix])
    y = torch.stack([data[i + 1:i + context_len + 1] for i in ix])
    return x.to(device), y.to(device)
