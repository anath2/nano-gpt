import hashlib
import json
import os

import torch

BATCH_SIZE = 64
CHUNK_SIZE = 256
TOKEN_DTYPE = torch.int16


def load_tokens(bin_path: str, merges_path: str) -> torch.Tensor:
    """binary produced by `scripts/tokenize-corpus.py`"""
    meta_path = bin_path + '.meta.json'
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f'{meta_path} not found — {bin_path} has no sidecar; run '
            'scripts/tokenize-corpus.py to produce both')

    with open(meta_path) as f:
        meta = json.load(f)

    with open(merges_path, 'rb') as f:
        merges_sha256 = hashlib.sha256(f.read()).hexdigest()

    if meta['merges_sha256'] != merges_sha256:
        raise ValueError(
            f'{bin_path} was tokenized against a different merges file: sidecar '
            f'records {meta["merges_sha256"]}, but {merges_path} hashes to '
            f'{merges_sha256}. Re-run scripts/tokenize-corpus.py --force so the '
            '.bin matches the current tokenizer.')

    filesize = os.path.getsize(bin_path)
    expected_bytes = meta['n_tokens'] * TOKEN_DTYPE.itemsize

    if expected_bytes != filesize:
        raise ValueError(
            f'{bin_path} is {filesize} bytes but {meta_path} claims '
            f'{meta["n_tokens"]} tokens ({expected_bytes} bytes expected) — '
            're-run scripts/tokenize-corpus.py --force')

    return torch.from_file(bin_path, shared=True, size=meta['n_tokens'], dtype=TOKEN_DTYPE)


class DataLoader:
    """Samples random training windows from token ids."""

    def __init__(self, batch_size=BATCH_SIZE, chunk_size=CHUNK_SIZE):
        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self.dtrain = None
        self.dval = None

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def chunk_size(self):
        return self._chunk_size

    def load(self, train_bin, val_bin, merges_path):
        self.dtrain = load_tokens(train_bin, merges_path)
        self.dval = load_tokens(val_bin, merges_path)

    def get_batch(self, split, device):
        data = self.dtrain if split == 'train' else self.dval
        assert data is not None, f'no {split} data: call load() first'
        ix = torch.randint(len(data) - self.chunk_size, (self.batch_size,))
        x = torch.stack([data[i: i + self.chunk_size] for i in ix]).to(torch.int64)
        y = torch.stack([data[i + 1:i + self.chunk_size + 1] for i in ix]).to(torch.int64)
        return x.to(device), y.to(device)
