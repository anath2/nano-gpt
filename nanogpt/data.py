import os
import time

import pyarrow.parquet as pq
import torch

from nanogpt.tokenizer import BPETokenizer


BATCH_SIZE = 64
CHUNK_SIZE = 256
VOCAB_SIZE = 256


def load_dataset(path):
    """Return the full corpus as one string, decoded from a parquet `text` column."""
    chunks = []
    for batch in pq.ParquetFile(path).iter_batches(columns=['text'], batch_size=100_000):
        chunks.append(''.join(batch.column('text').to_pylist()))
    return ''.join(chunks)


def build_bpe_tokenizer(dataset, merges_path, vocab_size=VOCAB_SIZE):
    if os.path.exists(merges_path):
        tok = BPETokenizer.load(merges_path)
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

    def load(self, train_ids, val_ids):
        self.dtrain = torch.tensor(train_ids, dtype=torch.long)
        self.dval = torch.tensor(val_ids, dtype=torch.long)

    def get_batch(self, split, device):
        data = self.dtrain if split == 'train' else self.dval
        assert data is not None, f'no {split} data: call load() first'
        ix = torch.randint(len(data) - self.chunk_size, (self.batch_size,))
        x = torch.stack([data[i: i + self.chunk_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.chunk_size + 1] for i in ix])
        return x.to(device), y.to(device)
