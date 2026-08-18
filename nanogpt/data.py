import pyarrow.parquet as pq
import torch


BATCH_SIZE = 64
CHUNK_SIZE = 256
VOCAB_SIZE = 256


def load_dataset(path):
    """Return the full corpus as one string, decoded from a parquet `text` column."""
    chunks = []
    for batch in pq.ParquetFile(path).iter_batches(columns=['text'], batch_size=100_000):
        chunks.append(''.join(batch.column('text').to_pylist()))
    return ''.join(chunks)


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
