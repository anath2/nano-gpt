from abc import ABC, abstractmethod
from typing import Optional

import tiktoken


class TokenizerBase(ABC):

    def __init__(self, corpus: Optional[str] = None):
        self.corpus = corpus

    @abstractmethod
    def encode(self, string: str) -> list[int]:
        ...

    @abstractmethod
    def decode(self, idx_list: list[int]) -> str:
        ...

    @abstractmethod
    def get_vocab_size(self) -> int:
        ...


class CharTokenizer(TokenizerBase):

    def __init__(self, corpus: str):
        super().__init__(corpus)
        assert self.corpus is not None
        vocab_set = set(list(self.corpus))
        self.vocab = sorted(list(vocab_set))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itoc = {i:ch for ch, i in self.stoi.items()}

    def encode(self, string: str) -> list[int]:
        return [self.stoi[c] for c in string]

    def decode(self, idx_list: list[int]) -> str:
        return ''.join([self.itoc[i] for i in idx_list])

    def get_vocab_size(self) -> int:
        return len(self.vocab)


class GPT2Tokenizer(TokenizerBase):

    def __init__(self):
        super().__init__()
        self.tokenizer = tiktoken.get_encoding('gpt2')

    def encode(self, string: str) -> list[int]:
        return self.tokenizer.encode(string)

    def decode(self, idx_list: list[int]) -> str:
        return self.tokenizer.decode(idx_list)

    def get_vocab_size(self) -> int:
        return self.tokenizer.max_token_value + 1


class BPETokenizer(TokenizerBase):

    def __init__(self, corpus: Optional[str] = None, vocab_size: int = 512):
        super().__init__(corpus)
        self.vocab_size = vocab_size
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}

    def encode(self, string: str) -> list[int]:
        assert self.vocab, 'tokenizer is untrained: call train() or load() first'
        ids = list(string.encode('utf-8'))
        while len(ids) >= 2:
            counts = self.count_pairs(ids)
            min_p = min(counts, key=lambda p: self.merges.get(p, float('inf')))
            if min_p not in self.merges:
                break
            min_id = self.merges[min_p]
            ids = self.merge(ids, min_p, min_id)
        return ids

    def decode(self, idx_list: list[int]) -> str:
        assert self.vocab, 'tokenizer is untrained: call train() or load() first'
        bps = b''.join(self.vocab[i] for i in idx_list)
        # a generated id can end mid-codepoint, so never raise on partial utf-8
        return bps.decode('utf-8', errors='replace')

    def get_vocab_size(self) -> int:
        return 256 + len(self.merges)

    def train(self) -> None:
        assert self.vocab_size >= 256
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        assert isinstance(self.corpus, str)
        ids = list(self.corpus.encode('utf-8'))
        num_merges = self.vocab_size - 256 # baseline is 1 byte

        for n in range(num_merges):
            counts = self.count_pairs(ids)

            if not counts:
                break
            top_pair = max(counts, key=lambda pair: counts[pair])

            if counts[top_pair] < 2:
                break

            new_id = 256 + n
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            ids = self.merge(ids, top_pair, new_id)

    # note we only write merges
    def save(self, outpath: str) -> None:
        with open(outpath, 'wt') as fd:
            fd.write(f'Number of merges {len(self.merges)}\n')
            for a, b in self.merges.keys():
                fd.write(f'{a} {b}\n')

    @classmethod
    def load(cls, filepath: str) -> 'BPETokenizer':
        t = cls(corpus=None)
        vocab = {i: bytes([i]) for i in range(256)}
        merges = {}

        with open(filepath) as fd:
            lines = fd.readlines()

        # first line is metadata; check it before building
        expected = int(lines[0].strip().split()[-1])
        data = [ln for ln in lines[1:] if ln.strip()]
        assert expected == len(data), \
            f'{filepath} is corrupt: header says {expected} merges, found {len(data)}'

        for line in data:
            a, b = map(int, line.split())
            new_id = 256 + len(merges)
            merges[(a, b)] = new_id
            vocab[new_id] = vocab[a] + vocab[b]

        t.vocab = vocab
        t.merges = merges
        t.vocab_size = 256 + len(merges)
        return t

    @staticmethod
    def count_pairs(ids: list[int]) -> dict[tuple[int, int], int]:
        counts = {}
        for x, y in zip(ids, ids[1:]):
            counts[(x, y)] = counts.get((x, y), 0) + 1
        return counts

    @staticmethod
    def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        i, out = 0, []
        while i < len(ids):
            if (i + 1) < len(ids) and (ids[i], ids[i+1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out
