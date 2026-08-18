import base64
import json
import os
from collections import Counter
from abc import ABC, abstractmethod
from typing import Optional

import regex
import tiktoken


PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""


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
    """Byte-level BPE with a single `ranks` table (token bytes -> id)"""

    def __init__(self, ranks: dict[bytes, int], pat_str: str):
        super().__init__()
        self.ranks = ranks
        self.pat_str = pat_str
        self._decoder: dict[int, bytes] = {rank: token for token, rank in ranks.items()}
        self._pat = regex.compile(pat_str)
        self._cache: dict[str, list[int]] = {}

    def get_vocab_size(self) -> int:
        """Return the number of tokens in the vocabulary."""
        return len(self.ranks)

    def decode(self, idx_list: list[int]) -> str:
        """Convert `idx_list` back to a string by looking up each token in `self._decoder`."""
        bps = b''.join(self._decoder[i] for i in idx_list)
        return bps.decode('utf-8', errors='replace')

    @classmethod
    def train(cls, text: str, vocab_size: int, pat_str: str = PAT_STR) -> 'BPETokenizer':
        """Train a fresh tokenizer on `text` to `vocab_size` ranks."""
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (256 single-byte tokens)")

        ranks = {bytes([i]): i for i in range(256)}  # seed ranks with single-byte tokens
        word_freq = Counter(m.group(0) for m in regex.finditer(pat_str, text))
        items = list(word_freq.items())
        pieces = [tuple(bytes([b]) for b in w.encode('utf-8')) for w, _ in items]
        freqs = [f for _, f in items]

        while len(ranks) < vocab_size:
            # find the most common pair of bytes and merge it
            pair_counts = Counter()
            for piece, freq in zip(pieces, freqs):
                for i in range(len(piece) - 1):
                    pair = piece[i:i+2]
                    pair_counts[pair] += freq

            if not pair_counts:
                raise ValueError(
                    f'train(): corpus exhausted at {len(ranks)}/{vocab_size} ranks — '
                    'no adjacent pairs remain to merge. Use a larger corpus or a '
                    'smaller vocab_size; a silently short table would mis-size the '
                    'embedding downstream.')

            best = max(pair_counts, key=lambda pair: pair_counts[pair])

            if pair_counts[best] < 2:
                raise ValueError(
                    f'train(): corpus exhausted at {len(ranks)}/{vocab_size} ranks — '
                    f'the best remaining pair {best!r} occurs only '
                    f'{pair_counts[best]} time(s). Use a larger corpus or a smaller '
                    'vocab_size; a silently short table would mis-size the embedding '
                    'downstream.')

            new_token = best[0] + best[1]
            ranks[new_token] = len(ranks)

            for i, piece in enumerate(pieces):
                pieces[i] = cls.apply_merge(piece, best, new_token)

        return cls(ranks, pat_str)

    def encode(self, string: str) -> list[int]:
        """Split `string` with `self._pat`, encode each piece independently, and
        cache the result per unique piece in `self._cache`."""
        out: list[int] = []

        for match in self._pat.finditer(string):
            key = match.group(0)
            cached = self._cache.get(key)

            if cached is not None:
                out.extend(cached)
                continue

            parts: list[bytes] = [bytes([b]) for b in key.encode('utf-8')]

            while len(parts) > 1:
                best_rank, best_i = None, None
                for i in range(len(parts) - 1):
                    r = self.ranks.get(parts[i] + parts[i + 1])
                    if r is not None and (best_rank is None or r < best_rank):
                        best_rank, best_i = r, i

                if best_i is None:
                    break

                parts[best_i: best_i + 2] = [parts[best_i] + parts[best_i + 1]]
            ids = [self.ranks[b] for b in parts]
            self._cache[key] = ids
            out.extend(ids)
        return out

    def save(self, path: str, extra: Optional[dict] = None) -> None:
        """Write the merges file and its metadata together, so the two
        can never drift apart."""
        with open(path, 'wb') as f:
            for token, rank in sorted(self.ranks.items(), key=lambda kv: kv[1]):
                f.write(base64.b64encode(token) + b' ' + str(rank).encode() + b'\n')

        meta = {'pat_str': self.pat_str, 'vocab_size': self.get_vocab_size()}
        if extra:
            meta.update(extra)
        meta_path = os.path.splitext(path)[0] + '.meta.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Read the merges file and its metadata together; raise if either is
        missing rather"""
        meta_path = os.path.splitext(path)[0] + '.meta.json'
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} not found')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f'{meta_path} not found — a merges file without its sidecar has no '
                'recorded pat_str (ADR 0015/0018)')

        with open(meta_path) as f:
            meta = json.load(f)

        ranks: dict[bytes, int] = {}
        with open(path, 'rb') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token_b64, rank = line.split(b' ')
                ranks[base64.b64decode(token_b64)] = int(rank)

        tok = cls(ranks, meta['pat_str'])
        assert tok.get_vocab_size() == meta['vocab_size'], (
            f'{path} holds {tok.get_vocab_size()} ranks, '
            f'{meta_path} says {meta["vocab_size"]}')
        return tok

    @staticmethod
    def apply_merge(piece, pair, new_token):
        """Replace `pair` in `piece` with `new_token`; used during training."""
        out, i = [], 0
        while i < len(piece):
            if piece[i:i+2] == pair:
                out.append(new_token)
                i += 2
            else:
                out.append(piece[i])
                i += 1
        return tuple(out)


if __name__ == '__main__':
    import os
    import time

    HERE = os.path.dirname(os.path.abspath(__file__))
    SAMPLE_PATH = os.path.join(HERE, '..', 'data', 'shakespeare.txt')
    OUT_PATH = os.path.join(HERE, '..', 'data', 'bpe_merges_smoketest.txt')
    VOCAB_SIZE = 512
    TRAIN_CHARS = 200_000
    ENCODE_CHARS = 50_000

    with open(SAMPLE_PATH, encoding='utf-8') as f:
        full_text = f.read()

    train_text = full_text[:TRAIN_CHARS]
    print(f'Training on {len(train_text)} chars from {SAMPLE_PATH} '
          f'(vocab_size={VOCAB_SIZE}) ...')

    t0 = time.time()
    tok = BPETokenizer.train(train_text, VOCAB_SIZE, PAT_STR)
    print(f'  trained vocab {tok.get_vocab_size()} in {time.time() - t0:.1f}s')

    assert sorted(tok.ranks.values()) == list(range(tok.get_vocab_size())), (
        'ranks are not a contiguous 0..vocab_size-1 range')
    print('  rank contiguity OK')

    assert tok.encode('') == []
    assert tok.decode([]) == ''
    print('  empty-input edge cases OK')

    sample = train_text[:ENCODE_CHARS]
    t0 = time.time()
    ids = tok.encode(sample)
    assert max(ids) >= 256, 'no merged tokens in output — encode() is not merging'
    dt = time.time() - t0
    ratio = len(sample) / len(ids) if ids else float('nan')
    assert ratio > 2.0, f'compression too low ({ratio:.2f}) — merges not being applied'

    print(f'  encoded {len(sample)} chars -> {len(ids)} ids in {dt:.2f}s '
          f'({ratio:.2f} chars/token)')
    decoded = tok.decode(ids)
    assert decoded == sample, 'decode(encode(x)) != x'
    print('  round-trip OK')

    tok.save(OUT_PATH, extra={'smoketest': True})
    loaded = BPETokenizer.load(OUT_PATH)
    assert loaded.ranks == tok.ranks
    assert loaded.pat_str == tok.pat_str
    print(f'  save/load round-trip OK -> {OUT_PATH}')
    print('All smoke-test checks passed.')
