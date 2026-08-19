#!/usr/bin/env python3
"""Pre-tokenize a parquet corpus into a flat int16 .bin file. 

Command:
    uv run scripts/tokenize-corpus.py [dataset] [--merges PATH] [--out PATH]
        [--max-mb N] [--force]
"""

import argparse
import array
import hashlib
import json
import os
import sys
import time

import pyarrow.parquet as pq

from nanogpt.tokenizer import BPETokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
TRAIN_PARQUET = os.path.join(DATA_DIR, 'train.parquet')
MERGES_PATH = os.path.join(DATA_DIR, 'bpe_merges.txt')

BATCH_SIZE = 2000
MAX_TOKEN_ID = 1 << 15  # int16 range: ids must fall in [0, 32768) to avoid wrapping negative


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def tokenize(dataset: str, merges: str, out: str, max_mb: float | None = None) -> dict:
    """Stream `dataset` through `tok.encode()` row by row, appending int16 ids
    to `out` as they're produced, then write the `out.meta.json` sidecar."""
    tok = BPETokenizer.load(merges)
    pf = pq.ParquetFile(dataset)

    max_bytes = int(max_mb * 1_000_000) if max_mb is not None else None
    n_chars = 0
    n_bytes = 0
    n_tokens = 0

    with open(out, 'wb') as f:
        for batch in pf.iter_batches(columns=['text'], batch_size=BATCH_SIZE):
            done = False
            for text in batch.column('text').to_pylist():
                if max_bytes is not None and n_bytes >= max_bytes:
                    done = True
                    break
                ids = tok.encode(text)
                if ids and max(ids) >= MAX_TOKEN_ID:
                    raise ValueError(
                        f'token id {max(ids)} does not fit in int16 (vocab_size='
                        f'{tok.get_vocab_size()}); pick a smaller vocab or widen '
                        'TOKEN_DTYPE before writing a .bin this large.')
                f.write(array.array('h', ids).tobytes())  # 'h' = signed short = int16
                n_tokens += len(ids)
                n_chars += len(text)
                n_bytes += len(text.encode('utf-8'))
            if done:
                break

    meta = {
        'n_tokens': n_tokens,
        'dtype': 'int16',
        'vocab_size': tok.get_vocab_size(),
        'merges_sha256': sha256_file(merges),
        'source': os.path.abspath(dataset),
        'chars_per_token': n_chars / n_tokens if n_tokens else float('nan'),
    }
    meta_path = out + '.meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Pre-tokenize a parquet corpus into a flat int16 .bin file '
                    'with a JSON sidecar.')
    parser.add_argument('dataset', nargs='?', default=TRAIN_PARQUET,
                        help=f'source parquet file (default: {TRAIN_PARQUET})')
    parser.add_argument('--merges', default=MERGES_PATH,
                        help=f'BPE merges file to encode with (default: {MERGES_PATH})')
    parser.add_argument('--out', default=None,
                        help='output .bin path (default: data/<dataset-stem>.bin)')
    parser.add_argument('--max-mb', type=float, default=None,
                        help='stop after this many MB of source text (for smoke runs)')
    parser.add_argument('--force', action='store_true',
                        help='retokenize and overwrite even if --out already exists')
    args = parser.parse_args(argv)

    out = args.out or os.path.join(
        DATA_DIR, os.path.splitext(os.path.basename(args.dataset))[0] + '.bin')

    if not args.force and os.path.exists(out):
        print(f'Already present: {out} (use --force to retokenize).')
        return 0

    if not os.path.exists(args.dataset):
        print(f'error: {args.dataset} not found', file=sys.stderr)
        return 1
    if not os.path.exists(args.merges):
        print(f'error: {args.merges} not found — run '
              '`uv run scripts/train-bpe.py` first', file=sys.stderr)
        return 1

    print(f'Tokenizing {args.dataset} -> {out} ...')
    t0 = time.time()
    meta = tokenize(args.dataset, args.merges, out, args.max_mb)
    wall_time = time.time() - t0

    print(f'Wrote {meta["n_tokens"]} tokens -> {out} ({wall_time:.1f}s)')
    print(f'  chars per token: {meta["chars_per_token"]:.2f}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
