#!/usr/bin/env python3
"""Train a BPE tokenizer on a random-windowed slice of WikiText-103 and cache the
merges (ADR 0010, ADR 0011).

Command:
    uv run scripts/train-bpe.py [dataset] [--vocab-size N] [--slice-mb N]
        [--num-windows N] [--seed N] [--out PATH] [--force]
"""

import argparse
import os
import random
import sys
import time

import pyarrow.parquet as pq

from nanogpt.tokenizer import BPETokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
TRAIN_PARQUET = os.path.join(DATA_DIR, 'train.parquet')
MERGES_OUT = os.path.join(DATA_DIR, 'bpe_merges.txt')

SEED = 5024
VOCAB_SIZE = 4096
SLICE_MB = 20
N_WINDOW = 20
BATCH_SIZE = 2000
CHARS_PER_TOKEN_SAMPLE = 1_000_000
MAX_REJECTION_ATTEMPTS = 10_000


def load_window(
    path: str,
    budget_mb: float,
    n_window: int,
    seed: int,
    batch_size: int = BATCH_SIZE
) -> list[str]:
    """Sample `n_window` windows of text from the parquet corpus via
    minimum-separation rejection sampling."""

    budget_per_window = int(budget_mb * 1_000_000) // n_window
    pf = pq.ParquetFile(path)
    num_rows = pf.metadata.num_rows

    total_bytes = sum(pf.metadata.row_group(i).total_byte_size
                       for i in range(pf.metadata.num_row_groups))
    avg_bytes_per_row = total_bytes / num_rows
    min_separation = max(int(budget_per_window / avg_bytes_per_row), 1)

    rng = random.Random(seed)
    starts: list[int] = []
    for i in range(n_window):
        for _ in range(MAX_REJECTION_ATTEMPTS):
            candidate = rng.randrange(num_rows)
            if all(abs(candidate - s) >= min_separation for s in starts):
                starts.append(candidate)
                break
        else:
            utilization = (budget_per_window * n_window) / total_bytes
            raise RuntimeError(
                f'load_window: could not place window {i + 1}/{n_window} after '
                f'{MAX_REJECTION_ATTEMPTS} attempts. Utilization is {utilization:.1%} — '
                'rejection sampling degrades above roughly 30-50% (ADR 0013); reduce '
                '--slice-mb / raise --num-windows, or switch to the adaptive covered-set '
                'sampler ADR 0013/0014 describe for the high-utilization regime.')
    starts.sort()

    windows = [{'start': s, 'chunks': [], 'nbytes': 0} for s in starts]

    row_idx = 0
    for batch in pf.iter_batches(columns=['text'], batch_size=batch_size):
        for text in batch.column('text').to_pylist():
            for w in windows:
                if w['nbytes'] >= budget_per_window or row_idx < w['start']:
                    continue
                w['chunks'].append(text)
                w['nbytes'] += len(text.encode('utf-8'))
            row_idx += 1
        if all(w['nbytes'] >= budget_per_window for w in windows):
            break

    for w in windows:
        if w['nbytes'] < budget_per_window:
            print(f"  warning: window at row {w['start']} only collected "
                  f"{w['nbytes'] / 1e6:.2f}MB of {budget_per_window / 1e6:.2f}MB "
                  "(ran off the end of the file)", file=sys.stderr)

    return [''.join(w['chunks']) for w in windows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Train a BPE tokenizer on a random-windowed slice of WikiText-103 '
                    'and cache the merges.')
    parser.add_argument('dataset', nargs='?', default=TRAIN_PARQUET,
                        help=f'source parquet file (default: {TRAIN_PARQUET})')
    parser.add_argument('--vocab-size', type=int, default=VOCAB_SIZE,
                        help=f'target vocabulary size (default: {VOCAB_SIZE})')
    parser.add_argument('--slice-mb', type=float, default=SLICE_MB,
                        help=f'megabytes of corpus to train on (default: {SLICE_MB})')
    parser.add_argument('--num-windows', type=int, default=N_WINDOW,
                        help=f'number of random windows to sample (default: {N_WINDOW})')
    parser.add_argument('--seed', type=int, default=SEED,
                        help=f'window-sampling seed (default: {SEED})')
    parser.add_argument('--out', default=MERGES_OUT,
                        help=f'merges file to write (default: {MERGES_OUT})')
    parser.add_argument('--force', action='store_true',
                        help='retrain and overwrite even if --out already exists')
    args = parser.parse_args(argv)

    if not args.force and os.path.exists(args.out):
        print(f'Already present: {args.out} (use --force to retrain).')
        return 0

    if not os.path.exists(args.dataset):
        print(f'error: {args.dataset} not found — run '
              '`uv run scripts/download-wikipedia.py` first', file=sys.stderr)
        return 1

    # train
    print(f'Sampling {args.slice_mb:g} MB across {args.num_windows} windows '
          f'from {args.dataset} (seed {args.seed}) ...')
    t0 = time.time()
    windows = load_window(args.dataset, args.slice_mb, args.num_windows, args.seed)
    text = ''.join(windows)
    tokenizer = BPETokenizer.train(text, args.vocab_size)
    wall_time = time.time() - t0
    print(f'  {len(text)} chars in {wall_time:.1f}s')

    # Sample to estimate compression ratio
    per_window = max(CHARS_PER_TOKEN_SAMPLE // len(windows), 1)
    sample = ''.join(w[:per_window] for w in windows)
    chars_per_token = len(sample) / len(tokenizer.encode(sample))

    # save
    tokenizer.save(args.out, extra={
           'input': os.path.abspath(args.dataset),
           'slice_mb': args.slice_mb,
           'num_windows': args.num_windows,
           'seed': args.seed,
           'chars_per_token': chars_per_token,
           'wall_time': wall_time,
    })

    print(f'Wrote merges -> {args.out}')
    print(f'  vocab size: {tokenizer.get_vocab_size()}')
    print(f'  chars per token: {chars_per_token:.2f}')
    print(f'  ')

    return 0


if __name__ == '__main__':
    sys.exit(main())
