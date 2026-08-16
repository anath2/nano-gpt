#!/usr/bin/env python3
"""Download the WikiText-103 raw corpus from HuggingFace into data/ as parquet.

Produces:
  data/train.parquet   ~307 MB  official training split (the two HF shards,
                                merged into one file)
  data/valid.parquet   ~0.7 MB  official validation split

Command:
    uv run download-datset-wikipedia
"""

import argparse
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

try:
    import pyarrow.parquet as pq
except ImportError:
    print('error: pyarrow is required — run this as `uv run download-dataset-wikipedia` '
          'or `uv sync` first', file=sys.stderr)
    sys.exit(1)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
TRAIN_OUT = os.path.join(DATA_DIR, 'train.parquet')
VAL_OUT = os.path.join(DATA_DIR, 'valid.parquet')

HF_BASE = ('https://huggingface.co/datasets/Salesforce/wikitext/resolve/'
           'main/wikitext-103-raw-v1')
TRAIN_SHARDS = [
    f'{HF_BASE}/train-00000-of-00002.parquet',
    f'{HF_BASE}/train-00001-of-00002.parquet',
]
VAL_SHARD = f'{HF_BASE}/validation-00000-of-00001.parquet'

STAGING_DIR = os.path.join(DATA_DIR, '.download')

MIN_TRAIN_BYTES = 200_000_000   # sanity floor; ~307 MB expected
MIN_VAL_BYTES = 300_000         # sanity floor; ~0.7 MB expected


def download(url, dest):
    """Fetch one file with a coarse progress readout; retry transient errors."""
    print(f'  {url}')
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                total = int(resp.headers.get('Content-Length') or 0)
                done = 0
                last_pct = -1
                with open(dest, 'wb') as out:
                    while True:
                        block = resp.read(256 * 1024)
                        if not block:
                            break
                        out.write(block)
                        done += len(block)
                        pct = int(100 * done / total) if total else -1
                        if pct != last_pct:
                            if total:
                                print(f'\r  {done / 1e6:6.1f}/{total / 1e6:.1f} MB ({pct}%)',
                                      end='', flush=True)
                            else:
                                print(f'\r  {done / 1e6:6.1f} MB', end='', flush=True)
                            last_pct = pct
                print()
                return
        except (urllib.error.URLError, OSError) as exc:
            if attempt == 3:
                raise
            print(f'    attempt {attempt} failed ({exc}); retrying...', file=sys.stderr)
            time.sleep(1)


def merge(shards, out):
    """Concatenate parquet shards into one file, preserving row order."""
    with pq.ParquetWriter(out, pq.read_schema(shards[0])) as writer:
        for shard in shards:
            writer.write_table(pq.read_table(shard))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Download WikiText-103 raw (train + valid) as parquet into data/.')
    parser.add_argument('--force', action='store_true',
                        help='re-download and rebuild even if the outputs already exist')
    args = parser.parse_args(argv)

    if not args.force and os.path.exists(TRAIN_OUT) and os.path.exists(VAL_OUT):
        print(f'Already present: {TRAIN_OUT} and {VAL_OUT} (use --force to re-download).')
        return 0

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(STAGING_DIR, exist_ok=True)
    try:
        print(f'Downloading WikiText-103 raw parquet from HuggingFace ({HF_BASE}):')
        train_dests = []
        for i, url in enumerate(TRAIN_SHARDS):
            dest = os.path.join(STAGING_DIR, f'train-{i}.parquet')
            download(url, dest)
            train_dests.append(dest)
        val_dest = os.path.join(STAGING_DIR, 'valid.parquet')
        download(VAL_SHARD, val_dest)

        # write to staging, then move into place, so a failure mid-merge never
        # leaves a truncated train.parquet behind
        merged = os.path.join(STAGING_DIR, 'train.parquet')
        print(f'Merging train shards -> {TRAIN_OUT} ...')
        merge(train_dests, merged)
        os.replace(merged, TRAIN_OUT)
        print(f'Moving validation split -> {VAL_OUT} ...')
        os.replace(val_dest, VAL_OUT)

        train_bytes = os.path.getsize(TRAIN_OUT)
        val_bytes = os.path.getsize(VAL_OUT)
        print(f'train: {train_bytes} bytes')
        print(f'valid: {val_bytes} bytes')
        if train_bytes < MIN_TRAIN_BYTES:
            print(f'warning: train split looks too small ({train_bytes} bytes) — '
                  'download may be truncated', file=sys.stderr)
        if val_bytes < MIN_VAL_BYTES:
            print(f'warning: valid split looks too small ({val_bytes} bytes) — '
                  'download may be truncated', file=sys.stderr)
        print(f'Done: {TRAIN_OUT}, {VAL_OUT}')
    finally:
        shutil.rmtree(STAGING_DIR, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
