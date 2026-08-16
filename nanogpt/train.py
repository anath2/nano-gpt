import os
import time

import modal
import torch

from nanogpt.data import DataLoader, build_bpe_tokenizer, load_dataset
from nanogpt.model import create_model


HERE = os.path.dirname(os.path.abspath(__file__))
MERGES_PATH = os.path.join(HERE, '..', 'data', 'bpe_merges.txt')
TRAIN_DATASET_PATH = os.path.join(HERE, '..', 'data', 'train.parquet')
VAL_DATASET_PATH = os.path.join(HERE, '..', 'data', 'valid.parquet')

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED = 567
LEARNING_RATE = 3e-4
MAX_ITERS = 5000
EVAL_INTERVAL = 500      # how often to estimate loss during training
EVAL_ITERS = 200         # batches averaged per loss estimate
LOG_INTERVAL = 100       # how often to print training throughput (ms/it)
GEN_TOKENS = 300         # tokens to generate after training


@torch.no_grad()
def estimate_loss(model, loader, device):
    out = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = loader.get_batch(split, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def fmt_loss(est, split):
    return f"{split} {est[split]:.4f}"


def run(device='cpu', train_path=TRAIN_DATASET_PATH, val_path=VAL_DATASET_PATH,
        merges_path=MERGES_PATH):
    print(f'Using device: {device}')
    torch.manual_seed(SEED)

    train_text = load_dataset(train_path)
    val_text = load_dataset(val_path)
    tok = build_bpe_tokenizer(train_text, merges_path=merges_path)
    vocab_size = tok.get_vocab_size()

    loader = DataLoader()
    model = create_model(vocab_size).to(device)

    assert model.context_len == loader.chunk_size, (
        f'model context_len ({model.context_len}) must match dataloader chunk_size '
        f'({loader.chunk_size})')

    nparams = sum(p.numel() for p in model.parameters())
    print(f'Model: {nparams / 1e6:.2f}M params | vocab {vocab_size}')

    t_enc = time.time()
    train_ids = tok.encode(train_text)
    val_ids = tok.encode(val_text)
    train_nbytes = len(train_text.encode('utf-8'))

    print(f'Encoded corpus in {(time.time() - t_enc) / 60:.1f} min: '
          f'train {train_nbytes} bytes -> {len(train_ids)} tokens '
          f'({train_nbytes / len(train_ids):.2f}x compression), val {len(val_ids)} tokens')

    loader.load(train_ids, val_ids)

    # train
    optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    t_start = time.time()
    t_mark = t_start
    for it in range(MAX_ITERS):
        xb, yb = loader.get_batch('train', device)
        logits, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        # throughput: CUDA kernels are async, so sync before timing or the
        # measurement just clocks Python enqueue time, not real GPU work.
        if it > 0 and it % LOG_INTERVAL == 0:
            if device == 'cuda':
                torch.cuda.synchronize()
            spi = (time.time() - t_mark) / LOG_INTERVAL
            print(f"iter {it:5d} | {spi * 1000:6.1f} ms/it "
                  f"| train-only proj ~{spi * MAX_ITERS / 60:.1f} min")
            t_mark = time.time()

        if it % EVAL_INTERVAL == 0:
            est = estimate_loss(model, loader, device)
            print(f"iter {it:5d} | {fmt_loss(est, 'train')} | {fmt_loss(est, 'val')}")

    est = estimate_loss(model, loader, device)
    print(f"final | {fmt_loss(est, 'train')} | {fmt_loss(est, 'val')}")
    print(f"total wall time: {(time.time() - t_start) / 60:.1f} min")

    # sample after training
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        print(tok.decode(model.generate(start, GEN_TOKENS)[0].tolist()))


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def main():
    from dotenv import load_dotenv

    repo_root = os.path.abspath(os.path.join(HERE, '..'))
    load_dotenv(os.path.join(repo_root, '.env'))
    os.chdir(repo_root)
    os.execvp('modal', ['modal', 'run', '--detach', os.path.join(HERE, 'train.py')])


# ---------------------------------------------------------------------------
# Modal: run training + inference on a remote GPU. Local entrypoint:
#   uv run nanogpt-train
# ---------------------------------------------------------------------------
REMOTE_TRAIN_DATASET = '/root/train.parquet'
REMOTE_VAL_DATASET = '/root/valid.parquet'
REMOTE_MERGES = '/root/bpe_merges.txt'

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch", "tiktoken", "pyarrow")  # pyarrow: read parquet in data.py
    .add_local_file(TRAIN_DATASET_PATH, REMOTE_TRAIN_DATASET)
    .add_local_file(VAL_DATASET_PATH, REMOTE_VAL_DATASET)
    .add_local_file(MERGES_PATH, REMOTE_MERGES)
    .add_local_python_source("nanogpt")
)

app = modal.App("nano-gpt", image=image)


@app.function(gpu="T4", timeout=7200)
def train_remote():
    run(device=get_device(), train_path=REMOTE_TRAIN_DATASET, val_path=REMOTE_VAL_DATASET,
        merges_path=REMOTE_MERGES)


@app.local_entrypoint()
def modal_main():
    train_remote.remote()
