import os
import time

import modal
import torch

from model import Transformer
from tokenizer import BPETokenizer


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATASET_PATH = os.path.join(HERE, '..', 'data', 'wikitext103.train.txt')
VAL_DATASET_PATH = os.path.join(HERE, '..', 'data', 'wikitext103.valid.txt')
VOCAB_SIZE = 256
MERGES_PATH = os.path.join(HERE, f'bpe_merges_{VOCAB_SIZE}.txt')
SEED = 567
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
MAX_ITERS = 5000
EVAL_INTERVAL = 500      # how often to estimate loss during training
EVAL_ITERS = 200         # batches averaged per loss estimate
LOG_INTERVAL = 100       # how often to print training throughput (ms/it)
GEN_TOKENS = 300         # tokens to generate after training
N_EMBED = 384
N_HEAD = 6
N_LAYER = 6
CONTEXT_LEN = 256


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


def get_batch(split, dtrain, dval, device):
    data = dtrain if split == 'train' else dval
    ix = torch.randint(len(data) - CONTEXT_LEN, (BATCH_SIZE,))
    x = torch.stack([data[i: i + CONTEXT_LEN] for i in ix])
    y = torch.stack([data[i + 1:i + CONTEXT_LEN + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, dtrain, dval, device):
    out = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split, dtrain, dval, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def fmt_loss(est, split):
    return f"{split} {est[split]:.4f}"



def run(device='cpu', train_path=TRAIN_DATASET_PATH, val_path=VAL_DATASET_PATH,
        merges_path=MERGES_PATH):
    """Train the model and sample from it. Works on CPU or CUDA."""
    print(f'Using device: {device}')
    torch.manual_seed(SEED)

    train_text = load_dataset(train_path)
    val_text = load_dataset(val_path)
    tok = build_bpe_tokenizer(train_text, merges_path=merges_path)
    vocab_size = tok.get_vocab_size()

    t_enc = time.time()
    train_ids = tok.encode(train_text)
    val_ids = tok.encode(val_text)
    train_nbytes = len(train_text.encode('utf-8'))
    print(f'Encoded corpus in {(time.time() - t_enc) / 60:.1f} min: '
          f'train {train_nbytes} bytes -> {len(train_ids)} tokens '
          f'({train_nbytes / len(train_ids):.2f}x compression), val {len(val_ids)} tokens')

    dtrain = torch.tensor(train_ids, dtype=torch.long)
    dval = torch.tensor(val_ids, dtype=torch.long)

    model = Transformer(CONTEXT_LEN, vocab_size, N_EMBED, N_LAYER, N_HEAD).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f'Model: {nparams / 1e6:.2f}M params | vocab {vocab_size} | ctx {CONTEXT_LEN} tokens')

    # train
    optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    t_start = time.time()
    t_mark = t_start
    for it in range(MAX_ITERS):
        xb, yb = get_batch('train', dtrain, dval, device)
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
            est = estimate_loss(model, dtrain, dval, device)
            print(f"iter {it:5d} | {fmt_loss(est, 'train')} | {fmt_loss(est, 'val')}")

    est = estimate_loss(model, dtrain, dval, device)
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


# ---------------------------------------------------------------------------
# Modal: run training + inference on a remote GPU
#   uv run modal run nano-gpt.py
# ---------------------------------------------------------------------------
REMOTE_TRAIN_DATASET = '/root/wikitext103.train.txt'
REMOTE_VAL_DATASET = '/root/wikitext103.valid.txt'
REMOTE_MERGES = f'/root/bpe_merges_{VOCAB_SIZE}.txt'

# NOTE: add_local_file bakes the dataset into the image layer, which is fine for
# small files but not for the 514 MB wikitext103 train split — this needs to move
# to a Modal Volume (see ROADMAP.md "Known gotchas") before running at full scale.
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch", "tiktoken")   # tiktoken: imported by tokenizer.py
    .add_local_file(TRAIN_DATASET_PATH, REMOTE_TRAIN_DATASET)
    .add_local_file(VAL_DATASET_PATH, REMOTE_VAL_DATASET)
    # bake the trained merges in so the GPU box loads them instead of spending
    # minutes retraining the tokenizer on CPU
    .add_local_file(MERGES_PATH, REMOTE_MERGES)
    .add_local_python_source("tokenizer")
)

app = modal.App("nano-gpt", image=image)


@app.function(gpu="T4", timeout=7200)
def train_remote():
    run(device=get_device(), train_path=REMOTE_TRAIN_DATASET, val_path=REMOTE_VAL_DATASET,
        merges_path=REMOTE_MERGES)


@app.local_entrypoint()
def modal_main():
    train_remote.remote()


if __name__ == '__main__':
    # Local run
    run(device=get_device())
