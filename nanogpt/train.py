import json
import os
import sys
import time

import modal
import torch

from nanogpt.data import DataLoader
from nanogpt.model import create_model, model_config
from nanogpt.tokenizer import BPETokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MERGES_PATH = os.path.join(HERE, '..', 'data', 'bpe_merges.txt')
TRAIN_BINARY    = os.path.join(HERE, '..', 'data', 'train.bin')
VAL_BINARY      = os.path.join(HERE, '..', 'data', 'valid.bin')

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED = 567
LEARNING_RATE = 3e-4
MAX_ITERS = 5000
EVAL_INTERVAL = 500      # how often to estimate loss during training
EVAL_ITERS = 50          # batches averaged per loss estimate
LOG_INTERVAL = 100       # how often to print training throughput (ms/it)
GEN_TOKENS = 300         # tokens to generate after training
CKPT_INTERVAL = 500      # how often to save a checkpoint


@torch.no_grad()
def estimate_loss(model, loader, device, eval_iters):
    out = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = loader.get_batch(split, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def fmt_loss(est, split):
    return f"{split} {est[split]:.4f}"


def train_config(lr, max_iters, batch_size, chunk_size, eval_iters, seed):
    """Hyperparameters that affect the trained model """
    return {'lr': lr, 'max_iters': max_iters, 'batch_size': batch_size,
            'chunk_size': chunk_size, 'eval_iters': eval_iters, 'seed': seed}


def save_checkpoint(path, model, optim, it, vocab_size, val_loss, on_save=None,
                     train_cfg=None, run_name=None):
    """Save checkpoint to disk, plus a `config.json`"""
    tmp = path + '.tmp'
    torch.save({
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'it': it,
        'vocab_size': vocab_size,
        'model_cfg': model_config(),
        'val_loss': val_loss,
        'seed': SEED,
        'train_cfg': train_cfg,
        'run_name': run_name,
    }, tmp)
    os.replace(tmp, path)

    config_path = os.path.join(os.path.dirname(path), 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'run_name': run_name,
            'it': it,
            'val_loss': val_loss,
            'model_cfg': model_config(),
            'train_cfg': train_cfg,
        }, f, indent=2)

    if on_save is not None:
        on_save()


def run(device='cpu', train_path=TRAIN_BINARY, val_path=VAL_BINARY,
        merges_path=MERGES_PATH, max_iters=MAX_ITERS, eval_interval=EVAL_INTERVAL,
        eval_iters=EVAL_ITERS, log_interval=LOG_INTERVAL, gen_tokens=GEN_TOKENS,
        checkpoint_path=None, on_save=None, ckpt_interval=CKPT_INTERVAL, resume_from=None,
        run_name=None
):
    print(f'Using device: {device}')
    torch.manual_seed(SEED)

    tok = BPETokenizer.load(merges_path)
    vocab_size = tok.get_vocab_size()

    loader = DataLoader()
    model = create_model(vocab_size).to(device)
    model_cfg = model_config()

    assert ckpt_interval % eval_interval == 0, (
        f'CKPT_INTERVAL ({ckpt_interval}) must be a multiple of eval_interval '
        f'({eval_interval}) or checkpoints record a stale val_loss')

    assert model_cfg["context_len"] == loader.chunk_size, (
        f'model context_len ({model_cfg["context_len"]}) must match dataloader chunk_size '
        f'({loader.chunk_size})')

    if checkpoint_path is not None:
        ckpt_dir = os.path.dirname(checkpoint_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

    nparams = sum(p.numel() for p in model.parameters())
    print(f'Model: {nparams / 1e6:.2f}M params | vocab {vocab_size}')

    loader.load(train_path, val_path, merges_path)

    # train
    optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    train_cfg = train_config(LEARNING_RATE, max_iters, loader.batch_size, loader.chunk_size,
                              eval_iters, SEED)

    start_iter = 0
    if resume_from and os.path.exists(resume_from):
        ck = torch.load(resume_from, map_location=device)
        assert ck['model_cfg'] == model_config(), (
            f"checkpoint model_cfg {ck['model_cfg']} != current {model_config()}")
        assert ck['vocab_size'] == vocab_size, (
            f"checkpoint vocab_size {ck['vocab_size']} != current {vocab_size}")
        model.load_state_dict(ck['model'])
        optim.load_state_dict(ck['optim'])
        start_iter = ck['it'] + 1
        print(f"resumed from iter {ck['it']} | val {ck['val_loss']:.4f}")

        ck_train_cfg = ck.get('train_cfg')
        if ck_train_cfg and ck_train_cfg != train_cfg:
            diffs = ', '.join(
                f'{k} {ck_train_cfg[k]} -> {train_cfg[k]}'
                for k in train_cfg if ck_train_cfg.get(k) != train_cfg[k]
            )
            print(f'WARNING: train_cfg differs from checkpoint: {diffs}')
    elif resume_from:
        print(f'{resume_from} not found; starting fresh')
    elif checkpoint_path is not None and os.path.exists(checkpoint_path):
        raise RuntimeError(
            f'{checkpoint_path} already exists — pass --resume to continue that run, '
            'or choose a different --run-name')

    if start_iter >= max_iters:
        print(f'checkpoint is already at iter {start_iter - 1} of {max_iters}; nothing to do')
        return

    t_start = time.time()
    t_mark = t_start
    est_loss = {'train': None, 'val': None}

    for it in range(start_iter, max_iters):
        if it % eval_interval == 0:
            est_loss = estimate_loss(model, loader, device, eval_iters)
            print(f"iter {it:5d} | {fmt_loss(est_loss, 'train')} | {fmt_loss(est_loss, 'val')}")

            # Checkpointing
            if it > 0 and checkpoint_path is not None and it % ckpt_interval == 0:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optim,
                    it,
                    vocab_size,
                    est_loss['val'],
                    on_save=on_save,
                    train_cfg=train_cfg,
                    run_name=run_name
                )

        xb, yb = loader.get_batch('train', device)
        logits, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        # throughput: CUDA kernels are async, so sync before timing or the
        # measurement just clocks Python enqueue time, not real GPU work.
        if it > 0 and it % log_interval == 0:
            if device == 'cuda':
                torch.cuda.synchronize()
            spi = (time.time() - t_mark) / log_interval
            print(f"iter {it:5d} | {spi * 1000:6.1f} ms/it "
                  f"| train-only proj ~{spi * max_iters / 60:.1f} min")
            t_mark = time.time()

    est_loss = estimate_loss(model, loader, device, eval_iters)
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            model,
            optim,
            max_iters,
            vocab_size,
            est_loss['val'],
            on_save=on_save,
            train_cfg=train_cfg,
            run_name=run_name
        )
    print(f"final | {fmt_loss(est_loss, 'train')} | {fmt_loss(est_loss, 'val')}")
    print(f"total wall time: {(time.time() - t_start) / 60:.1f} min")
    if device == 'cuda':
        print(f'peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')

    # sample after training
    start = torch.tensor([tok.encode('\n')], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        print(tok.decode(model.generate(start, gen_tokens)[0].tolist()))


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def main():
    from dotenv import load_dotenv

    repo_root = os.path.abspath(os.path.join(HERE, '..'))
    load_dotenv(os.path.join(repo_root, '.env'))
    os.chdir(repo_root)
    os.execvp('modal', ['modal', 'run', '--detach', os.path.join(HERE, 'train.py')]
              + sys.argv[1:])


# ---------------------------------------------------------------------------
# Modal: run training + inference on a remote GPU. Local entrypoint:
#   uv run nanogpt-train
# ---------------------------------------------------------------------------
REMOTE_TRAIN_DATASET = '/root/train.bin'
REMOTE_VAL_DATASET = '/root/valid.bin'
REMOTE_MERGES = '/root/bpe_merges.txt'
REMOTE_MERGES_META = os.path.splitext(REMOTE_MERGES)[0] + '.meta.json'
MERGES_META_PATH = os.path.splitext(MERGES_PATH)[0] + '.meta.json'
REMOTE_RUNS_DIR = '/runs'

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch", "regex", "pyarrow", "tiktoken")
    .add_local_file(TRAIN_BINARY, REMOTE_TRAIN_DATASET)
    .add_local_file(VAL_BINARY, REMOTE_VAL_DATASET)
    .add_local_file(MERGES_PATH, REMOTE_MERGES)
    .add_local_file(TRAIN_BINARY+'.meta.json', REMOTE_TRAIN_DATASET+'.meta.json')
    .add_local_file(VAL_BINARY+'.meta.json', REMOTE_VAL_DATASET+'.meta.json')
    .add_local_file(MERGES_META_PATH, REMOTE_MERGES_META)
    .add_local_python_source("nanogpt")
)

volume = modal.Volume.from_name('nano-gpt-runs', create_if_missing=True)

app = modal.App("nano-gpt", image=image, volumes={'/runs': volume})

@app.function(gpu="T4", timeout=7200)
def train_remote(run_name: str, resume: bool = False):
    ckpt = os.path.join(REMOTE_RUNS_DIR, run_name, 'ckpt_latest.pt')
    run(device=get_device(), train_path=REMOTE_TRAIN_DATASET, val_path=REMOTE_VAL_DATASET,
        merges_path=REMOTE_MERGES, run_name=run_name, checkpoint_path=ckpt,
        on_save=volume.commit, resume_from=ckpt if resume else None)


@app.function(gpu="T4", timeout=1800)
def smoke_remote():
    run(device=get_device(), train_path=REMOTE_TRAIN_DATASET, val_path=REMOTE_VAL_DATASET,
        merges_path=REMOTE_MERGES, run_name='smoke',
        max_iters=200, eval_interval=50, eval_iters=20, log_interval=25, gen_tokens=100)


@app.local_entrypoint()
def modal_main(smoke: bool = False, run_name: str = '', resume: bool = False):
    if not run_name:
        run_name = time.strftime('%Y%m%d-%H%M')
    print(f'run_name: {run_name}')
    if smoke:
        smoke_remote.remote()
    else:
        train_remote.remote(run_name, resume)
