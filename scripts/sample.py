#!/usr/bin/env python3
"""Sample text from a trained nanogpt checkpoint.

Command:
    uv run scripts/sample.py [--ckpt PATH] [--merges PATH] [--prompt STR]
        [--max-tokens N] [--temperature F] [--top-k N] [--repetition-penalty F]
        [--num-samples N] [--seed N] [--modal]
"""

import argparse
import os
import sys

import modal
import torch

from nanogpt.model import create_model, model_config
from nanogpt.tokenizer import BPETokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
CKPT_PATH = os.path.join(REPO_ROOT, 'checkpoints', '20260820-1957', 'ckpt_latest.pt')
MERGES_PATH = os.path.join(DATA_DIR, 'bpe_merges.txt')
MERGES_META_PATH = os.path.splitext(MERGES_PATH)[0] + '.meta.json'

MAX_TOKENS = 300
TEMPERATURE = 0.8
TOP_K = 40
REPETITION_PENALTY = 1.15
NUM_SAMPLES = 1


def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(ckpt_path: str, merges_path: str, device: str):
    tok = BPETokenizer.load(merges_path)
    ck = torch.load(ckpt_path, map_location=device)
    assert ck['model_cfg'] == model_config(), (
        f"checkpoint model_cfg {ck['model_cfg']} != current {model_config()}")
    assert ck['vocab_size'] == tok.get_vocab_size(), (
        f"checkpoint vocab_size {ck['vocab_size']} != current {tok.get_vocab_size()}")

    model = create_model(ck['vocab_size']).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, tok, ck


def generate_samples(model, tok, prompt, max_tokens, temperature, top_k,
                      repetition_penalty, num_samples, seed, device) -> list[str]:
    if seed is not None:
        torch.manual_seed(seed)

    ids = tok.encode(prompt)
    assert ids, f'tok.encode(prompt) produced no tokens for prompt={prompt!r}'
    start = torch.tensor([ids], dtype=torch.long, device=device)

    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            out = model.generate(start, max_tokens, temperature=temperature, top_k=top_k,
                                  repetition_penalty=repetition_penalty)
            samples.append(tok.decode(out[0].tolist()))
    return samples


def stream_sample(model, tok, prompt, max_tokens, temperature, top_k,
                  repetition_penalty, device) -> None:
    ids = tok.encode(prompt)
    assert ids, f'tok.encode(prompt) produced no tokens for prompt={prompt!r}'
    start = torch.tensor([ids], dtype=torch.long, device=device)

    out = list(ids)
    text = tok.decode(out)
    print(text, end='', flush=True)
    printed = len(text)

    with torch.no_grad():
        for next_tok in model.generate_stream(start, max_tokens, temperature=temperature,
                                              top_k=top_k,
                                              repetition_penalty=repetition_penalty):
            out.append(next_tok.item())
            text = tok.decode(out)
            if text.endswith('\ufffd'):
                continue  # incomplete multi-byte char; wait for the next token
            print(text[printed:], end='', flush=True)
            printed = len(text)

    print(tok.decode(out)[printed:])  # flush any held-back tail, plus a newline


def print_header(it, val_loss, temperature, top_k, repetition_penalty, device):
    print(f'ckpt iter {it} | val {val_loss:.4f} | temp {temperature} | '
          f'top_k {top_k} | rep_penalty {repetition_penalty} | device {device}')


# -----------------------------
# Modal: sample on a remote T4.
# -----------------------------
REMOTE_MERGES = '/root/bpe_merges.txt'
REMOTE_MERGES_META = os.path.splitext(REMOTE_MERGES)[0] + '.meta.json'
REMOTE_RUNS_DIR = '/runs'

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch", "regex")
    .add_local_file(MERGES_PATH, REMOTE_MERGES)
    .add_local_file(MERGES_META_PATH, REMOTE_MERGES_META)
    .add_local_python_source("nanogpt")
)

volume = modal.Volume.from_name('nano-gpt-runs')  # must already exist

app = modal.App("nano-gpt-sample", image=image)


@app.function(gpu="T4", timeout=600, volumes={'/runs': volume})
def sample_remote(run_name, prompt, max_tokens, temperature, top_k, repetition_penalty,
                   num_samples, seed):
    device = get_device()
    ckpt_path = os.path.join(REMOTE_RUNS_DIR, run_name, 'ckpt_latest.pt')
    model, tok, ck = load_model(ckpt_path, REMOTE_MERGES, device)
    samples = generate_samples(model, tok, prompt, max_tokens, temperature, top_k,
                                repetition_penalty, num_samples, seed, device)
    return {'it': ck['it'], 'val_loss': ck['val_loss'], 'samples': samples, 'device': device}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Sample text from a trained nanogpt checkpoint.')
    parser.add_argument('--ckpt', default=CKPT_PATH,
                        help=f'checkpoint path, run-scoped layout (default: {CKPT_PATH})')
    parser.add_argument('--merges', default=MERGES_PATH,
                        help=f'BPE merges file (default: {MERGES_PATH})')
    parser.add_argument('--prompt', default='\n', help="generation prompt (default: '\\n')")
    parser.add_argument('--max-tokens', type=int, default=MAX_TOKENS,
                        help=f'tokens to generate (default: {MAX_TOKENS})')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE,
                        help=f'sampling temperature (default: {TEMPERATURE})')
    parser.add_argument('--top-k', type=int, default=TOP_K,
                        help=f'keep only the top-k logits, 0 disables (default: {TOP_K})')
    parser.add_argument('--repetition-penalty', type=float, default=REPETITION_PENALTY,
                        help='CTRL-style penalty on repeated tokens, 1.0 disables '
                             f'(default: {REPETITION_PENALTY})')
    parser.add_argument('--num-samples', type=int, default=NUM_SAMPLES,
                        help=f'number of samples to draw (default: {NUM_SAMPLES})')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed (default: unset)')
    parser.add_argument('--modal', action='store_true',
                        help='sample on a remote Modal T4 instead of locally')
    args = parser.parse_args(argv)

    prompt = args.prompt if args.prompt.strip() else '\n'
    top_k = args.top_k if args.top_k > 0 else None

    if args.modal:
        run_name = os.path.basename(os.path.dirname(args.ckpt))
        if not run_name:
            print('error: --modal needs a run-scoped --ckpt, e.g. '
                  'checkpoints/<run_name>/ckpt_latest.pt', file=sys.stderr)
            return 1
        with app.run():
            result = sample_remote.remote(
                run_name, prompt, args.max_tokens, args.temperature, top_k,
                args.repetition_penalty, args.num_samples, args.seed)
        it, val_loss = result['it'], result['val_loss']
        samples, device = result['samples'], result['device']
    else:
        if not os.path.exists(args.ckpt):
            print(f'error: {args.ckpt} not found', file=sys.stderr)
            return 1
        if not os.path.exists(args.merges):
            print(f'error: {args.merges} not found', file=sys.stderr)
            return 1

        device = get_device()
        model, tok, ck = load_model(args.ckpt, args.merges, device)
        it, val_loss = ck['it'], ck['val_loss']

        # Local path streams: header first, then tokens as they arrive. Seed once for
        # the whole loop, matching generate_samples' single-stream behaviour.
        print_header(it, val_loss, args.temperature, top_k, args.repetition_penalty, device)
        if args.seed is not None:
            torch.manual_seed(args.seed)
        for i in range(args.num_samples):
            if args.num_samples > 1:
                print(f'--- sample {i + 1} ---')
            stream_sample(model, tok, prompt, args.max_tokens, args.temperature,
                          top_k, args.repetition_penalty, device)
        return 0

    print_header(it, val_loss, args.temperature, top_k, args.repetition_penalty, device)
    for i, sample in enumerate(samples):
        if args.num_samples > 1:
            print(f'--- sample {i + 1} ---')
        print(sample)

    return 0


if __name__ == '__main__':
    sys.exit(main())
