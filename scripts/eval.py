"""Capability evaluation for a nanoGPT checkpoint.

    uv run python scripts/eval.py
    uv run python scripts/eval.py --ckpt checkpoints/<run>/ckpt_latest.pt --json
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

from nanogpt.data import TOKEN_DTYPE
from nanogpt.model import create_model, model_config
from nanogpt.tokenizer import BPETokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
CKPT_PATH = os.path.join(REPO_ROOT, 'checkpoints', '20260821-0235', 'ckpt_latest.pt')
MERGES_PATH = os.path.join(DATA_DIR, 'bpe_merges.txt')
VAL_PATH = os.path.join(DATA_DIR, 'valid.bin')

# Hand-written minimal pairs: (phenomenon, grammatical, ungrammatical). Small and
# author-written, so treat the score as indicative rather than as a benchmark.
MINIMAL_PAIRS = [
    ('agreement', ' The player was named the most valuable player .',
                  ' The player were named the most valuable player .'),
    ('agreement', ' The songs were released in 2005 .',
                  ' The songs was released in 2005 .'),
    ('agreement', ' He has written several novels .',
                  ' He have written several novels .'),
    ('determiner', ' These books are popular .', ' This books are popular .'),
    ('determiner', ' Many people attended the ceremony .',
                   ' Many person attended the ceremony .'),
    ('reflexive', ' She taught herself to play .', ' She taught themselves to play .'),
    ('reflexive', ' They defended themselves .', ' They defended himself .'),
    ('irregular_past', ' The team won the championship .',
                       ' The team winned the championship .'),
    ('irregular_past', ' He went to the university .', ' He goed to the university .'),
    ('word_order', ' The film was released in June .',
                   ' The film released was in June .'),
    ('word_order', ' He is a famous writer .', ' He a famous is writer .'),
    ('aux_verb', ' She did not know the answer .', ' She did not knew the answer .'),
    ('article', ' He was an officer in the army .', ' He was a officer in the army .'),
    ('plural', ' Two children were playing .', ' Two child were playing .'),
]

FACTUAL_PROMPTS = [
    ' The capital of France is',
    ' The capital of Germany is',
    ' William Shakespeare was an English',
    ' The Earth orbits the',
    ' World War II ended in',
    ' The largest ocean is the',
]

BINDING_PROMPTS = [
    ' The password is zebra . The password is',
    ' Tom is taller than Sam . Sam is taller than Bob . The tallest is',
    ' All birds can fly . A robin is a bird . Therefore a robin can',
    ' John gave the book to Mary . She thanked',
]


def load(ckpt_path, merges_path, device):
    tok = BPETokenizer.load(merges_path)
    ck = torch.load(ckpt_path, map_location=device)
    if ck['model_cfg'] != model_config():
        print(f"warning: checkpoint model_cfg {ck['model_cfg']} != current "
              f"{model_config()}", file=sys.stderr)
    model = create_model(ck['vocab_size']).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, tok, ck


@torch.no_grad()
def sequence_logprob(model, tok, text):
    ids = tok.encode(text)[:model.context_len]
    if len(ids) < 2:
        return 0.0
    x = torch.tensor([ids])
    logits, _ = model(x)
    lp = F.log_softmax(logits[0, :-1], dim=-1)
    return lp.gather(1, x[0, 1:, None]).sum().item()


@torch.no_grad()
def top_next(model, tok, text, n=5):
    ids = tok.encode(text)[-model.context_len:]
    logits, _ = model(torch.tensor([ids]))
    probs = F.softmax(logits[0, -1], dim=-1)
    v, i = probs.topk(n)
    return [[tok.decode([j.item()]), round(p.item(), 4)] for p, j in zip(v, i)]


def eval_grammar(model, tok):
    """Fraction of minimal pairs where the grammatical member scores higher."""
    results, by_kind = [], {}
    for kind, good, bad in MINIMAL_PAIRS:
        margin = sequence_logprob(model, tok, good) - sequence_logprob(model, tok, bad)
        results.append({'phenomenon': kind, 'margin': round(margin, 3),
                        'correct': margin > 0, 'good': good.strip()})
        hit, tot = by_kind.get(kind, (0, 0))
        by_kind[kind] = (hit + (margin > 0), tot + 1)
    correct = sum(r['correct'] for r in results)
    return {
        'correct': correct,
        'total': len(results),
        'accuracy': round(correct / len(results), 4),
        'mean_margin': round(sum(r['margin'] for r in results) / len(results), 3),
        'by_phenomenon': {k: f'{h}/{t}' for k, (h, t) in sorted(by_kind.items())},
        'pairs': results,
    }


@torch.no_grad()
def eval_context_use(model, val, n=1000, seed=1):
    """NLL of one fixed target token as the visible context grows.

    The target is held constant across context lengths on purpose: an earlier
    version grew the window and moved the target together, which made the curve
    look flat past 16 tokens and suggested the model ignored long context.
    """
    ctx_len = model.context_len
    torch.manual_seed(seed)
    starts = torch.randint(0, len(val) - ctx_len - 2, (n,))
    full = torch.stack([val[s:s + ctx_len].to(torch.int64) for s in starts])
    target = full[:, -1]

    def nll(block):
        logits, _ = model(block)
        lp = F.log_softmax(logits[:, -2], dim=-1)
        return -lp.gather(1, target[:, None]).squeeze(1)

    curve = {}
    for k in (1, 2, 4, 8, 16, 32, 64, 128, ctx_len - 1):
        curve[k] = round(nll(full[:, -(k + 1):]).mean().item(), 4)

    # How much is the distant context worth, and how much of that is word order?
    keep = min(128, ctx_len - 1)
    ctx = full[:, -(keep + 1):]
    distant = keep - 8
    true_nll = nll(ctx).mean().item()
    shuffled = ctx.clone()
    shuffled[:, :distant] = shuffled[:, :distant][:, torch.randperm(distant)]
    alien = ctx.clone()
    alien[:, :distant] = full[torch.randperm(n), :distant]
    shuffled_nll = nll(shuffled).mean().item()
    alien_nll = nll(alien).mean().item()

    return {
        'nll_by_context_length': curve,
        'controls_at_context': keep,
        'true_context_nll': round(true_nll, 4),
        'shuffled_distant_nll': round(shuffled_nll, 4),
        'foreign_distant_nll': round(alien_nll, 4),
        'distant_context_worth_nats': round(alien_nll - true_nll, 4),
        'of_which_order_dependent': round(shuffled_nll - true_nll, 4),
        'of_which_topic_only': round(alien_nll - shuffled_nll, 4),
        'n_positions': n,
    }


@torch.no_grad()
def eval_induction(model, vocab_size, trials=20, length=60, seed=0):
    """Can it exploit a sequence it has already seen verbatim in context?

    Random tokens are far off-distribution, so both numbers sit above ln(vocab);
    what matters is the gap between the first and second copy.
    """
    torch.manual_seed(seed)
    firsts, seconds = [], []
    for _ in range(trials):
        seq = torch.randint(0, vocab_size, (length,))
        doubled = torch.cat([seq, seq])[None, :]
        logits, _ = model(doubled)
        lp = F.log_softmax(logits[0, :-1], dim=-1)
        nll = -lp.gather(1, doubled[0, 1:, None]).squeeze(1)
        firsts.append(nll[:length - 1].mean().item())
        seconds.append(nll[length:].mean().item())
    first = sum(firsts) / len(firsts)
    second = sum(seconds) / len(seconds)
    return {
        'first_copy_nll': round(first, 4),
        'second_copy_nll': round(second, 4),
        'improvement_nats': round(first - second, 4),
        'ln_vocab_reference': round(math.log(vocab_size), 4),
        'note': 'a strong induction head would drive second_copy_nll toward 0',
    }


@torch.no_grad()
def eval_val_loss(model, val, n=200, seed=2):
    ctx = model.context_len
    torch.manual_seed(seed)
    starts = torch.randint(0, len(val) - ctx - 2, (n,))
    total = 0.0
    for s in starts:
        block = val[s:s + ctx + 1].to(torch.int64)[None, :]
        _, loss = model(block[:, :-1], block[:, 1:])
        total += loss.item()
    return total / n


def eval_prompts(model, tok, prompts):
    return [{'prompt': p.strip(), 'top': top_next(model, tok, p, 5)} for p in prompts]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', default=CKPT_PATH)
    parser.add_argument('--merges', default=MERGES_PATH)
    parser.add_argument('--val', default=VAL_PATH)
    parser.add_argument('--out', default=None,
                        help='where to write eval.json (default: beside the checkpoint)')
    parser.add_argument('--json', action='store_true', help='print the JSON too')
    parser.add_argument('--positions', type=int, default=1000,
                        help='sampled positions for the context-use eval')
    args = parser.parse_args(argv)

    for path in (args.ckpt, args.merges, args.val):
        if not os.path.exists(path):
            print(f'error: {path} not found', file=sys.stderr)
            return 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok, ck = load(args.ckpt, args.merges, device)

    meta_path = args.val + '.meta.json'
    chars_per_token = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        chars_per_token = meta.get('chars_per_token')
        val = torch.from_file(args.val, shared=True, size=meta['n_tokens'],
                              dtype=TOKEN_DTYPE)
    else:
        print(f'error: {meta_path} not found', file=sys.stderr)
        return 1

    val_nll = eval_val_loss(model, val)
    report = {
        'checkpoint': os.path.relpath(args.ckpt, REPO_ROOT),
        'run_name': ck.get('run_name'),
        'iteration': ck.get('it'),
        'model_cfg': ck.get('model_cfg'),
        'train_cfg': ck.get('train_cfg'),
        'loss': {
            'val_nll': round(val_nll, 4),
            'recorded_val_loss': ck.get('val_loss'),
            'bits_per_char': (round(val_nll / math.log(2) / chars_per_token, 4)
                              if chars_per_token else None),
        },
        'grammar': eval_grammar(model, tok),
        'context_use': eval_context_use(model, val, n=args.positions),
        'induction': eval_induction(model, ck['vocab_size']),
        'factual_recall': eval_prompts(model, tok, FACTUAL_PROMPTS),
        'binding_and_deduction': eval_prompts(model, tok, BINDING_PROMPTS),
    }

    g, c, i = report['grammar'], report['context_use'], report['induction']
    print(f"checkpoint       {report['checkpoint']}")
    print(f"run / iteration  {report['run_name']} @ {report['iteration']}")
    print(f"val NLL          {report['loss']['val_nll']}  "
          f"({report['loss']['bits_per_char']} bits/char)")
    print(f"grammar          {g['correct']}/{g['total']} "
          f"(mean margin {g['mean_margin']} nats)")
    print(f"  by phenomenon  {g['by_phenomenon']}")
    print(f"context use      NLL {c['nll_by_context_length'][1]} @1 tok -> "
          f"{c['nll_by_context_length'][model.context_len - 1]} @{model.context_len - 1} tok")
    print(f"  distant ctx    worth {c['distant_context_worth_nats']} nats "
          f"({c['of_which_order_dependent']} order, {c['of_which_topic_only']} topic)")
    print(f"induction        {i['first_copy_nll']} -> {i['second_copy_nll']} "
          f"({i['improvement_nats']:+} nats)")
    print('factual recall')
    for row in report['factual_recall']:
        top = ', '.join(f'{t!r}:{p}' for t, p in row['top'][:3])
        print(f"  {row['prompt'][:36]:38s} {top}")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), 'eval.json')
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nwrote {os.path.relpath(out, REPO_ROOT)}')
    if args.json:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
