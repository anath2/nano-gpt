import pytest
import torch
import torch.nn.functional as F

from nanogpt.model import apply_sampling_controls


def _reference_generate(model, x, seq_len):
    for _ in range(seq_len):
        x_context = x[:, -model.context_len:]
        logits, _ = model(x_context)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        x = torch.concat((x, torch.multinomial(probs, num_samples=1)), dim=1)
    return x


def test_bare_call_matches_the_original_implementation(tiny_model, start_ids):
    torch.manual_seed(3)
    expected = _reference_generate(tiny_model, start_ids, 20)
    torch.manual_seed(3)
    got = tiny_model.generate(start_ids, 20)
    assert torch.equal(expected, got)


def test_explicit_defaults_match_the_bare_call(tiny_model, start_ids):
    torch.manual_seed(3)
    bare = tiny_model.generate(start_ids, 20)
    torch.manual_seed(3)
    explicit = tiny_model.generate(start_ids, 20, temperature=1.0, top_k=None,
                                   repetition_penalty=1.0)
    assert torch.equal(bare, explicit)


def test_top_k_zero_means_no_filtering_and_does_not_crash(tiny_model, start_ids):
    """Regression: torch.topk(logits, 0) returns empty, and [:, -1] indexed off it."""
    torch.manual_seed(3)
    raw = tiny_model.generate(start_ids, 20)
    torch.manual_seed(3)
    zero_k = tiny_model.generate(start_ids, 20, top_k=0)
    assert torch.equal(raw, zero_k)


def test_top_k_larger_than_vocab_is_clamped(tiny_model, start_ids):
    torch.manual_seed(3)
    tiny_model.generate(start_ids, 5, top_k=10 ** 6)  # must not raise


def test_top_k_restricts_the_support(tiny_model, start_ids):
    """With top_k=1 every sampled token must be the argmax."""
    x = start_ids
    for _ in range(10):
        logits, _ = tiny_model(x[:, -tiny_model.context_len:])
        expected = logits[0, -1].argmax().item()
        x = tiny_model.generate(x, 1, top_k=1)
        assert x[0, -1].item() == expected


@pytest.mark.parametrize('kwargs', [
    {'temperature': 0}, {'temperature': -1.0}, {'repetition_penalty': 0},
    {'repetition_penalty': -0.5},
])
def test_invalid_controls_raise(tiny_model, start_ids, kwargs):
    with pytest.raises(ValueError):
        tiny_model.generate(start_ids, 1, **kwargs)


def test_repetition_penalty_lowers_every_seen_logit_regardless_of_sign():
    """A flat `l / penalty` would move negative logits *up* toward zero, making
    already-seen unlikely tokens more likely -- the opposite of the intent.
    Synthetic logits so both branches are guaranteed to be exercised."""
    logits = torch.tensor([[2.0, -2.0, 0.5, -0.5, 8.0]])
    x_context = torch.tensor([[0, 1, 2, 3]])          # token 4 is unseen
    before = logits.clone()
    after = apply_sampling_controls(logits.clone(), x_context, repetition_penalty=2.0)
    seen = [0, 1, 2, 3]
    assert (after[0, seen] < before[0, seen]).all(), 'every seen logit must move down'
    assert after[0, 4] == before[0, 4], 'unseen logits must be untouched'


def test_repetition_penalty_lowers_probability_of_a_repeated_token(tiny_model):
    x = torch.tensor([[7, 7, 7, 7, 7, 7, 7, 7]], dtype=torch.long)
    logits, _ = tiny_model(x)
    base = F.softmax(logits[:, -1], dim=-1)[0, 7].item()
    shaped = apply_sampling_controls(logits[:, -1].clone(), x, repetition_penalty=1.5)
    assert F.softmax(shaped, dim=-1)[0, 7].item() < base


def test_temperature_sharpens_and_flattens():
    logits = torch.tensor([[3.0, 1.0, 0.0]])
    x_context = torch.tensor([[9]])
    def top(t):
        shaped = apply_sampling_controls(logits.clone(), x_context, temperature=t)
        return F.softmax(shaped, dim=-1)[0, 0].item()

    assert top(0.5) > top(1.0) > top(2.0)


def test_generate_is_generate_stream_accumulated(tiny_model, start_ids):
    torch.manual_seed(11)
    full = tiny_model.generate(start_ids, 15, temperature=0.8, top_k=4)
    torch.manual_seed(11)
    x = start_ids
    for tok in tiny_model.generate_stream(start_ids, 15, temperature=0.8, top_k=4):
        x = torch.concat((x, tok), dim=1)
    assert torch.equal(full, x)


def test_generate_stream_yields_one_token_at_a_time(tiny_model, start_ids):
    toks = list(tiny_model.generate_stream(start_ids, 7))
    assert len(toks) == 7
    assert all(t.shape == (1, 1) for t in toks)


def test_generate_returns_prompt_plus_new_tokens(tiny_model, start_ids):
    out = tiny_model.generate(start_ids, 12)
    assert out.shape == (1, start_ids.shape[1] + 12)
    assert torch.equal(out[:, :start_ids.shape[1]], start_ids)


def test_generation_respects_the_context_window(tiny_model):
    """Prompt longer than context_len must not blow up the position embedding."""
    long_prompt = torch.randint(0, 256, (1, tiny_model.context_len + 20))
    out = tiny_model.generate(long_prompt, 5)
    assert out.shape[1] == long_prompt.shape[1] + 5
