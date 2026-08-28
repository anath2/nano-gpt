import math

import pytest

from nanogpt.train import lr_at

PEAK, MIN, WARM, TOTAL = 3e-4, 3e-5, 100, 15000


def test_returns_plain_floats_across_the_whole_range():
    for it in (0, 1, WARM - 1, WARM, WARM + 1, TOTAL // 2, TOTAL - 1, TOTAL, TOTAL * 2):
        lr = lr_at(it, PEAK, WARM, TOTAL, MIN)
        assert isinstance(lr, float)
        assert math.isfinite(lr)


def test_warmup_is_linear_and_never_starts_at_zero():
    assert lr_at(0, PEAK, WARM, TOTAL, MIN) > 0, 'iter 0 must not be a dead step'
    assert lr_at(WARM - 1, PEAK, WARM, TOTAL, MIN) == pytest.approx(PEAK)
    # equal spacing between consecutive warmup steps
    steps = [lr_at(i, PEAK, WARM, TOTAL, MIN) for i in range(WARM)]
    deltas = [b - a for a, b in zip(steps, steps[1:])]
    assert all(d == pytest.approx(deltas[0]) for d in deltas)


def test_peak_reached_at_end_of_warmup():
    assert lr_at(WARM, PEAK, WARM, TOTAL, MIN) == pytest.approx(PEAK)


def test_decays_monotonically_after_warmup():
    prev = lr_at(WARM, PEAK, WARM, TOTAL, MIN)
    for it in range(WARM, TOTAL, 37):
        cur = lr_at(it, PEAK, WARM, TOTAL, MIN)
        assert cur <= prev + 1e-12
        prev = cur


def test_stays_in_the_min_peak_band_after_warmup():
    """`min_lr` floors the *decay*, not the warmup ramp -- warmup legitimately
    starts at peak/warmup_iters, which is below min_lr."""
    for it in range(WARM, TOTAL + 1, 53):
        assert MIN - 1e-12 <= lr_at(it, PEAK, WARM, TOTAL, MIN) <= PEAK + 1e-12


def test_warmup_ramps_up_from_below_min_lr():
    assert lr_at(0, PEAK, WARM, TOTAL, MIN) < MIN
    assert lr_at(WARM, PEAK, WARM, TOTAL, MIN) == pytest.approx(PEAK)


def test_ends_at_min_lr_and_stays_there():
    assert lr_at(TOTAL, PEAK, WARM, TOTAL, MIN) == pytest.approx(MIN)
    assert lr_at(TOTAL * 3, PEAK, WARM, TOTAL, MIN) == pytest.approx(MIN)


def test_short_run_traverses_the_whole_cosine():
    """A smoke run passes max_iters=200; it must see the full schedule, not a
    sliver of a 15000-iter curve."""
    assert lr_at(200, PEAK, 10, 200, MIN) == pytest.approx(MIN)
    assert lr_at(10, PEAK, 10, 200, MIN) == pytest.approx(PEAK)
