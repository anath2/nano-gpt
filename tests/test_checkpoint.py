"""Checkpoint writing and the resume-time config diff"""
import json
import os

import torch

from nanogpt.train import save_checkpoint, train_cfg_diff, train_config


def _cfg(**over):
    base = dict(lr=3e-4, max_iters=15000, batch_size=64, chunk_size=256,
                eval_iters=50, seed=567, warmup_iters=100, min_lr=3e-5)
    base.update(over)
    return train_config(**base)


def test_diff_is_empty_for_identical_configs():
    assert train_cfg_diff(_cfg(), _cfg()) == ''


def test_diff_reports_changed_values():
    out = train_cfg_diff(_cfg(), _cfg(lr=1e-4))
    assert 'lr' in out and '0.0001' in out


def _save(tmp_path, model, **kw):
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / 'run' / 'ckpt_latest.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(path), model, optim, it=500, vocab_size=256, val_loss=3.5,
                    train_cfg=_cfg(), run_name='testrun', **kw)
    return path


def test_writes_checkpoint_and_config_sidecar(tmp_path, tiny_model):
    path = _save(tmp_path, tiny_model)
    assert path.exists()
    assert (path.parent / 'config.json').exists()


def test_leaves_no_temp_file_behind(tmp_path, tiny_model):
    path = _save(tmp_path, tiny_model)
    assert not (path.parent / 'ckpt_latest.pt.tmp').exists()
    assert sorted(p.name for p in path.parent.iterdir()) == ['ckpt_latest.pt', 'config.json']


def test_checkpoint_round_trips(tmp_path, tiny_model):
    path = _save(tmp_path, tiny_model)
    ck = torch.load(str(path), map_location='cpu')
    assert ck['it'] == 500
    assert ck['vocab_size'] == 256
    assert ck['val_loss'] == 3.5
    assert ck['run_name'] == 'testrun'
    assert ck['train_cfg'] == _cfg()
    assert 'model' in ck and 'optim' in ck


def test_weights_survive_the_round_trip(tmp_path, tiny_model):
    path = _save(tmp_path, tiny_model)
    ck = torch.load(str(path), map_location='cpu')
    for k, v in tiny_model.state_dict().items():
        assert torch.equal(ck['model'][k], v), k


def test_config_sidecar_is_json_and_carries_no_tensors(tmp_path, tiny_model):
    path = _save(tmp_path, tiny_model)
    cfg = json.loads((path.parent / 'config.json').read_text())
    assert cfg['run_name'] == 'testrun'
    assert cfg['it'] == 500
    assert cfg['train_cfg'] == _cfg()
    assert 'model' not in cfg and 'optim' not in cfg


def test_on_save_hook_fires_after_the_file_is_in_place(tmp_path, tiny_model):
    seen = []
    path = tmp_path / 'run' / 'ckpt_latest.pt'
    path.parent.mkdir(parents=True)
    optim = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    save_checkpoint(str(path), tiny_model, optim, it=1, vocab_size=256, val_loss=1.0,
                    on_save=lambda: seen.append(os.path.exists(str(path))),
                    train_cfg=_cfg(), run_name='r')
    assert seen == [True]


def test_overwriting_an_existing_checkpoint_keeps_it_loadable(tmp_path, tiny_model):
    path = _save(tmp_path, tiny_model)
    optim = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    save_checkpoint(str(path), tiny_model, optim, it=1000, vocab_size=256,
                    val_loss=3.0, train_cfg=_cfg(), run_name='testrun')
    assert torch.load(str(path), map_location='cpu')['it'] == 1000
