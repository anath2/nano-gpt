import hashlib
import json
import os

import pytest

from nanogpt.data import load_tokens
from nanogpt.tokenizer import BPETokenizer


def test_encode_decode_round_trips_ascii(tiny_tokenizer):
    for text in ('hello world', ' = = Section = = ', 'a @-@ b', '12345 . , ; !'):
        assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text


def test_encode_decode_round_trips_multibyte(tiny_tokenizer):
    text = ' 1636 – 30 café — naïve résumé'  # characters spanning mulitple bytes
    assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text


def test_decoding_tokens_one_at_a_time_can_corrupt_multibyte_characters(tiny_tokenizer):
    ids = tiny_tokenizer.encode(' café — naïve')
    per_token = ''.join(tiny_tokenizer.decode([i]) for i in ids)
    assert per_token != tiny_tokenizer.decode(ids)
    assert '�' in per_token


def test_streaming_hold_back_is_lossless(tiny_tokenizer):
    """The exact loop `scripts/sample.py:stream_sample` runs."""
    src = ' 1636 – 30 café — naïve résumé'
    ids = tiny_tokenizer.encode(src)
    out, printed, emitted = [], 0, ''
    for i in ids:
        out.append(i)
        text = tiny_tokenizer.decode(out)
        if text.endswith('�'):
            continue
        emitted += text[printed:]
        printed = len(text)
    emitted += tiny_tokenizer.decode(out)[printed:]
    assert emitted == src
    assert '�' not in emitted


def test_merges_sidecar_replaces_the_extension(tmp_path, tiny_tokenizer):
    path = tmp_path / 'merges.txt'
    tiny_tokenizer.save(str(path))
    assert (tmp_path / 'merges.meta.json').exists(), 'splitext convention'
    assert not (tmp_path / 'merges.txt.meta.json').exists(), 'not the append convention'


def test_save_load_round_trips(tmp_path, tiny_tokenizer):
    path = tmp_path / 'merges.txt'
    tiny_tokenizer.save(str(path))
    loaded = BPETokenizer.load(str(path))
    assert loaded.get_vocab_size() == tiny_tokenizer.get_vocab_size()
    assert loaded.pat_str == tiny_tokenizer.pat_str
    text = ' the quick brown fox'
    assert loaded.encode(text) == tiny_tokenizer.encode(text)


def test_load_without_its_sidecar_raises(tmp_path, tiny_tokenizer):
    path = tmp_path / 'merges.txt'
    tiny_tokenizer.save(str(path))
    os.remove(tmp_path / 'merges.meta.json')
    with pytest.raises(FileNotFoundError, match='meta.json'):
        BPETokenizer.load(str(path))


def _write_corpus(tmp_path, tokens, merges_sha, n_tokens=None):
    bin_path = tmp_path / 'train.bin'
    with open(bin_path, 'wb') as f:
        f.write(b''.join(int(t).to_bytes(2, 'little', signed=True) for t in tokens))
    meta = {'n_tokens': n_tokens if n_tokens is not None else len(tokens),
            'dtype': 'int16', 'vocab_size': 256, 'merges_sha256': merges_sha}
    with open(str(bin_path) + '.meta.json', 'w') as f:
        json.dump(meta, f)
    return bin_path


def test_bin_sidecar_appends_to_the_full_filename(tmp_path, tiny_tokenizer):
    merges = tmp_path / 'merges.txt'
    tiny_tokenizer.save(str(merges))
    sha = hashlib.sha256(merges.read_bytes()).hexdigest()
    bin_path = _write_corpus(tmp_path, [1, 2, 3, 4], sha)
    assert (tmp_path / 'train.bin.meta.json').exists(), 'append convention'
    data = load_tokens(str(bin_path), str(merges))
    assert data.tolist() == [1, 2, 3, 4]


def test_bin_with_mismatched_merges_hash_raises(tmp_path, tiny_tokenizer):
    merges = tmp_path / 'merges.txt'
    tiny_tokenizer.save(str(merges))
    bin_path = _write_corpus(tmp_path, [1, 2, 3, 4], 'deadbeef' * 8)
    with pytest.raises(ValueError, match='different merges file'):
        load_tokens(str(bin_path), str(merges))


def test_bin_with_wrong_token_count_raises(tmp_path, tiny_tokenizer):
    merges = tmp_path / 'merges.txt'
    tiny_tokenizer.save(str(merges))
    sha = hashlib.sha256(merges.read_bytes()).hexdigest()
    bin_path = _write_corpus(tmp_path, [1, 2, 3, 4], sha, n_tokens=99)
    with pytest.raises(ValueError):
        load_tokens(str(bin_path), str(merges))


def test_bin_without_sidecar_raises(tmp_path, tiny_tokenizer):
    merges = tmp_path / 'merges.txt'
    tiny_tokenizer.save(str(merges))
    sha = hashlib.sha256(merges.read_bytes()).hexdigest()
    bin_path = _write_corpus(tmp_path, [1, 2, 3], sha)
    os.remove(str(bin_path) + '.meta.json')
    with pytest.raises(FileNotFoundError):
        load_tokens(str(bin_path), str(merges))
