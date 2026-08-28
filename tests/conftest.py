import pytest
import torch

from nanogpt.model import Transformer
from nanogpt.tokenizer import BPETokenizer

VOCAB = 256
CTX = 32


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    model = Transformer(context_len=CTX, vocab_size=VOCAB, n_embed=32, n_layer=2, n_head=2)
    model.eval()
    return model


@pytest.fixture
def tiny_tokenizer():
    ranks = {bytes([i]): i for i in range(256)}
    return BPETokenizer(ranks, pat_str=r"\S+|\s+")


@pytest.fixture
def start_ids():
    return torch.tensor([[1, 2, 3]], dtype=torch.long)
