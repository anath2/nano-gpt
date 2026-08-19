import torch
import torch.nn as nn
import torch.nn.functional as F

DROPOUT = 0.2
CONTEXT_LEN = 256
N_EMBED = 384
N_HEAD = 6
N_LAYER = 6


class FeedForward(nn.Module):
    """Feed forward block"""

    def __init__(self, n_embed):
        super().__init__()
        proj = nn.Linear(4 * n_embed, n_embed)
        proj.RESIDUAL_PROJ = True  # depth-scaled init, see Transformer._init_weights
        self.net = nn.Sequential(
           nn.Linear(n_embed,  4 * n_embed),  # GPT2 paper
           nn.ReLU(),
           proj,
           nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    """self attention head"""

    # register_buffer() assigns through Module.__setattr__, so the type has to
    # be declared here for tril to read as a Tensor rather than a Module
    tril: torch.Tensor

    def __init__(self, head_size, n_embed):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, self.head_size, bias=False)
        self.query = nn.Linear(n_embed, self.head_size, bias=False)
        self.value = nn.Linear(n_embed, self.head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)
        self.register_buffer('tril', torch.tril(torch.ones(CONTEXT_LEN, CONTEXT_LEN)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadedAttn(nn.Module):
    """Multiheaded attention"""

    def __init__(self, head_size, n_heads):
        super().__init__()
        n_embed = head_size * n_heads
        self.heads = nn.ModuleList([Head(head_size, n_embed) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.proj.RESIDUAL_PROJ = True  # depth-scaled init, see Transformer._init_weights
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, n_embed, n_head):
        super().__init__()
        assert n_embed % n_head == 0
        head_size = n_embed // n_head
        self.mha = MultiHeadedAttn(head_size, n_head)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class Transformer(nn.Module):
    """Naive GPT2 transformer"""

    def __init__(self, context_len, vocab_size, n_embed, n_layer, n_head):
        super().__init__()
        self._context_len = context_len
        self.n_layer = n_layer
        self.embed = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Embedding(self.context_len, n_embed)
        self.blocks = nn.Sequential(
          *[Block(n_embed, n_head) for
              _ in range(n_layer)],
          nn.LayerNorm(n_embed)
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.apply(self._init_weights)

        assert self.lm_head.weight.shape == self.embed.weight.shape
        self.lm_head.weight = self.embed.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, 'RESIDUAL_PROJ', False):
                std *= (2 * self.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # nn.LayerNorm is left alone: PyTorch already inits weight=1, bias=0.

    @property
    def context_len(self):
        return self._context_len

    def forward(self, idx, target=None):
        B, T  = idx.shape
        embs = self.embed(idx)
        pos_embs = self.pos_embed(torch.arange(T, device=idx.device))
        idx =  embs + pos_embs
        idx = self.blocks(idx)

        logits = self.lm_head(idx)

        if target is not None:
            # Calculate cross-entropy loss
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(-1)
            loss = F.cross_entropy(logits, target)
        else:
            loss = None

        return logits, loss

    def generate(self, x, seq_len):
        for _ in range(seq_len):
            x_cond = x[:, -self.context_len:]
            logits, _ = self(x_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            x = torch.concat((x, next_tok), dim=1)

        return x


def create_model(vocab_size):
    return Transformer(CONTEXT_LEN, vocab_size, N_EMBED, N_LAYER, N_HEAD)


def model_config():
    return {
        'context_len': CONTEXT_LEN,
        'n_embed': N_EMBED,
        'n_layer': N_LAYER,
        'n_head': N_HEAD,
    }
