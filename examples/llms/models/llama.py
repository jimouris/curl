import curl
import curl.nn as nn
import torch

from dataclasses import dataclass


@dataclass
class LlamaConfig:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    multiple_of: int
    ffn_dim_multiplier: float
    norm_eps: float
    rope_theta: float
    head_dim: int
    max_seq_len: int
    use_scaled_rope: bool


class RMSNorm(nn.Module):
    def __init__(self, norm_eps, norm_weights_size):
        super(RMSNorm, self).__init__()
        self.norm_eps = norm_eps
        self.norm_weights = nn.Parameter(curl.cryptensor(torch.ones(norm_weights_size)))

    def forward(self, tensor):
        return (
            tensor * (tensor.pow(2).mean(-1, keepdim=True) + self.norm_eps).inv_sqrt()
        ) * self.norm_weights(None)


class RotaryEmbedding(nn.Module):
    def __init__(self, rope_theta, head_dim, max_seq_len):
        super(RotaryEmbedding, self).__init__()
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.register_buffer("freqs_cis", self.calculate_rope_frequencies())

    def calculate_rope_frequencies(self):
        freqs = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float()
                / self.head_dim
            )
        )
        t = torch.arange(self.max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        freqs_cis = torch.view_as_real(freqs_cis)
        return freqs_cis

    def forward(self, q):
        q_split = q.view(*q.shape[:-1], -1, 2)
        a = q_split[..., 0]
        b = q_split[..., 1]
        c = self.freqs_cis[: q.size(0), :, 0]
        d = self.freqs_cis[: q.size(0), :, 1]

        q_rotated = curl.stack([a * c - b * d, b * c + a * d], dim=-1)
        q_rotated = q_rotated.view(q.size())
        return q_rotated


def repeat_kv(x, n_rep):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, config, cache=False):
        super(Attention, self).__init__()
        self.inner_dim = config.dim // config.n_heads
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.inner_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.inner_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.inner_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.inner_dim, config.dim, bias=False)

        self.rope = RotaryEmbedding(config.rope_theta, config.head_dim, config.max_seq_len)

        self.cache = cache
        if cache:
            self.cache_k = curl.cryptensor(torch.tensor([]))
            self.cache_v = curl.cryptensor(torch.tensor([]))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads, self.inner_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.inner_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.inner_dim)

        xq = self.rope(xq)
        xk = self.rope(xk)

        if self.cache:
            self.cache_k = curl.cat([self.cache_k, xk], dim=1)
            self.cache_v = curl.cat([self.cache_v, xv], dim=1)

            xk = self.cache_k
            xv = self.cache_v

        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = xq.matmul(keys.transpose(2, 3))  * self.inner_dim**(-0.5)
        scores = scores.softmax(dim=-1)

        output = scores.matmul(values)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.rms = RMSNorm(config.norm_eps, config.dim)
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)

    def forward(self, x):
        swish = self.w1(x).silu()
        x_v = self.w3(x)
        x = swish * x_v
        x = self.w2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, config, cache=False):
        super(Transformer, self).__init__()
        self.attn_norm = RMSNorm(config.norm_eps, config.dim)
        self.attention = Attention(config, cache)
        self.ffn_norm = RMSNorm(config.norm_eps, config.dim)
        self.ffn = FeedForward(config)

    def forward(self, x):
        h = self.attn_norm(x)
        h = self.attention(h)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Llama1B(nn.Module):
    def __init__(self, seq_len, full=False, cache=False):
        super(Llama1B, self).__init__()
        self.full = full
        self.embed_dim, self.head_dim = 2048, 32
        self.config = LlamaConfig(self.embed_dim, 16, self.head_dim, 8, 128256, 256, 1.5, 1e-05, 500000.0,
                                  self.embed_dim//self.head_dim, seq_len, True)
        self.layers = nn.ModuleList(
            [Transformer(self.config, cache) for _ in range(self.config.n_layers)]
        )
        if full:
            self.embedding = nn.Embedding(self.config.vocab_size, self.config.dim)
            self.output = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)

    def forward(self, x):
        if self.full:
            x = self.embedding(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        if self.full:
            x = self.output(x)
        return x
