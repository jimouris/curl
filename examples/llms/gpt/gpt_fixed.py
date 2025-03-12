import torch
import torch.nn as nn

PRECISION = 16

def to_fixed_point(tensor: torch.tensor, precision: int) -> torch.tensor:
    return (tensor * 2**precision).long()

def fixed_to_float(tensor: torch.tensor, precision: int) -> torch.tensor:
    return tensor.float() / 2**precision

class GPTLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

    def forward(self, x):
        weight = to_fixed_point(self.weight, PRECISION)
        bias = to_fixed_point(self.bias, PRECISION) * 2**PRECISION
        x = to_fixed_point(x, PRECISION)
        fixed_output = x.matmul(weight).add(bias)
        fixed_output = fixed_to_float(fixed_output, 2 * PRECISION)
        return fixed_output

class GPTAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(GPTAttention, self).__init__()

        assert embed_dim % num_heads == 0, "invalid heads and embedding dimension"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.search_dim = embed_dim // num_heads

        self.c_attn = GPTLinear(embed_dim, 3 * embed_dim)
        self.c_proj = GPTLinear(embed_dim, embed_dim)

    def forward(self, x, mask=True):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        query, key, value = self.c_attn(x).split(self.embed_dim, dim=2)
        query = query.reshape(batch_size, seq_len, self.num_heads, self.search_dim).transpose(1, 2)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.search_dim).permute(0, 2, 3, 1)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.search_dim).transpose(1, 2)

        query = to_fixed_point(query, PRECISION)
        key = to_fixed_point(key, PRECISION)
        attn = query.matmul(key)
        attn = attn >> PRECISION
        attn = attn * to_fixed_point(torch.tensor(1 / query.size(-1) ** 0.5), PRECISION)
        if mask:
            attn = attn * torch.tril(torch.ones_like(attn, dtype=torch.long), diagonal=0)
            attn = attn + -2**47 * torch.triu(torch.ones_like(attn, dtype=torch.long), diagonal=1)
        attn = fixed_to_float(attn, 2 * PRECISION)
        attn = attn.softmax(dim=-1)
        attn = to_fixed_point(attn, PRECISION)
        value = to_fixed_point(value, PRECISION)
        y = attn.matmul(value).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        y = y >> PRECISION
        y = fixed_to_float(y, PRECISION)
        y = self.c_proj(y)
        return y

class GPTMLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.c_fc = GPTLinear(embed_dim, embed_dim * 4)
        self.gelu = nn.GELU()
        self.c_proj = GPTLinear(embed_dim * 4, embed_dim)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(GPTBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = GPTAttention(embed_dim, num_heads)
        self.mlp = GPTMLP(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_blocks, vocab_size, seq_len):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim

        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(seq_len, embed_dim)

        self.h = nn.Sequential(
            *[GPTBlock(embed_dim, num_heads) for _ in range(num_blocks)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)

    def forward(self, x, target=None):
        tok_embedding = self.wte(x)
        pos_embedding = self.wpe.weight[:x.size()[1], :].reshape(x.size()[0], x.size()[1], -1)
        x = fixed_to_float(to_fixed_point(tok_embedding, PRECISION) + to_fixed_point(pos_embedding, PRECISION), PRECISION)
        x = self.h(x)
        x = self.ln_f(x)
        return x

class GPT(nn.Module):
    def __init__(self, embed_dim, num_heads, num_blocks, vocab_size, seq_len):
        super(GPT, self).__init__()
        self.transformer = Transformer(embed_dim, num_heads, num_blocks, vocab_size, seq_len)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x, target=None):
        x = self.transformer(x, target)
        x = to_fixed_point(x, PRECISION)
        weight = to_fixed_point(self.lm_head.weight, PRECISION)
        x = x.matmul(weight.t())
        return fixed_to_float(x, 2 * PRECISION)

class GPT2LMHead(GPT):
    def __init__(self, seq_len=1024):
        super(GPT2LMHead, self).__init__(embed_dim=768, num_heads=12, num_blocks=12, vocab_size=50257, seq_len=seq_len)
