import curl.nn as nn
import torch

class GPTLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GPTLinear, self).__init__()

        pytorch_module = torch.nn.Linear(in_features, out_features, bias=bias)
        self.register_parameter("weight", pytorch_module.weight.t(), requires_grad=False)
        if bias:
            self.register_parameter("bias", pytorch_module.bias)

    def forward(self, x):
        output = x.matmul(self.weight)
        if hasattr(self, "bias"):
            output = output.add(self.bias)
        return output


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Attention, self).__init__()

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

        attn = query.matmul(key) / query.size(-1) ** 0.5
        if mask:
            attn.share = attn.share * torch.tril(torch.ones_like(attn.share, dtype=torch.long), diagonal=0)
            attn.share = attn.share + -2**46 * torch.triu(torch.ones_like(attn.share, dtype=torch.long), diagonal=1)
        attn = attn.softmax(dim=-1)

        y = attn.matmul(value).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        y = self.c_proj(y)
        return y


class GPTMLP(nn.Module):
    def __init__(self, embed_dim):
        super(GPTMLP, self).__init__()
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
        self.attn = Attention(embed_dim, num_heads)
        self.mlp = GPTMLP(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_blocks, vocab_size, seq_len, full):
        super(Transformer, self).__init__()
        self.full = full
        self.embed_dim = embed_dim
        self.h = nn.Sequential(
            *[GPTBlock(embed_dim, num_heads) for _ in range(num_blocks)]
        )
        if full:
            self.wte = nn.Embedding(vocab_size, embed_dim)
            self.wpe = nn.Embedding(seq_len, embed_dim)
            self.ln_f = nn.LayerNorm(embed_dim)

    def forward(self, x, target=None):
        if self.full:
            tok_embedding = self.wte(x)
            pos_embedding = self.wpe.weight[:x.size()[1], :].reshape(x.size()[0], x.size()[1], -1)
            x = tok_embedding + pos_embedding
        x = self.h(x)
        if self.full:
            x = self.ln_f(x)
        return x


class GPT(nn.Module):
    def __init__(self, embed_dim, num_heads, num_blocks, vocab_size, seq_len, full=True):
        super(GPT, self).__init__()
        self.full = full
        self.embed_dim = embed_dim
        self.transformer = Transformer(embed_dim, num_heads, num_blocks, vocab_size, seq_len, full)
        if full:
            self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x, target=None):
        x = self.transformer(x, target)
        if self.full:
            x = self.lm_head(x)
        return x


class GPT2(GPT):
    def __init__(self, seq_len, full=True):
        super(GPT2, self).__init__(embed_dim=768, num_heads=12, num_blocks=12, vocab_size=50257, seq_len=seq_len, full=full)

class GPT2LMHead(GPT):
    def __init__(self, seq_len=1024):
        super(GPT2LMHead, self).__init__(embed_dim=768, num_heads=12, num_blocks=12, vocab_size=50257, seq_len=seq_len)
