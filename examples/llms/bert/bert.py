import logging
import curl
import curl.nn as nn
import torch


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Attention, self).__init__()

        assert embed_dim % num_heads == 0, "invalid heads and embedding dimension"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.search_dim = embed_dim // num_heads

        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=False):
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

class Bert(nn.Module):
    class Block(nn.Module):
        def __init__(self, embed_dim, num_heads):
            super(Bert.Block, self).__init__()
            self.ln1 = nn.LayerNorm(embed_dim)
            self.ln2 = nn.LayerNorm(embed_dim)
            self.attn = Attention(embed_dim, num_heads)
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim),
            )

        def forward(self, x):
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ff(x))
            return x

    def __init__(self, embed_dim, num_heads, num_blocks, vocab_size, seq_len, full=True):
        super(Bert, self).__init__()
        self.full = full
        self.embed_dim = embed_dim

        if full:
            self.tok_embed = nn.Embedding(vocab_size, embed_dim)
            self.pos_embed = nn.Parameter(curl.cryptensor(torch.zeros(1, seq_len, embed_dim)))
        self.ln = nn.LayerNorm
        self.blocks = nn.Sequential(
            *[Bert.Block(embed_dim, num_heads) for _ in range(num_blocks)]
        )
        self.ln = nn.LayerNorm(embed_dim)
        if full:
            self.fc = nn.Linear(embed_dim, vocab_size)
            self.softmax = nn.Softmax(-1)

    def forward(self, x, target=None):
        if self.full:
            tok_embedding = self.tok_embed(x)
            pos_embedding = self.pos_embed(x)[:, :x.size()[1], :]
            x = tok_embedding + pos_embedding
        x = self.ln(x)
        x = self.blocks(x)
        if self.full:
            x = self.fc(x)
            x = self.softmax(x)
        return x

class BertTiny(Bert):
    def __init__(self, seq_len, full=True):
        super(BertTiny, self).__init__(embed_dim=128, num_heads=2, num_blocks=2, vocab_size=30522, seq_len=seq_len, full=full)

class BertBase(Bert):
    def __init__(self, seq_len, full=True):
        super(BertBase, self).__init__(embed_dim=768, num_heads=12, num_blocks=12, vocab_size=30522, seq_len=seq_len, full=full)

class BertLarge(Bert):
    def __init__(self, seq_len, full=True):
        super(BertLarge, self).__init__(embed_dim=1024, num_heads=16, num_blocks=24, vocab_size=30522, seq_len=seq_len, full=full)
