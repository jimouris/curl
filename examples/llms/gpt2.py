import curl
import curl.nn as nn
import torch

class Transformer(nn.Module):
    class Full(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.c_fc = nn.AttentionLinear(embed_dim, embed_dim * 4)
            self.gelu = nn.GELU()
            self.c_proj = nn.AttentionLinear(embed_dim * 4, embed_dim)

        def forward(self, x):
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)
            return x

    class Block(nn.Module):
        def __init__(self, embed_dim, num_heads):
            super(Transformer.Block, self).__init__()
            self.ln_1 = nn.LayerNorm(embed_dim)
            self.ln_2 = nn.LayerNorm(embed_dim)
            self.attn = nn.Attention(embed_dim, num_heads)
            self.mlp = Transformer.Full(embed_dim)

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

    def __init__(self, embed_dim, num_heads, num_blocks, vocab_size, seq_len):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim

        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(seq_len, embed_dim)

        self.h = nn.Sequential(
            *[Transformer.Block(embed_dim, num_heads) for _ in range(num_blocks)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        # self.fc = nn.Linear(embed_dim, vocab_size)
        # self.softmax = nn.Softmax(-1)

    def forward(self, x, target=None):
        tok_embedding = self.wte(x)
        pos_embedding = self.wpe.weight[:x.size()[1], :].reshape(x.size()[0], x.size()[1], -1)
        x = tok_embedding + pos_embedding
        x = self.h(x)
        x = self.ln_f(x)
        # x = self.fc(x)
        # x = self.softmax(x)
        return x

class GPT(nn.Module):
    def __init__(self, embed_dim, num_heads, num_blocks, vocab_size, seq_len):
        super(GPT, self).__init__()
        self.transformer = Transformer(embed_dim, num_heads, num_blocks, vocab_size, seq_len)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x, target=None):
        x = self.transformer(x, target)
        x = self.lm_head(x)
        return x

class GPT2(GPT):
    def __init__(self, seq_len=1024):
        super(GPT2, self).__init__(embed_dim=768, num_heads=12, num_blocks=12, vocab_size=50257, seq_len=seq_len)
