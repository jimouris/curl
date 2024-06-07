import crypten
import crypten.nn as nn
import torch

class GPT(nn.Module):
    class Block(nn.Module):
        def __init__(self, embed_dim, num_heads):
            super(GPT.Block, self).__init__()
            embed_dim = embed_dim
            self.ln1 = nn.LayerNorm(embed_dim)
            self.ln2 = nn.LayerNorm(embed_dim)
            self.attn = nn.Attention(embed_dim, num_heads)
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim),
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
            return x

    def __init__(self, embed_dim, num_heads, num_blocks, vocab_size, seq_len, full=True):
        super(GPT, self).__init__()
        self.full = full
        if full:
            self.tok_embed = nn.Embedding(vocab_size, embed_dim)
            self.pos_embed = crypten.cryptensor(torch.zeros(1, seq_len, embed_dim))

        self.blocks = nn.Sequential(
            *[GPT.Block(embed_dim, num_heads) for _ in range(num_blocks)]
        )
        if full:
            self.ln = nn.LayerNorm(embed_dim)
            self.fc = nn.Linear(embed_dim, vocab_size)
            self.softmax = nn.Softmax(-1)

    def forward(self, x, target=None):
        if self.full:
            tok_embedding = self.tok_embed(x)
            pos_embedding = self.pos_embed[:, :x.size()[1], :]
            x = tok_embedding + pos_embedding
        x = self.blocks(x)
        if self.full:
            x = self.ln(x)
            x = self.fc(x)
            x = self.softmax(x)
        return x

class GPT2(GPT):
    def __init__(self, full=True):
        super(GPT2, self).__init__(embed_dim=768, num_heads=12, num_blocks=12, vocab_size=50257, seq_len=128, full=full)

class GPTNeo(GPT):
    def __init__(self, full=True):
        super(GPTNeo, self).__init__(embed_dim=2048, num_heads=16, num_blocks=24, vocab_size=50257, seq_len=128, full=full)

