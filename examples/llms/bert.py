import crypten
import crypten.nn as nn
import torch

class Bert(nn.Module):
    class Block(nn.Module):
        def __init__(self, embed_dim, num_heads):
            super(Bert.Block, self).__init__()
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
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ff(x))
            return x

    def __init__(self, embed_dim, num_heads, num_blocks, vocab_size, seq_len, full=True):
        super(Bert, self).__init__()
        self.full = full
        if full:
            self.tok_embed = nn.Embedding(vocab_size, embed_dim)
            self.pos_embed = crypten.cryptensor(torch.zeros(1, seq_len, embed_dim))
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
            pos_embedding = self.pos_embed[:, :x.size()[1], :]
            x = tok_embedding + pos_embedding
        x = self.ln(x)
        x = self.blocks(x)
        if self.full:
            x = self.fc(x)
            x = self.softmax(x)
        return x

class BertTiny(Bert):
    def __init__(self, full=True):
        super(BertTiny, self).__init__(embed_dim=128, num_heads=2, num_blocks=2, vocab_size=30522, seq_len=128, full=full)

class BertBase(Bert):
    def __init__(self, full=True):
        super(BertBase, self).__init__(embed_dim=768, num_heads=12, num_blocks=12, vocab_size=30522, seq_len=128, full=full)

class BertLarge(Bert):
    def __init__(self, full=True):
        super(BertLarge, self).__init__(embed_dim=1024, num_heads=16, num_blocks=24, vocab_size=30522, seq_len=128, full=full)
