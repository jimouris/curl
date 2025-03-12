import torch
import torch.nn as nn

from curl.cuda import CUDALongTensor

PRECISION = 16

def to_fixed_point(tensor: torch.tensor, precision: int) -> torch.tensor:
    return CUDALongTensor((tensor * 2**precision).long())

def fixed_to_float(tensor: torch.tensor, precision: int) -> torch.tensor:
    return tensor.tensor().float() / 2**precision

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, layer_id):
        super(Attention, self).__init__()
        self.attention = GPTAttention(embed_dim, num_heads, layer_id)

    def forward(self, x, mask=True):
        return self.attention(x, mask)

class GPTAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, layer_id):
        super(GPTAttention, self).__init__()

        assert embed_dim % num_heads == 0, "invalid heads and embedding dimension"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.search_dim = embed_dim // num_heads
        self.layer_id = layer_id

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.search_dim)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, mask=True):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        x = to_fixed_point(x, PRECISION)

        weight = to_fixed_point(self.q_proj.weight, PRECISION)
        query = x.matmul(weight.t())

        weight = to_fixed_point(self.k_proj.weight, PRECISION)
        key = x.matmul(weight.t())

        weight = to_fixed_point(self.v_proj.weight, PRECISION)
        value = x.matmul(weight.t())

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        query = to_fixed_point(query, PRECISION)
        key = to_fixed_point(key, PRECISION)

        attn = query.matmul(key.transpose(-1, -2))
        if mask:
            if self.layer_id % 2 != 0:
                # Create position indices
                rows = torch.arange(attn.shape[2]).unsqueeze(0)
                # Calculate distance between positions
                distance = torch.abs(rows - rows.T)
                # Create sliding window mask (1 for positions within window, 0 outside)
                window_mask = (
                    (distance <= 128).unsqueeze(0).unsqueeze(0).to(attn.device)
                )
                # Combine with existing mask
                attn = attn.masked_fill(window_mask.logical_not(), -2**47)
            attn = attn * torch.tril(torch.ones_like(attn, dtype=torch.long), diagonal=0)
            attn = attn + -2**47 * torch.triu(torch.ones_like(attn, dtype=torch.long), diagonal=1)
        attn = fixed_to_float(attn, 2 * PRECISION)
        attn = attn.softmax(dim=-1)
        attn = to_fixed_point(attn, PRECISION)
        attn = attn >> PRECISION
        y = attn.matmul(value).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        weight = to_fixed_point(self.out_proj.weight, PRECISION)
        y = y.matmul(weight.t())
        return y

class GPTMLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, embed_dim * 4)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

class GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, layer_id):
        super(GPTBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, layer_id)
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
            *[GPTBlock(embed_dim, num_heads, layer_id) for layer_id in range(num_blocks)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)

    def forward(self, x, target=None):
        tok_embedding = self.wte(x)
        pos_embedding = self.wpe.weight[:x.size()[1], :].reshape(x.size()[0], x.size()[1], -1)
        x = tok_embedding + pos_embedding
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

class GPTNeoLMHead(GPT):
    def __init__(self, seq_len=2048):
        super(GPTNeoLMHead, self).__init__(embed_dim=2048, num_heads=16, num_blocks=24, vocab_size=50257, seq_len=seq_len)
