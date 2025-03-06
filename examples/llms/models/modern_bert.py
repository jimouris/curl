import curl
import curl.nn as nn
import torch

class ModernBertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, bias=False)

    def forward(self, input_ids, token_type_ids):
        return self.norm(self.tok_embeddings(input_ids))


class ModernBertRotaryEmbedding(nn.Module):
    def __init__(self, dim, base):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))

    def forward(self, x):
        position_ids = torch.unsqueeze(torch.arange(0, x.size(1), dtype=torch.long).to(x.device), 0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(x.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return curl.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class ModernBertAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, layer_id):
        super().__init__()
        self.local_attention = 128
        self.global_attn_every_n_layers = 3
        self.hidden_size = hidden_size
        self.layer_id = layer_id

        if self.layer_id % self.global_attn_every_n_layers != 0:
            base = 10_000
        else:
            base = 160_000

        self.n_heads = n_heads
        self.Wqkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.head_dim = hidden_size // n_heads
        self.rotary_emb = ModernBertRotaryEmbedding(self.head_dim, base)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        qkv = self.Wqkv(hidden_states)
        cos, sin = self.rotary_emb(qkv)

        query, key, value = qkv.split(self.hidden_size, dim=2)
        query = query.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        attn = query.matmul(key.transpose(2, 3)) / query.size(-1) ** 0.5

        if self.layer_id % self.global_attn_every_n_layers != 0:
            # Create position indices
            rows = torch.arange(attn.shape[2]).unsqueeze(0)
            # Calculate distance between positions
            distance = torch.abs(rows - rows.T)
            # Create sliding window mask (1 for positions within window, 0 outside)
            window_mask = (
                (distance <= self.local_attention // 2).unsqueeze(0).unsqueeze(0).to(attn.device)
            )
            # Combine with existing mask
            attn.share = attn.share.masked_fill(window_mask.logical_not(), -2**46)

        attn = attn.softmax(dim=-1)

        emb_rich = attn.matmul(value).transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        emb_rich = self.Wo(emb_rich)
        return emb_rich


class ModernBertMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.Wi = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.act = nn.GELU()
        self.Wo = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        input_state, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.act(input_state) * gate)


class ModernBertLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, n_heads, layer_id):
        super().__init__()
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(hidden_size, bias=False)
        self.attn = ModernBertAttention(hidden_size, n_heads, layer_id)
        self.mlp_norm = nn.LayerNorm(hidden_size, bias=False)
        self.mlp = ModernBertMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class ModernBertModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, intermediate_size, seq_len, n_heads, n_layers, full=True):
        super().__init__()
        self.full = full
        self.embeddings = ModernBertEmbeddings(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [ModernBertLayer(hidden_size, intermediate_size, n_heads, layer_id) for layer_id in range(n_layers)])
        self.final_norm = nn.LayerNorm(hidden_size, bias=False)

    def forward(self, input_ids, input_embeds=None):
        hidden_states = self.embeddings(input_ids, input_embeds) if self.full else input_ids
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        return hidden_states


class ModernBertPredictionHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size, bias=False)

    def forward(self, hidden_states):
        hidden_states = hidden_states[:, 0]
        out = self.norm(self.act(self.dense(hidden_states)))
        return out


class ModernBertForSequenceClassification(nn.Module):
    def __init__(self, vocab_size, hidden_size, intermediate_size, seq_len, n_heads, n_layers, classes=2):
        super().__init__()
        self.model = ModernBertModel(vocab_size, hidden_size, intermediate_size, seq_len, n_heads, n_layers)
        self.head = ModernBertPredictionHead(hidden_size)
        self.classifier = nn.Linear(hidden_size, classes)

    def forward(self, input_ids, token_type_ids=None):
        pooled_out = self.model(input_ids, token_type_ids)
        out = self.head(pooled_out)
        return self.classifier(out)


class ModernBert(ModernBertModel):
    def __init__(self, seq_len, full):
        super().__init__(vocab_size=50368, hidden_size=768, intermediate_size=1152, seq_len=seq_len, n_heads=12, n_layers=22, full=full)

class ModernBertForTokenClassification(ModernBertForSequenceClassification):
    def __init__(self):
        super().__init__(vocab_size=50368, hidden_size=768, intermediate_size=1152, seq_len=8192, n_heads=12, n_layers=22)

class ModernBertLargeForTokenClassification(ModernBertForSequenceClassification):
    def __init__(self):
        super().__init__(vocab_size=50368, hidden_size=1024, intermediate_size=2624, seq_len=8192, n_heads=16, n_layers=24, classes=2)
