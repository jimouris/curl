from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import json
import torch
from dataclasses import dataclass
from torch import nn
from typing import List, Mapping, Any
import time


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


class LlamaTokenizer:
    def __init__(self, tokenizer_path: str):
        self.special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
        tokenizer_path = tokenizer_path + "tokenizer.model"
        self.mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
        self.tokenizer = tiktoken.Encoding(
            name=Path(tokenizer_path + "tokenizer.model").name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=self.mergeable_ranks,
            special_tokens={
                token: len(self.mergeable_ranks) + i
                for i, token in enumerate(self.special_tokens)
            },
        )

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def tokenize(self, prompt: str) -> torch.Tensor | None:
        tokens = [128000] + self.encode(prompt)
        return torch.tensor(tokens)

class Embedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(vocab_size, dim)

    def forward(self, tokens):
        token_embeddings_unnormalized = self.embedding_layer(tokens)
        return token_embeddings_unnormalized

    def load_weights(self, model):
        self.embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])

class RMSNorm(nn.Module):
    def __init__(self, norm_eps, norm_weights_size = 2048):
        super().__init__()
        self.norm_eps = norm_eps
        self.norm_weights = nn.Parameter(torch.ones(norm_weights_size))

    def forward(self, tensor):
        return (
            tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        ) * self.norm_weights
    
    def load_weights(self, model, layer):
        self.norm_weights.data.copy_(model[layer])

class RotaryEmbedding(nn.Module):


    def __init__(self, rope_theta, head_dim, max_seq_len):
        super().__init__()
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
        return freqs_cis

    def forward(self, q_or_k):
        q_split = q_or_k.float().view(q_or_k.size(0), -1, 2)
        q_complex = torch.view_as_complex(q_split)
        q_rotated_complex = q_complex * self.freqs_cis[: q_or_k.size(0)]
        q_rotated = torch.view_as_real(q_rotated_complex).view(q_or_k.size())
        return q_rotated

    def load_weights(self, model, layer):
        pass

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.dim // config.n_heads
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.wq = nn.Parameter(torch.ones(config.n_heads, inner_dim, config.dim))
        self.wk = nn.Parameter(torch.ones(config.n_kv_heads, inner_dim, config.dim))
        self.wv = nn.Parameter(torch.ones(config.n_kv_heads, inner_dim, config.dim))

    def forward(self, x, head):
        q = self.wq[head]
        k = self.wk[head // (self.n_heads // self.n_kv_heads)]
        v = self.wv[head // (self.n_heads // self.n_kv_heads)]
        q_per_token = torch.matmul(x, q.T)
        k_per_token = torch.matmul(x, k.T)
        v_per_token = torch.matmul(x, v.T)
        return q_per_token, k_per_token, v_per_token

    def forward_all(self, x):
        qkv_store = []
        for head in range(self.n_heads):
            qkv_store.append(self.forward_head(x, head))
        return qkv_store

    def load_weights(self, model, layer):
        self.wq.data.copy_(model[f"layers.{layer}.attention.wq.weight"].view(self.wq.size()))
        self.wk.data.copy_(model[f"layers.{layer}.attention.wk.weight"].view(self.wk.size()))
        self.wv.data.copy_(model[f"layers.{layer}.attention.wv.weight"].view(self.wv.size()))

class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.rms = RMSNorm(config.norm_eps)
        self.rope = RotaryEmbedding(config.rope_theta, config.head_dim, config.max_seq_len)
        self.attention = Attention(config)
        self.ff = FeedForward(config)

    def forward(self, x):
        qkv_attention_store = []
        layer_embedding_norm = self.rms(x)
        for head in range(self.n_heads):
            qt, kt, vt = self.attention(layer_embedding_norm, head)
            q_rotated = self.rope(qt)
            k_rotated = self.rope(kt)
            qk = Transformer.compute_qk_attention(q_rotated, k_rotated)
            qk_masked = Transformer.apply_attention_mask(qk, x.size(0))
            attention_weights = torch.nn.functional.softmax(qk_masked, dim=1)#.to(torch.bfloat16)
            qkv_attention = torch.matmul(attention_weights, vt)
            qkv_attention_store.append(qkv_attention)

        stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
        return self.ff(stacked_qkv_attention, x)

    @staticmethod
    def compute_qk_attention(q_rotated, k_rotated):
        qk = torch.matmul(q_rotated, k_rotated.T) / (q_rotated.size(-1) ** 0.5)
        return qk

    @staticmethod
    def apply_attention_mask(qk, token_count):
        mask = torch.full((token_count, token_count), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        return qk + mask

    def load_weights(self, model, layer):
        self.rms.load_weights(model, f"layers.{layer}.attention_norm.weight")
        self.rope.load_weights(model, layer)
        self.attention.load_weights(model, layer)
        self.ff.load_weights(model, layer)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rms = RMSNorm(config.norm_eps)
        hidden_dim = 4 * config.dim
        self.w0 = nn.Parameter(torch.ones(config.dim, config.dim))
        self.w1 = nn.Parameter(torch.ones(hidden_dim, config.dim))
        self.w2 = nn.Parameter(torch.ones(config.dim, hidden_dim))
        self.w3 = nn.Parameter(torch.ones(hidden_dim, config.dim))

    def forward(self, x, final_embedding):
        embedding_delta = torch.matmul(x, self.w0.T)
        embedding_after_edit = final_embedding + embedding_delta
        embedding_after_norm = self.rms(embedding_after_edit)

        feedforward_output = torch.matmul(
            torch.functional.F.silu(torch.matmul(embedding_after_norm, self.w1.T))
            * torch.matmul(embedding_after_norm, self.w3.T),
            self.w2.T,
        )
        return embedding_after_edit + feedforward_output

    def load_weights(self, model, layer):
        self.rms.load_weights(model, f"layers.{layer}.ffn_norm.weight")
        self.w0.data.copy_(model[f"layers.{layer}.attention.wo.weight"])
        self.w1.data.copy_(model[f"layers.{layer}.feed_forward.w1.weight"])
        self.w2.data.copy_(model[f"layers.{layer}.feed_forward.w2.weight"])
        self.w3.data.copy_(model[f"layers.{layer}.feed_forward.w3.weight"])

class Llama(nn.Module):
    def __init__(self, base_path: str):
        super().__init__()
        self.config = Llama.load_config(base_path)
        self.layers = nn.ModuleList([Transformer(self.config) for _ in range(self.config.n_layers)])
        self.embedding = Embedding(self.config.vocab_size, self.config.dim)
        self.output_weight = nn.Parameter(torch.ones(self.config.vocab_size, self.config.dim))

    @staticmethod
    def load_model(path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(path + "consolidated.00.pth", map_location=device)
        return model

    @staticmethod
    def load_config(path):
        with open(path + "params.json", "r") as f:
            dic = json.load(f)
            head_dim = dic["dim"] // dic["n_heads"]
            max_seq_len = 2048
            config = LlamaConfig(**dic, head_dim=head_dim, max_seq_len=max_seq_len)
        return config

    def forward(self, tokens: torch.Tensor):
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        logits = torch.matmul(x[-1], self.output_weight.T)
        next_token = torch.argmax(logits, dim=-1)
        return next_token

    def load_weights(self, path):
        model = Llama.load_model(path)
        self.embedding.load_weights(model)
        for layer in range(self.config.n_layers):
            self.layers[layer].load_weights(model, layer)
        self.output_weight.data.copy_(model["output.weight"])

def main():
    PATH = "Llama3.2-1B/"
    tokenizer = LlamaTokenizer(PATH)
    model = Llama(PATH)
    model.load_weights(PATH)
    prompt = (
        "quiero comer pizza de "
    )
    tokens = tokenizer.tokenize(prompt)
    #print(prompt, end="\n>")
    a = time.time()
    for i in range(100):

        next_token = model(tokens)

        #tokens = torch.cat([tokens, next_token.unsqueeze(-1)], dim=-1)
        #print(tokenizer.decode([next_token.item()]), end = "")
    b = time.time()
    print(f"Time: {b - a}")
    print("RESULT:", tokenizer.decode([next_token.item()]))


if __name__ == "__main__":
    main()
