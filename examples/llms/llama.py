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

        self.mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
        self.tokenizer = tiktoken.Encoding(
            name=Path(tokenizer_path).name,
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


class RMSNorm(nn.Module):
    def __init__(self, norm_eps, norm_weights_size = 2048):
        super().__init__()
        self.norm_eps = norm_eps
        self.norm_weights = nn.Parameter(torch.ones(norm_weights_size))

    def forward(self, tensor, norm_weights):
        return (
            tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        ) * norm_weights
    
    def load_weights(self, weights):
        self.norm_weights.data.copy_(weights)

class RoPEEmbedding(nn.Module):
    def __init__(self, rope_theta, max_seq_len):
        super().__init__()
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len

class Llama(nn.Module):
    @staticmethod
    def rms_norm(tensor, norm_weights, norm_eps):
        print(norm_weights.shape)
        return (
            tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)
        ) * norm_weights

    def __init__(self, base_path: str):
        super().__init__()
        self.tokenizer = LlamaTokenizer(base_path + "tokenizer.model")
        self.model, self.config = Llama.load_model_and_config(base_path)
        self.freqs_cis = Llama.calculate_rope_frequencies(self.config)

    def forward(self, tokens: torch.Tensor):
        token_embeddings_unnormalized = self.initialize_embeddings(tokens)
        final_embedding = self.perform_attention_and_feedforward(
            token_embeddings_unnormalized,
        )

        next_token = self.generate_next_token(final_embedding)
        return next_token

    @staticmethod
    def load_model_and_config(path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(path + "consolidated.00.pth", map_location=device)
        with open(path + "params.json", "r") as f:
            dic = json.load(f)
            head_dim = dic["dim"] // dic["n_heads"]
            max_seq_len = 2048
            config = LlamaConfig(**dic, head_dim=head_dim, max_seq_len=max_seq_len)
        return model, config

    def initialize_embeddings(self, tokens):
        vocab_size = self.config.vocab_size
        embedding_layer = torch.nn.Embedding(vocab_size, self.config.dim)
        embedding_layer.weight.data.copy_(self.model["tok_embeddings.weight"])
        token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)
        return token_embeddings_unnormalized

    @staticmethod
    def calculate_rope_frequencies(config):
        freqs = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2)[: (config.head_dim // 2)].float()
                / config.head_dim
            )
        )
        t = torch.arange(config.max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def perform_attention_and_feedforward(self, token_embeddings):
        n_layers = self.config.n_layers
        n_heads = self.config.n_heads
        n_kv_heads = self.config.n_kv_heads
        norm_eps = self.config.norm_eps
        model = self.model

        final_embedding = token_embeddings
        for layer in range(n_layers):
            qkv_attention_store = []
            layer_embedding_norm = Llama.rms_norm(
                final_embedding,
                model[f"layers.{layer}.attention_norm.weight"],
                norm_eps,
            )
            q_layer = model[f"layers.{layer}.attention.wq.weight"].view(
                n_heads, -1, layer_embedding_norm.size(-1)
            )
            k_layer = model[f"layers.{layer}.attention.wk.weight"].view(
                n_kv_heads, -1, layer_embedding_norm.size(-1)
            )
            v_layer = model[f"layers.{layer}.attention.wv.weight"].view(
                n_kv_heads, -1, layer_embedding_norm.size(-1)
            )
            w_layer = model[f"layers.{layer}.attention.wo.weight"]

            for head in range(n_heads):
                q = q_layer[head]
                k = k_layer[head // (n_heads // n_kv_heads)]
                v = v_layer[head // (n_heads // n_kv_heads)]
                q_per_token = torch.matmul(layer_embedding_norm, q.T)
                k_per_token = torch.matmul(layer_embedding_norm, k.T)
                v_per_token = torch.matmul(layer_embedding_norm, v.T)
                q_rotated = self.apply_rope(q_per_token)
                k_rotated = self.apply_rope(k_per_token)
                qk = Llama.compute_qk_attention(q_rotated, k_rotated)
                qk_masked = Llama.apply_attention_mask(qk, token_embeddings.size(0))
                attention_weights = torch.nn.functional.softmax(qk_masked, dim=1).to(
                    torch.bfloat16
                )
                qkv_attention = torch.matmul(attention_weights, v_per_token)
                qkv_attention_store.append(qkv_attention)

            stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
            embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
            final_embedding = self.update_embedding(
                final_embedding, embedding_delta, layer
            )

        return final_embedding

    def apply_rope(self, q_or_k):
        q_split = q_or_k.float().view(q_or_k.size(0), -1, 2)
        q_complex = torch.view_as_complex(q_split)
        q_rotated_complex = q_complex * self.freqs_cis[: q_or_k.size(0)]
        q_rotated = torch.view_as_real(q_rotated_complex).view(q_or_k.size())
        return q_rotated

    @staticmethod
    def compute_qk_attention(q_rotated, k_rotated):
        qk = torch.matmul(q_rotated, k_rotated.T) / (q_rotated.size(-1) ** 0.5)
        return qk

    @staticmethod
    def apply_attention_mask(qk, token_count):
        mask = torch.full((token_count, token_count), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        return qk + mask

    def update_embedding(self, final_embedding, embedding_delta, layer):
        embedding_after_edit = final_embedding + embedding_delta
        embedding_after_norm = Llama.rms_norm(
            embedding_after_edit,
            self.model[f"layers.{layer}.ffn_norm.weight"],
            self.config.norm_eps,
        )
        w1, w2, w3 = (
            self.model[f"layers.{layer}.feed_forward.w{i}.weight"] for i in (1, 2, 3)
        )
        feedforward_output = torch.matmul(
            torch.functional.F.silu(torch.matmul(embedding_after_norm, w1.T))
            * torch.matmul(embedding_after_norm, w3.T),
            w2.T,
        )
        return embedding_after_edit + feedforward_output

    def generate_next_token(self, final_embedding):
        logits = torch.matmul(final_embedding[-1], self.model["output.weight"].T)
        next_token = torch.argmax(logits, dim=-1)
        return next_token


def main():
    PATH = "Llama3.2-1B/"
    tokenizer = LlamaTokenizer(PATH + "tokenizer.model")
    model = Llama(PATH)
    prompt = (
        "the answer to the ultimate question of life, the universe, and everything is "
    )
    tokens = tokenizer.tokenize(prompt)
    next_token = model(tokens)
    print("RESULT:", tokenizer.decode([next_token.item()]))


if __name__ == "__main__":
    main()
