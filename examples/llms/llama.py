from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import json
import torch
from dataclasses import dataclass
import curl.nn as nn
from typing import List, Mapping, Any
import time
import curl


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

def transpose(x: curl.CrypTensor):
    # #print("[Transpose] x:", x.shape)
    ret = x.permute(*torch.arange(len(x.shape) - 1, -1, -1))
    #ret = x.view(x.size()[::-1])
    # #print("[Transpose] ret:", ret.shape)
    return ret


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


class LlamaEmbedding(nn.Embedding):
    def load_weights(self, model):
        self.weight = model["tok_embeddings.weight"]


class RMSNorm(nn.Module):
    def __init__(self, norm_eps, norm_weights_size=2048):
        super().__init__()
        self.norm_eps = norm_eps
        self.norm_weights = None

    def forward(self, tensor):
        #print("[RMSNorm] Tensor:", type(tensor))
        # a = tensor.pow(2).mean(-1, keepdim=True)
        # #print("[RMSNorm] a:", type(a))
        # b = a + self.norm_eps
        # #print("[RMSNorm] b:", type(b))
        # c = b.inv_sqrt()
        # #print("[RMSNorm] c:", type(c))
        # d = tensor * c
        # #print("[RMSNorm] d:", type(d))
        # #print(self.norm_weights)
        # e = d * self.norm_weights(None)
        # #print("[RMSNorm] e:", type(e))
        # return e
        return (
            tensor * (tensor.pow(2).mean(-1, keepdim=True) + self.norm_eps).inv_sqrt()
        ) * self.norm_weights(None)

        # return (
        #        tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        # ) * self.norm_weights

    def load_weights(self, model, layer):
        self.norm_weights = nn.Parameter(curl.cryptensor(model[layer]))


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
        freqs_cis = torch.polar(torch.empty_like(freqs), freqs)
        freqs_cis = torch.view_as_real(freqs_cis)
        #print("[RotaryEmbedding] freqs_cis:", type(freqs_cis), freqs_cis.shape)
        return freqs_cis

    def forward(self, q_or_k):
        q_split = q_or_k.view(q_or_k.size(0), -1, 2)
        #print("[RotaryEmbedding] q_split:", type(q_split), q_split.shape)
        #print("[RotaryEmbedding] freqs_cis:", type(self.freqs_cis), self.freqs_cis.size())
        # This is the original torch code:
        ## q_complex = torch.view_as_complex(q_split)
        ## q_rotated_complex = q_complex * self.freqs_cis[: q_or_k.size(0)]
        ## q_rotated = torch.view_as_real(q_rotated_complex).view(q_or_k.size())
        a = q_split[:, :, 0]
        b = q_split[:, :, 1]
        c = self.freqs_cis[: q_or_k.size(0), :, 0]
        d = self.freqs_cis[: q_or_k.size(0), :, 1]

        #print("[RotaryEmbedding] a:", type(a), a.shape)
        #print("[RotaryEmbedding] b:", type(b), b.shape)
        #print("[RotaryEmbedding] c:", type(c), c.shape)
        #print("[RotaryEmbedding] d:", type(d), d.shape)

        # complex number multiplication: (a + bi) * (c + di) = (ac - bd) + (bc + ad)i
        q_or_k_rotated = torch.stack([a * c - b * d, b * c + a * d], dim=-1)
        q_or_k_rotated = q_or_k_rotated.view(q_or_k.size())
        return q_or_k_rotated

    def load_weights(self, model, layer):
        pass


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.dim // config.n_heads
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.wq = (config.n_heads, inner_dim, config.dim)
        self.wk = (config.n_kv_heads, inner_dim, config.dim)
        self.wv = (config.n_kv_heads, inner_dim, config.dim)

    def forward(self, x, head):
        q = self.wq(None)[head]
        k = self.wk(None)[head // (self.n_heads // self.n_kv_heads)]
        v = self.wv(None)[head // (self.n_heads // self.n_kv_heads)]
        #print("Q:", q.shape, " K:", k.shape, " V:", v.shape)
        q_per_token = x.matmul(transpose(q))
        k_per_token = x.matmul(transpose(k))
        v_per_token = x.matmul(transpose(v))
        #print("Q per token:", q_per_token.shape, " K per token:", k_per_token.shape, " V per token:", v_per_token.shape)
        return q_per_token, k_per_token, v_per_token

    def forward_all(self, x):
        qkv_store = []
        for head in range(self.n_heads):
            qkv_store.append(self.forward_head(x, head))
        return qkv_store

    def load_weights(self, model, layer):
        self.wq = nn.Parameter(
            curl.cryptensor(model[f"layers.{layer}.attention.wq.weight"].view(self.wq))
        )
        self.wk = nn.Parameter(
            curl.cryptensor(model[f"layers.{layer}.attention.wk.weight"].view(self.wk))
        )
        self.wv = nn.Parameter(
            curl.cryptensor(model[f"layers.{layer}.attention.wv.weight"].view(self.wv))
        )


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.rms = RMSNorm(config.norm_eps)
        self.rope = RotaryEmbedding(
            config.rope_theta, config.head_dim, config.max_seq_len
        )
        self.attention = Attention(config)
        self.ff = FeedForward(config)

    def forward(self, x):
        #print("[Transformer] x:", x.shape)
        qkv_attention_store = []
        layer_embedding_norm = self.rms(x)
        for head in range(self.n_heads):
            qt, kt, vt = self.attention(layer_embedding_norm, head)

            q_rotated = self.rope(qt)
            k_rotated = self.rope(kt)
            #print("Q rotated:", q_rotated.shape, " K rotated:", k_rotated.shape)
            qk = Transformer.compute_qk_attention(q_rotated, k_rotated)
            qk_masked = Transformer.apply_attention_mask(qk, x.size(0))
            attention_weights = qk_masked.softmax(dim=1)  # .to(torch.bfloat16)
            qkv_attention = attention_weights.matmul(vt)
            qkv_attention_store.append(qkv_attention)

        stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
        return self.ff(stacked_qkv_attention, x)

    @staticmethod
    def compute_qk_attention(q_rotated, k_rotated):
        #print("Q rotated:", q_rotated.shape, " K rotated:", k_rotated.shape)
        qk = q_rotated.matmul(transpose(k_rotated)) / (q_rotated.size(-1) ** 0.5)
        return qk

    @staticmethod
    def apply_attention_mask(qk, token_count):# -> Any:
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
        self.w0 = None
        self.w1 = None
        self.w2 = None
        self.w3 = None

    def forward(self, x, final_embedding):
        #print("[FeedForward] x:", x.shape, " Final embedding:", final_embedding.shape)
        embedding_delta = x.matmul(transpose(self.w0(None)))
        embedding_after_edit = final_embedding + embedding_delta
        embedding_after_norm = self.rms(embedding_after_edit)
        #print("Embedding after norm:", embedding_after_norm.shape)


        feedforward_output = (
           (embedding_after_norm.matmul(transpose(self.w1(None)))).silu()
            * embedding_after_norm.matmul(transpose(self.w3(None)))
        ).matmul(transpose(self.w2(None)))
        return embedding_after_edit + feedforward_output

    def load_weights(self, model, layer):
        self.rms.load_weights(model, f"layers.{layer}.ffn_norm.weight")
        self.w0 = nn.Parameter(
            curl.cryptensor(model[f"layers.{layer}.attention.wo.weight"])
        )
        self.w1 = nn.Parameter(
            curl.cryptensor(model[f"layers.{layer}.feed_forward.w1.weight"])
        )
        self.w2 = nn.Parameter(
            curl.cryptensor(model[f"layers.{layer}.feed_forward.w2.weight"])
        )
        self.w3 = nn.Parameter(
            curl.cryptensor(model[f"layers.{layer}.feed_forward.w3.weight"])
        )


class Llama(nn.Module):
    def __init__(self, base_path: str):
        super().__init__()
        self.config = Llama.load_config(base_path)
        self.layers = nn.ModuleList(
            [Transformer(self.config) for _ in range(self.config.n_layers)]
        )
        self.embedding = LlamaEmbedding(self.config.vocab_size, self.config.dim)
        self.output_weight = nn.Parameter(
            curl.cryptensor(torch.empty(self.config.vocab_size, self.config.dim))
        )

    @staticmethod
    def load_model(path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(path + "consolidated.00.pth", map_location=device)

        for key, value in model.items():
            model[key] = value.to(torch.float32)
        return model

    @staticmethod
    def load_config(path):
        with open(path + "params.json", "r") as f:
            dic = json.load(f)
            head_dim = dic["dim"] // dic["n_heads"]
            max_seq_len = 2048
            config = LlamaConfig(**dic, head_dim=head_dim, max_seq_len=max_seq_len)
        return config

    def forward(self, tokens: curl.CrypTensor):
        start = time.time()
        tokens = tokens.view(-1)
        #print("[Forward] Tokens:", type(tokens), tokens.shape)
        x = self.embedding(tokens)
        #print("[Forward] Embedding:", type(x))
        for i, layer in enumerate(self.layers):
            layer_start_time = time.time()
            #print(f"[Forward] Layer Input {i}:", type(layer), x.shape)
            x = layer(x)
            #print(f"[Forward] Layer output {i}:", type(x), x.shape)
            print(f"[Forward] Layer {i} time:", time.time() - layer_start_time)
        logits = x[-1].matmul(transpose(self.output_weight(None)))
        next_token = logits.argmax(dim=-1)
        print("Time: ", time.time() - start)
        return next_token

    def load_weights(self, path):
        model = Llama.load_model(path)
        self.embedding.load_weights(model)
        for layer in range(self.config.n_layers):
            self.layers[layer].load_weights(model, layer)
        self.output_weight.data.copy_(curl.cryptensor(model["output.weight"]))


class Llama1B(Llama):
    def __init__(self, *args, **kwargs):
        #print("Initializing 1B weights")
        super().__init__("Llama3.2-1B/")
        #print("CONFIG:", self.config)
        #print("Loading 1B weights")
        self.load_weights("Llama3.2-1B/")
        #print("1B loaded")


class Llama8B(Llama):
    def __init__(self, *args, **kwargs):
        super().__init__("Llama3.1-8B/")
        self.load_weights("Llama3.1-8B/")
        #print("8B loaded")


def main():
    PATH = "Llama3.2-1B/"
    tokenizer = LlamaTokenizer(PATH)
    model = Llama(PATH)
    model.load_weights(PATH)
    prompt = (
        "the answer to the ultimate question of life, the universe, and everything is "
    )
    tokens = tokenizer.tokenize(prompt)
    # #print(prompt, end="\n>")
    a = time.time()
    next_token = None
    for i in range(100):
        next_token = model(tokens)

        # tokens = torch.cat([tokens, next_token.unsqueeze(-1)], dim=-1)
        # #print(tokenizer.decode([next_token.item()]), end = "")
    b = time.time()
    #print(f"Time: {b - a}")
    #print("RESULT:", tokenizer.decode([next_token.item()]))


if __name__ == "__main__":
    curl.init()
    main()
