import curl.nn as nn


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, emb_size, max_seq_length):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, emb_size)
        self.position_embeddings = nn.Embedding(max_seq_length, emb_size)
        self.token_type_embeddings = nn.Embedding(2, emb_size)
        self.LayerNorm = nn.LayerNorm(emb_size)

    def forward(self, input_ids, token_type_ids):
        word_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings.weight[:input_ids.size()[1], :]
        type_emb = self.token_type_embeddings(token_type_ids)

        emb = word_emb + pos_emb + type_emb
        emb = self.LayerNorm(emb)
        return emb


class BertSelfAttention(nn.Module):
    def __init__(self, emb_size, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = emb_size // self.n_heads
        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)

    def forward(self, emb):
        B, T, C = emb.shape  # batch size, sequence length, embedding size

        q = self.query(emb).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = self.key(emb).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = self.value(emb).view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        weights = q @ k.transpose(-2, -1) * self.head_size**-0.5
        weights = weights.softmax(dim=-1)

        emb_rich = weights @ v
        emb_rich = emb_rich.transpose(1, 2).reshape(B, T, C)
        return emb_rich


class BertSelfOutput(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.dense = nn.Linear(emb_size, emb_size)
        self.LayerNorm = nn.LayerNorm(emb_size)

    def forward(self, emb_rich, emb):
        x = self.dense(emb_rich)
        x = x + emb
        out = self.LayerNorm(x)
        return out


class BertAttention(nn.Module):
    def __init__(self, emb_size, n_heads):
        super().__init__()
        self.self = BertSelfAttention(emb_size, n_heads)
        self.output = BertSelfOutput(emb_size)

    def forward(self, emb):
        emb_rich = self.self(emb)
        out = self.output(emb_rich, emb)
        return out


class BertIntermediate(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.dense = nn.Linear(emb_size, 4 * emb_size)
        self.gelu = nn.GELU()

    def forward(self, att_out):
        x = self.dense(att_out)
        out = self.gelu(x)
        return out


class BertOutput(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.dense = nn.Linear(4 * emb_size, emb_size)
        self.LayerNorm = nn.LayerNorm(emb_size)

    def forward(self, intermediate_out, att_out):
        x = self.dense(intermediate_out)
        x = x + att_out
        out = self.LayerNorm(x)
        return out


class BertLayer(nn.Module):
    def __init__(self, emb_size, n_heads ):
        super().__init__()
        self.attention = BertAttention(emb_size, n_heads)
        self.intermediate = BertIntermediate(emb_size)
        self.output = BertOutput(emb_size)

    def forward(self, emb):
        att_out = self.attention(emb)
        intermediate_out = self.intermediate(att_out)
        out = self.output(intermediate_out, att_out)
        return out


class BertEncoder(nn.Module):
    def __init__(self, emb_size, n_heads, n_layers):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(emb_size, n_heads) for i in range(n_layers)])

    def forward(self, x):
        for l in self.layer:
            x = l(x)
        return x

class BertPooler(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.dense = nn.Linear(emb_size, emb_size)
        self.tanh = nn.Tanh()

    def forward(self, encoder_out):
        pool_first_token = encoder_out[:, 0]
        out = self.dense(pool_first_token)
        out = self.tanh(out)
        return out

class BertModel(nn.Module):
    def __init__(self, vocab_size, emb_size, seq_len, n_heads, n_layers):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, emb_size, seq_len)
        self.encoder = BertEncoder(emb_size, n_heads, n_layers)
        self.pooler = BertPooler(emb_size)

    def forward(self, input_ids, token_type_ids):
        emb = self.embeddings(input_ids, token_type_ids)
        out = self.encoder(emb)
        pooled_out = self.pooler(out)
        return out, pooled_out

class BertForSequenceClassification(nn.Module):
    def __init__(self, vocab_size, emb_size, seq_len, n_heads, n_layers):
        super().__init__()
        self.bert = BertModel(vocab_size, emb_size, seq_len, n_heads, n_layers)
        self.classifier = nn.Linear(emb_size, 2)

    def forward(self, input_ids, token_type_ids):
        _, pooled_out = self.bert(input_ids, token_type_ids)
        logits = self.classifier(pooled_out)
        return logits


class BertTinyForSequenceClassification(BertForSequenceClassification):
    def __init__(self):
        super(BertTinyForSequenceClassification, self).__init__(vocab_size=30522, emb_size=128, seq_len=512, n_heads=2, n_layers=2)


class BertBaseForSequenceClassification(BertForSequenceClassification):
    def __init__(self):
        super(BertBaseForSequenceClassification, self).__init__(vocab_size=28996, emb_size=768, seq_len=512, n_heads=12, n_layers=12)