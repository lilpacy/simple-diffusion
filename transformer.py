import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(size):
    mask = torch.full((size, size), -1e9)
    mask = torch.triu(mask, diagonal=1)
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class TransformerLayer(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(TransformerLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim),
        )
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, num_layers):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        self.layers = nn.ModuleList(
            [TransformerLayer(model_dim, num_heads) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(model_dim, vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        mask = causal_mask(x.size(1)).to(x.device)
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        logits = self.output_layer(x)
        return logits

    def sample_text(
        self, tokenizer, start_text, max_length=50, top_k=50, temperature=1.0
    ):
        self.eval()
        with torch.no_grad():
            input_ids = tokenizer.encode(start_text, return_tensors="pt").to(
                next(self.parameters()).device
            )
            for _ in range(max_length):
                logits = self(input_ids)
                logits = logits[:, -1, :] / temperature
                if top_k > 0:
                    values, _ = torch.topk(logits, top_k)
                    min_values = values[:, -1].unsqueeze(1)
                    logits = torch.where(
                        logits < min_values, torch.full_like(logits, -1e9), logits
                    )
                probs = F.softmax(logits, dim=-1)
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    break
                next_token_id = torch.multinomial(probs, num_samples=1)
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
            return tokenizer.decode(input_ids[0], skip_special_tokens=True)
