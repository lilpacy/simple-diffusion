import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast
import tqdm
import random

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch.manual_seed(42)
random.seed(42)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.examples = []
        for t in texts:
            enc = tokenizer.encode(t, truncation=True, max_length=max_length)
            if len(enc) < 2:
                continue
            self.examples.append(enc)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        enc = self.examples[idx]
        input_ids = enc[:-1]
        target_ids = enc[1:]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(
            target_ids, dtype=torch.long
        )

def collate_fn(batch):
    input_ids = [b[0] for b in batch]
    target_ids = [b[1] for b in batch]
    input_ids = nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    target_ids = nn.utils.rnn.pad_sequence(
        target_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    return input_ids, target_ids

def causal_mask(size):
    mask = torch.full((size, size), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim)
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
            nn.ReLU(),
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

    def forward(self, x):
        mask = causal_mask(x.size(1)).to(x.device)
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        logits = self.output_layer(x)
        return logits


def load_data():
    # 任意のテキストデータを使用可能。ここではwikitext-2-rawを使用。
    # 必要な場合はdatasetsをインストールし、以下コメントアウトを解除
    from datasets import load_dataset

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    texts = dataset["train"]["text"]
    # 簡易的なテキスト例（ユーザ環境で置換可能）
    # texts = ["Once upon a time there was a brave princess who"] * 2000
    return texts

texts = load_data()
train_dataset = TextDataset(texts, tokenizer, max_length=128)
train_dataloader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
)

model = SimpleTransformer(
    vocab_size=len(tokenizer), model_dim=256, num_heads=4, num_layers=4
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def sample_text(model, start_text, max_length=50, top_k=50, temperature=1.0):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(start_text, return_tensors="pt").to(device)
        for _ in range(max_length):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            filtered_logits = next_token_logits
            if top_k > 0:
                values, _ = torch.topk(next_token_logits, top_k)
                min_values = values[:, -1].unsqueeze(1)
                filtered_logits = torch.where(
                    next_token_logits < min_values,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )
            probs = F.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break
        return tokenizer.decode(input_ids[0], skip_special_tokens=True)

for epoch in range(3):
    model.train()
    total_loss = 0
    for input_ids, target_ids in tqdm.tqdm(train_dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=tokenizer.pad_token_id,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("Epoch:", epoch + 1, "Loss:", total_loss / len(train_dataloader))
    print(sample_text(model, "Once upon a time"))

# save
torch.save(model.state_dict(), "transformer_model.pth")
