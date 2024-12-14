import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast, get_linear_schedule_with_warmup
from datasets import load_dataset
import tqdm
import random

# デバイス設定
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

torch.manual_seed(42)
random.seed(42)

# トークナイザ設定（GPT-2）
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# GPT-2はpad_tokenがないので独自に追加する
# 特殊トークンを追加する場合、model側でも再度embeddingsをresizeする必要がある
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id

# データセット読み込み
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
train_texts = dataset["train"]["text"]
val_texts = dataset["validation"]["text"]


# TextDatasetクラス: トークン化とフォーマットを担当
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.input_ids_list = []
        for t in texts:
            # 空行や極端に短い行をスキップ
            if t.strip() == "":
                continue
            enc = tokenizer(
                t,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)  # shape: (max_length,)
            # 少なくとも2トークン以上必要
            if len((input_ids != pad_token_id).nonzero()) < 2:
                continue
            self.input_ids_list.append(input_ids)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        input_ids = self.input_ids_list[idx]
        # Causal LM用: input_idsを[x1, x2, x3, ..., xN], target_idsを[x2, x3, ..., xN, <pad>]
        # ただし末尾は使わないので実質input_ids[:-1], target_ids[:-1]
        # ただしパディングはあらかじめ含まれているので、そのままshiftして対応
        target_ids = torch.roll(input_ids, shifts=-1)
        target_ids[-1] = pad_token_id  # 最後をpadに
        return input_ids, target_ids

def collate_fn(batch):
    # Datasetは既にパディング済みなので単純にstack
    input_ids = torch.stack([b[0] for b in batch], dim=0)
    target_ids = torch.stack([b[1] for b in batch], dim=0)
    return input_ids, target_ids

# Causalマスク関数
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
    model, tokenizer, start_text, max_length=50, top_k=50, temperature=1.0, device="cpu"
):
    model.eval()
    with torch.no_grad():
        enc = tokenizer(start_text, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]
        for _ in range(max_length):
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(1)
                logits = torch.where(
                    logits < min_values, torch.full_like(logits, float("-inf")), logits
                )
            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                break
            next_token_id = torch.multinomial(probs, num_samples=1)
            if next_token_id.item() == eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
        return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def evaluate(model, dataloader, tokenizer, device="cpu"):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=pad_token_id,
            )
            total_loss += loss.item() * input_ids.size(0)
            total_count += input_ids.size(0)
    avg_loss = total_loss / total_count
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


# データセットインスタンス化
train_dataset = TextDataset(train_texts, tokenizer, max_length=128)
val_dataset = TextDataset(val_texts, tokenizer, max_length=128)

train_dataloader = DataLoader(
    train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
)

# モデル生成
model_dim = 128
num_heads = 4
num_layers = 4
model = SimpleTransformer(
    vocab_size=len(tokenizer),
    model_dim=model_dim,
    num_heads=num_heads,
    num_layers=num_layers,
).to(device)

# トークナイザ拡張分の埋め込みサイズ再調整
model.embedding.weight.data.normal_(mean=0.0, std=0.02)

# オプティマイザとスケジューラ
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
total_steps = len(train_dataloader) * 3
warmup_steps = total_steps // 10
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# 学習ループ
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_count = 0
    for input_ids, target_ids in tqdm.tqdm(train_dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=pad_token_id,
        )
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * input_ids.size(0)
        total_count += input_ids.size(0)

    avg_train_loss = total_loss / total_count
    train_ppl = math.exp(avg_train_loss)

    val_loss, val_ppl = evaluate(model, val_dataloader, tokenizer, device)
    print(f"Epoch: {epoch+1}")
    print(f"Train Loss: {avg_train_loss:.4f}, Train PPL: {train_ppl:.2f}")
    print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
    print("Sample:", sample_text(model, tokenizer, "Once upon a time", device=device))

# モデル保存
torch.save(model.state_dict(), "transformer_model.pth")
