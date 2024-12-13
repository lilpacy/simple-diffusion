import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from datasets import load_dataset
from transformers import BertTokenizer
import math
import tqdm

# デバイスの設定
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# データセットのロード
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
train_iter = dataset["train"]["text"]

# トークナイザーの準備
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# データセットの準備
def data_process(raw_text_iter):
    data = [
        torch.tensor(
            tokenizer.encode(
                item, add_special_tokens=True, max_length=512, truncation=True
            ),
            dtype=torch.long,
        )
        for item in raw_text_iter
    ]
    return pad_sequence(data, batch_first=True, padding_value=tokenizer.pad_token_id)


train_data = data_process(train_iter)
train_data = train_data.to(device)

# データローダーの作成
batch_size = 16
dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


# Positional Encoding
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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


# Transformer Layer
class TransformerLayer(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(TransformerLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.ReLU(),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_output, _ = self.self_attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)
        return x


# Simple Transformer Model
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(model_dim, num_heads) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(model_dim, input_dim)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


# モデルのインスタンス化
model = SimpleTransformer(
    input_dim=len(tokenizer), model_dim=512, num_heads=8, num_layers=6
)
model.to(device)

# オプティマイザーの設定
optimizer = Adam(model.parameters(), lr=1e-4)


# テキスト生成関数
def generate_text(model, start_text, max_length=50):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(start_text, return_tensors="pt").to(device)
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            if next_token_id == tokenizer.sep_token_id:
                break
        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


# トレーニングループ
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        optimizer.zero_grad()
        input_ids = batch.to(device)
        output = model(input_ids)
        loss = F.cross_entropy(
            output.view(-1, output.size(-1)),
            input_ids.view(-1),
            ignore_index=tokenizer.pad_token_id,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # エポックごとにテキスト生成
    start_text = "Once upon a time"
    generated_text = generate_text(model, start_text)
    print(f"Generated Text after Epoch {epoch + 1}: {generated_text}")

    # エポックごとにモデルを保存
    torch.save(model.state_dict(), "transformer_model.pth")
