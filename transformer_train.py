import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast, get_linear_schedule_with_warmup
import tqdm
import random
from concurrent.futures import ThreadPoolExecutor

from transformer import SimpleTransformer

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")
torch.manual_seed(42)
random.seed(42)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.examples = []
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(
                    lambda t: tokenizer.encode(
                        t, truncation=True, max_length=max_length
                    ),
                    texts,
                )
            )
        for enc in results:
            if len(enc) < 2:
                continue
            self.examples.append(enc)
        print(f"len(self.examples): {len(self.examples)}")

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


def load_data():
    from datasets import load_dataset

    dataset = load_dataset("globis-university/aozorabunko-clean")
    texts = dataset["train"]["text"]
    # texts = ["Once upon a time there was a brave princess who"] * 2000
    return texts


texts = load_data()
print(f"texts[0]: {texts[0]}")
train_dataset = TextDataset(texts, tokenizer, max_length=128)
train_dataloader = DataLoader(
    train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
)

model = SimpleTransformer(
    vocab_size=len(tokenizer), model_dim=256, num_heads=4, num_layers=4
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

total_steps = len(train_dataloader) * 5
warmup_steps = total_steps // 10
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# print(model.sample_text(tokenizer, "Once upon a time"))
print(model.sample_text(tokenizer, "むかしむかしあるところに"))
for epoch in range(10):
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
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print("Epoch:", epoch + 1, "Loss:", total_loss / len(train_dataloader))
    # print(model.sample_text(tokenizer, "Once upon a time"))
    print(model.sample_text(tokenizer, "むかしむかしあるところに"))

torch.save(model.state_dict(), "transformer_model.pth")
