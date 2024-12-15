import torch
from transformers import GPT2TokenizerFast
from transformer import SimpleTransformer

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = SimpleTransformer(
    vocab_size=len(tokenizer), model_dim=32, num_heads=2, num_layers=2
).to(device)

model.load_state_dict(
    torch.load("transformer_model.pth", map_location=device, weights_only=True)
)

start_text = "Once upon a time"
generated_text = model.sample_text(
    tokenizer, start_text, max_length=50, top_k=50, temperature=1.0
)

print("Generated Text:", generated_text)
