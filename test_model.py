import torch
from model import BigramLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

chars = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def decode(indices):
    return ''.join([itos[i] for i in indices])

model = BigramLanguageModel(
    vocab_size=65,
    n_embed=384,
    block_size=256,
    n_layer=6,
    n_head=6,
    dropout=0.2,
    device=device
)

model.load_state_dict(torch.load("gpt_pretrained_shakespeare.pt", map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully!")

context = torch.zeros((1, 1), dtype=torch.long, device=device)

with torch.no_grad():
    generated = model.generate(context, max_new_tokens=1000)
print("\n--- Generated Text ---\n")
print(decode(generated[0].tolist()))
