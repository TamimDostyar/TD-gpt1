import torch
from model import BigramLanguageModel

# Set seed and device
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

# Hyperparameters based on device
if device == 'cuda':
    batch_size = 64
    max_iteration = 5000
    block_size = 256
    eval_iters = 200
    learning_rate = 1e-3
    eval_interval = 500
    n_embed = 384
    dropout = 0.2
    n_head = 6
    n_layer = 6
else:
    batch_size = 32
    max_iteration = 3000
    block_size = 128
    eval_iters = 100
    learning_rate = 3e-4
    eval_interval = 300
    n_embed = 128
    dropout = 0.1
    n_head = 4
    n_layer = 4

# Load input data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Build vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Index mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encoding and decoding functions
def encoder(text):
    return [stoi[ch] for ch in text]

def decoder(indices):
    return ''.join([itos[i] for i in indices])

# Test encoding/decoding
encoded = encoder("Tamim")
decoded = decoder(encoded)
print(encoded)
print(decoded)

# Encode the entire text file
data = torch.tensor(encoder(text), dtype=torch.long)
print(data.dtype, data.shape)
print(data[:100])

# Split data into train and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Demo: show context-target pairs
demo_block = 8
x = train_data[:demo_block]
y = train_data[1:demo_block + 1]

for t in range(demo_block):
    context = x[:t + 1]
    print("Current Context", context)
    target = y[t]
    print("Current Target", target)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch('train')
print('input -->', xb.shape)
print('target -->', yb.shape)
print('___' * 20)

for b in range(min(2, batch_size)):
    for t in range(min(8, block_size)):
        context = xb[b, :t + 1]
        target = yb[b, t]
        print(f"input: {context.tolist()} target: {target}")

# Create model
model = BigramLanguageModel(
    vocab_size=vocab_size,
    n_embed=n_embed,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
    device=device
)
m = model.to(device)

print(f"Model parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iteration):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train')
    
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate text
start_idx = torch.zeros((1, 1), dtype=torch.long, device=device)
gen_idx = m.generate(start_idx, max_new_tokens=500)

generated_text = decoder(gen_idx[0].tolist())
print("generated text:", generated_text)

with open('generated_text.txt', 'w') as f:
    f.write(generated_text)
print("generated text saved to generated_text.txt")
