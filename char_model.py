import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

# A. Load the text and build the vocabulary
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Total characters:", len(text))
print(text[:200])

chars = sorted(set(text))          # every unique character, in order
vocab_size = len(chars)
print("Vocabulary:", "".join(chars))
print("Vocab size:", vocab_size)

# B. Map characters <-> integers
stoi = {ch: i for i, ch in enumerate(chars)}   # string to integer
itos = {i: ch for i, ch in enumerate(chars)}   # integer to string
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

print(encode("hello"))
print(decode(encode("hello")))

# Whole dataset as one long tensor of integers
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

# 90% to train on, 10% held back to test on unseen text
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# C. Batches
torch.manual_seed(1337)        # same random numbers every run, so results are repeatable
device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 8                 # how many characters of context the model sees
batch_size = 32                # how many sequences we process at once

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))      # 32 random start positions
    x = torch.stack([d[i:i + block_size] for i in ix])          # inputs
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])  # targets: shifted by 1
    return x.to(device), y.to(device)

xb, yb = get_batch("train")
print("x shape:", xb.shape, "y shape:", yb.shape, "device:", xb.device)
print(xb[0])
print(yb[0])

# D. Bigram model: predicts the next char from the current char only
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # A table of vocab_size x vocab_size learnable numbers.
        # Row i = scores for "which char comes after char i".
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)   # (B, T, C)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]                           # scores for the last position only
            probs = F.softmax(logits, dim=-1)                   # scores -> probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # sample one char
            idx = torch.cat((idx, idx_next), dim=1)             # append and repeat
        return idx

model = BigramLanguageModel(vocab_size).to(device)
logits, loss = model(xb, yb)
print("logits shape:", logits.shape)
print("starting loss:", loss.item())

# E. Training
learning_rate = 1e-2
max_iters = 3000
eval_interval = 300
eval_iters = 200

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()                      # no gradient tracking: we're only measuring
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

wandb.init(
    project="llm-from-scratch",
    name="bigram-baseline",
    config={"model": "bigram", "block_size": block_size, "batch_size": batch_size,
            "learning_rate": learning_rate, "max_iters": max_iters},
)
for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        wandb.log({"train_loss": losses["train"], "val_loss": losses["val"]}, step=it)

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)   # clear old gradients
    loss.backward()                         # work out how to nudge every number
    optimizer.step()                        # nudge them

# Generate 500 characters starting from a single newline (ID 0)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
wandb.finish()