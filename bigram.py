import torch
import torch.nn as nn
from torch.nn import functional as F
import torch_xla.core.xla_model as xm

# this file is a lightly modified copy of https://github.com/karpathy/ng-video-lecture/blob/master/bigram.py

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = xm.xla_device() if 'xla' in str(xm.xla_device()) else None
print(f'device: {str(device)}')
eval_iters = 200
# ------------

torch.manual_seed(1337)

print("loading organism names to determine characters...")
with open('data/shuffled_organism_names.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))

if '>' in chars:
    print('sorry, input text contains >, which will be used for a start-of-text marker')
    exit(1)
if '~' in chars:
    print('sorry, input text contains ~, which will be used for an end-of-text marker')
    exit(1)

chars.remove('\n')
chars.append('>') # prompt character at beginning of names
chars.append('~') # padding character at end of names
# chars is all the unique characters that occur in this text
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
print('done')

#print(encode('Onitis coxalis'))
#print(decode(encode('Onitis coxalis')))

with open('data/shuffled_organism_names.txt', 'r') as f:
    data = f.readlines()

max_length = max(len(line) for line in data)

print("loading organism names to create training data...")
for i, line in enumerate(data):
    line = line.strip()
    #print(f'{i} {line}')
    data[i] = torch.tensor(encode(f'>{line}' + ('~' * (1 + max_length - len(line)))))
    #print(data[i])
print('done')

# train and test splits sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 4
block_size = 1 + max_length

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data), (batch_size,))
    x = torch.stack([data[i][:block_size-1] for i in ix])
    y = torch.stack([data[i][1:block_size] for i in ix])
    if device is not None:
        x, y = x.to(device), y.to(device)
    return x, y

#xb, yb = get_batch('train')
#print('inputs:')
#print(xb.shape)
#print(xb)
#print('targets:')
#print(yb.shape)
#print(yb)

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus on only the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = BigramLanguageModel()
if device is not None:
    model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        # in a TPU v2 environment, estimate_loss() was taking about 80% of training loop time
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))