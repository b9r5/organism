import torch
import torch.nn as nn
from torch.nn import functional as F

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
print(chars)
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode('Onitis coxalis'))
print(decode(encode('Onitis coxalis')))

with open('data/shuffled_organism_names.txt', 'r') as f:
    data = f.readlines()

max_length = max(len(line) for line in data)

for i, line in enumerate(data):
    line = line.strip()
    #print(f'{i} {line}')
    data[i] = torch.tensor(encode(f'>{line}' + ('_' * (1 + max_length - len(line)))))
    #print(data[i])

# split into train and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 4
block_size = 1 + max_length

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data), (batch_size,))
    x = torch.stack([data[i][:block_size-1] for i in ix])
    y = torch.stack([data[i][1:block_size] for i in ix])
    return x, y

xb, yb = get_batch('train')

print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

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

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(torch.full((1, 1), stoi['>']), max_new_tokens=max_length)[0].tolist()))

