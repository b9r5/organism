import os.path
import torch
from bigram import BigramModel

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
train_split = 0.9
# ------------

torch.manual_seed(1337)

# chars is all the unique characters that occur in this text
chars = []

chars_file = 'data/chars.txt'
organism_names_file = 'data/organism_names.txt'
if os.path.exists(chars_file):
    print(f'loading {chars_file}...')
    with open(chars_file, 'r') as f:
        for line in f.readlines():
            chars.append(line.strip('\n'))
    print('done')
else:
    print(f"loading {organism_names_file} to determine characters...")
    with open(organism_names_file, 'r') as f:
        text = f.read()
    chars = sorted(list(set(text)))

    if '>' in chars:
        print('sorry, input text contains >, which will be used for a start-of-text marker')
        exit(1)
    if '~' in chars:
        print('sorry, input text contains ~, which will be used for an end-of-text marker')
        exit(1)

    chars.remove('\n')
    chars.append('>')  # prompt character at beginning of names
    chars.append('~')  # padding character at end of names

    with open(chars_file, 'w') as f:
        for char in chars:
            f.write(f'{char}\n')
    print('done')

vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# max_length is the maximum length of any line in organism_names.txt
max_length = -1

max_length_file = 'data/max_length.txt'
if os.path.exists(max_length_file):
    with open(max_length_file, 'r') as f:
        max_length = int(f.read())
else:
    print(f'writing {max_length_file}...')
    with open(organism_names_file, 'r') as f:
        data = f.readlines()
    max_length = max(len(line.strip()) for line in data)
    with open(max_length_file, 'w') as f:
        f.write(f'{max_length}')
    print('done')

data_file = 'data/organism_names.pt'
if os.path.exists(data_file):
    print(f"loading {data_file}...")
    data0 = torch.load(data_file)
    print('done')
else:
    print(f"loading {organism_names_file} to create {data_file}...")
    with open(organism_names_file, 'r') as f:
        lines = f.readlines()
    data0 = torch.empty(len(lines), max_length + 2, dtype=torch.int64)
    for i, line in enumerate(lines):
        line = line.strip()
        data0[i] = torch.tensor(encode(f'>{line}' + ('~' * (1 + max_length - len(line)))), dtype=torch.int64)
    torch.save(data0, data_file)
    print('done')

# train and test splits
n = int(train_split * len(data0))
train_data = data0[:n]
val_data = data0[n:]

block_size = 1 + max_length

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    batch_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(batch_data), (batch_size,))
    x = torch.stack([batch_data[i][:block_size-1] for i in ix])
    y = torch.stack([batch_data[i][1:block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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

model = BigramModel(vocab_size)
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