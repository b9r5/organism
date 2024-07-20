import os.path
import torch

from codec import device, encode, decode, organism_names_file, max_length, vocab_size, block_size
from gpt import GPTModel
import hyperparms

torch.manual_seed(1337)

data_file = 'data/organism_names.pt'
if os.path.exists(data_file):
    print(f"loading {data_file}...")
    data = torch.load(data_file)
    print('done')
else:
    print(f"loading {organism_names_file} to create {data_file}...")
    with open(organism_names_file, 'r') as f:
        lines = f.readlines()
    data = torch.empty(len(lines), max_length + 2, dtype=torch.int64)
    for i, line in enumerate(lines):
        line = line.strip()
        data[i] = encode(line)
    torch.save(data, data_file)
    print('done')

# train and test splits
n = int(hyperparms.train_split * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    batch_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(batch_data), (hyperparms.batch_size,))
    x = torch.stack([batch_data[i][:block_size-1] for i in ix])
    y = torch.stack([batch_data[i][1:block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(hyperparms.eval_iters)
        for k in range(hyperparms.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# was: model = bigram.BigramModel(vocab_size)

model = GPTModel(vocab_size=vocab_size, block_size=block_size, n_embd=hyperparms.n_embd, n_head=hyperparms.n_head,
                 n_layer=hyperparms.n_layer, dropout=hyperparms.dropout, device=device)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparms.learning_rate)

print('training...')

for iter in range(hyperparms.max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % hyperparms.eval_interval == 0:
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

print('done')

print('saving model...')
torch.save(model.state_dict(), 'data/model.pt')
print('done')

# generate from the model

n_samples = 10
context = encode('', for_training=False, batch_size=n_samples)
predictions = model.generate(context, max_new_tokens=max_length)

for i in range(n_samples):
    print(decode(predictions[i].tolist()))
