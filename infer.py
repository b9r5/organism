import torch
from sys import stdin

from gpt import GPTModel
import hyperparms
from codec import vocab_size, device, block_size, max_length, encode, decode

print('loading model...')
model = GPTModel(vocab_size=vocab_size, block_size=block_size, n_embd=hyperparms.n_embd, n_head=hyperparms.n_head,
                 n_layer=hyperparms.n_layer, dropout=hyperparms.dropout, device=device)
model.load_state_dict(torch.load('data/model.pt'))
model = model.to(device)
print('done')

print('Enter the beginning of a scientific name at the > prompt.')
print('Examnple:')
print('> Lactobacillus')
print('Enter ctrl-d when done.')

while True:
    print('> ', end='')
    line = stdin.readline()
    if not line:
        break

    n_continuations = 5
    context = encode(line.strip(), for_training=False, batch_size=n_continuations)
    prediction = model.generate(context, max_new_tokens=max_length)

    for i in range(n_continuations):
        answer = decode(prediction[i].tolist())
        answer = answer.lstrip('>')
        answer = answer.rstrip('~')
        print(answer)

print('bye!')
