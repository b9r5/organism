import os
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def decode(l):
    """Given a list of ints representing a decoded string, output the string"""
    return ''.join([itos[i] for i in l])

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

block_size = 1 + max_length


def encode(string, for_training=True, batch_size=1):
    string = '>' + string
    if for_training:
        string = string + ('~' * (1 + max_length - len(string)))
    ints = [stoi[c] for c in string]
    if for_training and len(ints) < max_length:
        ints = ints + ([0] * (max_length - len(ints)))

    result = torch.tensor(ints, dtype=torch.long)

    if not for_training:
        result = result.unsqueeze(0).repeat(batch_size, 1)
        result = result.to(device)

    return result
