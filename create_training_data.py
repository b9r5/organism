import random
import re

with open('data/raw_organism_names.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

random.seed(42)

# shuffle the names, since we intend on using the first 90% for training, and we don't want to bias training to any
# structure that might exist in the ordering of the original list (it's unlikely that this matters much)
random.shuffle(lines)

with open('data/organism_names.txt', 'w') as f:
    for line in lines:
        try:
            # we will have enough training data with ASCII names only, so exclude non-ASCII
            line.encode('ascii')
            re.sub(r'[^\S\n]', ' ', line)
            f.write(line)
        except UnicodeEncodeError:
            print(f'skipping {line.strip()}')
