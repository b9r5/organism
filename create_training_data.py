import random

with open('data/all_dinosaurs.txt', 'r') as f:
    lines = f.readlines()

random.seed(42)

random.shuffle(lines)

with open('data/shuffled_dinosaurs.txt', 'w') as f:
    f.writelines(lines)

split = 0.9

def line_to_xy(line):
    """Given a line, returns the x,y training data for the line."""
    line = line.strip()
    x = line
    y = f'{line[1:]}$' # $ signifies end of dinosaur name
    return f'{x},{y}\n'

with open('data/training_dinosaurs.csv', 'w', encoding='utf-8') as f:
    for i in range (int(0.9 * len(lines))):
        f.write(line_to_xy(lines[i]))

with open('data/validation_dinosaurs.csv', 'w', encoding='utf-8') as f:
    for i in range(int(0.9 * len(lines)), len(lines)):
        f.write(line_to_xy(lines[i]))
