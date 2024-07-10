with open('data/all_dinosaurs.txt', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.splitlines()

with open('data/dinosaurs_for_training.csv', 'w', encoding='utf-8') as f:
    for line in lines:
        for i in range(len(line)):
            prefix = line[:i+1]
            char = "#" if i+1 == len(line) else line[i+1:i+2]
            f.write(f"{prefix},{char}\n")
