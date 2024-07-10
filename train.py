import torch
from torch.utils.data import Dataset
import pandas as pd

# split into train and validation sets
#n = int(0.9 * len(lines))
#train_data = data[:n]
#val_data = data[n:]

class DinosaurDataset(Dataset):
    def __init__(self, csv_file):
        self.dinosaur_names = pd.read_csv(csv_file, header=None)
        self.max_x_length = self.dinosaur_names.iloc[:, 0].str.len().max()

        chars = set()
        for col in self.dinosaur_names.columns:
            for s in self.dinosaur_names[col]:
                for char in s:
                    chars.update(char)
        self.chars = sorted(list(chars))
        #print(self.chars)

        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}

    def __len__(self):
        return len(self.dinosaur_names)

    def __getitem__(self, idx):
        row = self.dinosaur_names.iloc[idx]
        x = torch.tensor(self.__encode__(row[0]))
        y = torch.tensor(self.char_to_idx[row[1][0]])
        return x, y

    def __encode__(self, str):
        prefix = [self.char_to_idx[char] for char in str]
        return prefix + ([-1] * (self.max_x_length - len(prefix)))

    def __decode__(self, tensor):
        if tensor.dim() == 0:
            return self.idx_to_char[tensor.item()]
        else:
            return ''.join(['' if idx == -1 else self.idx_to_char[idx] for idx in tensor.tolist()])

dinosaur_dataset = DinosaurDataset(csv_file='data/dinosaurs_for_training.csv')

item4 = dinosaur_dataset.__getitem__(4)
print(item4)
print(dinosaur_dataset.__decode__(item4[0])) # Aardo
print(dinosaur_dataset.__decode__(item4[1])) # n
