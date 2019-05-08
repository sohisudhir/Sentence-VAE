from pprint import pprint
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class PennTreeData(Dataset):
    def __init__(self, data_path, seq_length):
        self.seq_length = seq_length
        self.data = self.remove_tags(open(data_path, "r").read())
        self.vocab = list(set(self.data))
        self.data_size, self.vocab_size  = len(self.data), len(self.vocab) 
        self.word2idx= {w:i for i,w in enumerate(self.vocab)}
        self.idx2word = {i:w for i,w in enumerate(self.vocab)}

    def __getitem__(self, idx):
        offset = np.random.randint(0, len(self.data)-self.seq_length-2)
        inputs =  [self.word2idx[w] for w in self.data[offset:offset+self.seq_length]]
        targets = [self.word2idx[w] for w in self.data[offset+1:offset+self.seq_length+1]]
        return inputs, targets

    def remove_tags(self, data):
        """
        Remove tree tags from Penn Treebank dataset
        """
        data = data.split(")")
        out = []
        for e in data:
            e = e.split(" ")
            if e[-1] != "":
                out.append(e[-1])
        return out

    def convert_to_string(self, char_idx):
        return ''.join(self.idx2word[idx] for idx in char_idx)

    def convert_to_idx(self, chars):
        return [self.word2idx[char] for char in chars]

    def __len__(self):
        return self.data_size


if __name__ == "__main__":
    path = "../data/23.auto.clean"
    penntree = PennTreeData(data_path = path, seq_length = 5)
    data = DataLoader(penntree, batch_size=1, num_workers=1)
    for x, y in data:
        print(x)
        print(y)
        break



        
        
    