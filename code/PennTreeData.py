from pprint import pprint
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import Counter

class PennTreeData(Dataset):
    def __init__(self, data_path, rare_threshold=1):
        self.rare_threshold = rare_threshold
        self.data = self.remove_tags(open(data_path, "r").read())
        self.SOS = "|new"
        self.EOS = "."
        self.pad = "<_>"
        self.rare = "RARE"
        self.data.extend([self.SOS for i in range(self.rare_threshold + 1)])
        self.data.extend([self.pad for i in range(self.rare_threshold + 1)])
        self.data.extend([self.rare for i in range(self.rare_threshold + 1)])

        self.vocab = list(set(self.data))        
        self.define_mappings()

        self.sentences = [sentence for sentence in self.read_sentence()]
        self.max_sentence_len = self.largest_sentence()

    def define_mappings(self):
        self.word_counter = Counter(self.data)
        self.word2idx = {}
        self.idx2word = {}
        idx = 0
        for w in self.vocab:
            if self.word_counter[w] > self.rare_threshold:
                self.word2idx[w] = idx
                self.idx2word[idx] = w
                idx+=1
        self.vocab_size = len(self.word2idx)
        
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        inputs =  [self.word2idx[w] if self.word_counter[w] > self.rare_threshold else self.word2idx[self.rare] for w in sentence]
        targets = [self.word2idx[w] if self.word_counter[w] > self.rare_threshold else self.word2idx[self.rare] for w in sentence[1:]]
        # add padding
        inputs.extend([self.word2idx[self.pad] for i in range(self.max_sentence_len - len(inputs))])
        targets.extend([self.word2idx[self.pad] for i in range(self.max_sentence_len - len(targets))])
        return inputs, targets, len(sentence)

    def read_sentence(self):
        sentence = [self.SOS]
        for word in self.data:
            sentence.append(word)
            if word == self.EOS:
                yield sentence
                sentence = [self.SOS]

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

    def largest_sentence(self):
        max_len = 0
        for sentence in self.read_sentence():
            if len(sentence) > max_len:
                max_len = len(sentence)
        return max_len

    def convert_to_string(self, char_idx):
        return ' '.join(self.idx2word[idx] for idx in char_idx)

    def convert_to_idx(self, chars):
        return [self.word2idx[char] for char in chars]

    def __len__(self):
        return len(self.sentences)


if __name__ == "__main__":
    #path = "../data/02-21.10way.clean"
    path = "../data/sample.clean"
    penntree = PennTreeData(data_path = path, rare_threshold = 1)
    
    data = DataLoader(penntree, batch_size=1, num_workers=1)
    for x, y in data:
        print(x)
        break
        
        
        
    
    
        
        



        
        
    