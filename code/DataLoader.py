import re

class DataLoader:
    def __init__(self, data_path):
        self.data = open(data_path, "r").readlines()

    def generate_sentences(self):
        for sentence in self.data:
            parsed_sentence = self.parse_sentence(sentence)
            yield parsed_sentence[:-1]

    def parse_sentence(self, sentence):
        sentence = sentence.split(")")
        out = []
        for e in sentence:
            e = e.split(" ")
            if e[-1] != "":
                out.append(e[-1])
        return out


if __name__ == "__main__":
    path = "../data/23.auto.clean"
    d = DataLoader(path)
    for sentence in d.generate_sentences():
        print(sentence)
        
        
    