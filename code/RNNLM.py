import torch
import torch.nn as nn


class RNNLM(nn.Module):
    def __init__(self, seq_length, vocabulary_size, embedding_dim=32,
                 lstm_num_hidden=256, lstm_num_layers=3, device='cuda:0'):
        super(RNNLM, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                      embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=lstm_num_hidden,
                            num_layers=lstm_num_layers,
                            batch_first=False)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)
        
    def forward(self, x, hidden = None):
        x = self.embedding(x)
        all_hidden, hidden = self.lstm(x, hidden)
        out = self.linear(all_hidden)

        return out, hidden

if __name__ == "__main__":
    seq_length = 5
    batch_size = 32
    vocabulary_size = 100
    x = torch.rand(seq_length,batch_size).type(torch.LongTensor)
    model = RNNLM(seq_length=seq_length,
                  vocabulary_size=vocabulary_size)
    out, h = model(x)
    print(out.size())
        
