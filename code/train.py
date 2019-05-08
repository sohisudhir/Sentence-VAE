import os
import time
from datetime import datetime
import argparse

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from PennTreeData import PennTreeData
from RNNLM import RNNLM


TEXT_FILE = "../data/23.auto.clean"

################################################################################
def calculate_perplexity(predictions, targets):
    # TODO
    pass

def temperature_sample(h, temperature):
    h = h.squeeze()
    distribution = torch.softmax(h/temperature, dim=0)    
    return torch.multinomial(distribution,1).item()

def generate_samples(model, vocab_size, device, temperature = 0.5, method = "greedy"):
    x = torch.randint(low=0,
                  high=vocab_size,
                  size=(1, 1),
                  dtype=torch.long,
                  device=device)
    hidden = None
    with torch.no_grad():
        generated_sentence = [x.item()]
        for seq in range(config.seq_length):
            out,hidden  = model(x, hidden)
            if method == "greedy":
                sample_idx = torch.argmax(out, dim=2).item()
            else:
                sample_idx = temperature_sample(out, temperature)
            x = torch.tensor(sample_idx, dtype=torch.long, device=device).view(1,1)
            generated_sentence.append(sample_idx)
    
    return generated_sentence

def generate_sentence(model, sentence, device):
    out_sentence = [char for char in sentence]
    with torch.no_grad():
        hidden = None
        sentence = torch.tensor(sentence, dtype=torch.long, device=device).unsqueeze(1)
        out, hidden = model(sentence, hidden)
        last_char = out[-1, : , :].squeeze()
        last_char = torch.argmax(last_char).unsqueeze(0).unsqueeze(0)

        for char in range(config.chars_to_generate):
            out, hidden = model(last_char, hidden)
            last_char = out[-1, : , :].squeeze()
            last_char = torch.argmax(last_char).unsqueeze(0).unsqueeze(0)
            
            out_sentence.append(last_char.item())
    return out_sentence

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PennTreeData(TEXT_FILE, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    
    # Initialize the model that we are going to use
    model = RNNLM(seq_length=config.seq_length,
                  embedding_dim=config.embedding_dim,
                  vocabulary_size=dataset.vocab_size,
                  lstm_num_hidden=config.lstm_num_hidden,
                  lstm_num_layers=config.lstm_num_layers,
                  device=device)
    model.to(device)

    #setup output_file
    # f = open("../sample_results/samples_{}_{}.txt".format(config.sample_method, config.temperature),"w+")

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    losses = []
    accuracies = []
    for epoch in range(config.n_epochs):
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            # Only for time measurement of step through network
            t1 = time.time()

            batch_inputs = torch.stack(batch_inputs).to(device)
            batch_targets = torch.stack(batch_targets).to(device).view(-1)
            output, _ = model(batch_inputs)

            output = output.view(batch_inputs.size()[0]*batch_inputs.size()[1], -1)
            # accuracy = calculate_accuracy(output, batch_targets)
            accuracy = 0.0  #fix me

            loss = criterion(output, batch_targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # For plotting
            # accuracies.append(accuracy)
            # losses.append(loss.item())
            
            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)
            if step % config.print_every == 0:

                print("[{}] Epoch {} Train Step {:00f}, Batch Size = {}, Examples/Sec = {:.2f}, "
                    "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), epoch, step,
                        config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            if step % config.sample_every == 0:
                generated_sample = generate_samples(model,
                                                    vocab_size=dataset.vocab_size,
                                                    device=device,
                                                    temperature=config.temperature,
                                                    method=config.sample_method)
                print("SAMPLE:", dataset.convert_to_string(generated_sample))

            #     # f.write(str(step) + ": " + out + "\n")

            #     sentence = dataset.convert_to_idx(config.gen_sentence)
            #     out_sentence = generate_sentence(model,
            #                                      sentence,
            #                                      device=device,
            #                                      temperature=config.temperature,
            #                                      method=config.sample_method)
            #     print("SENTENCE:",dataset.convert_to_string(out_sentence))

    print('Done training.')
    # f.close()

    # plt.plot(losses, label = "Train Loss")
    # plt.legend()
    # plt.savefig("./plots/Loss_{}_{}.eps".format(config.sample_method, config.temperature))
    # plt.close()
    
    # plt.plot(accuracies, label = "Train Accuracy")
    # plt.legend()
    # plt.savefig("./plots/Accuracies_{}_{}.eps".format(config.sample_method, config.temperature))


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    # parser.add_argument('--txt_file', type=str, required=False, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Size of each embedding vector')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--n_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Sample parameters
    parser.add_argument('--sample_method', type=str, default="greedy", help='sampling method for character generation')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature for non-greedy sampling')
    parser.add_argument('--gen_sentence', type=str, default="Sleeping beauty is", help='sentence to start with generating')
    parser.add_argument('--chars_to_generate', type=int, default=30, help='len of new sentence')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=50, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    config = parser.parse_args()

    # Train the model
    train(config)
