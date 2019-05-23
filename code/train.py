import os
import time
from datetime import datetime
import argparse

import numpy as np
from scipy.misc import logsumexp
from matplotlib import pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PennTreeData import PennTreeData
from RNNLM import RNNLM


TRAIN_TEXT_FILE = "../data/02-21.10way.clean"
EVAL_TEXT_FILE = "../data/22.auto.clean"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

################################################################################

def dataset_metrics(dataloader, model):
    n_sentences = 0
    total_negative_ll = 0
    accuracy = 0.0
    with torch.no_grad():
        for step, (batch, sentence_len) in enumerate(dataloader):
        
            batch_inputs, batch_targets = batch
            batch_inputs = torch.stack(batch_inputs).to(device)
            flat_targets = torch.stack(batch_targets).to(device).view(-1)
            batch_targets = torch.stack(batch_targets).to(device)

            output, _ = model(batch_inputs)
            negative_ll, batch_size = calculate_perplexity(output , batch_targets, sentence_len)
            output = output.view(batch_inputs.size()[0]*batch_inputs.size()[1], -1)
            accuracy += calculate_accuracy(output, flat_targets)
    
            total_negative_ll += negative_ll
            n_sentences += batch_size

    total_perplexity = np.exp(total_negative_ll/n_sentences)

    return total_perplexity, total_negative_ll, accuracy/(step + 1)


def calculate_accuracy(predictions, targets):
    accurates = torch.argmax(predictions, dim=1) == targets
    accuracy = accurates.type(torch.FloatTensor).mean().item()
    return accuracy


def calculate_perplexity(predictions, targets, sentences_len):
    predictions = F.softmax(predictions, dim=2)    
    likelihoods = torch.gather(predictions, 2, targets.unsqueeze(dim=2).type(torch.LongTensor).to(device)).squeeze(dim=2)
    batch_size = predictions.size()[1]
    
    total_likelihood = 0
    for idx in range(batch_size): 
        sentence = likelihoods[:, idx]
        sentence_len = sentences_len[idx]
        sentence = sentence[:sentence_len]
        sentence_likelihood = logsumexp(sentence.detach().cpu().numpy())
        total_likelihood += sentence_likelihood

    return -np.log(total_likelihood), batch_size
        

def temperature_sample(h, temperature):
    h = h.squeeze()
    distribution = F.softmax(h/temperature, dim=0)    
    return torch.multinomial(distribution,1).item()

def generate_samples(model, dataset, device, temperature = 0.5, method = "greedy"):
    x = torch.tensor(dataset.word2idx[dataset.SOS],
                  dtype=torch.long,
                  device=device).view(-1,1)
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

    # Initialize the dataset and data loader (note the +1)
    train_dataset = PennTreeData(TRAIN_TEXT_FILE)
    train_data_loader = DataLoader(train_dataset, config.batch_size, num_workers=1)
    eval_dataset = PennTreeData(EVAL_TEXT_FILE)
    eval_data_loader = DataLoader(eval_dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = RNNLM(seq_length=train_dataset.max_sentence_len,
                  embedding_dim=config.embedding_dim,
                  vocabulary_size=train_dataset.vocab_size,
                  lstm_num_hidden=config.lstm_num_hidden,
                  lstm_num_layers=config.lstm_num_layers,
                  device=device)
    model.to(device)

    #setup output_file
    # f = open("../sample_results/samples_{}_{}.txt".format(config.sample_method, config.temperature),"w+")

    # Setup the loss and optimizer
    padding_index = train_dataset.word2idx[train_dataset.pad]
    criterion = nn.CrossEntropyLoss(ignore_index = padding_index)
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    perplexity = 0
    losses = []
    accuracies = []
    for epoch in range(config.n_epochs):
        for step, (batch, sentence_len) in enumerate(train_data_loader):
            # Only for time measurement of step through network
            t1 = time.time()
            batch_inputs, batch_targets = batch
            batch_inputs = torch.stack(batch_inputs).to(device)
            flat_targets = torch.stack(batch_targets).to(device).view(-1)

            output, _ = model(batch_inputs)
            output = output.view(batch_inputs.size()[0]*batch_inputs.size()[1], -1)
            accuracy = calculate_accuracy(output, flat_targets)
            
            loss = criterion(output, flat_targets)
            perplexity += np.exp(loss.item())/ torch.sum(sentence_len).item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #total_perplexity, total_negative_ll, accuracy = dataset_metrics(eval_data_loader, model)
            #print("total_perplexity:{}, total_negative_ll:{}, accuracy:{}".format(total_perplexity, total_negative_ll, accuracy))

            # For plotting
            # accuracies.append(accuracy)
            # losses.append(loss.item())
            
            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)
            if step % config.print_every == 0:

                print("[{}] Epoch {} Train Step {:00f}, Batch Size = {}, Examples/Sec = {:.2f},"
                    "Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), epoch, step,
                        config.batch_size, examples_per_second,
                        loss
                ))

            if step % config.sample_every == 0:                
                generated_sample = generate_samples(model,
                                                    dataset=train_dataset,
                                                    device=device,
                                                    temperature=config.temperature,
                                                    method=config.sample_method)
                print("SAMPLE:", train_dataset.convert_to_string(generated_sample))
        print(perplexity)
        # total_perplexity, total_negative_ll, accuracy = dataset_metrics(eval_data_loader, model)
        # print("total_perplexity:{}, total_negative_ll:{}, accuracy:{}".format(
        #     total_perplexity, total_negative_ll, accuracy))

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
