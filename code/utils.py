import os
import time
from datetime import datetime
import argparse

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def calculate_accuracy(predictions, targets):
    accurates = torch.argmax(predictions, dim=1) == targets
    accuracy = accurates.type(torch.FloatTensor).mean().item()
    return accuracy


def temperature_sample(h, temperature):
    h = h.squeeze()
    distribution = F.softmax(h/temperature, dim=0)    
    return torch.multinomial(distribution,1).item()


def generate_samples(model, dataset, device, seq_length, temperature = 0.5, method = "greedy"):
    x = torch.tensor(dataset.word2idx[dataset.SOS],
                  dtype=torch.long,
                  device=device).view(-1,1)
    hidden = None
    with torch.no_grad():
        generated_sentence = [x.item()]
        for seq in range(seq_length):
            out,hidden  = model(x, hidden)
            if method == "greedy":
                sample_idx = torch.argmax(out, dim=2).item()
            else:
                sample_idx = temperature_sample(out, temperature)
            x = torch.tensor(sample_idx, dtype=torch.long, device=device).view(1,1)
            generated_sentence.append(sample_idx)
    
    return generated_sentence


def generate_sentence(model, sentence, device, config):
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


def write_samples_to_file(f, model, dataset, device, seq_length):
    greedy_sample = generate_samples(model, dataset, device, seq_length, temperature = 0.5, method = "greedy")
    f.write("SAMPLE Greedy:{} \n ".format(dataset.convert_to_string(greedy_sample)))
    for sample in range(10):
        temp_1 = generate_samples(model, dataset, device, seq_length, temperature = 0.5, method = "temp")
        temp_2 = generate_samples(model, dataset, device, seq_length, temperature = 1, method = "temp")
        temp_3 = generate_samples(model, dataset, device, seq_length, temperature = 2, method = "temp")
        f.write("SAMPLE Temp 0.5:{} \n ".format(dataset.convert_to_string(temp_1)))
        f.write("SAMPLE Temp 1:{} \n ".format(dataset.convert_to_string(temp_2)))
        f.write("SAMPLE Temp 2:{} \n ".format(dataset.convert_to_string(temp_3))) 
    f.write("\n")


def make_plot(train, eval, name):
    plt.plot(train, label = "Train {}".format(name))
    plt.plot(eval, label = "Eval {}".format(name))
    plt.legend()
    plt.savefig("../plots/BenchMark_{}.eps".format(name))
    plt.close()


def run_epoch(model, data_loader, criterion, optimizer, device):
    negative_lls = []
    accuracies = []
    negative_ll = 0
    denominator = 0
    
    for step, (batch, sentence_len) in enumerate(data_loader):
    
        batch_inputs, batch_targets = batch
        batch_inputs = torch.stack(batch_inputs).to(device)
        flat_targets = torch.stack(batch_targets).to(device).view(-1)

        output, _ = model(batch_inputs)
        output = output.view(batch_inputs.size()[0]*batch_inputs.size()[1], -1)
        
        accuracy = calculate_accuracy(output, flat_targets)
        loss = criterion(output, flat_targets)
        
        negative_ll += loss.item()
        denominator += torch.sum(sentence_len).item()
        
        if model.training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()

        accuracies.append(accuracy)
        negative_lls.append(negative_ll)

    perplexity = np.exp(negative_ll / denominator)

    return np.mean(accuracies), np.mean(negative_lls), perplexity