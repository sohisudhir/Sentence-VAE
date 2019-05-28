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


def write_samples_to_file(f, model, dataset, device, seq_length):
    greedy_sample = generate_samples(model, dataset, device, seq_length, temperature = 0.5, method = "greedy")
    f.write("SAMPLE Greedy:{} \n ".format(dataset.convert_to_string(greedy_sample)))
    for sample in range(10):
        temp_01 = generate_samples(model, dataset, device, seq_length, temperature = 0.1, method = "temp")
        temp_015 = generate_samples(model, dataset, device, seq_length, temperature = 0.15, method = "temp")
        temp_002= generate_samples(model, dataset, device, seq_length, temperature = 0.2, method = "temp")
        temp_0025 = generate_samples(model, dataset, device, seq_length, temperature = 0.25, method = "temp")
        temp_0030 = generate_samples(model, dataset, device, seq_length, temperature = 0.30, method = "temp")
        temp_0035 = generate_samples(model, dataset, device, seq_length, temperature = 0.35, method = "temp")
        temp_05 = generate_samples(model, dataset, device, seq_length, temperature = 0.5, method = "temp")
        f.write("SAMPLE Temp 0.1:{} \n ".format(dataset.convert_to_string(temp_01)))
        f.write("SAMPLE Temp 0.15:{} \n ".format(dataset.convert_to_string(temp_015)))
        f.write("SAMPLE Temp 0.20:{} \n ".format(dataset.convert_to_string(temp_002)))
        f.write("SAMPLE Temp 0.25:{} \n ".format(dataset.convert_to_string(temp_0025)))
        f.write("SAMPLE Temp 0.30:{} \n ".format(dataset.convert_to_string(temp_0030)))
        f.write("SAMPLE Temp 0.35:{} \n ".format(dataset.convert_to_string(temp_0035)))
        f.write("SAMPLE Temp 0.5:{} \n ".format(dataset.convert_to_string(temp_05)))     
    f.write("\n")


def make_plot(train, eval, name):
    plt.plot(train, label = "Train {}".format(name))
    plt.plot(eval, label = "Eval {}".format(name))
    plt.legend()
    plt.savefig("../plots/BenchMark_3_{}.eps".format(name))
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