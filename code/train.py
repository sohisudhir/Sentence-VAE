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
from utils import calculate_accuracy, generate_samples, run_epoch

from PennTreeData import PennTreeData
from RNNLM import RNNLM


TRAIN_TEXT_FILE = "../data/02-21.10way.clean"
EVAL_TEXT_FILE = "../data/22.auto.clean"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train(config):
    # Initialize the dataset and data loader
    train_dataset = PennTreeData(TRAIN_TEXT_FILE)
    train_data_loader = DataLoader(train_dataset, config.batch_size, num_workers=1)
    eval_dataset = PennTreeData(EVAL_TEXT_FILE)
    eval_data_loader = DataLoader(eval_dataset, config.batch_size, num_workers=1)

    # Initialize the model
    model = RNNLM(seq_length=train_dataset.max_sentence_len,
                embedding_dim=config.embedding_dim,
                vocabulary_size=train_dataset.vocab_size,
                lstm_num_hidden=config.lstm_num_hidden,
                lstm_num_layers=config.lstm_num_layers,
                device=device)
    model.to(device)

    # Setup output file
    os.makedirs("../sample_results/", exist_ok=True)
    f = open("../sample_results/benchmark_samples.txt", "w+")

    # Setup the loss and optimizer
    padding_index = train_dataset.word2idx[train_dataset.pad]
    criterion = nn.CrossEntropyLoss(ignore_index=padding_index, reduction="sum")
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    train_nnl = []
    train_accuracies = []        
    eval_nnl = []
    eval_accuracies = []
    for epoch in range(config.n_epochs):
        model.train()
        train_accuracy, train_negative_ll, train_perplexity = run_epoch(model,
                                                                        train_data_loader,
                                                                        criterion,
                                                                        optimizer,
                                                                        device)
        model.eval()
        eval_accuracy, eval_negative_ll, eval_perplexity = run_epoch(model,
                                                                     eval_data_loader,
                                                                     criterion,
                                                                     optimizer,
                                                                     device)

        train_accuracies.append(train_accuracy)
        train_nnl.append(train_negative_ll)
        eval_accuracies.append(eval_accuracy)
        eval_nnl.append(eval_negative_ll)

        # generated_sample = generate_samples(model,
        #                                     dataset=train_dataset,
        #                                     device=device,
        #                                     config.seq_length,
        #                                     temperature=config.temperature,
        #                                     method=config.sample_method)
        # print("SAMPLE:", train_dataset.convert_to_string(generated_sample))
        print("EPOCH:{}".format(epoch)) 
        print("| Train_accuracy:{:.3f} | Train_nnl:{:.3f} | Train_ppl:{:.3f}".format(train_accuracy, train_negative_ll, train_perplexity))
        print("| Eval_accuracy:{:.3f} | Eval_nnl:{:.3f} | Eval_ppl:{:.3f}".format(eval_accuracy, eval_negative_ll, eval_perplexity))


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--embedding_dim', type=int, default=32, help='Size of each embedding vector')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    parser.add_argument('--n_epochs', type=int, default=20, help='Number of training epochs')

    # Sample parameters
    parser.add_argument('--seq_length', type=int, default=30, help='Length of a sample sentence')
    parser.add_argument('--sample_method', type=str, default="greedy", help='sampling method for character generation')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature for non-greedy sampling')
    parser.add_argument('--gen_sentence', type=str, default="Sleeping beauty is", help='sentence to start with generating')
    parser.add_argument('--chars_to_generate', type=int, default=30, help='len of new sentence')

    config = parser.parse_args()

    # Train the model
    train(config)
