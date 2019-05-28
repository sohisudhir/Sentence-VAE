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
from utils import *

from PennTreeData import PennTreeData
from RNNLM import RNNLM


TRAIN_TEXT_FILE = "../data/02-21.10way.clean"
EVAL_TEXT_FILE = "../data/22.auto.clean"
TEST_TEXT_FILE = "../data/23.auto.clean"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train(config):
    # Initialize the dataset and data loader
    train_dataset = PennTreeData(TRAIN_TEXT_FILE)
    train_data_loader = DataLoader(train_dataset, config.batch_size, num_workers=1)
    eval_dataset = PennTreeData(EVAL_TEXT_FILE)
    eval_data_loader = DataLoader(eval_dataset, config.batch_size, num_workers=1)
    test_dataset = PennTreeData(EVAL_TEXT_FILE)
    test_data_loader = DataLoader(test_dataset, config.batch_size, num_workers=1)

    # Initialize the model
    model = RNNLM(seq_length=train_dataset.max_sentence_len,
                embedding_dim=config.embedding_dim,
                vocabulary_size=train_dataset.vocab_size,
                lstm_num_hidden=config.lstm_num_hidden,
                lstm_num_layers=config.lstm_num_layers,
                device=device)
    model.to(device)

    # Setup output folders
    os.makedirs("../sample_results/", exist_ok=True)
    os.makedirs("../checkpoints/", exist_ok=True)
    os.makedirs("../plots/", exist_ok=True)
    
    # Setup output file
    f = open("../sample_results/benchmark_samples_3.txt", "w+")

    # Setup the loss and optimizer
    padding_index = train_dataset.word2idx[train_dataset.pad]
    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    train_nnl = []
    train_accuracies = []
    train_perplexities = []        
    eval_nnl = []
    eval_accuracies = []
    eval_perplexities = []
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
        train_perplexities.append(train_perplexity)
        eval_accuracies.append(eval_accuracy)
        eval_nnl.append(eval_negative_ll)
        eval_perplexities.append(eval_perplexity)

        # Write to file
        print("EPOCH:{}".format(epoch)) 
        print("| Train_accuracy:{:.3f} | Train_nnl:{:.3f} | Train_ppl:{:.3f}".format(train_accuracy, train_negative_ll, train_perplexity))
        print("| Eval_accuracy:{:.3f} | Eval_nnl:{:.3f} | Eval_ppl:{:.3f}".format(eval_accuracy, eval_negative_ll, eval_perplexity))
        f.write("\n")
        f.write("EPOCH:{} \n".format(epoch))
        f.write("| Train_accuracy:{:.3f} | Train_nnl:{:.3f} | Train_ppl:{:.3f} \n".format(train_accuracy, train_negative_ll, train_perplexity))
        f.write("| Eval_accuracy:{:.3f} | Eval_nnl:{:.3f} | Eval_ppl:{:.3f} \n".format(eval_accuracy, eval_negative_ll, eval_perplexity))
        write_samples_to_file(f, model, train_dataset, device, config.seq_length)

        # Save Model
        torch.save(model.state_dict(), "../checkpoints/checkpoint_3_{}.pt".format(epoch))


    print("Done Training") 
    
    # Test
    test_accuracy, test_negative_ll, test_perplexity = run_epoch(model,
                                                                 test_data_loader,
                                                                 criterion,
                                                                 optimizer,
                                                                 device)
    f.write("| Test_accuracy:{:.3f} | Test_nnl:{:.3f} | Test_ppl:{:.3f} \n".format(test_accuracy, test_negative_ll, test_perplexity))
    f.close()

    # Plot
    make_plot(train_accuracies, eval_accuracies, "Accuracy")
    make_plot(train_nnl, eval_nnl, "NLL")
    make_plot(train_perplexities, eval_perplexities, "Perplexity")
    


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
    parser.add_argument('--seq_length', type=int, default=50, help='Length of a sample sentence')
    parser.add_argument('--sample_method', type=str, default="greedy", help='sampling method for character generation')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature for non-greedy sampling')
    parser.add_argument('--sentence', type=str, default="Sleeping beauty is", help='sentence to start with generating')
    parser.add_argument('--chars_to_generate', type=int, default=50, help='len of new sentence')

    config = parser.parse_args()

    # Train the model
    train(config)
