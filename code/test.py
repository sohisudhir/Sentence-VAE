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

from PennTreeData import PennTreeData
from RNNLM import RNNLM
from utils import *


MODEL_PATH = "../checkpoints/checkpoint_7.pt"

TRAIN_TEXT_FILE = "../data/02-21.10way.clean"
TEST_PATH = "../data/23.auto.clean"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(config):

    train_dataset = PennTreeData(TRAIN_TEXT_FILE)
    train_data_loader = DataLoader(train_dataset, config.batch_size, num_workers=1)
    test_dataset = PennTreeData(TEST_PATH)
    test_data_loader = DataLoader(test_dataset, 64, num_workers=1)

    model = RNNLM(seq_length=train_dataset.max_sentence_len,
                embedding_dim=config.embedding_dim,
                vocabulary_size=train_dataset.vocab_size,
                lstm_num_hidden=config.lstm_num_hidden,
                lstm_num_layers=config.lstm_num_layers,
                device=device)

    model.load_state_dict(torch.load(MODEL_PATH,  map_location='cpu'))
    model.eval()

    padding_index = test_dataset.word2idx[test_dataset.pad]
    criterion = nn.CrossEntropyLoss(ignore_index=padding_index, reduction="sum")
    optimizer = optim.RMSprop(model.parameters())

    test_accuracy, test_negative_ll, test_perplexity = run_epoch(
        model, test_data_loader, criterion, optimizer, device)

    return test_accuracy, test_negative_ll, test_perplexity

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

    # Evaluate the model
    print(test_model(config))
