from __future__ import print_function, division
import os
import urllib
import torch
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.utils
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


def get_model(modelname, size_of_vocab=None, embedding_dim=None, num_hidden_nodes=None, num_output_nodes=None,
              num_layers=None,
              bidirectional=None, dropout=None, pad_idx=None):
    if modelname == 'googlenet':
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load('pytorch/vision:v0.10.0', modelname, pretrained=True)
    elif modelname == 'Textclassifier':
        model = Textclassifier(vocab_size=size_of_vocab, embedding_dim=embedding_dim,
                               hidden_dim=num_hidden_nodes, output_dim=num_output_nodes, n_layers=num_layers,
                               bidirectional=bidirectional, dropout=dropout, pad_idx=pad_idx)
    return model


class Textclassifier(nn.Module):

    # define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx, freeze=False):

        # Constructor
        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)

        # dense layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # activation function
        self.act = nn.Sigmoid()
        self.freeze = freeze

    def forward(self, text, text_lengths):

        # text = [batch size,sent_length]
        if not self.freeze:
            embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]
        else:
            embedded = text
        # packed sequence

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=False)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [batch size, num layers * num directions,hid dim]
        # cell = [batch size, num layers * num directions,hid dim]

        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)

        # Final activation function
        outputs = self.act(dense_outputs)

        return outputs
