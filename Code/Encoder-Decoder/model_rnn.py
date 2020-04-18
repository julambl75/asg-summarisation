from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DEVICE


class EncoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(embedding_size, self.hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=bidirectional)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size, device=DEVICE),
                torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size, device=DEVICE))

    def init_outputs(self, seq_length):
        return torch.zeros(seq_length, self.hidden_size * (1 + int(self.bidirectional)), device=DEVICE)


class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, seq_length, dropout_p=0.1, num_layers=1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.num_layers = num_layers

        self.embedding = nn.Embedding(embedding_size, self.hidden_size)
        self.attn = nn.Linear(hidden_size * 2, seq_length)
        self.attn_combine = nn.Linear(hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers)
        self.out = nn.Linear(self.hidden_size, embedding_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
