from __future__ import unicode_literals, print_function, division

import random
import time

import torch
import torch.nn as nn
from torch import optim

from utils import DEVICE, tensors_from_pair, time_since, show_plot

TEACHER_FORCING_RATIO = 0.5


class Trainer:
    def __init__(self, lang, encoder, decoder, pairs, seq_length):
        self.lang = lang
        self.encoder = encoder
        self.decoder = decoder
        self.pairs = pairs
        self.seq_length = seq_length

    def train(self, input_tensor, target_tensor, encoder_optimizer, decoder_optimizer, criterion):
        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs = self.encoder.init_outputs(self.seq_length)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss = 0

        for ei in range(self.seq_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[self.lang.seq_start_id]], device=DEVICE)
        decoder_hidden = encoder_hidden

        if self.encoder.bidirectional:
            decoder_hidden = (torch.sum(decoder_hidden[0], dim=0).unsqueeze(0),
                              torch.sum(decoder_hidden[0], dim=0).unsqueeze(0))

        use_teacher_forcing = random.random() < TEACHER_FORCING_RATIO

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(self.seq_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden,
                                                                                 encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(self.seq_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden,
                                                                                 encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == self.lang.seq_end_id:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / self.seq_length

    def train_iters(self, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        print(f'Starting training...')
        training_pairs = [tensors_from_pair(self.lang, random.choice(self.pairs), self.seq_length) for _ in
                          range(n_iters)]

        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            # TODO use BLEU score for loss
            loss = self.train(input_tensor, target_tensor, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            # TODO early stopping
            # if print_loss_total / print_every < EARLY_STOP_LOSS:
            #     break

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (
                    time_since(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        show_plot(plot_losses)
