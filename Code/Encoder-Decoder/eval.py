import random

import torch

from lang import *
from utils import DEVICE, tensor_from_sequence


class Evaluator:
    def __init__(self, lang, encoder, decoder):
        self.lang = lang
        self.encoder = encoder
        self.decoder = decoder

        self.pairs, self.max_seq_length = prepare_data(TEST, self.lang)

    def evaluate(self, sentence, max_length=MAX_LENGTH):
        with torch.no_grad():
            input_tensor = tensor_from_sequence(self.lang, sentence, self.max_seq_length)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.init_hidden()

            encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size, device=DEVICE)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[self.lang.seq_start_id]], device=DEVICE)

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(max_length, max_length)

            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == self.lang.seq_end_id:
                    decoded_words.append(SEQ_END_TOKEN)
                    break
                else:
                    decoded_words.append(self.lang.id_to_sequence[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]

    def evaluate_randomly(self, n=10):
        for i in range(n):
            pair = random.choice(self.pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = self.evaluate(pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')
