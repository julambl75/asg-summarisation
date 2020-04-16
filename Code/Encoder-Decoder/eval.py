import random

import torch

from lang import SEQ_END_TOKEN
from utils import DEVICE, tensor_from_sequence


class Evaluator:
    def __init__(self, lang, encoder, decoder, pairs, seq_length):
        self.lang = lang
        self.encoder = encoder
        self.decoder = decoder
        self.pairs = pairs
        self.seq_length = seq_length

    def evaluate(self, sentence):
        with torch.no_grad():
            input_tensor = tensor_from_sequence(self.lang, sentence, self.seq_length)
            encoder_hidden = self.encoder.init_hidden()

            encoder_outputs = torch.zeros(self.seq_length, self.encoder.hidden_size, device=DEVICE)

            for ei in range(self.seq_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[self.lang.seq_start_id]], device=DEVICE)

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(self.seq_length, self.seq_length)

            for di in range(self.seq_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == self.lang.seq_end_id:
                    decoded_words.append(SEQ_END_TOKEN)
                    break
                else:
                    bert_id = self.lang.emb2bert[topi.item()]
                    decoded_words.append(self.lang.bert_id_to_sequence(bert_id))

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
