import random
import sys
from os import path

import torch

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from helper import Helper
from lang import SEQ_PAD_TOKEN, SEQ_END_TOKEN, SEQ_START_TOKEN
from score_summary import SummaryScorer
from utils import DEVICE, tensor_from_sequence


class Evaluator:
    def __init__(self, lang, encoder, decoder, pairs, seq_length):
        self.lang = lang
        self.encoder = encoder
        self.decoder = decoder
        self.pairs = pairs
        self.seq_length = seq_length

        self.helper = Helper()

    def evaluate(self, sentence):
        with torch.no_grad():
            input_tensor = tensor_from_sequence(self.lang, sentence, self.seq_length)

            encoder_hidden = self.encoder.init_hidden()
            encoder_outputs = self.encoder.init_outputs(self.seq_length)

            for ei in range(self.seq_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[self.lang.seq_start_id]], device=DEVICE)
            decoder_hidden = encoder_hidden

            if self.encoder.bidirectional:
                decoder_hidden = (torch.sum(decoder_hidden[0], dim=0).unsqueeze(0),
                                  torch.sum(decoder_hidden[0], dim=0).unsqueeze(0))

            decoded_bert_ids = []
            decoder_attentions = torch.zeros(self.seq_length, self.seq_length)

            for di in range(self.seq_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden,
                                                                                 encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)

                bert_id = self.lang.emb2bert[topi.item()]
                decoded_bert_ids.append(bert_id)

                if topi.item() == self.lang.seq_end_id:
                    break
                decoder_input = topi.squeeze().detach()

            decoded_words = self.lang.bert_ids_to_sequence(decoded_bert_ids)

            return decoded_words, decoder_attentions[:di + 1]

    def evaluate_randomly(self, n=10, score_summary=False):
        bleu_scores_predicted = 0
        bleu_scores_expected = 0

        for i in range(n):
            pair = random.choice(self.pairs)
            pair = tuple(map(remove_seq_tokens, pair))

            predicted, attentions = self.evaluate(pair[0])
            predicted_sentence = remove_seq_tokens(predicted)

            bleu_score_predicted = self.helper.bleu_score(pair[0], predicted_sentence)
            bleu_score_expected = self.helper.bleu_score(*pair)
            bleu_scores_predicted += bleu_score_predicted
            bleu_scores_expected += bleu_score_expected

            print('>', pair[0])
            print('=', pair[1])
            print('<', predicted_sentence)
            print('| BLEU score (predicted): ', bleu_score_predicted)
            print('| BLEU score (expected): ', bleu_score_expected)

            if score_summary:
                summary_scorer_predicted = SummaryScorer(pair[0], predicted_sentence)
                summary_scorer_expected = SummaryScorer(*pair)
                print('| Summary score (predicted): ', summary_scorer_predicted.score())
                print('| Summary score (expected): ', summary_scorer_expected.score())
            print('')

        print('BLEU score average (predicted): ', bleu_scores_predicted / n)
        print('BLEU score average (expected): ', bleu_scores_expected / n)


def remove_seq_tokens(sequence):
    for seq_token in [SEQ_PAD_TOKEN, SEQ_START_TOKEN, SEQ_END_TOKEN]:
        sequence = sequence.replace(seq_token, '')
    return sequence.replace(' .', '.').replace('  ', ' ').strip()
