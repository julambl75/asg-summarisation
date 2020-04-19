import torch
from distutils.dir_util import mkpath

from eval import Evaluator
from lang import PATH, Lang, TRAIN, TEST
from model_rnn import EncoderRNN, AttnDecoderRNN
from train import Trainer
from utils import DEVICE

LEARNING_RATE = 0.01
HIDDEN_SIZE = 256
BIDIRECTIONAL = True

if __name__ == '__main__':
    lang = Lang()

    train_pairs, train_seq_length = lang.prepare_data(TRAIN)
    test_pairs, test_seq_length = lang.prepare_data(TEST)
    seq_length = max(train_seq_length, test_seq_length)

    lang.prepare_embeddings()

    encoder = EncoderRNN(lang.n_words, HIDDEN_SIZE, bidirectional=BIDIRECTIONAL).to(DEVICE)
    attn_decoder = AttnDecoderRNN(lang.n_words, HIDDEN_SIZE, seq_length, dropout_p=0.1, bidirectional_encoder=BIDIRECTIONAL).to(DEVICE)

    trainer = Trainer(lang, encoder, attn_decoder, train_pairs, seq_length)
    trainer.train_iters(100000, print_every=500, learning_rate=LEARNING_RATE)

    mkpath(f'{PATH}/models/')
    torch.save(encoder.state_dict(), f'{PATH}/models/encoder.pt')
    torch.save(attn_decoder.state_dict(), f'{PATH}/models/decoder.pt')

    evaluator = Evaluator(lang, encoder, attn_decoder, test_pairs, seq_length)
    evaluator.evaluate_randomly(20)#, score_summary=True)
