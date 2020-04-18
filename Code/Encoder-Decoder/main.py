import torch
from distutils.dir_util import mkpath

from eval import Evaluator
from lang import PATH, Lang, TRAIN, TEST
from model_rnn import EncoderRNN, AttnDecoderRNN
from train import Trainer
from utils import DEVICE

HIDDEN_SIZE = 128

if __name__ == '__main__':
    lang = Lang()

    train_pairs, train_seq_length = lang.prepare_data(TRAIN)
    test_pairs, test_seq_length = lang.prepare_data(TEST)
    seq_length = max(train_seq_length, test_seq_length)

    lang.prepare_embeddings()

    encoder = EncoderRNN(lang.n_words, HIDDEN_SIZE, bidirectional=True).to(DEVICE)
    attn_decoder = AttnDecoderRNN(lang.n_words, HIDDEN_SIZE, seq_length, dropout_p=0.1).to(DEVICE)

    trainer = Trainer(lang, encoder, attn_decoder, train_pairs, seq_length)
    trainer.train_iters(30000, print_every=500)

    mkpath(f'{PATH}/models/')
    torch.save(encoder.state_dict(), f'{PATH}/models/encoder.pt')
    torch.save(attn_decoder.state_dict(), f'{PATH}/models/decoder.pt')

    evaluator = Evaluator(lang, encoder, attn_decoder, test_pairs, seq_length)
    evaluator.evaluate_randomly(20)#, score_summary=True)
