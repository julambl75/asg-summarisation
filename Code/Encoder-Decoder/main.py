from distutils.dir_util import mkpath

import torch

from eval import Evaluator
from lang import PATH, Lang, TRAIN, TEST
from model_rnn import EncoderDecoder
from train import Trainer

LEARNING_RATE = 0.01
EMBEDDING_SIZE = 64
HIDDEN_SIZE = 128

if __name__ == '__main__':
    lang = Lang()

    train_pairs, train_seq_length = lang.prepare_data(TRAIN)
    test_pairs, test_seq_length = lang.prepare_data(TEST)  # TODO test tokens not necessarily in train
    seq_length = max(train_seq_length, test_seq_length)

    # TODO David had 50,000 vocabulary size, 400,000 examples (300 tokens each), 40 iterations, 500 hours to train
    lang.prepare_embeddings()

    encoder, decoder = EncoderDecoder.create(lang.n_words, EMBEDDING_SIZE, HIDDEN_SIZE, seq_length)

    trainer = Trainer(lang, encoder, decoder, train_pairs, seq_length)
    trainer.train_iters(100000, print_every=500, learning_rate=LEARNING_RATE)

    mkpath(f'{PATH}/models/')
    torch.save(encoder.state_dict(), f'{PATH}/models/encoder.pt')
    torch.save(decoder.state_dict(), f'{PATH}/models/decoder.pt')

    evaluator = Evaluator(lang, encoder, decoder, test_pairs, seq_length)
    evaluator.evaluate_randomly(20)#, score_summary=True)
