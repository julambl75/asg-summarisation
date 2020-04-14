import torch

from lang import PATH
from model_rnn import EncoderRNN, AttnDecoderRNN
from utils import DEVICE
from train import LANG, train_iters
from eval import evaluate_randomly

if __name__ == '__main__':
    hidden_size = 256
    encoder1 = EncoderRNN(LANG.n_words, hidden_size).to(DEVICE)
    attn_decoder1 = AttnDecoderRNN(hidden_size, LANG.n_words, dropout_p=0.1).to(DEVICE)

    # train_iters(encoder1, attn_decoder1, 75000, print_every=500)
    #
    # torch.save(encoder1.state_dict(), f'{PATH}/models/encoder.pt')
    # torch.save(attn_decoder1.state_dict(), f'{PATH}/models/decoder.pt')
    #
    # evaluate_randomly(encoder1, attn_decoder1, 100)
