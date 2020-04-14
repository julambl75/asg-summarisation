import torch
from pytorch_pretrained_bert import BertTokenizer

from lang import PATH, Lang
from model_rnn import EncoderRNN, AttnDecoderRNN
from utils import DEVICE
from train import train_iters
from eval import evaluate_randomly

HIDDEN_SIZE = 128

if __name__ == '__main__':
    lang = Lang()

    encoder1 = EncoderRNN(lang.n_words, HIDDEN_SIZE).to(DEVICE)
    attn_decoder1 = AttnDecoderRNN(HIDDEN_SIZE, lang.n_words, dropout_p=0.1).to(DEVICE)

    train_iters(lang, encoder1, attn_decoder1, 10000, print_every=500)

    torch.save(encoder1.state_dict(), f'{PATH}/models/encoder.pt')
    torch.save(attn_decoder1.state_dict(), f'{PATH}/models/decoder.pt')

    evaluate_randomly(encoder1, attn_decoder1, 100)
