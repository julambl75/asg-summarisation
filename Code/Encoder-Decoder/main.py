import torch
from pytorch_pretrained_bert import BertTokenizer

from lang import PATH, Lang
from model_rnn import EncoderRNN, AttnDecoderRNN
from utils import DEVICE
from train import Trainer
from eval import Evaluator

HIDDEN_SIZE = 128

if __name__ == '__main__':
    lang = Lang()

    encoder = EncoderRNN(lang.n_words, HIDDEN_SIZE).to(DEVICE)
    attn_decoder = AttnDecoderRNN(HIDDEN_SIZE, lang.n_words, dropout_p=0.1).to(DEVICE)

    trainer = Trainer(lang, encoder, attn_decoder)
    trainer.train_iters(10000, print_every=500)

    torch.save(encoder.state_dict(), f'{PATH}/models/encoder.pt')
    torch.save(attn_decoder.state_dict(), f'{PATH}/models/decoder.pt')

    evaluator = Evaluator(lang, encoder, attn_decoder)
    evaluator.evaluate_randomly(encoder, attn_decoder, 100)
