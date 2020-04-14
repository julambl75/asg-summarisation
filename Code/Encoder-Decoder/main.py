import torch
from pytorch_pretrained_bert import BertTokenizer

from lang import PATH, Lang
from model_rnn import EncoderRNN, AttnDecoderRNN
from utils import DEVICE
from train import Trainer
from eval import Evaluator

EMBEDDINGS_SIZE = 768
HIDDEN_SIZE = 128

if __name__ == '__main__':
    lang = Lang()

    encoder = EncoderRNN(EMBEDDINGS_SIZE, HIDDEN_SIZE).to(DEVICE)
    attn_decoder = AttnDecoderRNN(HIDDEN_SIZE, EMBEDDINGS_SIZE, dropout=0.1).to(DEVICE)

    trainer = Trainer(lang, encoder, attn_decoder)
    trainer.train_iters(10000, print_every=500)

    torch.save(encoder.state_dict(), f'{PATH}/models/encoder.pt')
    torch.save(attn_decoder.state_dict(), f'{PATH}/models/decoder.pt')

    evaluator = Evaluator(lang, encoder, attn_decoder)
    evaluator.evaluate_randomly(100)
