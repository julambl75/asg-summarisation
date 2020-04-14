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
    vocab_size = len(lang.tokenizer.vocab)

    encoder = EncoderRNN(vocab_size, HIDDEN_SIZE).to(DEVICE)
    attn_decoder = AttnDecoderRNN(HIDDEN_SIZE, vocab_size, dropout_p=0.1).to(DEVICE)

    trainer = Trainer(lang, encoder, attn_decoder)
    trainer.train_iters(10000, print_every=500)

    torch.save(encoder.state_dict(), f'{PATH}/models/encoder.pt')
    torch.save(attn_decoder.state_dict(), f'{PATH}/models/decoder.pt')

    evaluator = Evaluator(lang, encoder, attn_decoder)
    evaluator.evaluate_randomly(100)
