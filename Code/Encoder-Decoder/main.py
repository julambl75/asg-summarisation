from eval import evaluate_randomly, evaluate
from rnn_model import EncoderRNN, AttnDecoderRNN
from rnn_utils import DEVICE, show_attention
from train import INPUT_LANG, OUTPUT_LANG, train_iters


def evaluate_and_show_attention(input_sentence):
    output_words, attentions = evaluate(encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)


if __name__ == '__main__':
    hidden_size = 256
    encoder1 = EncoderRNN(INPUT_LANG.n_words, hidden_size).to(DEVICE)
    attn_decoder1 = AttnDecoderRNN(hidden_size, OUTPUT_LANG.n_words, dropout_p=0.1).to(DEVICE)

    train_iters(encoder1, attn_decoder1, 75000, print_every=5000)

    evaluate_randomly(encoder1, attn_decoder1)

    # eval_stories = read_data(INPUT, EVAL)
    # evaluate_and_show_attention(eval_stories[0])
