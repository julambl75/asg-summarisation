import math
import time

import matplotlib.pyplot as plt
import torch

import matplotlib.ticker as ticker

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tensor_from_sequence(lang, sentences, max_seq_length):
    indexes = lang.sequence_to_ids(sentences)
    indexes += [lang.seq_pad_id] * (max_seq_length - len(indexes))
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


def tensors_from_pair(lang, pair, max_seq_length):
    input_tensor = tensor_from_sequence(lang, pair[0], max_seq_length)
    target_tensor = tensor_from_sequence(lang, pair[1], max_seq_length)
    return input_tensor, target_tensor


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()
