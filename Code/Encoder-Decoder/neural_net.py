import math
import time
import subprocess

import numpy as np

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, pack_sequence

###############
# Torch setup #
###############
print('Torch version: {}, CUDA: {}'.format(torch.__version__, torch.version.cuda))
cuda_available = torch.cuda.is_available()
if not torch.cuda.is_available():
  print('WARNING: You may want to change the runtime to GPU for faster training!')
  DEVICE = 'cpu'
else:
  DEVICE = 'cuda:0'

#######################
# Some helper functions
#######################
def fix_seed(seed=None):
  """Sets the seeds of random number generators."""
  if seed is None:
    # Take a random seed
    seed = time.time()
  seed = int(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  return seed

def readable_size(n):
  """Returns a readable size string for model parameters count."""
  sizes = ['K', 'M', 'G']
  fmt = ''
  size = n
  for i, s in enumerate(sizes):
    nn = n / (1000 ** (i + 1))
    if nn >= 1:
      size = nn
      fmt = sizes[i]
    else:
      break
  return '%.2f%s' % (size, fmt)

class Vocabulary(object):
  """Data structure representing the vocabulary of a corpus."""
  def __init__(self):
    # Mapping from tokens to integers
    self._word2idx = {}

    # Reverse-mapping from integers to tokens
    self.idx2word = []

    # 0-padding token
    self.add_word('<pad>')
    # sentence start
    self.add_word('<s>')
    # sentence end
    self.add_word('</s>')
    # Unknown words
    self.add_word('<unk>')

    self._pad_idx = self._word2idx['<pad>']
    self._bos_idx = self._word2idx['<s>']
    self._eos_idx = self._word2idx['</s>']
    self._unk_idx = self._word2idx['<unk>']

  def word2idx(self, word):
    """Returns the integer ID of the word or <unk> if not found."""
    return self._word2idx.get(word, self._unk_idx)

  def add_word(self, word):
    """Adds the `word` into the vocabulary."""
    if word not in self._word2idx:
      self.idx2word.append(word)
      self._word2idx[word] = len(self.idx2word) - 1

  def build_from_file(self, fname):
    """Builds a vocabulary from a given corpus file."""
    with open(fname) as f:
      for line in f:
        words = line.strip().split()
        for word in words:
          self.add_word(word)

  def convert_idxs_to_words(self, idxs, until_eos=False):
    """Converts a list of indices to words."""
    if until_eos:
      try:
        idxs = idxs[:idxs.index(self.word2idx('</s>'))]
      except ValueError:
        pass

    return ' '.join(self.idx2word[idx] for idx in idxs)

  def convert_words_to_idxs(self, words, add_bos=False, add_eos=False):
    """Converts a list of words to a list of indices."""
    idxs = [self.word2idx(w) for w in words]
    if add_bos:
      idxs.insert(0, self.word2idx('<s>'))
    if add_eos:
      idxs.append(self.word2idx('</s>'))
    return idxs

  def __len__(self):
    """Returns the size of the vocabulary."""
    return len(self.idx2word)

  def __repr__(self):
    return "Vocabulary with {} items".format(self.__len__())

class Multi30K:
  """A dataset wrapper for Multi30K."""
  def __init__(self, src_lang='en', trg_lang='fr'):
    self.src_lang = src_lang
    self.trg_lang = trg_lang

    # Create vocabularies
    self.src_vocab = Vocabulary()
    self.src_vocab.build_from_file(f'train.{src_lang}')
    self.trg_vocab = Vocabulary()
    self.trg_vocab.build_from_file(f'train.{trg_lang}')

    self.n_src_vocab = len(self.src_vocab)
    self.n_trg_vocab = len(self.trg_vocab)

    self._data = {}
    for split in ('train', 'val', 'test'):
      # Read sentences and map to indices using the vocabularies
      print(f'Reading {split} set')
      self._data[split] = self.read_sentences(split)

  def read_sentences(self, split):
    src_sents = []
    trg_sents = []

    # Read source side
    with open(f'{split}.{self.src_lang}') as f:
      for line in f:
        line = line.strip()
        assert line, "Empty line found, please fix this!"
        idxs = self.src_vocab.convert_words_to_idxs(line.split(), add_eos=True)
        src_sents.append(idxs)
    # Read source side
    with open(f'{split}.{self.trg_lang}') as f:
      for line in f:
        line = line.strip()
        assert line, "Empty line found, please fix this!"
        idxs = self.trg_vocab.convert_words_to_idxs(line.split(), add_bos=True, add_eos=True)
        trg_sents.append(idxs)

    assert len(src_sents) == len(trg_sents), "Files are not aligned!"
    return src_sents, trg_sents

  def get_batch(self, idxs, split='train'):
    """Returns padded torch tensors for source and target sample indices."""
    src_idxs = [torch.LongTensor(self._data[split][0][i]) for i in idxs]
    trg_idxs = [torch.LongTensor(self._data[split][1][i]) for i in idxs]

    ###################################
    # Pad sequences to longest sequence
    ###################################
    padded_src_idxs = pad_sequence(src_idxs, padding_value=self.src_vocab._pad_idx)
    padded_trg_idxs = pad_sequence(trg_idxs, padding_value=self.trg_vocab._pad_idx)
    return padded_src_idxs.to(DEVICE), padded_trg_idxs.to(DEVICE)

  def __repr__(self):
    s = f"Multi30K {self.src_lang} (# {self.n_src_vocab}) -> {self.trg_lang} (# {self.n_trg_vocab})\n"
    s += f" train: {len(self._data['train'][0])} sentences\n"
    s += f"   val: {len(self._data['val'][0])} sentences\n"
    s += f"  test: {len(self._data['test'][0])} sentences"
    return s

dataset = Multi30K(src_lang='fr', trg_lang='en')
print(dataset)

# Get a batch and see inside
src_idxs, trg_idxs = dataset.get_batch([0, 10, 234, 12, 7], split='val')
print(src_idxs)
for i in range(src_idxs.shape[1]):
  print('SRC: ', dataset.src_vocab.convert_idxs_to_words(src_idxs[:, i]))
  print('TRG: ', dataset.trg_vocab.convert_idxs_to_words(trg_idxs[:, i]))
  print()

class EncDecNMT(nn.Module):
  """Encoder-decoder NMT without attention."""
  def __init__(self, dataset, emb_dim, enc_dim, dec_dim,
               enc_bidirectional=False,
               init_dec='max',
               dropout=0.3, clip_gradient_norm=1.0, tie_weights=True,
               batch_size=64):
    # Call parent's __init__ first
    super(EncDecNMT, self).__init__()

    # Store arguments
    self.dataset = dataset
    self.emb_dim = emb_dim
    self.enc_dim = enc_dim
    self.enc_bidirectional = enc_bidirectional
    self.dec_dim = dec_dim
    self.init_dec = init_dec
    self.clip_gradient_norm = clip_gradient_norm
    self.p_dropout = dropout
    self.tie_weights = tie_weights
    self.batch_size = batch_size

    assert self.init_dec in ('max', 'avg'), \
      "init_dec argument contains unknown value!"

    # Since target sequences are <pad>'ded, we want to ignore the loss
    # values for those positions
    self.loss = nn.CrossEntropyLoss(
        reduction='none', ignore_index=self.dataset.trg_vocab._pad_idx)

    # Create the dropout
    self.drop = nn.Dropout(p=self.p_dropout)

    ###############################################
    # Create the source and target embedding layers
    ###############################################
    self.src_emb = nn.Embedding(
      num_embeddings=self.dataset.n_src_vocab, embedding_dim=self.emb_dim,
      padding_idx=self.dataset.src_vocab._pad_idx)
    self.trg_emb = nn.Embedding(
      num_embeddings=self.dataset.n_trg_vocab, embedding_dim=self.emb_dim,
      padding_idx=self.dataset.trg_vocab._pad_idx)

    ###########################################
    # QUESTION
    ###########################################
    # Create the encoder by using the arguments
    ###########################################
    self.enc = nn.GRU(input_size=self.emb_dim, hidden_size=self.enc_dim)

    ##################################################################
    # QUESTION
    ##################################################################
    # Compute encoder's output dim which may be different than enc_dim
    # because of bidirectionality
    ##################################################################
    self.enc_out_dim = self.enc_dim
    if self.enc_bidirectional:
      self.enc_out_dim = self.emb_dim

    ###############################################################
    # Let's use the `GRUCell` for decoder, which is designed to run for
    # single timesteps through an explicit for loop. This will make
    # it easier to understand the concepts shown at the lecture.
    ###############################################################
    self.dec = nn.GRUCell(input_size=self.emb_dim, hidden_size=self.dec_dim)

    #############################################################
    # Add a non-linear layer for decoder conditioning projection
    #############################################################
    self.ff_dec_init = nn.Sequential(
        nn.Linear(self.enc_out_dim, self.dec_dim),
        nn.Tanh(),
    )

    ####################################################################
    # Bottleneck layer maps decoder's hidden state to emb size for tying
    ####################################################################
    self.bneck = nn.Linear(self.dec_dim, self.emb_dim)

    ####################
    # Final output layer
    ####################
    self.out = nn.Linear(self.emb_dim, self.dataset.n_trg_vocab)

    ############################################
    # Tie input/output embeddings of the decoder
    ############################################
    if self.tie_weights:
      self.out.weight = self.trg_emb.weight

    # Reset padding embeddings to all 0s
    with torch.no_grad():
        self.src_emb.weight.data[0].fill_(0)
        self.trg_emb.weight.data[0].fill_(0)

  def __repr__(self):
    """String representation for pretty-printing."""
    n_params = 0
    for param in self.parameters():
      n_params += np.cumprod(param.data.size())[-1]
    n_params = readable_size(n_params)

    s = super(EncDecNMT, self).__repr__()
    return f"{s}\n# of parameters: {n_params} -- Decoder init: '{self.init_dec}'"

  def get_batch_indices(self, split='train', shuffle=True):
    """Returns the list of sample indices for a whole epoch."""
    # Get number of samples
    n_samples = len(self.dataset._data[split][0])

    # Get sample indices and batch them
    if shuffle:
      order = torch.randperm(n_samples)
    else:
      order = torch.arange(n_samples)

    start_offsets = range(0, n_samples, self.batch_size)
    return [order[i: i + self.batch_size] for i in start_offsets]

  ###################################
  # Decoder initialisation approaches
  ###################################
  def get_encoder_max_state(self, enc_states, input_mask):
    """Computes h_0 for the decoder based on the max-pooled encoding state."""
    # we fill the padded position values with small numbers so that
    # max-pooling is not able to return wrong max values
    masked_states = enc_states.masked_fill(
        input_mask.unsqueeze(-1).bool().logical_not(), -100000)
    max_state = masked_states.max(0)[0]
    return max_state

  def get_encoder_avg_state(self, enc_states, input_mask):
    """Computes the average of encoder states taking care of padding."""
    ##############################################################
    # QUESTION
    ##############################################################
    # Compute the average encoder states by taking into account
    # sentence length information through `input_mask`
    ##############################################################
    positions_filled = input_mask.sum(1)
    weighed_states = positions_filled.view(-1, 1, 1) * enc_states
    mean_state = weighed_states.sum(0) / positions_filled.sum()
    # masked_states = enc_states.masked_fill(
    #     input_mask.unsqueeze(-1).bool().logical_not(), 0)
    # mean_state = masked_states.mean(0)
    return mean_state

  def compute_decoder_state(self, enc_states, input_mask):
    """Calls the appropriate `init_dec` method, projects the `v`."""
    func = getattr(self, f'get_encoder_{self.init_dec}_state')
    # Get the vector `v`
    h_0 = func(enc_states, input_mask)
    # Project it with the FF
    return self.ff_dec_init(h_0)

  ####################################
  # Encodes a batch of input sentences
  ####################################
  def encode(self, x):
    """Encode tokens `x` to obtain the encoder states."""
    # Compute the mask to detect <pad>'s in the further parts
    # x is a padded tensor of shape (seq_len, batch_size)
    # mask: (seq_len, batch_size)
    self.mask = x.ne(self.dataset.src_vocab._pad_idx).long()

    # src_embs: (seq_len, batch_size, emb_dim)
    embs = self.drop(self.src_emb(x))

    # Pack the tensor so that RNN correctly computes the hidden
    # states by ignoring padded positions
    packed_inputs = pack_padded_sequence(
        embs, lengths=self.mask.sum(0).long(), enforce_sorted=False)

    # Encode -> unpack to obtain an ordinary tensor of hidden states
    # padded positions will now have explicit 0's in their hidden states
    # all_hids: (seq_len, batch_size, self.enc_out_dim)
    self.all_hids = pad_packed_sequence(self.enc(packed_inputs)[0])[0]
    return self.all_hids, self.mask

  def compute_loss_from_logits(self, logits, y):
    """Returns the scalar losses for every element in the batch."""
    return self.loss(logits, y)

  def compute_decoder_logits(self, dec_hid_state, y):
    # *Cell() functions return (batch_size, hidden_size) i.e.
    # a tensor containing the computed hidden state for each element in the batch
    dec_hid_state = self.dec(self.trg_emb(y), dec_hid_state)
    logits = self.out(self.bneck(self.drop(dec_hid_state)))
    return logits, dec_hid_state

  def forward(self, x, y):
    """Forward-pass of the model for training/ppl evaluation only."""
    # Encode source sentence and get encoder states
    enc_states, input_mask = self.encode(x)

    # Compute decoder's initial state h_0 for each sentence
    dec_hid_state = self.compute_decoder_state(enc_states, input_mask)

    # Manually compute the loss for each decoding timestep
    # We skip the last item since we don't want to input </s>
    loss = 0.0
    n_trg_tokens = 0
    for t in range(y.size(0) - 1):
      y_prev, y_next = y[t], y[t + 1]

      # Do a recurrence step
      logits, dec_hid_state = self.compute_decoder_logits(dec_hid_state, y_prev)

      # Accumulate the sequence loss
      loss += self.compute_loss_from_logits(logits, y_next).sum()

      # Compute number of valid positions by ignoring <pad>'s
      n_trg_tokens += y_next.ne(self.dataset.trg_vocab._pad_idx).sum()

    return loss, float(n_trg_tokens.item())

  def train_model(self, optim, n_epochs=5):
    """Trains the model."""
    train_ppls, val_ppls = [], []

    for eidx in range(1, n_epochs + 1):
      start_time = time.time()
      epoch_loss = 0
      epoch_items = 0

      # Enable training mode
      self.train()

      # Start training (will shuffle at each epoch)
      for iter_count, idxs in enumerate(self.get_batch_indices('train')):
        # Get x's and y's
        x, y = self.dataset.get_batch(idxs)

        # Clear the gradients
        optim.zero_grad()

        total_loss, n_items = self.forward(x, y)

        # Backprop the average loss and update parameters
        total_loss.div(n_items).backward()

        # Clip the gradients to avoid exploding gradients
        if self.clip_gradient_norm > 0:
          torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_gradient_norm)

        # Update parameters
        optim.step()

        # sum the loss for reporting, along with the denominator
        epoch_loss += total_loss.item()
        epoch_items += n_items

        # Overall epoch loss and ppl
        loss_per_token = epoch_loss / epoch_items
        ppl = math.exp(loss_per_token)

        if (iter_count + 1) % 100 == 0:
          # Print progress
          print(f'[Epoch {eidx:<3}] loss: {loss_per_token:6.2f}, perplexity: {ppl:6.2f}')

      time_spent = time.time() - start_time

      print(f'\n[Epoch {eidx:<3}] ended with train_loss: {loss_per_token:6.2f}, train_ppl: {ppl:6.2f}')
      train_ppls.append(ppl)

      # Evaluate on valid set
      valid_loss, valid_ppl = self.evaluate('val')
      val_ppls.append(valid_ppl)
      print(f'[Epoch {eidx:<3}] ended with valid_loss: {valid_loss:6.2f}, valid_ppl: {valid_ppl:6.2f}')
      print(f'[Epoch {eidx:<3}] completed in {time_spent:.2f} seconds\n')
      print(f'[Epoch {eidx:<3}] validation BLEU with greedy search:')
      bleu = self.greedy_search('val')
      print(f'[Epoch {eidx:<3}] {bleu}')

    ######################################
    # Evaluate the final model on test set
    ######################################
    test_loss, test_ppl = self.evaluate('test')
    print(f' ---> Final test set performance: {test_loss:6.2f}, test_ppl: {test_ppl:6.2f}')

  def evaluate(self, split):
    # Switch to eval mode
    self.eval()

    eval_loss = 0.
    eval_tokens = 0

    with torch.no_grad():
      for iter_count, idxs in enumerate(self.get_batch_indices(split, shuffle=False)):
        # Get x's and y's
        x, y = self.dataset.get_batch(idxs, split=split)

        total_loss, n_items = self.forward(x, y)
        eval_loss += total_loss.item()
        eval_tokens += n_items
    eval_loss /= eval_tokens

    return eval_loss, math.exp(eval_loss)

  def greedy_search(self, split, max_len=60):
    """Performs a greedy search, dumps the translations and computes BLEU."""
    # Switch to eval mode
    self.eval()

    bos = self.dataset.trg_vocab._bos_idx
    eos = self.dataset.trg_vocab._eos_idx

    # We keep the hypotheses for a batch in a tensor for efficiency
    # Although there's a hard-limit for decoding timesteps `max_len`,
    # the hypotheses will likely to produce </s> before reaching `max_len`.
    batch_hyps = torch.zeros(
      (max_len, self.batch_size), dtype=torch.long,
      device=self.trg_emb.weight.device)

    # Resulting sentences in dataset split order
    results = []

    with torch.no_grad():
      for iter_count, idxs in enumerate(self.get_batch_indices(split, shuffle=False)):
        # We don't care about `y` for translation decoding
        x, _ = self.dataset.get_batch(idxs, split=split)

        # Clear batch hypotheses tensor
        batch_hyps.zero_()

        # Get encoder hidden states
        enc_states, input_mask = self.encode(x)

        # Compute decoder's initial state h_0 for each sentence
        h = self.compute_decoder_state(enc_states, input_mask)

        # last batch could be smaller than the requested batch size
        cur_batch_size = h.size(0)

        # Start all sentences with <s>
        next_word_idxs = torch.full(
            (cur_batch_size, ), bos, dtype=torch.long, device=h.device)

        # Track sentences who already produced </s>
        track_fini = torch.zeros((cur_batch_size, ), device=h.device).bool()

        # A maximum of `max_len` decoding steps
        for t in range(max_len):
          if track_fini.all():
            # All hypotheses produced </s>, early stop!
            break

          # Get logits from the decoder
          logits, h = self.compute_decoder_logits(h, next_word_idxs)

          # Get next probabilities and argmax them for every sentence
          next_word_idxs = nn.functional.softmax(logits, dim=-1).argmax(dim=-1)

          # Update finished sentence tracker
          track_fini.add_(next_word_idxs.eq(eos))

          # Insert most probable words for timestep `t` into tensor
          batch_hyps[t, :cur_batch_size] = next_word_idxs

        # All finished, convert translations to python lists on CPU
        results.extend(batch_hyps[:, :cur_batch_size].t().cpu().tolist())

    # post-process results to convert them to actual sentences
    out_fname = f'{split}_translations.{self.dataset.trg_lang}'
    with open(out_fname, 'w') as f:
      for sent in results:
        sent_str = self.dataset.trg_vocab.convert_idxs_to_words(sent, True)
        f.write(sent_str + '\n')

    # Evaluate with BLEU
    reference = f'{split}.{self.dataset.trg_lang}'
    bleu = subprocess.check_output(
      ['./multi-bleu.perl', reference], stdin=open(out_fname),
      stderr=subprocess.DEVNULL, universal_newlines=True).strip()
    return bleu

def train_encdec_model(n_epochs=5, init_lr=0.0005, **kwargs):
  # Set the seed for reproducible results
  fix_seed(30494)

  base_params = {
    'dataset':dataset,
    'emb_dim':200,              # word embedding dim
    'enc_dim':200,              # hidden layer dim for the encoder
    'enc_bidirectional':False,  # True makes the encoder bidirectional
    'dec_dim':300,              # hidden layer dim for the decoder
    'clip_gradient_norm':1.0,   # gradient clip threshold
    'dropout':0.3,              # dropout probability
    'tie_weights':True,         # Weight typing for decoder inputs/outputs
    'batch_size':64,            # Batch size
    'init_dec':'max',          # Initialize the decoder with max or avg state
  }

  # Override with given arguments
  base_params.update(kwargs)

  model = EncDecNMT(**base_params)

  # move to device
  model.to(DEVICE)

  # Create the optimizer
  opt = torch.optim.Adam(model.parameters(), lr=init_lr)
  print(model)

  # Returns train, val and test perplexities
  model.train_model(opt, n_epochs=n_epochs)
  return model

# add more configurations to here to train other systems
param_set = [
  {'enc_bidirectional': False, 'init_dec': 'avg'},
]

bleu_scores = []

for params in param_set:
  # Train for 5 epochs
  model = train_encdec_model(n_epochs=5, init_lr=0.0005, **params)

  # Translate the test set and get the bleu scores
  bleu = model.greedy_search('test', max_len=60)
  bleu_scores.append(bleu)

print()

for config, score in zip(param_set, bleu_scores):
  print(f'{config}\n  {score}')

class AttentionNMT(EncDecNMT):
  """Encoder-decoder NMT with attention."""
  def __init__(self, **kwargs):
    # The internal dimension for the dot product i.e.
    # the common dimension that d_t and h_i's should be projected
    self.att_dim = kwargs.pop('att_dim')

    # The attention type to compute the similarity scores
    self.att_type = kwargs.pop('att_type')
    assert self.att_type in ('dot', 'mlp'), "att_type unknown."

    # Call parent's __init__ with the remaining arguments
    super(AttentionNMT, self).__init__(**kwargs)

    ############################################
    # QUESTION
    ############################################
    # Add decoder state (query) projection layer
    self.ff_q_proj = nn.Linear(self.dec_dim, self.att_dim)

    ############################################
    # QUESTION
    ############################################
    # Add encoder states projection layer for similarity computation
    ############################################
    self.ff_k_proj = nn.Linear(self.enc_dim, self.att_dim)

    ####################################################
    # Adaptor so that the output of attention can be fed
    # directly to the `self.bneck` layer
    ####################################################
    self.ff_enc2bneck = nn.Linear(self.enc_out_dim, self.dec_dim)

    if self.att_type == 'mlp':
      ###############################################################
      # First finish the dot attention and then try to implement MLP
      ###############################################################
      # QUESTION
      ##########
      # the only parameter you would add is linear layer (no bias) representing the
      # `a` vector in the lecture slides.
      ###############################################################
      self.mlp_att = nn.Linear("<TODO>")

  def encode(self, x):
    # Let's first call the EncDec's encode()
    all_hids, mask = super(AttentionNMT, self).encode(x)

    # This is to avoid projection of encoder states at each decoding step
    # since they can be precomputed at once
    self.e_proj = self.ff_k_proj(all_hids)

    return all_hids, mask

  def compute_decoder_logits(self, dec_hid_state, y):
    ###########################################################
    # This step is the same as encoder-decoder, we feed the embedding
    # and get `d_t` (query for attention) i.e. the hidden state of the decoder
    ###########################################################
    dec_hid_state = self.dec(self.trg_emb(y), dec_hid_state)

    # Below you'll have to do a lot of permute(), t(), squeeze(), unsqueeze()
    # operations to make dimensions compatible. Check PyTorch documents
    # if you are not familiar with these operations

    ###########################################################
    # QUESTION
    ###########################################################
    # Project `dec_hid_state to attention dim with `ff_q_proj` layer
    # Expected shape: (batch_size, att_dim, 1)
    ###########################################################
    proj_q = self.ff_q_proj(dec_hid_state).unsqueeze(2)

    ###########################################################
    # QUESTION
    ###########################################################
    # Permute the dimensions of already cached `self.e_proj`
    # so that the shape becomes: (batch_size, seq_len, att_dim)
    ###########################################################
    proj_e = self.e_proj.permute(1, 0, 2)

    ###########################################################
    # QUESTION (Dot attention)
    ###########################################################
    # Now that you have the queries for all the batch (proj_q)
    # and encoder states for all source positions in the batch (proj_e)
    # you can use `torch.bmm()` to compute all similarity scores at once.
    # `bmm` stands for "Batch matrix multiplication". If you have two 3D
    # tensors where first dimension represents the `batch_size`, `bmm`
    # computes the products for each element in the batch.
    ##########
    # Example:
    ##########
    # torch.bmm("tensor of size B x S x A" , "tensor of size B x A x 1")
    #    --> produces a tensor of "B x S x 1"
    ##########
    # Use this to obtain the similarity scores and use squeeze() and t()
    # to make it look like (seq_len, batch_size)
    if self.att_type == 'dot':
      scores = torch.bmm(proj_e, proj_q).squeeze(2).t()
    elif self.att_type == 'mlp':
      ###########################################################
      # QUESTION (MLP attention)
      ###########################################################
      # First finish the dot attention and then try to implement this
      ff_layer = nn.RNN(input_size=proj_e.size(), hidden_size=proj_q.size())
      output, _ = ff_layer(proj_e, proj_q)
      scores = torch.bmm(output, self.mlp_att).squeeze(2).t()
    #   scores = self.mlp_att.t() * nn.functional.tanh(Wd * proj_e + Ws * proj_q)

    #########################################################
    # we fill the padded positions with small numbers so that
    # softmax() does not assign probabilities to them.
    #########################################################
    scores.masked_fill_(self.mask.logical_not(), -1e8)

    #############################################
    # QUESTION
    #############################################
    # Use softmax() on `scores` to obtain alpha's / probabilities
    # expected shape: (seq_len, batch_size)
    alpha = nn.functional.softmax(scores)

    #############################################
    # QUESTION
    #############################################
    # Weigh the bidirectional encoder states `self.all_hids`
    # with `alpha`
    # expected shape: (batch_size, self.enc_out_dim)
    ctx = torch.bmm(self.all_hids.permute(1, 2, 0), alpha.t().unsqueeze(2)).squeeze(2)

    # Project the computed weighted context to `dec_dim` so that
    # the output layer works as espected
    # shape: (batch_size, dec_dim)
    c_t = self.ff_enc2bneck(ctx)

    ##################################################################
    # We sum the decoder's state `d_t` and the computed `c_t` together
    ##################################################################
    logits = self.out(self.bneck(self.drop(c_t + dec_hid_state)))
    return logits, dec_hid_state

def train_attention_model(n_epochs=5, init_lr=0.0005, **kwargs):
  # Set the seed for reproducible results
  fix_seed(30494)

  base_params = {
    'dataset':dataset,
    'emb_dim':200,              # word embedding dim
    'enc_dim':200,              # hidden layer dim for the encoder
    'enc_bidirectional':True,   # True makes the encoder bidirectional
    'dec_dim':300,              # hidden layer dim for the decoder
    'clip_gradient_norm':1.0,   # gradient clip threshold
    'dropout':0.3,              # dropout probability
    'tie_weights':True,         # Weight typing for decoder inputs/outputs
    'batch_size':64,            # Batch size
    'init_dec':'avg',           # Initialize the decoder with max or avg state
    'att_dim': 200,             # Dot product's inner dimension
    'att_type': 'mlp',          # att_type dot/mlp
  }

  # Override with given arguments
  base_params.update(kwargs)

  model = AttentionNMT(**base_params)

  # move to device
  model.to(DEVICE)

  # Create the optimizer
  opt = torch.optim.Adam(model.parameters(), lr=init_lr)
  print(model)

  # Returns train, val and test perplexities
  model.train_model(opt, n_epochs=n_epochs)
  return model

# add more configurations to here to train other systems
param_set = [
  {'enc_bidirectional': True, 'init_dec': 'avg', 'att_dim': 200, 'att_type': 'dot'},
]

bleu_scores = []

for params in param_set:
  # Train for 5 epochs
  model = train_attention_model(n_epochs=5, init_lr=0.0005, **params)

  # Translate the test set and get the bleu scores
  bleu = model.greedy_search('test', max_len=60)
  bleu_scores.append(bleu)

print()

for config, score in zip(param_set, bleu_scores):
  print(f'{config}\n  {score}')