"""Torch model/ missing utils

the missing utility library for pytorch

Tensor operations:
* `to_np`
* `length_to_mask`
* `lengths_to_squared_mask`
* 'mask_by_length'
* `ind_to_one_hot`
* `bow_to_k_hot`
* `seq_to_lens`
* `find_index`
* `seq_ends`
* `reverse_sequence`
* `batch_gather_last`
* `batch_index_select`
* `batch_index_put`
* `batch_index_fill`
* `batch_repeat`

NLP:
* `build_vocab`
* `pad_or_trunc_seq`

Probability:
* `cumsum`
* `sample_gumbel`
* `reparameterize_gumbel`
* `seq_gumbel_encode` # needs update
* `reparameterize_gaussian`
* `entropy`
* `kl_divergence`
* `js_divergence`

Model operations:
* `load_partial_state_dict`
* `print_params`
* `print_grad`

Miscellaneous:
* `print_args`
* `print_dict_nums`
* `str2bool`
* `BucketSampler`
"""

import os 
import shutil

import numpy as np 
import pandas as pd 

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from torch.utils.data import BatchSampler, SequentialSampler
from torch.nn.parameter import Parameter
from collections import OrderedDict, Counter


def to_np(x):
  """Export a tensor to numpy"""
  return x.detach().cpu().numpy()


def length_to_mask(length, max_len):
  """
  True = 1 = not masked, False = 0 = masked

  Args:
    length: type=torch.tensor(int), size=[batch]
    max_len: type=int

  Returns:
    mask: type=torch.tensor(bool), size=[batch, max_len]
  """
  batch_size = length.shape[0]
  device = length.device
  mask = torch.arange(max_len, dtype=length.dtype)\
    .expand(batch_size, max_len).to(device) < length.unsqueeze(1)
  return mask


def lengths_to_squared_mask(lengths, max_len):
  """
    True = 1 = not masked, False = 0 = masked

    e.g., lengths = [2], max_len = 3
    returns: [[1, 1, 0],
              [1, 1, 0],
              [0, 0, 0]]

    Args:
      length: type=torch.tensor(int), size=[batch]
      max_len: type=int

    Returns:
      mask: type=torch.tensor(bool), size=[batch, max_len, max_len]
    """
  batch_size = lengths.size(0)
  mask_ = length_to_mask(lengths, max_len)
  mask = mask_.view(batch_size, 1, max_len).repeat(1, max_len, 1)
  mask = mask * mask_.float().unsqueeze(-1)
  return mask.bool()
  

def mask_by_length(A, lens, mask_id=0.):
  """mask a batch of seq tensor by length
  
  Args:
    A: type=torch.tensor(), size=[batch, max_len, *]
    lens: type=torch.tensor(ing), size=[batch]
    mask_id: type=float

  Returns
    A_masked: type=torch.tensor(float), note the for whatever input, the output 
      type would be casted to float
  """
  mask = length_to_mask(lens, A.size(1))
  target_size = list(mask.size()) + [1] * (len(A.size()) - 2)
  mask = mask.view(target_size)

  A_masked = A.float() * mask.float() + A.float() * (1 - mask.float()) * mask_id
  return A_masked


def ind_to_one_hot(ind, max_len):
  """Index to one hot representation

  Args:
    ind: type=torch.tensor(int), size=[batch]
    max_len: type=int

  Returns:
    one_hot: type=torch.tensor(bool), size=[batch, max_len]

  Note: 
    by default, `one_hot.dtype = ind.dtype`, and there is no constraint on 
    `ind.dtype`. So it is also possible to pass `ind` with float type 
  """
  device = ind.device
  batch_size = ind.shape[0]
  one_hot = torch.arange(max_len, dtype=ind.dtype)\
    .expand(batch_size, max_len).to(device) == (ind).unsqueeze(1)
  one_hot = one_hot.float()
  return one_hot


def bow_to_k_hot(bow, vocab_size, pad_id=0, mask_pad=False):
  """Bag of words to one hot representation

  Args:
    bow: type=torch.tensor(int), size=[batch, max_bow]
    vocab_size: type=int
    pad_id: type=int

  Returns:
    one_hot: type=torch.tensor(int), size=[batch, vocab_size]
  """
  device = bow.device
  batch_size = bow.shape[0]
  bow = bow.view(-1).unsqueeze(1)  # [batch * bow, 1]
  k_hot = (bow == torch.arange(vocab_size).to(device).reshape(1, vocab_size))
  k_hot = k_hot.float().view(batch_size, -1, vocab_size)
  if(mask_pad):
    k_hot.index_fill_(
      dim=2, index=torch.tensor([pad_id]).to(device), value=0)
  k_hot = k_hot.sum(dim=1)
  return k_hot

def k_hot_to_bow(k_hot, max_bow_size, pad_id=0):
  """Convert a k-hot vector to an index vector padded by pad_id
  
  Args:
    k_hot: type=torch.tensor(int), size=[batch, vocab_size]
    max_bow_size:

  Returns:
    bow: type=torch.tensor(int), size=[batch, max_bow_size]
      max_bow_size may extend according to input

  Example:
    max_bow_size = 4
    k_hot = [0, 0, 1, 1, 0]
            [0, 1, 0, 1, 1]
    num_pad = [2, 1]
    k_hot_padded = [0, 0, 1, 1, 0, 1, 1, 0, 0]
                   [0, 1, 0, 1, 1, 1, 0, 0, 0]
    idx = [0, 0, 0, 0, 1, 1, 1, 1]
          [2, 3, 5, 6, 1, 3, 4, 5]
    bow = [2, 3, 0, 0]
          [1, 3, 4, 0]
    
  Warning:
    if [number of ones in the k hot vector] > max_bow_size
  """
  assert(len(k_hot.size()) == 2)
  device = k_hot.device
  batch_size = k_hot.size(0)
  vocab_size = k_hot.size(1)

  num_activate = k_hot.sum(1)
  if(num_activate.max() > max_bow_size): 
    print('WARNING: More 1s than max_bow_size = %d, extend max_bow_size to %d'
      % (max_bow_size, num_activate.max()))
    max_bow_size = num_activate.max()
  num_pad = max_bow_size - num_activate # [B]
  pad_ = length_to_mask(num_pad, max_bow_size)
  k_hot_padded = torch.cat([k_hot, pad_], dim=1) # [B, vocab_size + max_bow_size]

  bow = k_hot_padded.nonzero().transpose(0, 1)[1].view(batch_size, max_bow_size)
  bow = bow * (bow < vocab_size)
  return bow


def seq_to_lens(seq, pad_id=0):
  """Calculate sequence length
  
  Args:
    seq: type=torch.tensor(long), shape=[*, max_len]
    pad_id: pad index. 

  Returns:
    lens: type=torch.tensor(long), shape=[*]
  """
  lens = (seq != pad_id).sum(dim=-1).type(torch.long)
  return lens


def find_index(seq, val):
  """Find the first location index of a value 
  if there is no such value, return -1
  
  Args:
    seq: type=torch.tensor(long), shape=[batch, max_len]
    val: type=int 

  Returns:
    lens: type=torch.tensor(long), shape=[batch]
  """
  device = seq.device
  s_ = (seq == val).type(torch.float)
  seq_len = seq.size(-1)
  ind_ = torch.arange(seq_len).view(1, seq_len) + 1
  ind_ = ind_.to(device)
  s = (1 - s_) * 1e10 + s_ * ind_
  _, index = torch.min(s, dim=-1)
  index = index.type(torch.long)
  not_find = (s_.sum(-1) == 0)
  index.masked_fill_(not_find, -1)
  return index


def seq_ends(seq, end_id):
  """Calculate where the sequence ends
  if there is not end_id, return the last index 
  
  Args:
    seq: type=torch.tensor(long), shape=[batch, max_len]
    end_id: end index. 

  Returns:
    ends_at: type=torch.tensor(long), shape=[batch]
  """
  ends_at = find_index(seq, end_id)
  max_len = seq.size(1) - 1
  ends_at[ends_at == -1] = max_len
  return ends_at


def reverse_sequence(seq, seq_lens):
  """Reverse the sequence

  Examples:

  seq = [[1, 2, 3, 4, 5], [6, 7 ,8, 9, 0]], seq_lens = [3, 4]
  reverse_sequence(seq, seq_lens) = [[3, 2, 1, 4, 5], [9, 8, 7, 6, 0]]

  seq = [[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], 
         [[6, 6], [7, 7], [8, 8], [9, 9], [0, 0]]], 
  seq_lens = [3, 4]
  reverse_sequence(seq, seq_lens) = 
    [[[3, 3], [2, 2], [1, 1], [4, 4], [5, 5]], 
     [[9, 9], [8, 8], [7, 7], [6, 6], [0, 0]]]
  
  Args: 
    seq: size=[batch, max_len, *]
    seq_lens: size=[batch]

  Returns:
    reversed_seq
  """
  batch = seq.size(0)
  reversed_seq = seq.clone()
  for i in range(batch):
    ind = list(range(seq_lens[i]))
    ind.reverse()
    reversed_seq[i, :seq_lens[i]] = seq[i, ind]
  return reversed_seq


def batch_gather_last(seq, seq_lens):
  """Gather the last element of a given sequence"""
  return batch_index_select(seq, seq_lens - 1)


def batch_index_select(A, ind):
  """Batched index select
  
  Args:
    A: size=[batch, num_class, *] 
    ind: size=[batch, num_select] or [batch]

  Returns:
    A_selected: size=[batch, num_select, *] or [batch, *]
  """  
  batch_size = A.size(0)
  num_class = A.size(1)
  A_size = list(A.size())
  device = A.device
  A_ = A.clone().reshape(batch_size * num_class, -1)
  if(len(ind.size()) == 1): 
    batch_ind = (torch.arange(batch_size) * num_class)\
      .type(torch.long).to(device)
    ind_ = ind + batch_ind
    A_selected = torch.index_select(A_, 0, ind_)\
      .view([batch_size] + A_size[2:])
  else:
    batch_ind = (torch.arange(batch_size) * num_class)\
      .type(torch.long).to(device)
    num_select = ind.size(1)
    batch_ind = batch_ind.view(batch_size, 1)
    ind_ = (ind + batch_ind).view(batch_size * num_select)
    A_selected = torch.index_select(A_, 0, ind_)\
      .view([batch_size, num_select] + A_size[2:])
  return A_selected


def batch_index_put(A, ind, N):
  """distribute a batch of values to locations in a tensor

  Example:
    A = tensor([[0.1000, 0.9000],
                [0.2000, 0.8000]])
    ind = tensor([[1, 2],
                  [0, 3]])
    N = 5
  then:
    A_put = tensor([[0.0000, 0.1000, 0.9000, 0.0000, 0.0000],
                    [0.2000, 0.0000, 0.0000, 0.8000, 0.0000]])

  Args:
    A: size=[batch, M, *], * can be any list of dimensions
    ind: size=[batch, M]
    N: type=int. Maximum length of new tensor 

  Returns:
    A_put: size=[batch, N, *]
  """
  batch_size = A.size(0)
  M = A.size(1)
  As = list(A.size()[2:])
  device = A.device
  A_put = torch.zeros([batch_size * N] + As).to(device)
  ind_ = torch.arange(batch_size).view(batch_size, 1) * N
  ind_ = ind_.expand(batch_size, M).flatten().to(device)
  ind_ += ind.flatten()
  A_put[ind_] += A.view([batch_size * M] + As)
  A_put = A_put.view([batch_size, N] + As)
  return A_put


def batch_index_fill(A, ind, v):
  """Fill in values to a tensor

  Example:
    A = torch.zeros(2, 4, 2)
    ind = torch.LongTensor([[2, 3], [0, 1]])
    batch_index_fill(A, ind, 4) 
  Then:
    A_filled = tensor([[[0., 0.],
                        [0., 0.],
                        [4., 4.],
                        [4., 4.]],

                       [[4., 4.],
                        [4., 4.],
                        [0., 0.],
                        [0., 0.]]])
  
  Args:
    A: size=[batch, M, rest]
    ind: size=[batch] or [batch, N], N < M
    v: size=[rest] or 1

  Returns:
    A_filled: size=[batch, M, rest]
  """
  A = A.clone()
  batch_size = A.size(0)
  M = A.size(1)
  As = list(A.size()[2:])
  device = A.device

  A_ = A.view([batch_size * M] + As)

  if(len(ind.size()) == 1): ind = ind.unsqueeze(1)
  ind_ = (((torch.arange(batch_size)) * M).unsqueeze(1).to(device) + ind).flatten()
  A_[ind_] = v
  A_filled = A_.view([batch_size, M] + As)
  return A_filled


def batch_repeat(A, n):
  """
  Args:
    A: size=[batch, *], * can be any list of dimensions
    n: type=int

  Returns:
    A: size=[batch * n, *]
  """
  batch_size = A.size(0)
  As = list(A.size()[1:])
  A_ = A.view([batch_size, 1] + As)
  A_ = A_.repeat([1, n] + [1] * len(As))
  A_ = A_.view([batch_size * n] + As)
  return A_

def build_vocab(data: list, start_id: int, freq_thres: int=0):
  """
  Build vocabulary

  Args:
    data: ... 
    start_id: ... 
    freq_thres: ... 

  Returns:
    word2id: ... 
    id2word: ... 
  """
  word2id, id2word = {}, {}
  vocab = []
  for d in data: vocab.extend(d)
  vocab = Counter(vocab)
  id_ = start_id
  for w, c in vocab.most_common():
    if(c < freq_thres): break
    word2id[w] = id_
    id2word[id_] = w
    id_ += 1
  return word2id, id2word

def pad_or_trunc_seq(s: list, max_len: int, pad: int = 0) -> list:
  """
  If sequence longer than max_len, truncate it
  If sequence shorter than max_len, pad it to max_len
  """
  for _ in range(max_len - len(s)): s.append(pad)
  return s[: max_len]

def save_attn_figure(src, tgt, attn, fpath):
  """Save attention heatmap

  Args:
    src: a list of strings, each string is a source token 
    tgt: a list of strings, each string is a target token 
    attn: an attention matrix. A 2d np array 
    fpath: path to save the figure 
  """
  df = pd.DataFrame(attn) # target * source
  df.columns = src
  df.index = tgt
  fig = plt.figure()
  # sns.set(font_scale = 0.5)
  ax = sns.heatmap(df, 
                   annot=True, 
                   cmap='coolwarm', 
                   annot_kws={"size": 5},  
                   xticklabels=True, 
                   yticklabels=True
                   )
  plt.yticks(size=6)
  plt.xticks(rotation=270)
  plt.subplots_adjust(left=0.3, bottom=0.28)
  ax.figure.savefig(fpath + '.png', dpi=150)
  plt.close(fig)
  return 


def save_two_attn_figure(src, pred, ref, attn_pred, attn_ref, fpath):
  """Save attention heatmap

  Args:
  src: a list of strings, each string is a source token 
  tgt: a list of strings, each string is a target token 
  attn: an attention matrix. A 2d np array 
  fpath: path to save the figure 
  """
  attn_pred_len = attn_pred.shape[0]
  attn_ref_len = attn_ref.shape[0]
  attn_max_len = max(attn_pred_len, attn_ref_len)
  attn_src_len = attn_ref.shape[1]
  attn_pred_ = np.zeros(shape=(attn_max_len, attn_src_len))
  attn_pred_[:attn_pred_len] = attn_pred
  attn_ref_ = np.zeros(shape=(attn_max_len, attn_src_len))
  attn_ref_[:attn_ref_len] = attn_ref
  
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
  fig.tight_layout(pad=6.0)

  df = pd.DataFrame(attn_ref_) # target * source
  df.columns = src
  for _ in range(attn_max_len - len(ref)): ref.append('')
  df.index = ref
  sns.heatmap(df, 
            annot=True, 
            cmap='coolwarm', 
            annot_kws={"size": 5}, 
            ax=ax1,
            xticklabels=True, 
            yticklabels=True,
            cbar=False
            )
  ax1.set_yticklabels(ref, fontsize=7)
  ax1.set_xticklabels(src, rotation=300)

  df = pd.DataFrame(attn_pred_) # target * source
  df.columns = src
  for _ in range(attn_max_len - len(pred)): pred.append('')
  df.index = pred
  sns.heatmap(df, 
            annot=True, 
            cmap='coolwarm', 
            annot_kws={"size": 5}, 
            ax=ax2,
            xticklabels=True, 
            yticklabels=True
            )
  ax2.set_yticklabels(pred, fontsize=7)
  ax2.set_xticklabels(src, rotation=300)

  plt.savefig(fpath + '.png', dpi=200)
  plt.close(fig)
  return 

# def save_attn_pred_figure(src, pred, ref, attn_pred, attn_ref, pred_dist, vocab, fpath):
#     """Save attention heatmap

#     Args:
#     src: a list of strings, each string is a source token 
#     tgt: a list of strings, each string is a target token 
#     attn: an attention matrix. A 2d np array 
#     fpath: path to save the figure 
#     """
#     attn_pred_len = attn_pred.shape[0]
#     attn_ref_len = attn_ref.shape[0]
#     attn_max_len = max(attn_pred_len, attn_ref_len)
#     attn_src_len = attn_ref.shape[1]
#     attn_pred_ = np.zeros(shape=(attn_max_len, attn_src_len))
#     attn_pred_[:attn_pred_len] = attn_pred
#     attn_ref_ = np.zeros(shape=(attn_max_len, attn_src_len))
#     attn_ref_[:attn_ref_len] = attn_ref
    
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,6))
#     fig.tight_layout(pad=4.0)

#     df = pd.DataFrame(attn_ref_) # target * source
#     df.columns = src
#     for _ in range(attn_max_len - len(ref)): ref.append('')
#     df.index = ref
#     sns.heatmap(df, 
#               annot=True, 
#               cmap='Blues', 
#               annot_kws={"size": 5}, 
#               ax=ax1,
#               xticklabels=True, 
#               yticklabels=True,
#               cbar=False
#               )
#     ax1.set_yticklabels(ref, fontsize=7)
#     ax1.set_xticklabels(src, rotation=300)

#     df = pd.DataFrame(attn_pred_) # target * source
#     df.columns = src
#     for _ in range(attn_max_len - len(pred)): pred.append('')
#     df.index = pred
#     sns.heatmap(df, 
#               annot=True, 
#               cmap='Blues', 
#               annot_kws={"size": 5}, 
#               ax=ax2,
#               xticklabels=True, 
#               yticklabels=True,
#               cbar=False
#               )
#     ax2.set_yticklabels(pred, fontsize=7)
#     ax2.set_xticklabels(src, rotation=300)
    
#     df = pd.DataFrame(pred_dist) # target * source
#     df.columns = vocab
#     sns.heatmap(df, 
#               annot=True, 
#               cmap='Blues', 
#               annot_kws={"size": 5}, 
#               ax=ax3,
#               xticklabels=True, 
#               yticklabels=True,
#               cbar=False
#               )
#     ax3.set_yticklabels(range(len(pred_dist)), rotation=0, fontsize=7)
#     ax3.set_xticklabels(vocab, rotation=0, fontsize=5)
    
    
#     plt.subplots_adjust(left=0.1)
#     plt.savefig(fpath + '.png', dpi=200)
#     plt.close(fig)
#     return 

def save_attn_pred_figure(src, pred, ref, attn_pred, attn_ref, 
  pred_dist, pred_dist_ref, vocab, fpath):
  """Save attention heatmap

  Args:
  src: a list of strings, each string is a source token 
  tgt: a list of strings, each string is a target token 
  attn: an attention matrix. A 2d np array 
  fpath: path to save the figure 
  """
  attn_pred_len = attn_pred.shape[0]
  attn_ref_len = attn_ref.shape[0]
  attn_max_len = max(attn_pred_len, attn_ref_len)
  attn_src_len = attn_ref.shape[1]
  attn_pred_ = np.zeros(shape=(attn_max_len, attn_src_len))
  attn_pred_[:attn_pred_len] = attn_pred
  attn_ref_ = np.zeros(shape=(attn_max_len, attn_src_len))
  attn_ref_[:attn_ref_len] = attn_ref

  pred_dist_ref = pred_dist_ref[:, 2:]
  vocab = vocab[2:]
  pred_dist = pred_dist[:, 2:]
  
  fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,6))
  fig.tight_layout(pad=4.0)
  
  df = pd.DataFrame(pred_dist_ref) # target * source
  df.columns = vocab
  sns.heatmap(df, 
            annot=True, 
            cmap='Blues', 
            annot_kws={"size": 5}, 
            ax=ax1,
            xticklabels=True, 
            yticklabels=True,
            cbar=False
            )
  ax1.set_xticklabels(vocab, fontsize=6, rotation=0)
  ax1.set_yticklabels(range(len(pred_dist_ref)), rotation=0)

  df = pd.DataFrame(attn_ref_) # target * source
  df.columns = src
  for _ in range(attn_max_len - len(ref)): ref.append('')
  df.index = ref
  sns.heatmap(df, 
            annot=True, 
            cmap='Blues', 
            annot_kws={"size": 5}, 
            ax=ax2,
            xticklabels=True, 
            yticklabels=True,
            cbar=False
            )
  ax2.set_yticklabels(ref, fontsize=7)
  ax2.set_xticklabels(src, rotation=300)

  df = pd.DataFrame(attn_pred_) # target * source
  df.columns = src
  for _ in range(attn_max_len - len(pred)): pred.append('')
  df.index = pred
  sns.heatmap(df, 
            annot=True, 
            cmap='Blues', 
            annot_kws={"size": 5}, 
            ax=ax3,
            xticklabels=True, 
            yticklabels=True,
            cbar=False
            )
  ax3.set_yticklabels(pred, fontsize=7)
  ax3.set_xticklabels(src, rotation=300)
  
  df = pd.DataFrame(pred_dist) # target * source
  df.columns = vocab
  sns.heatmap(df, 
            annot=True, 
            cmap='Blues', 
            annot_kws={"size": 5}, 
            ax=ax4,
            xticklabels=True, 
            yticklabels=True,
            cbar=False
            )
  ax4.set_xticklabels(vocab, fontsize=6, rotation=0)
  ax4.set_yticklabels(range(len(pred_dist)), rotation=0)
  
  
  plt.savefig(fpath + '.png', dpi=200)
  plt.close(fig)
  return 


def sample_gumbel(shape, eps=1e-20):
  """Sample from a standard gumbel distribution"""
  U = torch.rand(shape)
  return -torch.log(-torch.log(U + eps) + eps)


def reparameterize_gumbel(logits, tau):
  """Reparameterize gumbel sampling

  Note: gumbel reparameterization will give you sample no matter tau. tau just 
  controls how close the sample is to one-hot 
  
  Args: 
    logits: shape=[*, vocab_size]
    tau: the temperature, typically start from 1.0 and anneal to 0.01

  Returns:
    y: shape=[*, vocab_size]
  """
  y = logits + sample_gumbel(logits.size()).to(logits.device)
  return F.softmax(y / tau, dim=-1)

def gumbel_topk(logits, k):
  """Gumbel-topk for sampling without replacements 
  Args: 
    logits: [*, vocab_size] 
    k: integer
  """
  y = logits + sample_gumbel(logits.size()).to(logits.device)
  val, ind = torch.topk(y, k, dim=-1)
  return val, ind


def seq_gumbel_encode(sample, sample_ids, embeddings, gumbel_st):
  """Encoding of gumbel sample. Given a sequence of relaxed one-hot 
  representations, return a sequence of corresponding embeddings

  TODO: drop `probs`, only use `sample`

  Args:
    sample: type=torch.tensor(torch.float), shape=[batch, max_len, vocab_size]
    sample_ids: type=torch.tensor(torch.long), shape=[batch, max_len]
    embeddings: type=torch.nn.Embeddings
    gumbel_st: type=bool, if use gumbel straight through estimator
  """
  batch_size = sample.size(0)
  max_len = sample.size(1)
  vocab_size = sample.size(2)
  if(gumbel_st):
    # straight-through version, to avoid training-inference gap 
    sample_emb = embeddings(sample_ids)
    sample_one_hot = ind_to_one_hot(
      sample_ids.view(-1), vocab_size)
    sample_one_hot =\
      sample_one_hot.view(batch_size, max_len, vocab_size)
    sample_soft = sample.masked_select(sample_one_hot)
    sample_soft = sample_soft.view(batch_size, max_len, 1)
    sample_emb *= (1 - sample_soft).detach() + sample_soft
  else:
    # original version, requires annealing in the end of training
    sample_emb = torch.matmul(
      sample.view(-1, vocab_size), embeddings.weight)
    embedding_size = sample_emb.size(-1)
    # [batch * max_len, embedding_size] -> [batch, max_len, embedding_size]
    sample_emb = sample_emb.view(
      batch_size, max_len, embedding_size)
  return sample_emb


def reparameterize_gaussian(mu, logvar):
  """Reparameterize the gaussian sample"""
  std = torch.exp(0.5 * logvar)
  eps = torch.randn_like(std)
  return mu + eps * std


def entropy(p, eps=1e-10, keepdim=False):
  """Calculate the entropy of a discrete distribution
  
  Args: 
    p: shape = [*, support_size]
  """
  ent = (-p * torch.log(p + eps)).sum(dim=-1)
  if(keepdim): return ent
  else: return ent.mean()


def kl_divergence(p0, p1, eps=1e-10):
  """Calculate the kl divergence between two distributions

  Args: 
    p0: size=[*, support_size]
    p1: size=[*, support_size]
  """
  kld = p0 * torch.log(p0 / (p1 + eps) + eps)
  kld = kld.sum(dim=-1)
  return kld


def js_divergence(p0, p1):
  """Calculate the Jensen-Shannon divergence between two distributions
  
  Args: 
    p0: size=[*, support_size]
    p1: size=[*, support_size]
  """
  p_ = (p0 + p1) / 2
  jsd = (kl_divergence(p0, p_) + kl_divergence(p1, p_)) / 2
  return jsd


def load_partial_state_dict(model, state_dict):
  """Load part of the model

  NOTE: NEED TESTING!!!

  Args:
    model: the model 
    state_dict: partial state dict
  """
  print('Loading partial state dict ... ')
  own_state = model.state_dict()
  own_params = set(own_state.keys())
  for name, param in state_dict.items():
    if name not in own_state:
      print('%s passed' % name)
      continue
    if isinstance(param, Parameter):
      # backwards compatibility for serialized parameters
      param = param.data
    print('loading: %s ' % name)
    own_params -= set(name)
    own_state[name].copy_(param)
  return 


def print_params(model):
  """Print the model parameters"""
  for name, param in model.named_parameters(): 
     print('  ', name, param.data.shape, 'requires_grad', param.requires_grad)
  return 


def print_grad(model, level='first'):
  """Print the gradient norm and std, for inspect training

  Note: the variance of gradient printed here is not the variance of a gradient 
  estimator
  """
  if(level == 'first'): print_grad_first_level(model)
  elif(level == 'second'): print_grad_second_level(model)
  else: 
    raise NotImplementedError(
      'higher level gradient inpection not implemented!')


def print_grad_first_level(model):
  """Print the gradient norm of model parameters, up to the first level name 
  hierarchy 
  """
  print('gradient of the model parameters:')

  grad_norms = OrderedDict()
  grad_std = OrderedDict()
  for name, param in model.named_parameters():
    splitted_name = name.split('.')
    first_level_name = splitted_name[0]

    if(first_level_name not in grad_norms): 
      grad_norms[first_level_name] = []
      grad_std[first_level_name] = []

    if(param.requires_grad and param.grad is not None):
      grad_norms[first_level_name].append(
        to_np(param.grad.norm()))
      grad_std[first_level_name].append(
        to_np(param.grad.var(unbiased=False)))

  for fn in grad_norms:
    if(isinstance(grad_norms[fn], list)):
      print(fn, np.average(grad_norms[fn]), np.average(grad_std[fn]))

  print('')
  return 


def print_grad_second_level(model):
  """Print the gradient norm of model parameters, up to the second level name 
  hierarchy 
  """
  print('gradient of the model parameters:')

  grad_norms = OrderedDict()
  grad_std = OrderedDict()
  for name, param in model.named_parameters():
    splitted_name = name.split('.')
    first_level_name = splitted_name[0]

    if(first_level_name not in grad_norms): 
      if(len(splitted_name) == 1):
        grad_norms[first_level_name] = []
        grad_std[first_level_name] = []
      else:
        grad_norms[first_level_name] = {}
        grad_std[first_level_name] = {}

    if(len(splitted_name) > 1):
      second_level_name = splitted_name[1]
      if(second_level_name not in grad_norms[first_level_name]):
        grad_norms[first_level_name][second_level_name] = []
        grad_std[first_level_name][second_level_name] = []

    if(param.requires_grad and param.grad is not None):
      # print(name, param.grad.norm(), param.grad.std())
      if(len(splitted_name) == 1):
        grad_norms[first_level_name].append(
          param.grad.norm().detach().cpu().numpy())
        grad_std[first_level_name].append(
          param.grad.std().detach().cpu().numpy())
      else: 
        grad_norms[first_level_name][second_level_name].append(
          param.grad.norm().detach().cpu().numpy())  
        grad_std[first_level_name][second_level_name].append(
          param.grad.std().detach().cpu().numpy())  

  # print(grad_norms.keys())
  for fn in grad_norms:
    if(isinstance(grad_norms[fn], list)):
      print(fn, np.average(grad_norms[fn]),
        np.average(grad_std[fn]))
    else: 
      for sn in grad_norms[fn]:
        print(fn, sn, 
          np.average(grad_norms[fn][sn]),
          np.average(grad_std[fn][sn]))

  print('')
  return 

def print_args(args):
  """Print commandline arguments nicely"""
  for arg in vars(args): print(arg, getattr(args, arg))
  return 

def pprint_dict_nums(d, index_list=False, prec=4):
  """Pretty printing a dictionary of numbers"""
  for s in d:
    if(isinstance(d[s], list)):
      print('  %s' % s)
      for i, dsi in enumerate(d[s]):
        if(prec == 4):
          if(index_list):
            print('    %d: %.4f' % (i, dsi))
          else:
            print('    %.4f' % dsi)
    else: 
      print('  %s: %.4f' % (s, d[s]))
  return 

def str2bool(v):
  """String to bool, used for setting up commandline arguments"""
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def refresh_dir(d):
  """Refresh a directory, if"""
  if(os.path.exists(d)): 
    print('removing existing dir: %s' % d)
    shutil.rmtree(d)
  print('creating dir: %s' % d)
  os.makedirs(d)
  return 


class BucketSampler(object):
  """Bucket Sampler for bucketing sentences with similar length"""
  def __init__(self, dataset, batch_size):
    """
    Args:
      dataset: a torch.data.Dataset instance. Assume sentences within the 
        dataset are sorted according to their lengths
    """
    indices = range(len(dataset))
    indices = list(BatchSampler(
      SequentialSampler(indices), batch_size=batch_size, drop_last=True))
    self.indices = indices
    self.ptr = 0
    return 

  def __len__(self):
    return len(self.indices)

  def __iter__(self):
    np.random.shuffle(self.indices)
    self.ptr = 0
    return self

  def __next__(self):
    if(self.ptr < len(self.indices)):
      result = self.indices[self.ptr]
      self.ptr += 1
      return result
    else: 
      raise StopIteration
