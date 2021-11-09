import torch 
import numpy as np

from tqdm import tqdm 
from torch import nn
from time import time
from frtorch import LinearChainCRF

crf = LinearChainCRF()

# scale up large space 
transition_scale = 0.1
emission_scale = 0.01
# transition_bias = -1
# emission_bias = -1
transition_bias = -3
emission_bias = -3
state_size = 762
num_state= 1000
max_len = 100

state_matrix = torch.normal(
  size=[num_state, state_size], mean=0.0, std=transition_scale)
transition = torch.matmul(
  state_matrix, state_matrix.transpose(1, 0)) / np.sqrt(state_size)\
  + transition_bias
emission_weight = torch.normal(
  size=[2, max_len, state_size], mean=0.0, std=emission_scale)
emission = torch.matmul(
  emission_weight, state_matrix.transpose(1, 0)) / np.sqrt(state_size)\
  + emission_bias
lens = torch.tensor([max_len, max_len])

## test viterbi 
y, s = crf.argmax(transition, emission, lens)
y_, _, s_ = crf.rargmax(transition, emission, lens)

## test FFBS
y, y_soft = crf.rsample(transition, emission, lens)
combined_index, y, y_soft, y_origin, y_emb =\
  crf.rsample_approx(state_matrix, emission, lens, 99, 'softmax')

# scale up large space 
transition_scale = 0.1
emission_scale = 0.01
# transition_bias = -1
# emission_bias = -1
transition_bias = -3
emission_bias = -3
state_size = 762
num_state= 2
max_len = 3

state_matrix = torch.normal(
  size=[num_state, state_size], mean=0.0, std=transition_scale)
transition = torch.matmul(
  state_matrix, state_matrix.transpose(1, 0)) / np.sqrt(state_size)\
  + transition_bias
emission_weight = torch.normal(
  size=[2, max_len, state_size], mean=0.0, std=emission_scale)
emission = torch.matmul(
  emission_weight, state_matrix.transpose(1, 0)) / np.sqrt(state_size)\
  + emission_bias
lens = torch.tensor([max_len, max_len])

y, y_soft, y_prob, y_prob_step = crf.rsample(
  transition, emission, lens, return_prob=True)
print(y)
print(y_prob.exp())

y_log_potential = crf.seq_log_potential(y, transition, emission, lens)
_, log_Z = crf.forward_sum(transition, emission, lens)
(y_log_potential - log_Z).exp()
