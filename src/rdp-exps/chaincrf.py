import torch
import numpy as np

from torch import nn 
from frtorch import torch_model_utils as tmu, LinearChainCRF
import torch.nn.functional as F

class ChainCRFSimulation(nn.Module):

  def __init__(self, batch_size, max_len, num_states, hidden_size, 
    norm_scale=10, device='cuda'):

    super().__init__()
    self.device = device

    self.state_matrix = nn.Parameter(torch.rand(num_states, hidden_size))
    self.word_emb = nn.Parameter(torch.rand(batch_size, max_len, hidden_size))
    self.lens = torch.tensor([max_len] * batch_size).long().to(self.device)

    self.crf = LinearChainCRF(
      potential_normalization='minmax', potential_scale=norm_scale)
    return 

  def log_partition(self):
    transition = torch.matmul(self.state_matrix, self.state_matrix.transpose(0, 1))
    emission = torch.matmul(self.word_emb, self.state_matrix.transpose(0, 1))

    _, log_Z = self.crf.forward_sum(transition, emission, self.lens)
    return log_Z

  def log_partition_approx(self, K1, K2=1, proposal='localglobal', topk_only=False):
    emission = torch.matmul(self.word_emb, self.state_matrix.transpose(0, 1))
    log_Z_est = self.crf.forward_approx(self.state_matrix, emission, self.lens, 
      sum_size=K1, sample_size=K2, proposal='softmax', transition_proposal='l1norm', 
      topk_sum=topk_only)
    return log_Z_est

  def entropy(self):
    transition = torch.matmul(self.state_matrix, self.state_matrix.transpose(0, 1))
    emission = torch.matmul(self.word_emb, self.state_matrix.transpose(0, 1))

    ent = self.crf.entropy(transition, emission, self.lens)
    return ent

  def entropy_approx(self, K1, K2=1, proposal='localglobal', topk_only=False):
    emission = torch.matmul(self.word_emb, self.state_matrix.transpose(0, 1))
    ent_est = self.crf.entropy_approx(self.state_matrix, emission, self.lens, 
      sum_size=K1, sample_size=K2, proposal='softmax', transition_proposal='l1norm', topk_sum=topk_only)
    return ent_est

  def entropy_local(self):
    emission = torch.matmul(self.word_emb, self.state_matrix.transpose(0, 1))
    emission = F.softmax(emission, dim=-1)
    ent = tmu.entropy(emission).mean()
    return ent
