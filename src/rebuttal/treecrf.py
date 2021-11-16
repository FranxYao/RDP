import torch
import numpy as np

from torch import nn 
from frtorch import torch_model_utils as tmu
import torch.nn.functional as F

class TreeCRFSimple(nn.Module):

  def __init__(self):
    super().__init__()
    return 

  def inside(self, log_potentials, lens):
    device = log_potentials.device
    batch_size = log_potentials.size(0)
    max_len = log_potentials.size(1)
    label_size = log_potentials.size(3)

    beta = torch.zeros_like(log_potentials).to(device)
    for i in range(max_len):
      beta[:, i, i] = log_potentials[:, i, i]
    for d in range(1, max_len):
      for i in range(max_len - d):
        j = i + d
        before_lse_1 = beta[:, i, i:j].view(batch_size, d, label_size, 1)
        before_lse_2 = beta[:, i + 1: j + 1, j].view(batch_size, d, 1, label_size)
        before_lse = (before_lse_1 + before_lse_2).reshape(batch_size, -1)
        after_lse = torch.logsumexp(before_lse, -1).view(batch_size, 1)
        beta[:, i, j] = log_potentials[:, i, j] + after_lse
    
    before_lse = tmu.batch_index_select(beta[:, 0], lens - 1)
    log_z = torch.logsumexp(before_lse, -1)
    return log_z


class TreeCRFSimulation(nn.Module):

  def __init__(self, batch_size, max_len, num_states, hidden_size, 
    norm_scale=10, device='cuda'):
    super().__init__()
    self.device = device
    self.norm_scale = norm_scale
    self.word_emb = nn.Parameter(torch.rand(batch_size, max_len, hidden_size))
    self.word_emb.to(device)
    self.state_matrix = nn.Parameter(torch.rand(num_states, hidden_size))
    self.state_matrix.to(device)
    self.treecrf = TreeCRFSimple()
    return 

  def normalize_potentials(self, log_potentials):
    log_potentials_max = log_potentials.max(-1, keepdim=True).values
    log_potentials_min = log_potentials.min(-1, keepdim=True).values
    log_potentials = (log_potentials - log_potentials_min)
    log_potentials = self.norm_scale * log_potentials / (log_potentials_max - log_potentials_min)
    return log_potentials

  def get_log_potentials(self):
    batch_size = self.word_emb.size(0)
    max_len = self.word_emb.size(1)
    hidden_size = self.word_emb.size(2)
    num_states = self.state_matrix.size(0)

    log_potentials = self.word_emb.view(batch_size, max_len, 1, 1, hidden_size) +\
                     self.state_matrix.view(1, 1, 1, num_states, hidden_size) +\
                     self.word_emb.view(batch_size, 1, max_len, 1, hidden_size)
    log_potentials = log_potentials.sum(-1)
    return log_potentials

  def log_partition(self):
    batch_size = self.word_emb.size(0)
    max_len = self.word_emb.size(1)
    device = self.word_emb.device
    lens = (torch.zeros(batch_size) + max_len).long().to(device)

    log_potentials = self.get_log_potentials()
    log_potentials = self.normalize_potentials(log_potentials)

    log_Z = self.treecrf.inside(log_potentials, lens)
    return log_Z

  def log_partition_approx(self, K1, K2=1, proposal='localglobal', topk_only=False):
    log_potentials = self.get_log_potentials()
    log_potentials = self.normalize_potentials(log_potentials)

    state_matrix = self.state_matrix
    num_state = state_matrix.size(0)
    max_len = log_potentials.size(1)
    batch_size = log_potentials.size(0)

    # step one, sample state
    proposal_local = F.softmax(log_potentials, -1)
    proposal_global = F.softmax(state_matrix.abs().sum(-1), dim=-1)
    if(proposal == 'localglobal'):
      proposal_combined = proposal_local + proposal_global.view(1, 1, 1, num_state)
    elif(proposal == 'local'):
      proposal_combined = proposal_local
    elif(proposal == 'global'):
      proposal_combined = 0 * proposal_local + proposal_global.view(1, 1, 1, num_state)
    elif(proposal == 'uniform'):
      proposal_combined = 0 * proposal_local + 1.
    else:
      raise NotImplementedError('proposal %s not implemented' % proposal)

    # top k1 summation 
    # [B, T, T, K1]
    _, sum_index = torch.topk(proposal_combined, K1, dim=-1)
    sum_potentials = tmu.batch_index_select(
      log_potentials.view(-1, num_state), 
      sum_index.view(-1, K1)
      )

    # sample k2 
    proposal_renorm = tmu.batch_index_fill(
      proposal_combined.view(batch_size * max_len * max_len, -1), 
      sum_index.view(batch_size * max_len * max_len, -1),
      1e-7)
    proposal_renorm /= proposal_renorm.sum(-1, keepdim=True)
    sampled_index = torch.multinomial(proposal_renorm, K2)
    sample_log_prob = tmu.batch_index_select(proposal_renorm, sampled_index)
    sample_log_prob = (sample_log_prob + 1e-8).log()

    sample_potentials = tmu.batch_index_select(
      log_potentials.view(-1, num_state), 
      sampled_index.view(-1, K2)
      )
    # bias correction
    sample_potentials -= sample_log_prob + np.log(K2)

    # step two, combine states and view them as if it is a new treecrf
    if(topk_only):
      combined_potentials = sum_potentials.reshape(
        batch_size, max_len, max_len, K1)
    else:
      combined_potentials = torch.cat([sum_potentials, sample_potentials], dim=-1)
      combined_potentials = combined_potentials.reshape(
        batch_size, max_len, max_len, K1 + K2)

    # step three, conventional inside
    lens = (torch.zeros(batch_size) + max_len).long().to(self.device)
    log_Z_est = self.treecrf.inside(combined_potentials, lens)
    return log_Z_est

  def entropy_approx(self):
    """"""
    log_potentials = self.get_log_potentials()
    # NOTE: to make the entropy controllable, it is important to set 
    # the normalization scale to 10 (not 1), otherwise Adam cannot decrease 
    # entropy
    log_potentials = self.normalize_potentials(log_potentials)
    ent_approx = F.softmax(log_potentials, -1)
    ent_approx = tmu.entropy(ent_approx)
    return ent_approx