
"""A implementation of Linear-chain CRF inference algorithms, including:

* Viterbi, relaxed Viterbi
* Perturb and MAP sampling, and its relaxed version 
* Forward algorithm
* Entropy 
* Forward Filtering Backward Sampling, and it Gumbelized version

"""

import torch
import copy

import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from .. import torch_model_utils as tmu

class LinearChainCRF(nn.Module):
  """Implemention of the linear chain CRF, since we only need the forward, 
  relaxed sampling, and entropy here, we emit other inference algorithms like 
  forward backward, evaluation, and viterbi"""

  def __init__(self):
    super(LinearChainCRF, self).__init__()
    return 

  def combine_potentials(self, transition, emission):
    """Mix the transition and emission scores

    Args:
      transition: 
      emission_potentials: torch.Tensor(float), 
        size=[batch, max_len, num_state]

    Returns:
      scores: size=[batch, len, num_state, num_state]
        scores := log phi(batch, x_t, y_{t-1}, y_t)
    """
    batch_size = emission.size(0)
    seq_len = emission.size(1)
    num_state = emission.size(2)

    # scores[batch, t, C, C] = log_potential(t, from y_{t-1}, to y_t)
    if(len(transition.size()) == 2):
      log_potentials = transition.view(1, 1, num_state, num_state)\
        .expand(batch_size, seq_len, num_state, num_state) + \
        emission.view(batch_size, seq_len, 1, num_state)\
        .expand(batch_size, seq_len, num_state, num_state)
    else: 
      log_potentials = transition + \
        emission.view(batch_size, seq_len, 1, num_state)\
        .expand(batch_size, seq_len, num_state, num_state)
    return log_potentials

  def forward_topk(self, 
                   state_matrix, 
                   emission_potentials, 
                   seq_lens, 
                   sum_size):
    device = emission_potentials.device
    batch_size = emission_potentials.size(0)
    max_len = emission_potentials.size(1)

    ## get proposal distribution from emission potential 
    proposal_p = F.softmax(emission_potentials, -1) 

    ## Get topk for exact marginalization 
    # [B, T, top_K]
    _, sum_index = torch.topk(proposal_p, sum_size, dim=-1)
    sum_emission = tmu.batch_index_select(
      emission_potentials.view(batch_size * max_len, -1), 
      sum_index.view(batch_size * max_len, -1)
      ).view(batch_size, max_len, -1)

    ## get renormalized proposal distribution 
    # proposal_renorm = tmu.batch_index_fill(
    #   proposal_p.view(batch_size * max_len, -1), 
    #   sum_index.view(batch_size * max_len, -1),
    #   0).view(batch_size, max_len, -1)
    # proposal_renorm /= proposal_renorm.sum(-1, keepdim=True)

    ## sample from the proposal. use one single sample for now 
    # [B, T, 1]
    # TODO: sample from multinomial; add transition prior to proposal 
    # sample_proposal_dist = Categorical(probs=proposal_renorm)
    # sampled_index = sample_proposal_dist.sample().to(device)
      
    # sampled_emission = tmu.batch_index_select(
    #   emission_potentials.view(batch_size * max_len, -1), 
    #   sampled_index.view(batch_size * max_len, -1)
    #   ).view(batch_size, max_len, sample_size)

    ##  debias sampled emission
    # sample_log_prob = sample_proposal_dist.log_prob(sampled_index)
    # sample_log_prob = sample_log_prob.view(batch_size, max_len, sample_size)
    # sampled_emission -= sample_log_prob

    ## Combine the emission 
    # [B, T, top_k + sample_K]
    combined_emission = sum_emission
    # torch.cat([sum_emission, sampled_emission], dim=-1) 
    # combined_emission = torch.cat([sum_emission, sampled_emission], dim=-1) 
    # [B, T, top_k + sample_K]
    combined_index = sum_index
    # combined_index = torch.cat([sum_index, sampled_index.unsqueeze(2)], dim=-1) 
    num_state_sampled = sum_size
    # num_state_sampled = sum_size + sample_size
    

    ## get the transition 
    # [B, T, top_k + sample_K, state_size]
    state_size = state_matrix.size(1)
    sampled_transition = torch.index_select(
      state_matrix, 0, combined_index.view(-1))\
      .view(batch_size, max_len, num_state_sampled, -1)
    # print(sampled_transition.size())
    # [B, T - 1, top_k + sample_K, top_k + sample_K]
    # sampled_transition = torch.matmul(sampled_transition[:, :-1], 
    #   sampled_transition[:, 1:].transpose(2, 3))  / np.sqrt(state_size)\
    #   + transition_bias
    sampled_transition = torch.matmul(sampled_transition[:, :-1], 
      sampled_transition[:, 1:].transpose(2, 3))
    # sampled_transition = sampled_transition + transition_bias

    # combine transition and emission 
    # [B, T, from_state, to_state]
    
    log_potentials = torch.zeros(
      batch_size, max_len, num_state_sampled, num_state_sampled).to(device)
    log_potentials[:, 1:] =\
      sampled_transition + combined_emission[:, 1:].unsqueeze(2)

    _, log_Z_est = self.forward_sum(
      None, combined_emission, seq_lens, log_potentials)
    return log_Z_est


  def sample_states(self,
                   state_matrix, 
                   emission_potentials, 
                   seq_lens, 
                   sum_size, 
                   proposal='softmax',
                   transition_proposal='none',
                   sample_size=1,):
    """Sample states for
    * Randomized Forward
    * Randomized FFBS

    Args:
    """


    device = emission_potentials.device
    batch_size = emission_potentials.size(0)
    max_len = emission_potentials.size(1)

    ## get proposal distribution from emission potential 
    proposal_p = F.softmax(emission_potentials, -1) 

    # use transition as prior 
    if(transition_proposal == 'none'):
      prior = 0.
    elif(transition_proposal == 'prod'):
      prior = torch.matmul(
        state_matrix.detach(), state_matrix.detach().transpose(1, 0))
      prior = F.softmax(prior, -1).mean(0)
      proposal_p += prior.view(1, 1, -1)
    elif(transition_proposal == 'abs_sum'):
      prior = F.softmax(state_matrix.abs().sum(-1), dim=-1)
      proposal_p += prior.view(1, 1, -1)
    elif(transition_proposal == 'prod_1'):
      prior = torch.matmul(
        state_matrix.detach(), state_matrix.detach().transpose(1, 0))
      prior = F.softmax(prior.mean(-1))
      proposal_p += prior.view(1, 1, -1)
    else: 
      raise NotImplementedError(
        'transition proposal %s not implemented' % transition_proposal)

    ## Get topk for exact marginalization 
    # [B, T, top_K]
    _, sum_index = torch.topk(proposal_p, sum_size, dim=-1)
    sum_emission = tmu.batch_index_select(
      emission_potentials.view(batch_size * max_len, -1), 
      sum_index.view(batch_size * max_len, -1)
      ).view(batch_size, max_len, -1)

    ## get renormalized proposal distribution 
    if(proposal == 'softmax'):
      pass
    elif(proposal == 'uniform'):
      proposal_p = 1. + torch.zeros_like(proposal_p)
    else:
      raise NotImplementedError('proposal %s not implemented' % proposal)
    
    # [B * T, V]
    proposal_renorm = tmu.batch_index_fill(
      proposal_p.view(batch_size * max_len, -1), 
      sum_index.view(batch_size * max_len, -1),
      1e-7)
    proposal_renorm /= proposal_renorm.sum(-1, keepdim=True)

    ## sample from the proposal. use one single sample for now 
    # [B * T, 1]
    # TODO: sample from multinomial; add transition prior to proposal 
    # sample_proposal_dist = Categorical(probs=proposal_renorm)
    sampled_index = torch.multinomial(proposal_renorm, sample_size)
    # [B * T, 1]
    sample_log_prob = tmu.batch_index_select(proposal_renorm, sampled_index)
    sample_log_prob = (sample_log_prob + 1e-8).log()
    sampled_emission = tmu.batch_index_select(
      emission_potentials.view(batch_size * max_len, -1), 
      sampled_index.view(batch_size * max_len, -1)
      ).view(batch_size, max_len, sample_size)

    ##  debias sampled emission
    sampled_index = sampled_index.view(batch_size, max_len, sample_size)
    sample_log_prob = sample_log_prob.view(batch_size, max_len, sample_size)
    sampled_emission -= sample_log_prob

    ## Combine the emission 
    # [B, T, top_k + sample_K]
    combined_emission = torch.cat([sum_emission, sampled_emission], dim=-1) 
    # [B, T, top_k + sample_K]
    combined_index = torch.cat([sum_index, sampled_index], dim=-1) 
    num_state_sampled = sum_size + sample_size
    

    ## get the transition 
    # [B, T, top_k + sample_K, state_size]
    state_size = state_matrix.size(1)
    sampled_states =  torch.index_select(
      state_matrix, 0, combined_index.view(-1))\
      .view(batch_size, max_len, num_state_sampled, -1)
    sampled_transition = sampled_states
    # print(sampled_transition.size())

    # NOTEL: alternative: do normalization.
    # [B, T - 1, top_k + sample_K, top_k + sample_K]
    # sampled_transition = torch.matmul(sampled_transition[:, :-1], 
    #   sampled_transition[:, 1:].transpose(2, 3))  / np.sqrt(state_size)\
    #   + transition_bias

    # NOTE: currently do not do bias, and do not normalize. Assume already 
    # normlized as unit vectors
    sampled_transition = torch.matmul(sampled_transition[:, :-1], 
      sampled_transition[:, 1:].transpose(2, 3))
    # sampled_transition = sampled_transition + transition_bias

    # combine transition and emission 
    # [B, T, from_state, to_state]
    
    log_potentials = torch.zeros(
      batch_size, max_len, num_state_sampled, num_state_sampled).to(device)
    log_potentials[:, 1:] =\
      sampled_transition + combined_emission[:, 1:].unsqueeze(2)
    return (combined_index, combined_emission, sampled_states, log_potentials, 
      proposal_renorm, sample_log_prob)

  def forward_approx(self, 
                     state_matrix, 
                     emission_potentials, 
                     seq_lens, 
                     sum_size, 
                     proposal='softmax',
                     transition_proposal='none',
                     sample_size=1
                     ):
    """
    Args:
      state_matrix: [num_state, state_emb_size]
      emission_potentials: [batch, seq_len, num_state]
    """
    (sampled_index, combined_emission, sampled_states, log_potentials, 
    proposal_renorm, sample_log_prob) = self.sample_states(
      state_matrix, emission_potentials, seq_lens, sum_size, proposal,
      transition_proposal, sample_size)
    alpha, log_Z_est = self.forward_sum(
      None, combined_emission, seq_lens, log_potentials)
    return (log_Z_est, sampled_index, sampled_states, combined_emission, alpha, 
      proposal_renorm, sample_log_prob)

  def forward_sum(self, transition_potentials, emission_potentials, seq_lens, 
    log_potentials=None):
    """The forward algorithm
    
    score = log(potential)

    Args:
      emission_potentials: size=[batch, max_len, num_state]
      seq_lens: size=[batch]

    Returns:
      alpha: size=[batch, max_len, num_state]
      log_Z: size=[batch]
    """
    device = emission_potentials.device

    if(log_potentials is None):
      log_potentials = self.combine_potentials(
        transition_potentials, emission_potentials)

    batch_size = log_potentials.size(0)
    seq_len = log_potentials.size(1)
    num_state = log_potentials.size(2)
    alpha = torch.zeros(batch_size, seq_len, num_state).to(device)

    # initialize the alpha with first emission
    alpha[:, 0, :] = emission_potentials[:, 0, :]

    for word_idx in range(1, seq_len):
      # batch_size, num_state, num_state
      before_log_sum_exp = alpha[:, word_idx - 1, :]\
        .view(batch_size, num_state, 1)\
        .expand(batch_size, num_state, num_state)\
        + log_potentials[:, word_idx, :, :]
      alpha[:, word_idx, :] = torch.logsumexp(before_log_sum_exp, 1)

    # batch_size x num_state
    last_alpha = tmu.batch_gather_last(alpha, seq_lens)
    log_Z = torch.logsumexp(last_alpha, -1)
    return alpha, log_Z

  def backward_sum(self, emission_potentials, seq_lens):
    """backward algorithm
    
    Args:
      emission_potentials: size=[batch, max_len, num_state]
      seq_lens: size=[batch]

    Returns:
      beta: size=[batch, max_len, num_state]
    """
    device = emission_potentials.device
    log_potentials = self.calculate_log_potentials(emission_potentials)

    batch_size = log_potentials.size(0)
    seq_len = log_potentials.size(1)

    # beta[T] initialized as 0
    beta = torch.zeros(batch_size, seq_len, self.num_state).to(device)

    # beta stored in reverse order
    # all score at i: phi(from class at L - i - 1, to class at L - i)
    log_potentials = tmu.reverse_sequence(log_potentials, seq_lens)
    for word_idx in range(1, seq_len):
      # beta[t + 1]: batch_size, t + 1, to num_state
      # indexing tricky here !! and different than the forward algo
      beta_t_ = beta[:, word_idx - 1, :]\
        .view(batch_size, 1, self.num_state)\
        .expand(batch_size, self.num_state, self.num_state)\

      # log_potentials[t]: batch, from_state t-1, to state t
      before_log_sum_exp = beta_t_ + log_potentials[:, word_idx - 1, :, :]
      beta[:, word_idx, :] = torch.logsumexp(before_log_sum_exp, 2)

    # reverse beta:
    beta = tmu.reverse_sequence(beta, seq_lens)
    # set the first beta to emission
    # beta[:, 0] = emission_potentials[:, 0]
    return beta

  def seq_log_potential(self, 
    seq, transition_potentials, emission_potentials, seq_lens):
    """Evaluate the probability of a sequence
    
    Args:
      seq: int tensor, [batch, max_len]
      transition_potentials: float tensor, [num_state, num_state]
      emission_potentials: float tensor, [batch, max_len, num_state]
      seq_lens: int tensor, [batch]

    Returns:
      seq_log_potential: float tensor, [batch]
    """
    device = emission_potentials.device
    max_len = seq.size(1)
    batch_size = seq.size(0)
    num_state = emission_potentials.size(2)
    score = torch.zeros(batch_size, max_len).to(device)
    
    for i in range(max_len):
      if(i == 0):
        score[:, i] += tmu.batch_index_select(
          emission_potentials[:, 0], # [batch, num_state]
          seq[:, 0]) # [batch]
      else: 
        transition_ = transition_potentials.view(1, num_state, num_state)
        transition_ = transition_.repeat(batch_size, 1, 1)
        prev_ind = seq[:, i - 1] # [batch] 
        current_ind = seq[:, i] # [batch]
        # select prev index
        transition_ = tmu.batch_index_select(transition_, prev_ind)
        # select current index
        transition_ = tmu.batch_index_select(transition_, current_ind)
        score[:, i] += transition_
        score[:, i] += tmu.batch_index_select(emission_potentials[:, i], current_ind)

    score = tmu.mask_by_length(score, seq_lens)
    seq_log_potential = score.sum(-1)
    return seq_log_potential

  # def marginal(self, seq, emission_potentials, seq_lens):
  #   """Marginal distribution with conventional forward-backward

  #   TODO: an autograd based implementation 

  #   Args:
  #     seq: size=[batch, max_len]
  #     emission_potentials: size=[batch, max_len, num_state]
  #     seq_lens: size=[batch]

  #   Returns:
  #     log_marginal: size=[batch, max_len]
  #   """
  #   alpha, log_Z = self.forward_score(emission_potentials, seq_lens)
  #   beta = self.backward_score(emission_potentials, seq_lens)

  #   # select weight according to the index to be evaluated
  #   batch_size = seq.size(0)
  #   max_len = seq.size(1)
  #   alpha_ = alpha.view(batch_size * max_len, -1)
  #   alpha_ = tmu.batch_index_select(alpha_, seq.view(-1))
  #   alpha_ = alpha_.view(batch_size, max_len)
  #   beta_ = beta.view(batch_size * max_len, -1)
  #   beta_ = tmu.batch_index_select(beta_, seq.view(-1))
  #   beta_ = beta_.view(batch_size, max_len)
  #   log_marginal = alpha_ + beta_ - log_Z.unsqueeze(1)
  #   return log_marginal

  def argmax(self, transition_potentials, emission_potentials, seq_lens):
    """Viterbi decoding.

    Args:
      transition_potentials: float tensor, [num_state, num_state]
      emission_potentials: float tensor, [batch, max_len, num_state]
      seq_lens: int tensor, [batch]

    Returns:
      y: int tensor, [batch, max_len]
      y_potential: float tensor, [batch]
    """
    device = emission_potentials.device

    log_potentials = self.combine_potentials(
        transition_potentials, emission_potentials)
    batch_size = log_potentials.size(0)
    seq_len = log_potentials.size(1)
    num_state = log_potentials.size(2)

    s = torch.zeros(batch_size, seq_len, num_state).to(device)
    s[:, 0] = emission_potentials[:, 0] 

    # [B, T, from C, to C]
    bp = torch.zeros(batch_size, seq_len, num_state).to(device)
    
    # forward viterbi
    for t in range(1, seq_len):
      s_ = s[:, t - 1].unsqueeze(2) + log_potentials[:, t] # [B, from C, to C]
      s[:, t] = s_.max(dim=1)[0] # [B, C]
      bp[:, t] = s_.argmax(dim=1) # [B, C]

    # backtracking
    s = tmu.reverse_sequence(s, seq_lens)
    bp = tmu.reverse_sequence(bp, seq_lens)
    y = torch.zeros(batch_size, seq_len).to(device).long()
    y[:, 0] = s[:, 0].argmax(dim=1)
    y_potential = s[:, 0].max(dim=1)
    for t in range(1, seq_len):
      y_ = y[:, t-1] # [B]
      y[:, t] = tmu.batch_index_select(bp[:, t-1], y_)
    y = tmu.reverse_sequence(y, seq_lens)
    return y, y_potential

  def rargmax(self, transition_potentials, emission_potentials, seq_lens, tau=1.0):
    """Relaxed Argmax from the CRF, Viterbi decoding.

    Everything is the same with pmsample except not using the gumbel noise

    Args:
      emission_potentials: type=torch.tensor(float), 
        size=[batch, max_len, num_state]
      seq_lens: type=torch.tensor(int), size=[batch]

    Returns:
      y_hard: size=[batch, max_len]
      y: size=[batch, max_len, num_state]
    """
    device = emission_potentials.device

    log_potentials = self.combine_potentials(transition_potentials, emission_potentials)
    batch_size = log_potentials.size(0)
    seq_len = log_potentials.size(1)
    num_state = log_potentials.size(2)

    s = torch.zeros(batch_size, seq_len, num_state).to(device)
    s[:, 0] = emission_potentials[:, 0] 

    # [B, T, from C, to C]
    bp = torch.zeros(batch_size, seq_len, num_state, num_state)
    bp = bp.to(device)
    
    # forward viterbi
    for t in range(1, seq_len):
      s_ = s[:, t - 1].unsqueeze(2) + log_potentials[:, t] # [B, from C, to C]
      s[:, t] = s_.max(dim=1)[0] # [B, C]
      bp[:, t] = torch.softmax(s_ / tau, dim=1)

    # backtracking
    s = tmu.reverse_sequence(s, seq_lens)
    bp = tmu.reverse_sequence(bp, seq_lens)
    y = torch.zeros(batch_size, seq_len, num_state).to(device)
    y[:, 0] = torch.softmax(s[:, 0] / tau, dim=1)
    s = s[:, 0].max(dim=1)
    for t in range(1, seq_len):
      y_ = y[:, t-1].argmax(dim=1) # [B]
      y[:, t] = tmu.batch_index_select(
        bp[:, t-1].transpose(1, 2), # [B, to C, from C]
        y_ 
        )
    y = tmu.reverse_sequence(y, seq_lens)
    y_hard = y.argmax(dim=2)
    return y_hard, y, s

  def rsample_approx(self, state_matrix, emission_potentials, seq_lens, 
    sum_size, proposal, transition_proposal='none', sample_size=1, tau=1.0, 
    return_ent=False):
    """Sampled-forward + Gumbel-FFBS

    Args:

    Returns:
    """

    batch_size = emission_potentials.size(0)
    max_len = emission_potentials.size(1)

    (combined_index, combined_emission, sampled_states, log_potentials, 
      proposal_renorm, sample_log_prob) =\
      self.sample_states(state_matrix, 
                        emission_potentials, 
                        seq_lens, 
                        sum_size, 
                        proposal,
                        transition_proposal,
                        sample_size)
    # compute entropy as side product
    if(return_ent):
      H = self.entropy(None, combined_emission, seq_lens, log_potentials)
    
    sample, relaxed_sample, sample_log_prob, _ = self.rsample(
      None, combined_emission, seq_lens, log_potentials, tau, return_prob=True)
    relaxed_sample_emb = torch.einsum(
      'ijk,ijkl->ijl', relaxed_sample, sampled_states)

    sum_sample_size = combined_index.size(2)
    sample_origin = tmu.batch_index_select(
      combined_index.view(batch_size * max_len, sum_sample_size),
      sample.view(batch_size * max_len)
      ).view(batch_size, max_len)

    ret = [combined_index, sample, relaxed_sample, sample_origin, 
      relaxed_sample_emb, sample_log_prob]
    if(return_ent): ret.append(H)
    return ret 

  def rsample(self, transition_potentials, emission_potentials, seq_lens, 
    log_potentials=None, tau=1.0, return_prob=False):
    """Reparameterized CRF sampling, a Gumbelized version of the 
    Forward-Filtering Backward-Sampling algorithm

    TODO: an autograd based implementation 
    requires to redefine the backward function over a relaxed-sampling semiring
    
    Args:
      emission_potentials: type=torch.tensor(float), 
        size=[batch, max_len, num_state]
      seq_lens: type=torch.tensor(int), size=[batch]
      tau: type=float, anneal strength

    Returns
      sample: size=[batch, max_len]
      relaxed_sample: size=[batch, max_len, num_state]
    """
    # Algo 2 line 1
    if(log_potentials is None):
      log_potentials = self.combine_potentials(
        transition_potentials, emission_potentials)
    alpha, log_Z = self.forward_sum(
      None, emission_potentials, seq_lens, log_potentials) 

    batch_size = emission_potentials.size(0)
    max_len = emission_potentials.size(1)
    num_state = emission_potentials.size(2)
    device = emission_potentials.device

    # Backward sampling start
    # The sampling still goes backward, but for simple implementation we
    # reverse the sequence, so in the code it still goes from 1 to T 
    relaxed_sample_rev = torch.zeros(batch_size, max_len, num_state).to(device)
    sample_prob = torch.zeros(batch_size, max_len).to(device)
    sample_rev = torch.zeros(batch_size, max_len).type(torch.long).to(device)
    alpha_rev = tmu.reverse_sequence(alpha, seq_lens).to(device)
    log_potentials_rev = tmu.reverse_sequence(log_potentials, seq_lens).to(device)
    
    # Algo 2 line 3, log space
    # w.shape=[batch, num_state]
    w = alpha_rev[:, 0, :].clone()
    w -= log_Z.view(batch_size, -1)
    p = w.exp()
    # switching regularization for longer chunk, not mentioned in the paper
    # so do no need to care. In the future this will be updated with posterior
    # regularization
    # if(return_switching): 
    #   switching = 0.
    
    # Algo 2 line 4
    relaxed_sample_rev[:, 0] = tmu.reparameterize_gumbel(w, tau)
    # Algo 2 line 5
    sample_rev[:, 0] = relaxed_sample_rev[:, 0].argmax(dim=-1)
    sample_prob[:, 0] = tmu.batch_index_select(p, sample_rev[:, 0]).flatten()
    mask = tmu.length_to_mask(seq_lens, max_len).type(torch.float)
    prev_p = p
    for i in range(1, max_len):
      # y_after_to_current[j, k] = log_potential(y_{t - 1} = k, y_t = j, x_t)
      # size=[batch, num_state, num_state]
      y_after_to_current = log_potentials_rev[:, i-1].transpose(1, 2)
      # w.size=[batch, num_state]
      w = tmu.batch_index_select(y_after_to_current, sample_rev[:, i-1])
      w_base = tmu.batch_index_select(alpha_rev[:, i-1], sample_rev[:, i-1])
      # Algo 2 line 7, log space
      w = w + alpha_rev[:, i] - w_base.view(batch_size, 1)
      p = F.softmax(w, dim=-1) # p correspond to pi in the paper
      # if(return_switching):
      #   switching += (tmu.js_divergence(p, prev_p) * mask[:, i]).sum()
      prev_p = p
      # Algo 2 line 8
      relaxed_sample_rev[:, i] = tmu.reparameterize_gumbel(w, tau)
      # Algo 2 line 9
      sample_rev[:, i] = relaxed_sample_rev[:, i].argmax(dim=-1)
      sample_prob[:, i] = tmu.batch_index_select(p, sample_rev[:, i]).flatten()

    # Reverse the sequence back
    sample = tmu.reverse_sequence(sample_rev, seq_lens)
    relaxed_sample = tmu.reverse_sequence(relaxed_sample_rev, seq_lens)
    sample_prob = tmu.reverse_sequence(sample_prob, seq_lens)
    sample_prob = sample_prob.masked_fill(mask == 0, 1.)
    sample_log_prob_stepwise = (sample_prob + 1e-10).log()
    sample_log_prob = sample_log_prob_stepwise.sum(dim=1)

    ret = [sample, relaxed_sample]
    # if(return_switching): 
    #   switching /= (mask.sum(dim=-1) - 1).sum()
    #   ret.append(switching)
    if(return_prob):
      ret.extend([sample_log_prob, sample_log_prob_stepwise])
    return ret

  def entropy_approx(self, 
                 state_matrix, 
                 emission_potentials, 
                 seq_lens, 
                 sum_size, 
                 proposal='softmax',
                 transition_proposal='none',
                 sample_size=1
                 ):
    """Approx entropy. Also based on sampled forward -- yet this one does not quite work"""
    _, combined_emission, _, log_potentials =\
      self.sample_states(state_matrix, 
                         emission_potentials, 
                         seq_lens, 
                         sum_size, 
                         proposal,
                         transition_proposal,
                         sample_size)

    H = self.entropy(None, combined_emission, seq_lens, log_potentials)
    return H

  def entropy(self, transition_potentials, emission_potentials, seq_lens, 
    log_potentials=None):
    """The entropy of the CRF, another DP algorithm. See the write up
    
    Args:
      emission_potentials:
      seq_lens:

    Returns:
      H_total: the entropy, type=torch.Tensor(float), size=[batch]
    """
    if(log_potentials is None):
      log_potentials = self.combine_potentials(
        transition_potentials, emission_potentials)

    alpha, log_Z = self.forward_sum(
      None, emission_potentials, seq_lens, log_potentials)

    batch_size = emission_potentials.size(0)
    max_len = emission_potentials.size(1)
    num_state = emission_potentials.size(2)
    device = emission_potentials.device

    H = torch.zeros(batch_size, max_len, num_state).to(device)
    for t in range(max_len - 1):
      # log_w.shape = [batch, from_class, to_class]
      log_w = log_potentials[:, t+1, :, :] +\
        alpha[:, t, :].view(batch_size, num_state, 1) -\
        alpha[:, t+1, :].view(batch_size, 1, num_state)
      w = log_w.exp()
      H[:, t+1, :] = torch.sum(
        w * (H[:, t, :].view(batch_size, num_state, 1) - log_w), dim=1)
    
    last_alpha = tmu.batch_gather_last(alpha, seq_lens)
    H_last = tmu.batch_gather_last(H, seq_lens)
    log_p_T = last_alpha - log_Z.view(batch_size, 1)
    p_T = log_p_T.exp()

    H_total = p_T * (H_last - log_p_T)
    H_total = H_total.sum(dim = -1)
    return H_total