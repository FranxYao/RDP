"""Recovering latent network structure from pretrained embeddings
"""

import torch 
import nltk
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np 

from tqdm import tqdm
from json import encoder
from torch import nn 
from torch.distributions import Uniform, Categorical
from torch.optim import Adam, SGD

from nltk.corpus import stopwords
from collections import Counter
# from transformers import BertModel, AdamW, get_constant_schedule_with_warmup

from transformers import GPT2Tokenizer, GPT2Model, AdamW
# Use adabelief, see if it SGD property helps optimization
# TODO: test it in the BertNet model
# from adabelief_pytorch import AdaBelief

from frtorch import FRModel, LinearChainCRF, LSTMDecoder, Attention
from frtorch import torch_model_utils as tmu
from nltk.translate.bleu_score import corpus_bleu


class GPT2NetModel(nn.Module):
  """Use scaled CRF to recover latent network structure from contextualized 
  embeddings, GPT2 version
  """
  def __init__(self,
               num_state=20,
               state_size=768,
               transition_init_scale=0.01,
               exact_rsample=False, 
               latent_scale=0.1, 
               sum_size=10,
               sample_size=10,
               proposal='softmax',
               transition_proposal='none',
               device='cpu',
               vocab_size=-1, 
               pad_id=-1,
               bos_id=-1, 
               max_dec_len=-1,
               crf_weight_norm='none', # do not change this
               latent_type='sampled_gumbel_crf', 
               ent_approx='softmax', # TODO: regularize transition
               use_bow=False,
               use_copy=True,
               task='', 
              #  use_bow_loss=False,
               word_dropout_decay=False,
               dropout=0.0,
               potential_normalization='none',
               potential_scale=1.0,
               n_sample_infer=100,
               mask_z=False,
               z_st=False,
               topk_sum=False,
               cache_dir=''
               ):
    """
    GPT2Net init
    """
    super(GPT2NetModel, self).__init__()

    # self.loss_type = loss_type
    self.state_size = state_size
    self.exact_rsample = exact_rsample
    self.latent_scale = latent_scale
    self.sum_size = sum_size
    self.proposal = proposal
    self.transition_proposal = transition_proposal
    self.sample_size = sample_size
    self.device = device
    self.num_state = num_state
    self.vocab_size = vocab_size
    self.crf_weight_norm = crf_weight_norm
    self.latent_type = latent_type
    self.ent_approx = ent_approx
    self.task = task
    self.use_bow = use_bow
    self.use_copy = use_copy
    # self.use_bow_loss = use_bow_loss
    self.word_dropout_decay = word_dropout_decay
    self.pad_id = pad_id
    self.bos_id = bos_id
    self.max_dec_len = max_dec_len
    self.n_sample_infer = n_sample_infer

    self.mask_z = mask_z
    self.z_st = z_st
    self.topk_sum = topk_sum
    
    if(cache_dir == ''):
      self.encoder = GPT2Model.from_pretrained('gpt2')
    else:
      self.encoder = GPT2Model.from_pretrained('gpt2', cache_dir=cache_dir)

    # do not update GPT model
    for param in self.encoder.parameters():
      param.requires_grad = False

    self.state_matrix = nn.Parameter(torch.normal(
      size=[num_state, state_size], mean=0.0, std=transition_init_scale))
    self.crf = LinearChainCRF(potential_normalization, potential_scale)

    self.embeddings = nn.Embedding(vocab_size, state_size)
    self.p_init_proj_h = nn.Linear(state_size, state_size)
    self.p_init_proj_c = nn.Linear(state_size, state_size)
    self.decoder = LSTMDecoder(vocab_size=vocab_size, 
                               state_size=state_size, 
                               embedding_size=state_size,
                               dropout=dropout,
                               use_attn=False,
                               use_hetero_attn=False
                               )
    self.p_copy_attn = Attention(state_size, state_size, state_size)
    self.p_copy_g = nn.Linear(state_size, 1)
    self.p_z_proj = nn.Linear(state_size, num_state)
    self.p_z_intermediate = nn.Linear(2 * state_size, state_size)
    return 

  def get_transition(self):
    """Return transition matrix"""
    transition = torch.matmul(
      self.state_matrix, self.state_matrix.transpose(1, 0))
    return self.state_matrix, transition

  def weight_norm(self, x_emb):
    """Restrict the latent embeddings within a subspace
    
    Args:
      x_emb: size=[batch, max_len, dim]

    Returns: 
      state_matrix: size=[state, dim]
      emission_seq: size=[batch, max_len, dim]
      transition: size=[state, state] # could be large
      emission: size=[batch, max_len, state]
    """
    if(self.crf_weight_norm == 'sphere'):
      emission_seq = x_emb / torch.sqrt((x_emb ** 2).sum(-1, keepdim=True))
      emission_seq = self.latent_scale * emission_seq
      state_matrix = self.state_matrix /\
        torch.sqrt((self.state_matrix ** 2).sum(-1, keepdim=True))
      state_matrix = self.latent_scale * state_matrix
    elif(self.crf_weight_norm == 'zscore'):
      raise NotImplementedError('z score normalization not implemented!')
    elif(self.crf_weight_norm == 'none'):
      emission_seq = x_emb
      state_matrix = self.state_matrix 
    else:
      raise ValueError('Invalid crf_weight_norm: %s' % self.crf_weight_norm)

    transition = torch.matmul(state_matrix, state_matrix.transpose(1, 0))
    emission = torch.matmul(emission_seq, state_matrix.transpose(1, 0))
    return state_matrix, emission_seq, transition, emission

  def prepare_dec_io(self, 
    z_sample_ids, z_sample_emb, sentences, bow_emb=None, x_lambd=0):
    """Prepare the decoder output g based on the inferred z from the CRF 
    NOTE: the index here is different than the BERT model

    Args:
      z_sample_ids: sampled z index. size=[batch, max_len]
      z_sample_emb: embedding of gumbel sampled z. size=[batch, max_len, dim]
      sentences: size=[batch, max_len]
      x_lambd: word dropout ratio. 1 = all dropped

    Returns:
      dec_inputs: size=[batch, max_len, state_size]
      dec_targets_x: size=[batch, max_len]
      dec_targets_z: size=[batch, max_len]
    """
    batch_size = sentences.size(0)
    max_len = sentences.size(1) - 1
    device = sentences.device

    sent_emb = self.embeddings(sentences[:, :-1])

    # word dropout ratio = x_lambd. 0 = no dropout, 1 = all drop out
    m = Uniform(0., 1.)
    mask = m.sample([batch_size, max_len]).to(device)
    mask = (mask > x_lambd).float().unsqueeze(2)

    if(self.mask_z): z_sample_emb *= 0

    if(bow_emb is not None): 
      max_bow_len = bow_emb.size(1)
      mask_bow = m.sample([batch_size, max_bow_len]).to(device)
      mask_bow = (mask_bow > x_lambd).float().unsqueeze(2)
      # print(bow_emb.size())
      # print(mask_bow.size())

    if(self.word_dropout_decay):
      dec_inputs = sent_emb * mask * (1 - x_lambd)
      dec_inputs[:, 1:] += z_sample_emb[:, :-1]
      if(bow_emb is not None): bow_emb = bow_emb * mask_bow * (1 - x_lambd)
    else: 
      dec_inputs = sent_emb * mask
      dec_inputs[:, 1:] += z_sample_emb[:, :-1]
      if(bow_emb is not None): bow_emb = bow_emb * mask_bow

    dec_targets_x = sentences[:, 1:]
    dec_targets_z = z_sample_ids
    if(self.mask_z): dec_targets_z *= 0
    return dec_inputs, dec_targets_x, dec_targets_z, bow_emb

  def decode_state_init(self, bow_emb, bow_mask, batch_size, state_size, device):
    """Initialize decoder state to BOW or zero"""
    if(self.use_bow):
      state = (bow_emb * bow_mask.unsqueeze(-1)).sum(1)/\
        bow_mask.sum(-1, keepdim=True)
      batch_size, state_dim = state.size()
      state_h = self.p_init_proj_h(state).view(1, batch_size, state_size)
      state_c = self.p_init_proj_c(state).view(1, batch_size, state_size)
      state = (state_h, state_c)
    else: 
      state = (
        torch.zeros(self.decoder.lstm_layers, batch_size, state_size).to(device), 
        torch.zeros(self.decoder.lstm_layers, batch_size, state_size).to(device))
    return state

  def decode_train(self, 
    dec_inputs, z_sample_emb, dec_targets_x, dec_targets_z, x_lens, 
    bow, bow_emb, bow_mask):
    """
    NOTE: the index here is different than the BERT model

    Args:
      dec_inputs: size=[batch, max_len, dim]
      z_sample_emb: size=[batch, max_len, dim]
      dec_target_x: size=[batch, max_len]
      dec_target_z: size=[batch, max_len]
      x_lens: size=[batch]

    Returns:
      log_prob: size=[batch, max_len]
      log_prob_x: size=[batch, max_len]
      log_prob_z: size=[batch, max_len]
    """
    max_len = dec_inputs.size(1)
    batch_size = dec_inputs.size(0)
    device = dec_inputs.device
    state_dim = dec_inputs.size(-1)

    dec_cell = self.decoder

    dec_inputs = dec_inputs.transpose(1, 0)
    dec_targets_x = dec_targets_x.transpose(1, 0)
    dec_targets_z = dec_targets_z.transpose(1, 0)
    z_sample_emb = z_sample_emb.transpose(1, 0) # start from z[1]

    state = self.decode_state_init(
      bow_emb, bow_mask, batch_size, self.state_size, device)
    log_prob_x, log_prob_z = [], []
    
    for i in range(max_len):
      if(self.use_bow):
        dec_out, state = dec_cell(
          dec_inputs[i], state, memory=bow_emb, mem_mask=bow_mask)
      else:
        dec_out, state = dec_cell(dec_inputs[i], state)
      dec_out = dec_out[0]
      z_logits = self.p_z_proj(dec_out)
      log_prob_z_i = -F.cross_entropy(
        z_logits, dec_targets_z[i], reduction='none')
      log_prob_z.append(log_prob_z_i)

      dec_intermediate = self.p_z_intermediate(
        torch.cat([dec_out, z_sample_emb[i]], dim=1))
      x_logits = dec_cell.output_proj(dec_intermediate)
      lm_prob = F.softmax(x_logits, -1)

      if(self.use_copy):
        _, copy_dist = self.p_copy_attn(dec_intermediate, bow_emb, bow_mask)
        copy_prob = tmu.batch_index_put(copy_dist, bow, self.vocab_size)
        copy_g = torch.sigmoid(self.p_copy_g(dec_intermediate))
        out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob
        x_logits = (out_prob + 1e-10).log()

      log_prob_x_i = -F.cross_entropy(
        x_logits, dec_targets_x[i], reduction='none')
      log_prob_x.append(log_prob_x_i)

    log_prob_x = torch.stack(log_prob_x).transpose(1, 0) # [B, T]
    log_prob_x = tmu.mask_by_length(log_prob_x, x_lens)
    log_prob_x = log_prob_x.sum(-1) 
    log_prob_z = torch.stack(log_prob_z).transpose(1, 0)
    log_prob_z = tmu.mask_by_length(log_prob_z, x_lens)
    log_prob_z = log_prob_z.sum(-1)
    log_prob = log_prob_x + log_prob_z

    log_prob_x = log_prob_x / x_lens
    log_prob_z = log_prob_z / x_lens

    log_prob_x = log_prob_x.mean()
    log_prob_z = log_prob_z.mean()
    return log_prob, log_prob_x, log_prob_z

  def forward(self, sent, bow, 
    tau=1.0, x_lambd=0.0, z_lambd=0.0, z_beta=1.0):
    """
    NOTE: the index here is different than the BertNet model

    Args:
      sent: input sentence, size=[batch, max_len]
      bow: input attention mask, size=[batch, max_len]

    Returns:
      loss
      out_dict
    """
    out_dict = {}
    pad_id = self.pad_id
    batch_size = sent.size(0)

    x_enc = sent[:, 1:] # no start symbol
    # no end symbol because end_id = pad_id
    attention_mask = (x_enc != pad_id).float() 
    x_lens = attention_mask.sum(-1).long()

    # encoding
    x_emb = self.encoder(x_enc, attention_mask=attention_mask).last_hidden_state
    # if(self.use_latent_proj):
    #   x_emb = self.latent_proj(x_emb)

    # latent 
    if(self.latent_type == 'softmax'):
      z_logits = torch.einsum('bij,kj->bik', x_emb, self.state_matrix)
      z_sample_relaxed = tmu.reparameterize_gumbel(z_logits, tau)
      z_sample = z_sample_relaxed.argmax(dim=-1)
      z_sample_emb = torch.matmul(z_sample_relaxed, self.state_matrix)
      ent = tmu.entropy(F.softmax(z_logits, dim=-1))
    elif(self.latent_type == 'sampled_gumbel_crf'):
      state_matrix, emission_seq, transition, emission = self.weight_norm(x_emb)
      if(self.exact_rsample):
        # raise NotImplementedError('TODO: normalize transition')
        # print('using exact sample')
        z_sample, z_relaxed = self.crf.rsample(
          transition, emission, x_lens, tau=tau, z_st=self.z_st)
        z_sample_emb = torch.einsum('ijk,kl->ijl', z_relaxed, state_matrix)
        # print('cuda memory point 0:', torch.cuda.memory_allocated())
        # ent = self.crf.entropy(transition, emission, x_lens)
        # ent = ent.mean()
        ent_sofmax = tmu.entropy(F.softmax(emission, -1), keepdim=True)
        ent_sofmax = (ent_sofmax * attention_mask).sum()
        ent = ent_sofmax / attention_mask.sum()
        z_relaxed_max = z_relaxed.max(-1).values
        z_relaxed_max = tmu.mask_by_length(z_relaxed_max, x_lens)
        z_relaxed_max = z_relaxed_max.sum(-1) / x_lens.float()
        out_dict['z_relaxed_max'] = z_relaxed_max.mean().item()
      else:
        # NOTE: pay attention to the index transform here, see details in the 
        # implementation
        # if(self.ent_approx == 'log_prob'):
        #   _, _, _, z_sample, z_sample_emb, z_sample_log_prob, inspect =\
        #     self.crf.rsample_approx(state_matrix, emission, x_lens, 
        #       self.sum_size, self.proposal, tau=tau, return_ent=False) 
        #   # use log prob as single estimate of entropy 
        #   ent = -z_sample_log_prob.mean()
        # elif(self.ent_approx == 'softmax_sample'):
        #   _, _, _, z_sample, z_sample_emb, z_sample_log_prob, inspect, ent =\
        #     self.crf.rsample_approx(state_matrix, emission, x_lens, 
        #       self.sum_size, self.proposal, tau=tau, return_ent=True) 
        #   ent_sofmax = tmu.entropy(F.softmax(emission, -1), keepdim=True)
        #   ent_sofmax = (ent_sofmax * attention_mask).sum()
        #   ent_sofmax = ent_sofmax / attention_mask.sum()

        #   # TODO: report the two terms respectively 
        #   ent = ent.mean() + ent_sofmax
        if(self.ent_approx == 'softmax'):
          _, _, z_relaxed, z_sample, z_sample_emb, z_sample_log_prob, inspect =\
            self.crf.rsample_approx(state_matrix, emission, x_lens, 
              self.sum_size, self.proposal, 
              transition_proposal=self.transition_proposal,
              sample_size=self.sample_size,
              tau=tau, return_ent=False, z_st=self.z_st, topk_sum=self.topk_sum) 
          # print('cuda memory point 0:', torch.cuda.memory_allocated())
          ent_sofmax = tmu.entropy(F.softmax(emission, -1), keepdim=True)
          ent_sofmax = (ent_sofmax * attention_mask).sum()
          ent = ent_sofmax / attention_mask.sum()
          z_relaxed_max = z_relaxed.max(-1).values
          z_relaxed_max = tmu.mask_by_length(z_relaxed_max, x_lens)
          z_relaxed_max = z_relaxed_max.sum(-1) / x_lens.float()
          out_dict['z_relaxed_max'] = z_relaxed_max.mean().item()
        else: 
          raise ValueError('Invalid value ent_approx: %s' % self.ent_approx)
    else: 
      raise NotImplementedError(
        'Latent type %s not implemented!' % self.latent_type)
    
    # decoding 
    bow_emb = self.embeddings(bow)
    bow_mask = (bow != pad_id).float()
    dec_inputs, dec_targets_x, dec_targets_z, bow_emb = self.prepare_dec_io(
      z_sample, z_sample_emb, sent, bow_emb, x_lambd)

    # x_lens + 1: add one end symbol
    x_lens = x_lens + 1
    p_log_prob, p_log_prob_x, p_log_prob_z = self.decode_train(
      dec_inputs, z_sample_emb, dec_targets_x, dec_targets_z, x_lens,
      bow, bow_emb, bow_mask)

    # by default we do maximization
    loss = p_log_prob_x + z_lambd * p_log_prob_z + z_beta * ent
    obj = p_log_prob_x + p_log_prob_z + ent

    # # TODO: add bag of words loss
    # if(self.use_bow_loss):
    #   pass

    # turn maximization to minimization
    loss = -loss
    
    out_dict['loss'] = loss.item()
    # out_dict['obj'] = obj.item()
    out_dict['ent'] = ent.item()
    out_dict['p_log_prob_x'] = p_log_prob_x.item()
    out_dict['p_log_prob_z'] = p_log_prob_z.item()
    out_dict['p_log_x_z'] = p_log_prob_x.item() + p_log_prob_z.item()
    out_dict['z_sample'] = tmu.to_np(z_sample)
    out_dict['input_ids'] = tmu.to_np(sent)
    # out_dict['p_t_min'] = inspect['p_t_min'].item()
    # out_dict['p_t_max'] = inspect['p_t_max'].item()
    # out_dict['p_t_mean'] = inspect['p_t_mean'].item()
    # out_dict['p_e_min'] = inspect['p_e_min'].item()
    # out_dict['p_e_max'] = inspect['p_e_max'].item()
    # out_dict['p_e_mean'] = inspect['p_e_mean'].item()
    out_dict['x_lens'] = (x_lens).float().mean().item()
    return loss, out_dict

  def sampled_forward_est(self, sent):
    """Sampled forward"""
    pad_id = self.pad_id
    batch_size = sent.size(0)
    x_enc = sent[:, 1:]
    attention_mask = (x_enc != pad_id).float()
    x_lens = attention_mask.sum(-1).long()

    # encoding
    # print('debug: x_enc:')
    # print(x_enc)
    # print('debug: attention_mask:')
    # print(attention_mask)
    x_emb = self.encoder(x_enc, attention_mask=attention_mask).last_hidden_state
    # print('debug: x_emb:')
    # print(x_emb[0, :5, :5])

    state_matrix, emission_seq, transition, emission = self.weight_norm(x_emb)
    _, log_z_exact = self.crf.forward_sum(transition, emission, x_lens)

    crf_vars = [x_emb, state_matrix, transition, emission, x_enc, x_lens]

    # print('debug, transition:')
    # print(transition[:5, :5])
    # print('debug, emission:')
    # print(emission[:5, :5])
    # print('debug, x_lens:')
    # print(x_lens)
    ent = self.crf.entropy(transition, emission, x_lens)
    ent = ent.mean().cpu().item()
    # print('debug, ent:', ent)

    x_lens_mean = x_lens.float().mean().cpu().item()

    log_z_exact = log_z_exact[0].cpu().item()

    log_z_est_non_trans = []
    for _ in range(100):
      est = self.crf.forward_approx(state_matrix, emission, x_lens, 
        sum_size=self.sum_size, proposal='softmax', 
        transition_proposal='none', sample_size=self.sample_size)
      log_z_est_non_trans.append(est[0].cpu().item())

    log_z_est_prod = []
    for _ in range(100):
      est = self.crf.forward_approx(state_matrix, emission, x_lens, 
        sum_size=self.sum_size, proposal='softmax', 
        transition_proposal='prod', sample_size=self.sample_size)
      log_z_est_prod.append(est[0].cpu().item())

    log_z_est_norm = []
    for _ in range(100):
      est = self.crf.forward_approx(state_matrix, emission, x_lens, 
        sum_size=self.sum_size, proposal='softmax', 
        transition_proposal='l1norm', sample_size=self.sample_size)
      log_z_est_norm.append(est[0].cpu().item())
    return (ent, x_lens_mean, log_z_exact, log_z_est_non_trans, log_z_est_prod, 
    log_z_est_norm, crf_vars)

  def forward_lm(self, sent):
    """Only use the decoder as an LSTM language model"""
    out_dict = {}
    batch_size = sent.size(0)
    device = sent.device
    dec_inputs = self.embeddings(sent[:, :-1]).transpose(0, 1)
    max_len = dec_inputs.size(0)

    dec_targets = sent[:, 1:].transpose(0, 1)
    # note that sent = [start_id, x1, ..., xt, pad_id, pad_id ...]
    # so mask =                  [1,  ..., 1,  1,      0 ...]
    mask = (sent != self.pad_id).float()[:, :-1] 
    x_lens = (sent != self.pad_id).float()[:, 1:].sum(-1) + 1

    log_probs = []
    
    state = self.decode_state_init(
      None, None, batch_size, self.state_size, device)
    dec_cell = self.decoder
    for i in range(max_len):
      dec_out, state = dec_cell(dec_inputs[i], state)
      dec_out = dec_out[0]
      x_logits = dec_cell.output_proj(dec_out)
      log_p = -F.cross_entropy(x_logits, dec_targets[i], reduction='none')
      log_probs.append(log_p)

    marginal = torch.stack(log_probs).transpose(0, 1) # [B, T]
    marginal = (marginal * mask).sum(-1)
    out_dict['marginal_all'] = tmu.to_np(marginal)
    out_dict['x_lens_all'] = tmu.to_np(x_lens)
    loss = -(marginal / x_lens)
    ppl = loss.exp().mean()
    loss = loss.mean()
    marginal = marginal.mean()
    log_prob_x = -loss

    out_dict['loss'] = loss.item(),
    out_dict['marginal'] = marginal.item(),
    out_dict['x_lens'] = x_lens.mean().item(),
    out_dict['log_prob_x']: log_prob_x.item()
    return loss, out_dict

  def forward_score_fn(self):
    """Score function estimator for learning the inference network"""
    return

  def infer_marginal(self, sent, bow=None):
    """Infer marginal likelihood"""
    out_dict = {}

    pad_id = self.pad_id
    batch_size = sent.size(0)

    x_enc = sent[:, 1:]
    attention_mask = (x_enc != pad_id).float()
    x_lens = attention_mask.sum(-1).long()

    x_emb = self.encoder(x_enc, attention_mask=attention_mask).last_hidden_state
    n_sample = self.n_sample_infer

    # exact sampling 
    state_matrix, _, transition, emission = self.weight_norm(x_emb)
    ent = self.crf.entropy(transition, emission, x_lens)
    ent = ent.mean()

    z_sample_idx, q_log_probs = self.crf.ksample(
      transition, emission, x_lens, n_sample)
    z_sample_idx = z_sample_idx.view(batch_size * n_sample, -1)
    max_len = z_sample_idx.size(1)
    
    z_sample_emb = state_matrix[z_sample_idx.view(-1)]
    z_sample_emb = z_sample_emb.view(batch_size * n_sample, max_len, -1)
    sent = tmu.batch_repeat(sent, n_sample)

    if(bow is not None):
      bow = tmu.batch_repeat(bow, n_sample)
      bow_emb = self.embeddings(bow)
      bow_mask = (bow != pad_id).float()
    else:
      bow, bow_emb, bow_mask = None, None, None

    dec_inputs, dec_targets_x, dec_targets_z, _ = self.prepare_dec_io(
      z_sample_idx, z_sample_emb, sent, bow_emb, 0)
    
    # [B * n_sample]
    x_lens = x_lens + 1
    x_lens_ = tmu.batch_repeat(x_lens, n_sample)
    p_log_probs, _, _ = self.decode_train(
      dec_inputs, z_sample_emb, dec_targets_x, dec_targets_z, x_lens_, 
      bow, bow_emb, bow_mask)
    p_log_probs = p_log_probs.view(batch_size, n_sample)

    if(self.mask_z): q_log_probs *= 0
    elbo = (p_log_probs - q_log_probs).mean() # this is wrong, did not consider entropy 
    joint = p_log_probs.mean()
    marginal = torch.logsumexp(p_log_probs - q_log_probs, -1) - np.log(n_sample)
    x_lens = x_lens.float()
    # print('debug: marginal', marginal[0])
    # print('debug: -marginal / x_lens', -marginal[0] / x_lens[0])
    # print('debug: x_lens', x_lens[0])
    # print('debug: exp(-marginal / x_lens)', (-marginal[0] / x_lens[0]).exp())
    out_dict['x_lens_all'] = tmu.to_np(x_lens)
    ppl = (-marginal / x_lens).exp().mean()
    out_dict['marginal_all'] = tmu.to_np(marginal)
    marginal = marginal.mean()

    out_dict['ent'] = ent.item()
    out_dict['p_log_probs'] = p_log_probs.mean().item()
    out_dict['q_log_probs'] = q_log_probs.mean().item()
    out_dict['elbo'] = elbo.item()
    out_dict['marginal'] = marginal.item()
    out_dict['joint'] = joint.item()
    out_dict['x_lens'] = x_lens.mean().item()
    return out_dict

  def decode_x_conditional(self, bow, z_sample):
    """Generate x given z"""
    pad_id = self.pad_id
    bos_id = self.bos_id
    bow_emb = self.embeddings(bow)
    bow_mask = (bow != pad_id).float()
    batch_size = z_sample.size(0)
    max_len = z_sample.size(1)
    device = z_sample.device
    
    inp = torch.zeros(batch_size).long().to(device) + bos_id
    inp = self.embeddings(inp)
    state = self.decode_state_init(
      bow_emb, bow_mask, batch_size, self.state_size, device)
    predictions_x = []
    z_sample_emb = self.state_matrix[z_sample.view(-1)]
    z_sample_emb = z_sample_emb.view(batch_size, max_len, -1).transpose(0, 1)
    dec_cell = self.decoder
    for i in range(max_len):
      if(self.use_bow):
        dec_out, state = dec_cell(
          inp, state, memory=bow_emb, mem_mask=bow_mask)
      else:
        dec_out, state = dec_cell(inp, state)
      dec_out = dec_out[0]

      dec_intermediate = self.p_z_intermediate(
        torch.cat([dec_out, z_sample_emb[i]], dim=1))
      x_logits = dec_cell.output_proj(dec_intermediate)
      lm_prob = F.softmax(x_logits, dim=-1)
      out_prob = lm_prob

      if(self.use_copy):
        _, copy_dist = self.p_copy_attn(dec_intermediate, bow_emb, bow_mask)
        copy_prob = tmu.batch_index_put(copy_dist, bow, self.vocab_size)
        copy_g = torch.sigmoid(self.p_copy_g(dec_intermediate))
        out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob

      x_pred_i = out_prob.argmax(-1)
      predictions_x.append(x_pred_i)

      inp = self.embeddings(x_pred_i) + z_sample_emb[i]
      
    predictions_x = torch.stack(predictions_x).transpose(0, 1) # -> [B, T]
    return predictions_x

  def decode_x_and_z(self, bow, dec_method='argmax'):
    """Given BOW decode both x and z"""
    pad_id = self.pad_id
    bos_id = self.bos_id
    bow_emb = self.embeddings(bow)
    bow_mask = (bow != pad_id).float()
    max_dec_len = self.max_dec_len
    batch_size = bow.size(0)
    device = bow.device
    
    inp = torch.zeros(batch_size).long().to(device) + bos_id
    inp = self.embeddings(inp)
    state = self.decode_state_init(
      bow_emb, bow_mask, batch_size, self.state_size, device)
    predictions_x = []
    predictions_z = []
    dec_cell = self.decoder
    for i in range(max_dec_len):
      if(self.use_bow):
        dec_out, state = dec_cell(
          inp, state, memory=bow_emb, mem_mask=bow_mask)
      else:
        dec_out, state = dec_cell(inp, state)
      dec_out = dec_out[0]
      z_logits = self.p_z_proj(dec_out)
      if(dec_method == 'argmax'):
        z_pred_i = z_logits.argmax(-1)
      elif(dec_method == 'sampling'):
        z_pred_i = Categorical(logits=z_logits).sample()
      else: 
        raise NotImplementedError(
          'Decoding method %s not implemented' % dec_method)
      predictions_z.append(z_pred_i)
      z_emb_i = self.state_matrix[z_pred_i]

      dec_intermediate = self.p_z_intermediate(
        torch.cat([dec_out, z_emb_i], dim=1))
      x_logits = dec_cell.output_proj(dec_intermediate)
      lm_prob = F.softmax(x_logits, dim=-1)
      out_prob = lm_prob

      if(self.use_copy):
        _, copy_dist = self.p_copy_attn(dec_intermediate, bow_emb, bow_mask)
        copy_prob = tmu.batch_index_put(copy_dist, bow, self.vocab_size)
        copy_g = torch.sigmoid(self.p_copy_g(dec_intermediate))
        out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob

      x_pred_i = out_prob.argmax(-1)
      predictions_x.append(x_pred_i)

      inp = self.embeddings(x_pred_i) + z_emb_i

    predictions_x = torch.stack(predictions_x).transpose(0, 1)
    predictions_z = torch.stack(predictions_z).transpose(0, 1)
    return predictions_x, predictions_z

  def predict_paraphrase(self, sent, bow, sample_size=3):
    """Predict paraphrase with given sentence
    Args:
      sent: [batch, max_len]
      bow: [batch, max_bow_len]
    Outputs:
    """
    out_dict = {}

    pad_id = self.pad_id
    batch_size = sent.size(0)
    x_enc = sent[:, 1:]
    attention_mask = (x_enc != pad_id).float()
    x_lens = attention_mask.sum(-1).long()

    # # encoding
    # x_emb = self.encoder(x_enc, attention_mask=attention_mask).last_hidden_state

    # # with sampled z
    # state_matrix, emission_seq, transition, emission = self.weight_norm(x_emb)
    # z_sample, q_log_probs = self.crf.ksample(
    #   transition, emission, x_lens, sample_size)
    # z_sample = z_sample.view(batch_size * sample_size, -1)

    # # decoding 
    # predictions_x = self.decode_x_conditional(bow, z_sample)
    # predictions_x = predictions_x.view(batch_size, sample_size, -1)
    # z_sample = z_sample.view(batch_size, sample_size, -1)
    # out_dict['predictions_x_q'] = tmu.to_np(predictions_x)
    # out_dict['predictions_z_q'] = tmu.to_np(z_sample)

    # with free z decoding 
    predictions_x, predictions_z = self.decode_x_and_z(bow, dec_method='argmax')
    predictions_x = predictions_x.view(batch_size, 1, -1)
    predictions_z = predictions_z.view(batch_size, 1, -1)

    bow = tmu.batch_repeat(bow, sample_size - 1)
    predictions_x_, predictions_z_ = self.decode_x_and_z(
      bow, dec_method='sampling')
    predictions_x_ = predictions_x_.view(batch_size, sample_size - 1, -1)
    predictions_x = torch.cat([predictions_x, predictions_x_], 1)
    predictions_z_ = predictions_z_.view(batch_size, sample_size - 1, -1)
    predictions_z = torch.cat([predictions_z, predictions_z_], 1)
    out_dict['predictions_x_p'] = tmu.to_np(predictions_x)
    out_dict['predictions_z_p'] = tmu.to_np(predictions_z)
    # temperary 
    out_dict['predictions_x_q'] = tmu.to_np(predictions_x)
    out_dict['predictions_z_q'] = tmu.to_np(predictions_z)
    return out_dict

class GPT2Net(FRModel):
  """GPT2Net model, train/dev/test wrapper"""
  def __init__(self, 
               model, 
               learning_rate=1e-3, 
               validation_criteria='loss', 
               num_batch_per_epoch=-1, 
               word_dropout=True,
               x_lambd_warm_end_epoch=1,
               x_lambd_warm_n_epoch=1,
               tau_anneal_start_epoch=18,
               tau_anneal_n_epoch=3,
               tokenizer=None,
               z_beta_init=1.0,
               z_beta_final=0.01,
               anneal_beta_with_lambd=False,
               anneal_z_prob=False,
               save_mode='full',
               optimizer_type='adam',
               space_token='',
               freeze_z_at_epoch=100000,
               change_opt_to_sgd_at_epoch=100000,
               model_path=''
               ):
    """"""
    super(GPT2Net, self).__init__()

    self.model = model
    self.task = model.task
    self.device = model.device
    self.aggregated_posterior = np.zeros(
      (self.model.num_state, self.model.vocab_size))
    self.tokenizer = tokenizer
    self.tau_anneal_start_epoch = tau_anneal_start_epoch
    self.tau_anneal_n_epoch = tau_anneal_n_epoch
    self.x_lambd_warm_end_epoch = x_lambd_warm_end_epoch
    self.x_lambd_warm_n_epoch = x_lambd_warm_n_epoch
    self.z_beta_init = z_beta_init
    self.z_beta_final = z_beta_final
    self.anneal_beta_with_lambd = anneal_beta_with_lambd
    self.save_mode = save_mode
    self.anneal_z_prob = anneal_z_prob
    self.word_dropout = word_dropout
    self.freeze_z_at_epoch = freeze_z_at_epoch
    self.model_path = model_path

    self.learning_rate = learning_rate
    self.change_opt_to_sgd_at_epoch = change_opt_to_sgd_at_epoch

    stopwords_ = stopwords.words('english')
    stopwords_.extend(['"', "'", '.', ',', '?', '!', '-', '[CLS]', '[SEP]', 
      ':', '@', '/', '[', ']', '(', ')', 'would', 'like'])
    stopwords_space = []
    for w in stopwords_:
      stopwords_space.append(w)
      stopwords_space.append(space_token + w)
    
    self.stopwords = set(stopwords_space)
    if(optimizer_type.lower() == 'adam'):
      self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
    elif(optimizer_type.lower() == 'adamw'):
      self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
    elif(optimizer_type.lower() == 'adabelief'):
      # TODO: tune parameters
      self.optimizer = AdaBelief(self.model.parameters(), lr=learning_rate)
    else:
      raise NotImplementedError('optimizer %s not implemented' % optimizer_type)

    # self.scheduler = get_constant_schedule_with_warmup(
    #   self.optimizer, num_warmup_steps=50)

    self.log_info = ['loss', 'obj', 'ent', 'p_log_prob_x', 'p_log_prob_z', 'p_log_x_z', 
      'x_lambd', 'tau', 'z_lambd', 'z_beta']
    self.validation_scores = ['x_lambd', 'tau', 'z_lambd', 'z_beta']
    self.validation_criteria = validation_criteria
    self.num_batch_per_epoch = num_batch_per_epoch
    return 

  def schedule(self, n_iter, ei, bi, mode='train'):
    """Schedule parameters
    * x_lambd: word dropout ratio
    * tau: gumbel temperature
    * z_lambd: 
    * z_beta:
    """
    if(mode == 'val'):
      bi = 0 # schedule according to train batch, not validate batch 
      ei = ei + 1

    x_lambd_warm_end_epoch = self.x_lambd_warm_end_epoch
    x_lambd_warm_n_epoch = self.x_lambd_warm_n_epoch
    if(ei < x_lambd_warm_end_epoch): x_lambd = 1.
    else: 
      step_interval = 1. / (self.num_batch_per_epoch * x_lambd_warm_n_epoch)
      num_step = (ei - x_lambd_warm_end_epoch) * self.num_batch_per_epoch + bi
      x_lambd = 1 - num_step * step_interval
    if(x_lambd < 0): x_lambd = 0

    if(self.word_dropout == False): x_lambd = 0

    if(self.anneal_beta_with_lambd):
      if(ei < x_lambd_warm_end_epoch): z_beta = self.z_beta_init
      else:
        step_interval = (self.z_beta_init - self.z_beta_final) /\
          (self.num_batch_per_epoch * x_lambd_warm_n_epoch)
        num_step = (ei - x_lambd_warm_end_epoch) * self.num_batch_per_epoch + bi
        z_beta = self.z_beta_init - num_step * step_interval
        if(z_beta < self.z_beta_final): z_beta = self.z_beta_final
    else:
      z_beta = self.z_beta_init

    if(self.anneal_z_prob): z_lambd = 1 - x_lambd
    else: z_lambd = 1.0

    tau_anneal_start_epoch = self.tau_anneal_start_epoch
    tau_anneal_n_epoch = self.tau_anneal_n_epoch
    if(ei < tau_anneal_start_epoch): tau = 1.0
    else: 
      step_interval = 1. / (
        self.num_batch_per_epoch * float(tau_anneal_n_epoch))
      num_step = (ei - tau_anneal_start_epoch) * self.num_batch_per_epoch + bi
      tau = 1 - num_step * step_interval
      if(tau < 0.01): tau = 0.01

    # TODO: BOW loss schedule
    return tau, x_lambd, z_lambd, z_beta

  def update_aggregated_posterior(self, ei, bi, out_dict):
    """Updata the aggregated posterior, only use the first sample in the batch
    """

    if(bi == 0): self.aggregated_posterior *= 0

    for x, z in zip(out_dict['input_ids'], out_dict['z_sample']):
      for xi, zi in zip(x[1:-1], z[1:-1]):
        self.aggregated_posterior[zi, xi] += 1
    return

  def train_step(self, batch, n_iter, ei, bi):
    self.model.zero_grad()
    tau, x_lambd, z_lambd, z_beta = self.schedule(n_iter, ei, bi, 'train')
    if(ei == self.freeze_z_at_epoch and bi == 0):
      print('epoch %d, freeze inference network' % ei)
      self.model.state_matrix.requires_grad = False
    if(ei >= self.freeze_z_at_epoch): z_beta = 0.

    if(ei == self.change_opt_to_sgd_at_epoch and bi == 0):
      print('epoch %d, change optimizer to SGD')
      self.optimizer = SGD(self.model.parameters(), lr=self.learning_rate)

    if(self.task != 'lm'):
      loss, out_dict = self.model(
        sent=batch['input_ids'].to(self.device),
        bow=batch['bow'].to(self.device),
        tau=tau,
        x_lambd=x_lambd,
        z_lambd=z_lambd,
        z_beta=z_beta
        )
    elif(self.task == 'lm'):
      loss, out_dict = self.model.forward_lm(sent=batch['input_ids'].to(self.device))
    else: raise ValueError('Invalid task %s' % self.task)
    # print('cuda memory point 1:', torch.cuda.memory_allocated())
    loss.backward()
    # print('cuda memory point 2:', torch.cuda.memory_allocated())
    self.optimizer.step()
    # print('cuda memory point 3:', torch.cuda.memory_allocated())
    # self.scheduler.step()
    if(self.task != 'lm'):
      self.update_aggregated_posterior(ei, bi, out_dict)
    out_dict['tau'] = tau
    out_dict['x_lambd'] = x_lambd
    out_dict['z_lambd'] = z_lambd
    out_dict['z_beta'] = z_beta
    return out_dict

  def val_step(self, batch, n_iter, ei, bi, dataset):
    tau, x_lambd, z_lambd, z_beta = self.schedule(n_iter, ei, bi, 'val')
    out_dict = {}
    out_dict['tau'] = tau
    out_dict['x_lambd'] = x_lambd
    out_dict['z_lambd'] = z_lambd
    out_dict['z_beta'] = z_beta

    with torch.no_grad():
      if(self.task == 'paraphrase'):
        out_dict_ = self.model.predict_paraphrase(
          sent=batch['input_ids'].to(self.device),
          bow=batch['bow'].to(self.device)
          )
        out_dict.update(out_dict_)
        out_dict_ = self.model.infer_marginal(
          sent=batch['input_ids'][:1].to(self.device),
          bow=batch['bow'][:1].to(self.device)
          )
        out_dict['marginal'] = out_dict_['marginal']
        out_dict['joint'] = out_dict_['joint']
        out_dict['references'] = batch['references']
        out_dict['input_ids'] = tmu.to_np(batch['input_ids'])
        return out_dict
      
      if(self.task == 'density'):
        out_dict_ = self.model.infer_marginal(
          sent=batch['input_ids'].to(self.device))
        out_dict.update(out_dict_)
        return out_dict

      if(self.task != 'lm'):
        _, out_dict_ = self.model(
          sent=batch['input_ids'].to(self.device),
          bow=batch['bow'].to(self.device),
          tau=tau,
          x_lambd=x_lambd,
          z_lambd=z_lambd,
          z_beta=z_beta
          )
      else:
        _, out_dict_ = self.model.forward_lm(sent=batch['input_ids'].to(self.device))

      out_dict.update(out_dict_)
    return out_dict

  def state_ngram_stats(self, outputs, dataset, ei, n, output_path_base):
    """Count state ngram stats and write to file"""
    z_ngram_nostop = dict()
    z_ngram_stop = dict()
    z_ngram_cnt_nostop = []
    z_ngram_cnt_stop = []
    for out_dict in tqdm(outputs):
      for z, x in zip(out_dict['z_sample'], out_dict['input_ids']):
        len_z = len(z)
        if(len_z < n): continue 
        for i in range(len_z - n + 1):
          z_b = '-'.join([str(z[j]) for j in range(i, i + n)])
          w_b = [
            dataset.id2word[x[j]] for j in range(i, i + n)]
          w_b_set = set(w_b)
          w_b = '-'.join(w_b)

          if(len(w_b_set & self.stopwords) > 0):
            z_ngram_cnt_stop.append(z_b)
            if(z_b not in z_ngram_stop):
              z_ngram_stop[z_b] = dict()
              z_ngram_stop[z_b][w_b] = 1
              z_ngram_stop[z_b]['total'] = 1
            else:
              z_ngram_stop[z_b]['total'] += 1
              if(w_b not in z_ngram_stop[z_b]): z_ngram_stop[z_b][w_b] = 1
              else: z_ngram_stop[z_b][w_b] += 1
          else:
            z_ngram_cnt_nostop.append(z_b)
            if(z_b not in z_ngram_nostop):
              z_ngram_nostop[z_b] = dict()
              z_ngram_nostop[z_b][w_b] = 1
              z_ngram_nostop[z_b]['total'] = 1
            else:
              z_ngram_nostop[z_b]['total'] += 1
              if(w_b not in z_ngram_nostop[z_b]): z_ngram_nostop[z_b][w_b] = 1
              else: z_ngram_nostop[z_b][w_b] += 1

    filename = output_path_base + '_epoch_%d_state_%dgram_sw.txt' % (ei, n)
    print('Writing state %dgram to %s, with stop words' % (n, filename))
    with open(filename, 'w') as fd:
      z_ngram_cnt_stop = Counter(z_ngram_cnt_stop)
      for zb, f in z_ngram_cnt_stop.most_common(2000):
        fd.write('state %dgram %s freq %d\n' % (n, zb, f))
        wb_list = [(wb, z_ngram_stop[zb][wb]) for wb in z_ngram_stop[zb]]
        wb_list.sort(key=lambda x: x[1])
        wb_list.reverse()
        for wb, f in wb_list:
          fd.write('%s %d | ' % (wb, f))
        fd.write('\n----\n')
        z_ngram_stop[zb] = wb_list

    filename = output_path_base + '_epoch_%d_state_%dgram_no_sw.txt' % (ei, n)
    print('Writing state %dgram to %s, no stop words' % (n, filename))
    with open(filename, 'w') as fd:
      z_ngram_cnt_nostop = Counter(z_ngram_cnt_nostop)
      write_cnt = 0
      for zb, f in z_ngram_cnt_nostop.most_common():
        wb_list = [(wb, z_ngram_nostop[zb][wb]) for wb in z_ngram_nostop[zb]]
        wb_list.sort(key=lambda x: x[1])
        wb_list.reverse()
        z_ngram_nostop[zb] = wb_list
        if(write_cnt < 2000):
          fd.write('state %dgram %s freq %d\n' % (n, zb, f))
          for wb, f in wb_list:
            fd.write('%s %d | ' % (wb, f))
          fd.write('\n----\n')
          write_cnt += 1
        
    return z_ngram_stop, z_ngram_nostop

  def val_end_density(self, 
    outputs, n_iter, ei, bi, dataset, mode, output_path_base):
    marginals = [out['marginal_all'] for out in outputs]
    marginals = np.concatenate(marginals, 0)
    np.save(output_path_base + '_e%d_marginals.npy' % ei, marginals)

    x_lens = [out['x_lens_all'] for out in outputs]
    x_lens = np.concatenate(x_lens, 0)
    np.save(output_path_base + '_e%d_lens.npy' % ei, x_lens)

    ppl = np.exp(-marginals.sum() / x_lens.sum())
    scores = {'ppl': ppl}
    return scores

  def bleu_scores(self, hyps, refs, refs_self):
    """Calculate bleu scores"""
    bleu_scores = {}
    # bleu_scores['p_bleu_1'] = corpus_bleu(
    #   refs, hyps_, weights=(1, 0, 0, 0))
    bleu_scores['bleu_2'] = corpus_bleu(
      refs, hyps, weights=(0.5, 0.5, 0, 0))
    bleu_scores['bleu_3'] = corpus_bleu(
      refs, hyps, weights=(0.333, 0.333, 0.333, 0))
    bleu_scores['bleu_4'] = corpus_bleu(
      refs, hyps, weights=(0.25, 0.25, 0.25, 0.25))
    # bleu_scores['p_self_bleu_1'] = corpus_bleu(
    #   refs_self, hyps_, weights=(1, 0, 0, 0))
    bleu_scores['sb_2'] = corpus_bleu(
      refs_self, hyps, weights=(0.5, 0.5, 0, 0))
    bleu_scores['sb_3'] = corpus_bleu(
      refs_self, hyps, weights=(0.333, 0.333, 0.333, 0))
    bleu_scores['sb_4'] = corpus_bleu(
      refs_self, hyps, weights=(0.25, 0.25, 0.25, 0.25))
    # bleu_scores['p_ib1'] =\
    #   0.9 * bleu_scores['p_bleu_1'] - 0.1 * bleu_scores['p_self_bleu_1']
    bleu_scores['ib2'] =\
      0.9 * bleu_scores['bleu_2'] - 0.1 * bleu_scores['sb_2']
    bleu_scores['ib3'] =\
      0.9 * bleu_scores['bleu_3'] - 0.1 * bleu_scores['sb_3']
    bleu_scores['ib4'] =\
      0.9 * bleu_scores['bleu_4'] - 0.1 * bleu_scores['sb_4']
    return bleu_scores

  def ter_scores(self):
    #  TBC 
    return 

  def val_end_paraphrase(self,
    outputs, n_iter, ei, bi, dataset, mode, output_path_base):
    # TODO: update multiple sample
    # fd_paraphrase_q = open(output_path_base + '_paraphrase_q_e%d.txt' % ei, 'w')
    fd_paraphrase_p = open(output_path_base + '_paraphrase_p_e%d.txt' % ei, 'w')
    input_ids = []
    inputs, references = [], []
    pred_p, pred_q = [], []
    states_p, states_q = [], []
    for out_dict in outputs:
      predictions_sent_q = out_dict['predictions_x_q']
      predictions_states_q = out_dict['predictions_z_q']
      predictions_sent_p = out_dict['predictions_x_p']
      predictions_states_p = out_dict['predictions_z_p']
      references.extend(out_dict['references'])
      for inp, s_q, z_q, s_p, z_p in zip(out_dict['input_ids'], 
        predictions_sent_q, predictions_states_q, 
        predictions_sent_p, predictions_states_p):
        pred_qi = []
        states_qi = []
        pred_pi = []
        states_pi = []
        s_ = dataset.decode_sent(inp)
        inputs.append(s_)
        for si_q, zi_q in zip(s_q, z_q):
          s_, states_ = dataset.decode_sent(si_q, zi_q)
          pred_qi.append(s_)
          states_qi.append(states_)
        for si_p, zi_p in zip(s_p, z_p):
          s_, states_ = dataset.decode_sent(si_p, zi_p)
          pred_pi.append(s_)
          states_pi.append(states_)
        pred_p.append(pred_pi)
        states_p.append(states_pi)
        pred_q.append(pred_qi)
        states_q.append(states_qi)
          
    for inp, s_p, s_q, z_p, z_q, ref in zip(
      inputs, pred_p, pred_q, states_p, states_q, references):
      # fd_paraphrase_q.write('in:\t%s\n' % inp)
      fd_paraphrase_p.write('in:\t%s\n' % inp)
      # for zi_q, si_q in zip(z_q, s_q):
      #   fd_paraphrase_q.write('states:\t')
      #   for z in zi_q:
      #     fd_paraphrase_q.write('%d ' % z)
      #   fd_paraphrase_q.write('\n')
      #   fd_paraphrase_q.write('out:\t%s\n' % si_q)
      for zi_p, si_p in zip(z_p, s_p):
        fd_paraphrase_p.write('states:\t')
        for z in zi_p:
          fd_paraphrase_p.write('%d ' % z)
        fd_paraphrase_p.write('\n')
        fd_paraphrase_p.write('out:\t%s\n' % si_p)
      for r in ref:
        # fd_paraphrase_q.write('ref:s`\t%s\n' % r)
        fd_paraphrase_p.write('ref:\t%s\n' % r)
      # fd_paraphrase_q.write('\n')
      fd_paraphrase_p.write('\n')

    # fd_paraphrase_q.close()
    fd_paraphrase_p.close()

    # bleu scores - p
    refs = []
    for r in references:
      r_ = []
      for ri in r: r_.append(ri.split())
      refs.append(r_)
    refs_self = []
    for r in inputs: refs_self.append([r.split()])
    preds_argmax = [x[0].split() for x in pred_p]
    preds_sample = [x[1].split() for x in pred_p]
    bleu_scores_argmax = self.bleu_scores(preds_argmax, refs, refs_self)
    bleu_scores_sample = self.bleu_scores(preds_sample, refs, refs_self)
    bleu_scores = {}
    for n in bleu_scores_argmax: bleu_scores['am_' + n] = bleu_scores_argmax[n]
    for n in bleu_scores_sample: bleu_scores['sp_' + n] = bleu_scores_sample[n]

    # bleu scores - q
    # hyps_ = [x[0].split() for in pred_q]
    # bleu_scores['q_bleu_1'] = corpus_bleu(
    #   refs_, hyps_, weights=(1, 0, 0, 0))
    # bleu_scores['q_bleu_2'] = corpus_bleu(
    #   refs_, hyps_, weights=(0.5, 0.5, 0, 0))
    # bleu_scores['q_bleu_3'] = corpus_bleu(
    #   refs_, hyps_, weights=(0.333, 0.333, 0.333, 0))
    # bleu_scores['q_bleu_4'] = corpus_bleu(
    #   refs_, hyps_, weights=(0.25, 0.25, 0.25, 0.25))
    # bleu_scores['q_self_bleu_1'] = corpus_bleu(
    #   refs_self, hyps_, weights=(1, 0, 0, 0))
    # bleu_scores['q_self_bleu_2'] = corpus_bleu(
    #   refs_self, hyps_, weights=(0.5, 0.5, 0, 0))
    # bleu_scores['q_self_bleu_3'] = corpus_bleu(
    #   refs_self, hyps_, weights=(0.333, 0.333, 0.333, 0))
    # bleu_scores['q_self_bleu_4'] = corpus_bleu(
    #   refs_self, hyps_, weights=(0.25, 0.25, 0.25, 0.25))
    # bleu_scores['q_ib1'] =\
    #   0.9 * bleu_scores['q_bleu_1'] + 0.1 * bleu_scores['q_self_bleu_1']
    # bleu_scores['q_ib2'] =\
    #   0.9 * bleu_scores['q_bleu_2'] + 0.1 * bleu_scores['q_self_bleu_2']
    # bleu_scores['q_ib3'] =\
    #   0.9 * bleu_scores['q_bleu_3'] + 0.1 * bleu_scores['q_self_bleu_3']
    # bleu_scores['q_ib4'] =\
    #   0.9 * bleu_scores['q_bleu_4'] + 0.1 * bleu_scores['q_self_bleu_4']
    return bleu_scores

  def val_end(self, outputs, n_iter, ei, bi, dataset, mode, output_path_base):
    """End of validation, output all state-word maps
    
    Returns:
      val_end_scores. a dictionary of validation scores 
    """
    scores = dict()
    z_freq_stats = self.aggregated_posterior.sum(-1)
    num_active_states = (z_freq_stats != 0).sum()
    scores['num_active_states'] = num_active_states

    if(self.task == 'density'): 
      scores = self.val_end_density(
        outputs, n_iter, ei, bi, dataset, mode, output_path_base) 
      return scores
    if(self.task == 'paraphrase'):
      scores_ = self.val_end_paraphrase(
        outputs, n_iter, ei, bi, dataset, mode, output_path_base) 
      scores.update(scores_)
      return scores
    if(self.task == 'lm'):
      scores = self.val_end_density(
        outputs, n_iter, ei, bi, dataset, mode, output_path_base) 
      return scores 

    # Write state to word
    filename = output_path_base + '_epoch_%d_s2w.txt' % ei
    print('Writing state-word aggregated posterior to %s' % filename)
    with open(filename, 'w') as fd:
      z_freq_stats_no_sw = np.array(self.aggregated_posterior)
      for w in self.stopwords:
        if(w in dataset.word2id):
          wid = dataset.word2id[w]
          z_freq_stats_no_sw[:, wid] = 0
      z_freq_stats_no_sw = z_freq_stats_no_sw.sum(-1)
      
      ind = np.argsort(z_freq_stats)[::-1]
      # TODO: draw state frequency figure with static / dynamic portion
      for i in tqdm(range(self.model.num_state)):
        z_i = ind[i]
        fd.write('state %d freq %d freq_no_sw %d\n' % 
          (z_i, z_freq_stats[z_i], z_freq_stats_no_sw[z_i]))
        w_ind = np.argsort(self.aggregated_posterior[z_i])[::-1]
        printed = 0
        for w_ij in w_ind:
          w = dataset.id2word[w_ij]
          w_freq = self.aggregated_posterior[z_i, w_ij]
          if(w not in self.stopwords and w_freq > 0):
            fd.write('%s %d | ' % (w, w_freq))
            printed += 1
          if(printed == 30): break
        fd.write('\n--------\n')

    # Write word to state 
    filename = output_path_base + '_epoch_%d_w2s.txt' % ei
    print('Writing word-state aggregated posterior to %s' % filename)
    with open(filename, 'w') as fd:
      aggregated_posterior_inv = np.transpose(self.aggregated_posterior, (1, 0))
      w_freq_stats = aggregated_posterior_inv.sum(-1)
      ind = np.argsort(w_freq_stats)[::-1]
      for i in tqdm(range(self.model.vocab_size)):
        w_i = ind[i]
        w = dataset.id2word[w_i]
        fd.write('word %s freq %d\n' % (w, w_freq_stats[w_i]))
        z_ind = np.argsort(aggregated_posterior_inv[w_i])[::-1]
        printed = 0
        for z_ij in z_ind:
          fd.write('s%d f%d | ' % (z_ij, aggregated_posterior_inv[w_i, z_ij]))
          printed += 1
          if(printed == 50): break
        fd.write('\n--------\n')

    # write state bigram 
    _, z_bigram_nostop = self.state_ngram_stats(
      outputs, dataset, ei, 2, output_path_base)

    # write state transition
    filename = output_path_base + '_epoch_%d_s2s.txt' % ei
    print('Writing state transition to %s' % filename)
    # TODO: add bigram instances
    with open(filename, 'w') as fd:
      with torch.no_grad():
        _, transition = self.model.get_transition()
        transition = tmu.to_np(transition)
        # np.save(filename + '_epoch_%d_transition' % ei, transition)
        ind = np.flip(np.argsort(transition, axis=-1), axis=-1)
        for si in range(ind.shape[0]):
          fd.write('state %d: \n' % si)
          for i in ind[si][:10]:
            fd.write('  to %d score %.4f\n    ' % (i, transition[si, i]))
            transition_str = '%d-%d' % (si, i)
            if(transition_str in z_bigram_nostop):
              wb_list = z_bigram_nostop[transition_str]
              for wb, f in wb_list:
                fd.write('%s %d | ' % (wb, f))
              fd.write('\n')
            else: fd.write('not in frequent state bigram\n')
          fd.write('\n----\n')

    # write state trigram 
    _, _ = self.state_ngram_stats(outputs, dataset, ei, 3, output_path_base)

    # write state four gram 
    _, _ = self.state_ngram_stats(outputs, dataset, ei, 4, output_path_base)
    return scores

  def inspect_step(self, batch, out_dict, n_iter, ei, bi, dataset):
    """Inspect the model during training"""
    print('epoch %d batch %d inspect model' % (ei, bi))

    # z_freq_stats = self.aggregated_posterior.sum(-1)
    # print('z_freq_stats')
    # print(z_freq_stats)

    # print('top 20 state top 15 words')
    # ind = np.argsort(z_freq_stats)[::-1]
    # for i in range(20):
    #   z_i = ind[i]
    #   w_ind = np.argsort(self.aggregated_posterior[z_i])[::-1]
    #   print('%d: ' % z_i, end='')
    #   printed = 0
    #   for w_ij in w_ind:
    #     w = dataset.tokenizer.ids_to_tokens[w_ij]
    #     if(w not in self.stopwords):
    #       print(' %s %d ' % (w, self.aggregated_posterior[z_i, w_ij]), end='|')
    #       printed += 1
    #     if(printed == 15): break
    #   print('')

    # forward estimate with different proposal 
    with torch.no_grad():
      (ent, x_lens_mean, log_z_exact, log_z_est_non_trans, log_z_est_prod, 
      log_z_est_norm, crf_vars) = self.model.sampled_forward_est(
        batch['input_ids'][0:1].to(self.device))

      print('entropy %.4f' % ent)
      print('avg length: %.2f' % x_lens_mean)
      print('log z exact %.4f' % log_z_exact)
      print('log z estimates no transition proposal mean %.4f std %.4f' % 
        (np.mean(log_z_est_non_trans), np.std(log_z_est_non_trans)))
      print('log z estimates transition prod proposal mean %.4f std %.4f' % 
        (np.mean(log_z_est_prod), np.std(log_z_est_prod)))
      print('log z estimates transition norm proposal mean %.4f std %.4f' % 
        (np.mean(log_z_est_norm), np.std(log_z_est_norm)))

      print('epoch %d batch %d save state matrix' % (ei, bi))
      save_path = self.model_path + 'state_matrix_e%d_b%d' % (ei, bi) + '.pt'
      state_dict = self.model.state_dict()
      state_dict_keys = list(state_dict.keys())
      for k in state_dict_keys:
        if(k not in ['state_matrix', 'encoder']):
          state_dict.pop(k)
      state_dict['input_ids'] = batch['input_ids']
      state_dict['x_emb'] = crf_vars[0]
      state_dict['state_matrix_norm'] = crf_vars[1]
      state_dict['transition'] = crf_vars[2]
      state_dict['emission'] = crf_vars[3]
      state_dict['x_enc'] = crf_vars[4]
      state_dict['x_lens'] = crf_vars[5]
      torch.save(state_dict, save_path)
    return

  def save(self, save_path):
    """Save the model, but do not save the BERT part"""
    print('Saving model to %s, mode %s' % (save_path, self.save_mode))
    state_dict = self.model.state_dict()
    state_dict_keys = list(state_dict.keys())
    if(self.save_mode == 'state_matrix'):
      for k in state_dict_keys:
        if(k != 'state_matrix'):
          state_dict.pop(k)
    elif(self.save_mode == 'full'):
      for k in state_dict_keys:
        if(k.split('.')[0] == 'encoder'):
          state_dict.pop(k)
    else:
      raise NotImplementedError('save mode %s not implemented' % self.save_mode)
    torch.save(state_dict, save_path)
    return 

  def load(self, load_path):
    """Load the model non bert parts"""
    state_dict = torch.load(load_path)
    tmu.load_partial_state_dict(self.model, state_dict)
    return 
