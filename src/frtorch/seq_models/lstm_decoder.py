

import torch

import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from frtorch import torch_model_utils as tmu 

def attention(query, memory, mem_mask, device):
  """The attention function, Transformer style, scaled dot product
  
  Args:
    query: the query vector, shape = [batch_size, state_size]
    memory: the memory, shape = [batch_size, max_mem_len, state_size]
    mem_mask: the memory mask, shape = [batch_size, max_mem_len]. 1 = not masked
      0 = masked

  Returns:
    context_vec: the context vector, shape = [batch_size, state_size]
    attn_dist: the attention distribution, shape = [batch_size, max_mem_len]
  """
  state_size = query.shape[-1]
  batch_size = query.shape[0]
  max_mem_len = memory.shape[1]

  memory_ = memory.transpose(2, 1) # [B, M, S] -> [B, S, M]
  query_ = query.unsqueeze(1) # [B, 1, S]
  
  attn_weights = torch.bmm(query_, memory_) 
  attn_weights /= torch.sqrt(torch.Tensor([state_size]).to(device)) # [B, 1, M]
  attn_weights = attn_weights.view(batch_size, max_mem_len) # [B, M]

  if(mem_mask is not None):
    attn_weights = attn_weights.masked_fill(mem_mask == 0, -1e9)
  
  attn_dist = F.softmax(attn_weights, -1)
  # print(attn_dist.shape)
  attn_dist = attn_dist.unsqueeze(2)

  context_vec = attn_dist * memory
  context_vec = context_vec.sum(1) # [B, S]
  return context_vec, attn_dist.squeeze(2)

class Attention(nn.Module):
  """Simple scaled product attention"""
  def __init__(self, q_state_size, m_state_size, embedding_size):
    super(Attention, self).__init__()

    self.query_proj = nn.Linear(q_state_size, m_state_size)
    self.attn_proj = nn.Linear(m_state_size, embedding_size)
    return 

  def forward(self, query, memory, mem_mask=None):
    """
    Args:
      query: size=[batch, state_size]
      memory: size=[batch, mem_len, state_size]
      mem_mask: size=[batch, mem_len]

    Returns:
      context_vec: size=[batch, state_size]
      attn_dist: size=[batch, mem_len]
    """
    # map the memory and the query to the same space 
    # print(query.shape)
    device = query.device
    query = self.query_proj(query)
    context_vec, attn_dist = attention(query, memory, mem_mask, device)
    context_vec = self.attn_proj(context_vec)
    return context_vec, attn_dist

class LSTMDecoder(nn.Module):
  """Simple attentive LSTM decoder"""

  def __init__(self, 
               pad_id=0, 
               start_id=-1, 
               vocab_size=-1, 
               max_dec_len=-1,
               embedding_size=100,
               state_size=100, 
               mem_state_size=100,
               dropout=0.0,
               lstm_layers=1,
               decode_strategy='greedy',
               sampling_topk_k=3,
               sampling_topp_gap=0.5,
               copy_decoder=False,
               device='cpu',
               use_attn=True,
               use_hetero_attn=True
               ):
    super(LSTMDecoder, self).__init__()

    self.state_size = state_size
    self.attn_entropy = 0.0
    self.device = device
    self.vocab_size = vocab_size
    self.pad_id = pad_id
    self.start_id = start_id
    self.max_dec_len = max_dec_len
    self.decode_strategy = decode_strategy 
    self.topk_k = sampling_topk_k
    self.topp_gap = sampling_topp_gap
    self.lstm_layers = lstm_layers
    self.use_attn = use_attn
    self.use_hetero_attn = use_hetero_attn

    if(lstm_layers == 1): dropout = 0.

    self.cell = nn.LSTM(input_size=embedding_size, 
                        hidden_size=state_size,
                        num_layers=lstm_layers,
                        dropout=dropout)

    if(self.use_attn):
      self.attention = Attention(
        state_size, mem_state_size, embedding_size)
      self.attn_cont_proj = nn.Linear(
        2 * embedding_size, embedding_size)
    if(self.use_hetero_attn):
      self.hetero_attn = Attention(
        state_size, mem_state_size, embedding_size)

    self.dropout = nn.Dropout(dropout)
    
    self.output_proj = nn.Linear(self.state_size, vocab_size)

    self.copy = copy_decoder
    if(self.copy):
      self.copy_g = nn.Linear(state_size, 1)
      self.copy_attn = Attention(
        state_size, state_size, state_size)
    return 

  def forward(self, inp, state, memory=None, mem_mask=None, 
    hetero_mem=None, hetero_mem_mask=None, return_attn=False):
    """
    Args: 
      state = (h, c)
        h: type = torch.tensor(Float)
           shape = [num_layers, batch, hidden_state]
        c: type = torch.tensor(Float)
           shape = [num_layers, batch, hidden_state]
    """
    inp = self.dropout(inp)
    device = inp.device
    query = state[0][0] # use the bottom layer output as query, as in GNMT
    context_vec = None
    if(memory is not None and self.use_attn):
      context_vec, attn_dist = self.attention(query, memory, mem_mask)
    if(hetero_mem is not None and self.use_hetero_attn):
      context_hetero, hetero_attn_dist =\
        self.hetero_attn(query, hetero_mem, hetero_mem_mask)
      context_vec += context_hetero

    if(context_vec is not None):
      inp = self.attn_cont_proj(torch.cat([inp, context_vec], dim=1))
    out, state = self.cell(inp.unsqueeze(0), state)

    out = self.dropout(out)
    if(return_attn):
      return out, state, attn_dist
    else: return out, state

  def decode_train(self, init_state, dec_inputs, dec_targets, 
    mem=None, mem_emb=None, mem_mask=None, hetero_mem=None, hetero_mask=None, 
    return_attn=False):
    """Decoder training loop
    
    Args:
      init_state: (h, c), h.size=[], c.size=[]
      mem: size=[batch, max_mem]
      mem_emb: size=[batch, max_mem, state]
      mem_mask: size=[batch, max_mem]
      dec_inputs: size=[batch, max_len, state]
      dec_targets: size=[batch, max_len]
      return_attn: if return attention distribution

    Returns:
      log_prob: per-token log probability, averaged over everything
      predictions: size=[batch, max_dec_len]
      attn_dist: size=[batch, max_dec_len, max_src_len]
    """
    batch_size = dec_inputs.size(0)
    max_dec_len = dec_targets.size(1)

    state = init_state
    dec_inputs = dec_inputs.transpose(0, 1)
    dec_targets = dec_targets.transpose(0, 1)
    log_prob = []
    predictions = []
    # attn_dist = []
    pred_dist = []
    for i in range(max_dec_len):
      dec_out, state = self.forward(
        dec_inputs[i], state, mem_emb, mem_mask, hetero_mem, hetero_mask, False)

      # attn_dist.append(attn_dist_t)

      dec_out = dec_out[0]
      lm_logits = self.output_proj(dec_out)
      lm_prob = F.softmax(lm_logits, dim=-1)

      if(self.copy):
        _, copy_dist = self.copy_attn(dec_out, mem_emb, mem_mask)
        copy_prob = tmu.batch_index_put(copy_dist, mem, self.vocab_size)
        copy_g = torch.sigmoid(self.copy_g(dec_out))
        out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob
        logits = out_prob.log()
        log_prob_i = -F.cross_entropy(logits, dec_targets[i], reduction='none')
      else:
        logits = lm_logits
        out_prob = lm_prob
        log_prob_i = -F.cross_entropy(logits, dec_targets[i], reduction='none')

      pred_prob_t = F.softmax(logits, dim=-1)
      pred_dist.append(pred_prob_t)

      log_prob.append(log_prob_i)
      predictions.append(logits.argmax(dim=-1))

    log_prob = torch.stack(log_prob) # [T, B]
    mask = dec_targets != self.pad_id
    log_prob.masked_fill_(mask == 0, 0.) 
    log_prob = log_prob.sum() / mask.sum()
    predictions = torch.stack(predictions).transpose(0, 1)
    # attn_dist = torch.stack(attn_dist).transpose(0, 1) # [B, T, M]
    pred_dist = torch.stack(pred_dist).transpose(0, 1) # [B, T, V]
    if(return_attn): return log_prob, predictions, attn_dist, pred_dist
    else: return log_prob, predictions

  def decode_predict(self, init_state, embeddings, 
    mem=None, mem_emb=None, mem_mask=None, 
    hetero_mem=None, hetero_mask=None, return_attn=False):
    """Decoding for prediction
    
    Args:
      init_state:
      embeddings:
      mem:
      mem_emb:
      mem_mask:
      hetero_mem:
      hetero_mask:

    Returns:
      predictions: size=[batch, length], dtype=long
    """
    batch_size = init_state[0].size(1)
    device = init_state[0].device

    state = init_state
    inp = torch.zeros(batch_size).type(torch.long).to(device) + self.start_id
    inp = embeddings(inp)
    predictions = []
    max_dec_len = self.max_dec_len
    attn_dist = []
    pred_dist = []
    for i in range(max_dec_len):
      dec_out, state, attn_dist_t = self.forward(
        inp, state, mem_emb, mem_mask, hetero_mem, hetero_mask, return_attn=True)
      attn_dist.append(attn_dist_t)
      dec_out = dec_out[0]
      lm_logits = self.output_proj(dec_out)
      lm_prob = F.softmax(lm_logits, dim=-1)
      if(self.copy):
        _, copy_dist = self.copy_attn(dec_out, mem_emb, mem_mask)
        copy_prob = tmu.batch_index_put(copy_dist, mem, self.vocab_size)
        copy_g = torch.sigmoid(self.copy_g(dec_out))
        out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob
        logits = out_prob.log()
      else:
        logits = lm_logits
        out_prob = lm_prob

      pred_prob_t = F.softmax(logits, dim=-1)
      pred_dist.append(pred_prob_t)

      if(self.decode_strategy == 'greedy'):
        out = logits.argmax(dim=-1)
      elif(self.decode_strategy == 'sampling_unconstrained'):
        dist = Categorical(logits=logits)
        out = dist.sample()
      elif(self.decode_strategy == 'sampling_topk'):
        prob, ind = torch.topk(out_prob, self.topk_k, -1)
        prob /= prob.sum(dim=-1, keepdim=True)
        dist = Categorical(probs=prob)
        out_ = dist.sample()
        out = tmu.batch_index_select(ind, out_).squeeze()
      elif(self.decode_strategy == 'sampling_topp_adapt'):
        # adaptive top p sampling 
        prob_max, _ = out_prob.max(dim=-1, keepdim=True)
        prob_baseline = prob_max - self.topp_gap
        prob_mask = out_prob < prob_baseline
        prob_sample = out_prob.masked_fill(prob_mask, 0.)
        prob_sample /= prob_sample.sum(dim=-1, keepdim=True)
        dist = Categorical(prob_sample)
        out = dist.sample()
      else:
        raise NotImplementedError('decoding method %s not implemented!' % 
          self.decode_strategy)

      inp = embeddings(out)
      predictions.append(out)

    attn_dist = torch.stack(attn_dist).transpose(0, 1) # [B, T, M]
    pred_dist = torch.stack(pred_dist).transpose(0, 1) # [B, T, V]
    predictions = torch.stack(predictions).transpose(0, 1) # [B, T]
    if(return_attn): return predictions, attn_dist, pred_dist
    else: return predictions