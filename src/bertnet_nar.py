"""Recovering latent network structure from contextualized embeddings

Non-autoregressive decoder
"""


import torch 
import nltk
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np 

from tqdm import tqdm
from json import encoder
from torch import nn 
from torch.distributions import Uniform
# from torch.optim import Adam

from nltk.corpus import stopwords
from collections import Counter
from transformers import BertModel, AdamW, get_constant_schedule_with_warmup

# from adabelief_pytorch import AdaBelief
from frtorch import FRModel, LinearChainCRF, LSTMDecoder
from frtorch import torch_model_utils as tmu


class BertNetModel(nn.Module):
  """Use scaled CRF to recover latent network structure from contextualized 
  embeddings. BERT version
  """
  def __init__(self,
               num_state=20,
               state_size=768,
               transition_init_scale=0.01,
               encoder_type='bert',
               loss_type='reconstruct_ent_bow',
               freeze_enc=True,
               exact_rsample=True, 
               latent_scale=0.1, 
               sum_size=10,
               sample_size=10,
               proposal='softmax',
               device='cpu',
               vocab_size=30522,
               use_latent_proj=False,
               proj_dim=200,
               crf_weight_norm='none', 
               latent_type='sampled_gumbel_crf', 
               ent_approx='softmax_sample',
               use_bow_loss=False,
               word_dropout_decay=False,
               dropout=0.0,
               potential_normalization=False,
               potential_scale=1.0,
               topk_sum=False,
               pad_id=0
               ):
    """
    Args:
    """
    super(BertNetModel, self).__init__()

    self.encoder_type = encoder_type
    self.loss_type = loss_type
    self.exact_rsample = exact_rsample
    self.latent_scale = latent_scale
    self.sum_size = sum_size
    self.sample_size = sample_size
    self.proposal = proposal
    self.device = device
    self.num_state = num_state
    self.vocab_size = vocab_size
    self.crf_weight_norm = crf_weight_norm
    self.latent_type = latent_type
    self.ent_approx = ent_approx
    self.use_bow_loss = use_bow_loss
    self.word_dropout_decay = word_dropout_decay
    self.topk_sum = topk_sum
    self.state_size = state_size
    if(topk_sum == True): print('Using topk sum!')
    # self.potential_normalization = potential_normalization
    # self.potential_scale = potential_scale
    

    if(self.encoder_type == 'bert'):
      print('Using pretrained BERT')
      self.encoder = BertModel.from_pretrained('bert-base-uncased')
    elif(self.encoder_type == 'bert_random'):
      print('Using randomly initialized BERT')
      config = BertConfig()
      self.encoder = BertModel(config)
    else:
      raise NotImplementedError('LSTM encoder to yet implemented!')

    if(freeze_enc):
      for param in self.encoder.parameters():
        param.requires_grad = False

    self.use_latent_proj = use_latent_proj
    if(use_latent_proj):
      state_size = proj_dim
      self.latent_proj = nn.Linear(768, proj_dim)

    self.state_matrix = nn.Parameter(torch.normal(
      size=[num_state, state_size], mean=0.0, std=transition_init_scale))
    self.crf = LinearChainCRF(potential_normalization, potential_scale)

    self.embeddings = nn.Embedding(vocab_size, state_size)
    self.decoder = LSTMDecoder(vocab_size=vocab_size, 
                               state_size=state_size, 
                               embedding_size=state_size,
                               dropout=dropout,
                               use_attn=False,
                               use_hetero_attn=False,
                               pad_id=pad_id
                               )
    # self.p_z_proj = nn.Linear(state_size, num_state)
    # self.p_z_intermediate = nn.Linear(2 * state_size, state_size)
    return 

  def get_transition(self):
    """Return transition matrix"""
    transition = torch.matmul(
      self.state_matrix, self.state_matrix.transpose(1, 0))
    return self.state_matrix, transition

  # def weight_norm(self, x_emb):
  #   """Normalize all embeddings to scaled unit vector
    
  #   Args:
  #     x_emb: size=[batch, max_len, dim]

  #   Returns: 
  #     state_matrix: size=[state, dim]
  #     emission_seq: size=[batch, max_len, dim]
  #     transition: size=[state, state] # could be large
  #     emission: size=[batch, max_len, state]
  #   """
  #   if(self.crf_weight_norm == 'sphere'):
  #     emission_seq = x_emb / torch.sqrt((x_emb ** 2).sum(-1, keepdim=True))
  #     emission_seq = self.latent_scale * emission_seq
  #     state_matrix = self.state_matrix /\
  #       torch.sqrt((self.state_matrix ** 2).sum(-1, keepdim=True))
  #     state_matrix = self.latent_scale * state_matrix
  #   elif(self.crf_weight_norm == 'zscore'):
  #     raise NotImplementedError('z score normalization not implemented!')
  #   elif(self.crf_weight_norm == 'none'):
  #     emission_seq = x_emb
  #     state_matrix = self.state_matrix 
  #   else:
  #     raise ValueError('Invalid crf_weight_norm: %s' % self.crf_weight_norm)

  #   transition = torch.matmul(state_matrix, state_matrix.transpose(1, 0))
  #   emission = torch.matmul(emission_seq, state_matrix.transpose(1, 0))
  #   return state_matrix, emission_seq, transition, emission

  def prepare_dec_io(self, 
    z_sample_ids, z_sample_emb, sentences, x_lambd):
    """Prepare the decoder output g based on the inferred z from the CRF 
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
    max_len = sentences.size(1)
    device = sentences.device

    sent_emb = self.embeddings(sentences)
    z_sample_emb[:, 0] *= 0. # mask out z[0]

    # word dropout ratio = x_lambd. 0 = no dropout, 1 = all drop out
    m = Uniform(0., 1.)
    mask = m.sample([batch_size, max_len]).to(device)
    mask = (mask > x_lambd).float().unsqueeze(2)

    if(self.word_dropout_decay):
      dec_inputs = z_sample_emb + sent_emb * mask * (1 - x_lambd)
    else: 
      dec_inputs = z_sample_emb + sent_emb * mask
    dec_inputs = dec_inputs[:, :-1]

    dec_targets_x = sentences[:, 1:]
    dec_targets_z = z_sample_ids[:, 1:]
    return dec_inputs, dec_targets_x, dec_targets_z

  # def decode_train(self, 
  #   dec_inputs, z_sample_emb, dec_targets_x, dec_targets_z, x_lens):
  #   """

  #   Args:
  #     dec_inputs: size=[batch, max_len, dim]
  #     z_sample_emb: size=[batch, max_len, dim]
  #     dec_target_x: size=[batch, max_len]
  #     dec_target_z: size=[batch, max_len]
  #     x_lens: size=[batch]

  #   Returns:
  #     log_prob: size=[batch, max_len]
  #     log_prob_x: size=[batch, max_len]
  #     log_prob_z: size=[batch, max_len]
  #   """
  #   max_len = dec_inputs.size(1)
  #   batch_size = dec_inputs.size(0)
  #   device = dec_inputs.device
  #   state_dim = dec_inputs.size(-1)

  #   dec_cell = self.decoder

  #   dec_inputs = dec_inputs.transpose(1, 0)
  #   dec_targets_x = dec_targets_x.transpose(1, 0)
  #   dec_targets_z = dec_targets_z.transpose(1, 0)
  #   z_sample_emb = z_sample_emb[:, 1:].transpose(1, 0) # start from z[1]

  #   state = (torch.zeros(dec_cell.lstm_layers, batch_size, state_dim).to(device), 
  #            torch.zeros(dec_cell.lstm_layers, batch_size, state_dim).to(device))
  #   log_prob_x, log_prob_z = [], []
    
  #   for i in range(max_len):
  #     dec_out, state = dec_cell(dec_inputs[i], state)
  #     dec_out = dec_out[0]
  #     z_logits = self.p_z_proj(dec_out)
  #     log_prob_z_i = -F.cross_entropy(
  #       z_logits, dec_targets_z[i], reduction='none')
  #     log_prob_z.append(log_prob_z_i)

  #     dec_intermediate = self.p_z_intermediate(
  #       torch.cat([dec_out, z_sample_emb[i]], dim=1))
  #     x_logits = dec_cell.output_proj(dec_intermediate)
  #     log_prob_x_i = -F.cross_entropy(
  #       x_logits, dec_targets_x[i], reduction='none')
  #     log_prob_x.append(log_prob_x_i)

  #   log_prob_x = torch.stack(log_prob_x).transpose(1, 0) # [B, T]
  #   log_prob_x = tmu.mask_by_length(log_prob_x, x_lens)
  #   log_prob_x = (log_prob_x.sum(-1) / x_lens).mean()

  #   log_prob_z = torch.stack(log_prob_z).transpose(1, 0)
  #   log_prob_z = tmu.mask_by_length(log_prob_z, x_lens)
  #   log_prob_z = (log_prob_z.sum(-1) / x_lens).mean()

  #   log_prob = log_prob_x + log_prob_z
  #   return log_prob, log_prob_x, log_prob_z

  def forward(self, x, attention_mask, 
    tau=1.0, x_lambd=0.0, z_lambd=0.0, z_beta=1.0):
    """

    Args:
      x: input sentence, size=[batch, max_len]
      attention_mask: input attention mask, size=[batch, max_len]

    Returns:
      loss
      out_dict
    """
    batch_size = x.size(0)
    x_lens = attention_mask.sum(-1)
    device = x.device

    # encoding
    x_emb = self.encoder(x, attention_mask=attention_mask)[0]
    if(self.use_latent_proj):
      x_emb = self.latent_proj(x_emb)

    # latent 
    if(self.latent_type == 'softmax'):
      z_logits = torch.einsum('bij,kj->bik', x_emb, self.state_matrix)
      z_sample_relaxed = tmu.reparameterize_gumbel(z_logits, tau)
      z_sample = z_sample_relaxed.argmax(dim=-1)
      z_sample_emb = torch.matmul(z_sample_relaxed, self.state_matrix)
      ent = tmu.entropy(F.softmax(z_logits, dim=-1))
    elif(self.latent_type == 'sampled_gumbel_crf'):
      # state_matrix, emission_seq, transition, emission = self.weight_norm(x_emb)
      state_matrix = self.state_matrix 
      emission = torch.matmul(x_emb, state_matrix.transpose(1, 0))
      if(self.exact_rsample):
        raise NotImplementedError('TODO: normalize transition')
        # z_sample, z_sample_relaxed = self.crf.rsample(
        #   transition, emission, x_lens, tau=tau)
        # z_sample_emb = torch.einsum('ijk,kl->ijl', z_sample_relaxed, state_matrix)
        # ent = self.crf.entropy(transition, emission, x_lens)
        # ent = ent.mean()
      else:
        # NOTE: pay attention to the index transform here, see details in the 
        # implementation
        if(self.ent_approx == 'log_prob'):
          pass
          # _, _, _, z_sample, z_sample_emb, z_sample_log_prob, inspect =\
          #   self.crf.rsample_approx(state_matrix, emission, x_lens, 
          #     self.sum_size, self.proposal, tau=tau, return_ent=False) 
          # # use log prob as single estimate of entropy 
          # ent = -z_sample_log_prob.mean()
        elif(self.ent_approx == 'softmax_sample'):
          pass
          # _, _, _, z_sample, z_sample_emb, z_sample_log_prob, inspect, ent =\
          #   self.crf.rsample_approx(state_matrix, emission, x_lens, 
          #     self.sum_size, self.proposal, tau=tau, return_ent=True) 
          # ent_sofmax = tmu.entropy(F.softmax(emission, -1), keepdim=True)
          # ent_sofmax = (ent_sofmax * attention_mask).sum()
          # ent_sofmax = ent_sofmax / attention_mask.sum()

          # # TODO: report the two terms respectively 
          # ent = ent.mean() + ent_sofmax
        elif(self.ent_approx == 'softmax'):
          _, _, _, z_sample, z_sample_emb, z_sample_log_prob, inspect =\
            self.crf.rsample_approx(state_matrix, emission, x_lens, 
              self.sum_size, self.proposal, sample_size=self.sample_size,
              tau=tau, return_ent=False, 
              topk_sum=self.topk_sum) 
          ent_sofmax = tmu.entropy(F.softmax(emission, -1), keepdim=True)
          ent_sofmax = (ent_sofmax * attention_mask).sum()
          ent = ent_sofmax / attention_mask.sum()
        else: 
          raise ValueError('Invalid value ent_approx: %s' % self.ent_approx)
    else: 
      raise NotImplementedError(
        'Latent type %s not implemented!' % self.latent_type)
    
    # decoding 
    # dec_inputs, dec_targets_x, dec_targets_z = self.prepare_dec_io(
    #   z_sample, z_sample_emb, x, x_lambd)

    # p_log_prob, p_log_prob_x, p_log_prob_z = self.decode_train(
    #   dec_inputs, z_sample_emb, dec_targets_x, dec_targets_z, x_lens)

    state = (torch.zeros(self.decoder.lstm_layers, batch_size, self.state_size).to(device), 
             torch.zeros(self.decoder.lstm_layers, batch_size, self.state_size).to(device))
    p_log_prob, _ = self.decoder.decode_train(state, z_sample_emb, x)

    # by default we do maximization
    loss = p_log_prob + z_beta * ent

    # turn maximization to minimization
    loss = -loss
    
    out_dict = {}
    out_dict['loss'] = loss.item()
    out_dict['ent'] = ent.item()
    out_dict['p_log_prob'] = p_log_prob.item()
    out_dict['z_sample'] = tmu.to_np(z_sample)
    out_dict['input_ids'] = tmu.to_np(x)
    out_dict['p_t_min'] = inspect['p_t_min'].item()
    out_dict['p_t_max'] = inspect['p_t_max'].item()
    out_dict['p_t_mean'] = inspect['p_t_mean'].item()
    out_dict['p_e_min'] = inspect['p_e_min'].item()
    out_dict['p_e_max'] = inspect['p_e_max'].item()
    out_dict['p_e_mean'] = inspect['p_e_mean'].item()
    return loss, out_dict

  # def sampled_forward_est(self, x, attention_mask):
  #   """Sampled forward"""
  #   x_emb = self.encoder(x, attention_mask=attention_mask)[0]
  #   x_lens = attention_mask.sum(-1)

  #   state_matrix, emission_seq, transition, emission = self.weight_norm(x_emb)
  #   _, log_z_exact = self.crf.forward_sum(transition, emission, x_lens)
  #   log_z_exact = log_z_exact[0].cpu().item()

  #   log_z_est_non_trans = []
  #   for _ in range(1000):
  #     est = self.crf.forward_approx(state_matrix, emission, x_lens, 
  #       sum_size=100, proposal='softmax', 
  #       transition_proposal='none', sample_size=100)
  #     log_z_est_non_trans.append(est[0].cpu().item())

  #   log_z_est_prod = []
  #   for _ in range(1000):
  #     est = self.crf.forward_approx(state_matrix, emission, x_lens, 
  #       sum_size=100, proposal='softmax', 
  #       transition_proposal='prod', sample_size=100)
  #     log_z_est_prod.append(est[0].cpu().item())

  #   log_z_est_norm = []
  #   for _ in range(1000):
  #     est = self.crf.forward_approx(state_matrix, emission, x_lens, 
  #       sum_size=100, proposal='softmax', 
  #       transition_proposal='abs_sum', sample_size=100)
  #     log_z_est_norm.append(est[0].cpu().item())
  #   return log_z_exact, log_z_est_non_trans, log_z_est_prod, log_z_est_norm

  # def infer_marginal(self, x, attention_mask):
  #   """Infer marginal likelihood"""
  #   out_dict = {}
  #   return out_dict

class BertNet(FRModel):
  """BertNet model, train/dev/test wrapper"""
  def __init__(self, 
               model, 
               learning_rate=1e-3, 
               validation_criteria='loss', 
               num_batch_per_epoch=-1, 
               x_lambd_warm_end_epoch=1,
               x_lambd_warm_n_epoch=1,
               tau_anneal_start_epoch=18,
               tau_anneal_n_epoch=3,
               tokenizer=None,
               z_beta_init=1.0,
               z_beta_final=0.01,
               anneal_beta_with_lambd=False,
               anneal_z_prob=False,
               save_mode='full'
               ):
    """"""
    super(BertNet, self).__init__()

    self.model = model
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

    self.stopwords = stopwords.words('english')
    self.stopwords.extend(['"', "'", '.', ',', '?', '!', '-', '[CLS]', '[SEP]', 
      ':', '@', '/', '[', ']', '(', ')', 'would', 'like'])
    self.stopwords = set(self.stopwords)
    
    # TODO: test AdaBelief optimizer
    self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
    self.scheduler = get_constant_schedule_with_warmup(
      self.optimizer, num_warmup_steps=50)

    # self.log_info = ['loss', 'obj', 'ent', 'p_log_prob_x', 'p_log_prob_z', 'p_log_x_z', 
    #   'x_lambd', 'tau', 'z_lambd', 'z_beta']
    # self.validation_scores = ['loss', 'obj', 'ent', 'p_log_prob_x', 'p_log_prob_z', 
    #   'p_log_x_z', 'x_lambd', 'tau', 'z_lambd', 'z_beta']
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

    loss, out_dict = self.model(
      x=batch['input_ids'].to(self.device),
      attention_mask=batch['attention_mask'].to(self.device),
      tau=tau,
      x_lambd=x_lambd,
      z_lambd=z_lambd,
      z_beta=z_beta
      )
    loss.backward()
    self.optimizer.step()
    self.scheduler.step()

    self.update_aggregated_posterior(ei, bi, out_dict)
    out_dict['tau'] = tau
    # out_dict['x_lambd'] = x_lambd
    # out_dict['z_lambd'] = z_lambd
    out_dict['z_beta'] = z_beta
    return out_dict

  def val_step(self, batch, n_iter, ei, bi, dataset):
    tau, x_lambd, z_lambd, z_beta = self.schedule(n_iter, ei, bi, 'val')

    # TODO: likelihood of tail words

    with torch.no_grad():
      _, out_dict = self.model(
        x=batch['input_ids'].to(self.device),
        attention_mask=batch['attention_mask'].to(self.device), 
        tau=tau,
        x_lambd=x_lambd,
        z_lambd=z_lambd,
        z_beta=z_beta
        )
      out_dict['tau'] = tau
      out_dict['x_lambd'] = x_lambd
      out_dict['z_lambd'] = z_lambd
      out_dict['z_beta'] = z_beta
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
            dataset.tokenizer.ids_to_tokens[x[j]] for j in range(i, i + n)]
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

  def val_end(self, outputs, n_iter, ei, bi, dataset, mode, output_path_base):
    """End of validation, output all state-word maps"""
    # TODO: differentiate words with single / multiple states
    # TODO: add Viterbi decoding 
    # TODO: write sentence-state pairs 
    scores = dict()

    filename = output_path_base + '_epoch_%d_s2w.txt' % ei
    print('Writing state-word aggregated posterior to %s' % filename)
    with open(filename, 'w') as fd:
      fd_sw = open(output_path_base + '_epoch_%d_s2w_sw.txt' % ei, 'w')
      z_freq_stats = self.aggregated_posterior.sum(-1)
      z_ppl = z_freq_stats / z_freq_stats.sum()
      z_ppl = np.exp((-z_ppl * np.log(z_ppl + 1e-8)).sum())
      scores['z_ppl'] = z_ppl
      z_freq_stats_no_sw = np.array(self.aggregated_posterior)
      for w in self.stopwords:
        wid = self.tokenizer.vocab[w]
        z_freq_stats_no_sw[:, wid] = 0
      z_freq_stats_no_sw = z_freq_stats_no_sw.sum(-1)
      num_active_states = (z_freq_stats != 0).sum()
      scores['num_active_states'] = num_active_states
      ind = np.argsort(z_freq_stats)[::-1]
      # TODO: draw state frequency figure with static / dynamic portion
      for i in tqdm(range(self.model.num_state)):
        z_i = ind[i]
        # write state
        fd.write('state %d freq %d freq_no_sw %d\n' % 
          (z_i, z_freq_stats[z_i], z_freq_stats_no_sw[z_i]))
        fd_sw.write('state %d freq %d freq_sw %d freq_no_sw %d\n' % 
            (z_i, z_freq_stats[z_i], 
             z_freq_stats[z_i] - z_freq_stats_no_sw[z_i], 
             z_freq_stats_no_sw[z_i]
            )
          )
        # write word
        w_ind = np.argsort(self.aggregated_posterior[z_i])[::-1]
        printed = 0
        printed_sw = 0
        for w_ij in w_ind:
          w = dataset.tokenizer.ids_to_tokens[w_ij]
          w_freq = self.aggregated_posterior[z_i, w_ij]

          if(printed_sw < 60):
            fd_sw.write('%s %d | ' % (w, w_freq))
            printed_sw += 1

          if(w not in self.stopwords and w_freq > 0):
            fd.write('%s %d | ' % (w, w_freq))
            printed += 1
          if(printed == 60): break

        fd.write('\n--------\n')
        fd_sw.write('\n--------\n')
    fd_sw.close()

    filename = output_path_base + '_epoch_%d_w2s.txt' % ei
    print('Writing word-state aggregated posterior to %s' % filename)
    with open(filename, 'w') as fd:
      aggregated_posterior_inv = np.transpose(self.aggregated_posterior, (1, 0))
      w_freq_stats = aggregated_posterior_inv.sum(-1)
      ind = np.argsort(w_freq_stats)[::-1]
      for i in tqdm(range(self.model.vocab_size)):
        w_i = ind[i]
        w = dataset.tokenizer.ids_to_tokens[w_i]
        fd.write('word %s freq %d\n' % (w, w_freq_stats[w_i]))
        z_ind = np.argsort(aggregated_posterior_inv[w_i])[::-1]
        printed = 0
        for z_ij in z_ind:
          fd.write('s%d f%d | ' % (z_ij, aggregated_posterior_inv[w_i, z_ij]))
          printed += 1
          if(printed == 50): break
        fd.write('\n--------\n')

    ## Compute F1 score 

    ## Compute V score

    # # write state bigram 
    # _, z_bigram_nostop = self.state_ngram_stats(
    #   outputs, dataset, ei, 2, output_path_base)

    # filename = output_path_base + '_epoch_%d_s2s.txt' % ei
    # print('Writing state transition to %s' % filename)
    # # TODO: add bigram instances
    # with open(filename, 'w') as fd:
    #   with torch.no_grad():
    #     _, transition = self.model.get_transition()
    #     transition = tmu.to_np(transition)
    #     np.save(filename + '_epoch_%d_transition' % ei, transition)
    #     ind = np.flip(np.argsort(transition, axis=-1), axis=-1)
    #     for si in range(ind.shape[0]):
    #       fd.write('state %d: \n' % si)
    #       for i in ind[si][:10]:
    #         fd.write('  to %d score %.4f\n    ' % (i, transition[si, i]))
    #         transition_str = '%d-%d' % (si, i)
    #         if(transition_str in z_bigram_nostop):
    #           wb_list = z_bigram_nostop[transition_str]
    #           for wb, f in wb_list:
    #             fd.write('%s %d | ' % (wb, f))
    #           fd.write('\n')
    #         else: fd.write('not in frequent state bigram\n')
    #       fd.write('\n----\n')

    # # write state trigram 
    # _, _ = self.state_ngram_stats(outputs, dataset, ei, 3, output_path_base)
    # # write state four gram 
    # _, _ = self.state_ngram_stats(outputs, dataset, ei, 4, output_path_base)
    return scores

  def inspect_step(self, batch, out_dict, n_iter, ei, bi, dataset):
    """Inspect the model during training"""

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
    # with torch.no_grad():
    #   log_z_exact, log_z_est_non_trans, log_z_est_prod, log_z_est_norm =\
    #   self.model.sampled_forward_est(
    #     x=batch['input_ids'][0:1].to(self.device),
    #     attention_mask=batch['attention_mask'][0:1].to(self.device)
    #     )
    #   print('log z exact %.4f' % log_z_exact)
    #   print('log z estimates no transition proposal mean %.4f std %.4f' % 
    #     (np.mean(log_z_est_non_trans), np.std(log_z_est_non_trans)))
    #   print('log z estimates transition prod proposal mean %.4f std %.4f' % 
    #     (np.mean(log_z_est_prod), np.std(log_z_est_prod)))
    #   print('log z estimates transition norm proposal mean %.4f std %.4f' % 
    #     (np.mean(log_z_est_norm), np.std(log_z_est_norm)))
    return

  def save(self, save_path):
    """Save the model, but do not save the BERT part"""
    print('Saving model to %s, BERT part not saved' % save_path)
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
