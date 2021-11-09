"""Sequence to sequence"""
from json import encoder
import torch 
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np 

from torch import nn 
# from torch.optim import Adam

from transformers import BertModel, AdamW, get_constant_schedule_with_warmup
from frtorch import FRModel, LinearChainCRF 
from frtorch import torch_model_utils as tmu


def evaluate(y_pred, y, y_lens):
  support = tmu.mask_by_length((y > 0).float(), y_lens).sum()
  pred = tmu.mask_by_length((y_pred > 0).float(), y_lens).sum()
  overlap = tmu.mask_by_length(((y_pred > 0) * (y > 0)).float(), y_lens).sum()

  acc = tmu.mask_by_length((y_pred == y).float(), y_lens).sum() / y_lens.sum()
  if(pred == 0): prec = torch.tensor(0.)
  else: prec = overlap / pred
  recl = overlap / support

  if(prec == 0.): f1 = torch.tensor(0.)
  else: f1 = 2 * prec * recl / (prec + recl)
  return support, pred, overlap, acc, prec, recl, f1


class CRFScaleModel(nn.Module):
  def __init__(self, 
               state_size=768,
               num_z_state=10, 
               approx_state_r=0.1, 
               norm_scale_transition=0.01,
               norm_scale_emission=0.01, 
               lambd_softmax=0.1, 
               encoder_type='bert',
               loss_type='approx_crf'
               ):
    """Pytorch seq2seq model"""
    super().__init__()
    
    self.encoder_type = encoder_type
    if(self.encoder_type == 'bert'):
      self.bert = BertModel.from_pretrained('bert-base-uncased')
    else:
      raise NotImplementedError('LSTM encoder to yet implemented!')
    self.state_matrix = nn.Parameter(torch.normal(
      mean=0., std=norm_scale_transition, size=[num_z_state, state_size]))
    self.state_size = state_size

    self.num_z_state = num_z_state
    self.approx_state_r = approx_state_r
    self.norm_scale_transition = norm_scale_transition
    self.norm_scale_emission = norm_scale_emission
    self.lambd_softmax = lambd_softmax
    self.loss_type = loss_type
    return 

  def forward(self, x, attention_mask, x_gather_id, y, y_lens):
    """
    Args: 
      x: size=[batch, max_len], sentece tokenized by BERT tokenizer
      attention_mask: size=[batch, max_len], returned from BERT tokenized
      x_gather_id: size=[batch, max_label_len], map word piece to original word
      y: size=[batch, max_label_len]
      y_lens: size=[batch]
    """
    batch_size = x.size(0)
    max_label_len = y.size(1)
    num_z_state = self.num_z_state
    state_size = self.state_size

    # batch, max_len, emb_size
    if(self.encoder_type == 'bert'):
      x_emb = self.bert(x, attention_mask=attention_mask)[0] 
      x_emb = tmu.batch_index_select(x_emb, x_gather_id) 
    else:
      x_emb = ...
      raise NotImplementedError('LSTM encoder not yet implemented!')

    # contextualized embedding normalization, applied at single word level  
    # x_emb = x_emb - x_emb.mean(dim=-1, keepdim=True)
    # x_emb = x_emb / x_emb.std(dim=-1, keepdim=True)
    # x_emb = self.norm_scale_emission * x_emb

    # state embedding normalization, applied to all stetes
    state_emb = self.state_matrix
    # state_emb = self.state_matrix - self.state_matrix.mean()
    # state_emb = state_emb / state_emb.std()
    # state_emb = self.norm_scale_transition * state_emb

    # emission potential = scaled vec prod of word-state emb
    # [B, T, V] * [V, V] -> [B, T, V]
    emission_potentials = torch.matmul(x_emb, 
      state_emb.transpose(1, 0).unsqueeze(0)) / np.sqrt(state_size)
    emission_mean = emission_potentials.mean()
    emission_std = emission_potentials.std()
    # emission_potentials = torch.matmul(x_emb, 
    #   state_emb.transpose(1, 0).unsqueeze(0))

    # transition potential = scaled vec prod of state-state emb 
    transition_potentials = torch.matmul(state_emb, 
      state_emb.transpose(1, 0)) / np.sqrt(state_size)
    transition_mean = transition_potentials.mean()
    transition_std = transition_potentials.std()

    crf = LinearChainCRF()

    # exact forward
    _, log_Z = crf.forward_sum(
      transition_potentials, emission_potentials, y_lens)
    log_Z_out = log_Z.mean()

    # approximate forward
    num_state_approx = int(self.approx_state_r * self.num_z_state)
    log_Z_approx = crf.approximate_forward(
      state_emb, emission_potentials, y_lens, num_state_approx)
    log_Z_approx_out = log_Z_approx.mean()
    
    # seq potential 
    log_y_potential = crf.seq_log_potential(
      y, transition_potentials, emission_potentials, y_lens)
    seq_log_prob_exact = -(log_y_potential - log_Z).mean()
    seq_log_prob_approx = -(log_y_potential - log_Z_approx).mean()

    # seq_prob with softmax, baseline model
    seq_log_prob_local = F.log_softmax(emission_potentials, -1)
    max_prob = torch.max(F.softmax(emission_potentials, -1), -1)[0]
    max_prob = tmu.mask_by_length(max_prob, y_lens).sum() / y_lens.sum()
    
    seq_log_prob_local = tmu.batch_index_select(
      seq_log_prob_local.view(batch_size * max_label_len, -1),
      y.view(batch_size * max_label_len)
      ).view(batch_size, max_label_len)
    seq_log_prob_local = tmu.mask_by_length(
      seq_log_prob_local, y_lens)
    seq_log_prob_local = seq_log_prob_local.sum() / y_lens.sum()

    # imbalanced classification, only promote positive
    # seq_log_prob_local = seq_log_prob_local * (y > 0).float()
    # seq_log_prob_local = seq_log_prob_local.sum() / (y > 0).float().sum()

    ## loss
    if(self.loss_type == 'softmax'):
      loss = -seq_log_prob_local
      # loss = loss_softmax
    elif(self.loss_type == 'exact_crf'):
      loss = seq_log_prob_exact + self.lambd_softmax * seq_log_prob_local
    elif(self.loss_type == 'approx_crf'):
      loss = seq_log_prob_approx + self.lambd_softmax * seq_log_prob_local
    else:
      raise NotImplementedError('loss %s not implemented' % self.loss_type)

    ## output and evaluation 
    # TODO: look at the AEFT paper and look at their metrics
    # crf 
    crf_output, _ = crf.argmax(transition_potentials, emission_potentials, y_lens)
    _, crf_pred, crf_overlap, crf_acc, crf_prec, crf_recl, crf_f1 =\
      evaluate(crf_output, y, y_lens)
    # softmax
    softmax_output = emission_potentials.argmax(-1)
    (softmax_support, softmax_pred, softmax_overlap, 
     softmax_acc, softmax_prec, softmax_recl, softmax_f1) =\
      evaluate(softmax_output, y, y_lens)

    out_dict = {'loss': loss.item(), 
                'max_prob_softmax': max_prob.item(),
                'seq_log_prob_local': seq_log_prob_local.item(),
                'seq_log_prob_exact': seq_log_prob_exact.item(),
                'seq_log_prob_approx': seq_log_prob_approx.item(),
                'log_Z': log_Z_out.item(),
                'log_Z_approx': log_Z_approx_out.item(),
                'transition_mean': transition_mean.item(),
                'transition_std': transition_std.item(),
                'emission_mean': emission_mean.item(),
                'emission_std': emission_std.item(), 
                'crf_pred': crf_pred.item(),
                'crf_overlap': crf_overlap.item(),
                'crf_acc': crf_acc.item(), 
                'crf_prec': crf_prec.item(),
                'crf_recl': crf_recl.item(),
                'crf_f1': crf_f1.item(),
                'softmax_support': softmax_support.item(),
                'softmax_pred': softmax_pred.item(), 
                'softmax_overlap': softmax_overlap.item(), 
                'softmax_acc': softmax_acc.item(),
                'softmax_prec': softmax_prec.item(),
                'softmax_recl': softmax_recl.item(),
                'softmax_f1': softmax_f1.item()}
    return loss, out_dict

class CRFScale(FRModel):
  def __init__(self, model, learning_rate, validation_criteria):
    self.model = model 
    self.learning_rate = learning_rate
    self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
    self.scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=50)

    self.log_info = [ 'loss', 
                      'max_prob_softmax',
                      'seq_log_prob_local', 
                      'seq_log_prob_exact', 
                      'seq_log_prob_approx',
                      'log_Z',
                      'log_Z_approx',
                      'transition_mean',
                      'transition_std',
                      'emission_mean',
                      'emission_std', 
                      'crf_pred', 
                      'crf_overlap', 
                      'crf_acc', 
                      'crf_prec',
                      'crf_recl',
                      'crf_f1',
                      'softmax_support',
                      'softmax_pred', 
                      'softmax_overlap', 
                      'softmax_acc',
                      'softmax_prec',
                      'softmax_recl',
                      'softmax_f1']
    self.validation_scores = ['crf_acc', 
                              'crf_prec',
                              'crf_recl',
                              'crf_f1',
                              'softmax_acc',
                              'softmax_prec',
                              'softmax_recl',
                              'softmax_f1']
    self.validation_criteria = validation_criteria
    return 

  def train_step(self, batch, n_iter, ei, bi):
    self.model.zero_grad()
    loss, out_dict = self.model(x=batch['sent_tokens'],
                                attention_mask=batch['attention_masks'],
                                x_gather_id=batch['token_maps'],
                                y=batch['labels'],
                                y_lens=batch['label_lens'])
    loss.backward()
    self.optimizer.step()
    self.scheduler.step()
    return out_dict

  def val_step(self, batch, n_iter, ei, bi):
    with torch.no_grad():
      _, out_dict = self.model(x=batch['sent_tokens'],
                                  attention_mask=batch['attention_masks'],
                                  x_gather_id=batch['token_maps'],
                                  y=batch['labels'],
                                  y_lens=batch['label_lens'])
    return out_dict

  def inspect_step(self, batch_dict, n_iter, ei, bi):
    return 