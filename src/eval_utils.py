"""Evaluation"""
import spacy 
import codecs
import torch 
import numpy as np 
import pickle
from time import time 
from sklearn.metrics import v_measure_score
from tqdm import tqdm
from collections import Counter, OrderedDict

def align_tags(latent_tags, defined_tags, thres=0.9):
  """Align tags associated with BERT tokenization to predefined tags
  we say a latent state i is associated with a predefined tag j if there exist
  a set of words (w1, w2, ..., wk), where the wi are representative words of i 
  and j and occ(w1 with state i) + ... + occ(wk with state i) > 
                                                    .9 * total occ of state i
  E.g., if occ(state 5) = 1000
           occ(good with state 5) + occ(worse with state 5) = 910
           good and worse are representative words for 'adjective'
           then we say latent state 5 align with POS tag 'adjective'
  """
  latent_to_defined = {}
  for l in latent_tags:
    l_occ = 0
    for w in latent_tags[l]: l_occ += latent_tags[l][w]
    for d in defined_tags:
      d_repr_words = set(defined_tags[d].keys())
      overlap = 0
      for w in latent_tags[l]:
        if(w in d_repr_words): overlap += latent_tags[l][w]
      # print(overlap, l_occ, l, d)
      if(overlap / l_occ > thres): 
        latent_to_defined[l] = d
        break
  return latent_to_defined

def compute_representative_words(tag_word_dict, thres=0.9):
  """A set of words are said to be the representative words for a given state 
  if the occurrence of these words under this state is more than .9 of the 
  total occurrence of the state
  E.g., if occ(state 5) = 1000
           occ(w1 with state 5) + occ(w2 with state 5) = 910
           then (w1, w2) is the representative word of the state
  """
  tag_word_dict_repr = {}
      
  for k in tag_word_dict:
    total_freq = 0.
    for w in tag_word_dict[k]: total_freq += tag_word_dict[k][w]
    cumsum = 0
    tag_word_dict_repr[k] = OrderedDict()
    for w, c in tag_word_dict[k].most_common():
      tag_word_dict_repr[k][w] = c
      cumsum += c
      if(cumsum / total_freq > thres): break
      # print(cumsum / total_freq)
  return tag_word_dict_repr

def compute_tag_word_dict(tag_all, token_all):
    tag_word_dict = {}
    for tags, tokens in zip(tag_all, token_all):
        for tag, tok in zip(tags, tokens):
            if(tag not in tag_word_dict): tag_word_dict[tag] = [tok]
            else: tag_word_dict[tag].append(tok)
    for t in tag_word_dict: tag_word_dict[t] = Counter(tag_word_dict[t])
    return tag_word_dict

def compute_covered_word_occ(latent_to_defined, latent_word_dict, full_occ):
  covered = 0
  for l in latent_to_defined:
    covered += np.sum(list(latent_word_dict[l].values()))
  covered_occ_r = float(covered) / full_occ
  return covered_occ_r

def compute_prec_recl(defined_tags_all_, latent_tags_spacy_, 
                      defined_to_id, id_to_defined, latent_to_defined):
  """Compute precision and recall"""
  pred = 0
  recl = 0
  prec = 0
  for e, l in zip(defined_tags_all_, latent_tags_spacy_):
    if(l in latent_to_defined):
        el = defined_to_id[latent_to_defined[l]]
        prec += 1
        if(el == e and e in id_to_defined): 
            pred += 1
    if(e in id_to_defined): recl += 1
  prec = pred / (float(prec) + 0.001)
  recl = pred / (float(recl) + 0.001)
  return prec, recl

class LatentStateEval(object):

  def __init__(self, file_path='', stopwords=None):
    """Evaluation suite"""
    self.stopwords = stopwords
    print('Loading evaluation suite ...')
    start_time = time()

    # NOTE: these .pkl files are from scripts/compare_tag_dev.ipynb
    self.spacy_tokenized_all = pickle.load(open(file_path + 'spacy_tokenized.pkl', 'rb'))
    self.bert_tokenized_all = pickle.load(open(file_path + 'bert_tokenized.pkl', 'rb'))
    self.pos_tags_all = pickle.load(open(file_path + 'pos_tags.pkl', 'rb'))
    self.ent_tags_all = pickle.load(open(file_path + 'ent_tags.pkl', 'rb'))
    self.fine_pos_tags_all = pickle.load(open(file_path + 'fine_pos_tags.pkl', 'rb'))

    self.pos_tags_all_ = []
    for l in self.pos_tags_all: self.pos_tags_all_.extend(l)
    self.fine_pos_tags_all_ = []
    for l in self.fine_pos_tags_all: self.fine_pos_tags_all_.extend(l)
    self.ent_tags_all_ = []
    for l in self.ent_tags_all: self.ent_tags_all_.extend(l)

    self.bert_to_spacy_all = pickle.load(open(file_path + 'bert_to_spacy.pkl', 'rb'))
    self.pos_word_dict_repr = pickle.load(open(file_path + 'pos_word_dict_repr.pkl', 'rb'))
    self.fine_pos_word_dict_repr = pickle.load(open(file_path + 'fine_pos_word_dict_repr.pkl', 'rb'))
    self.ent_word_dict_repr = pickle.load(open(file_path + 'ent_word_dict_repr.pkl', 'rb'))
    self.bcn_word_dict_repr = pickle.load(open(file_path + 'bcn_word_dict_repr.pkl', 'rb'))
    self.ccg_word_dict_repr = pickle.load(open(file_path + 'ccg_word_dict_repr.pkl', 'rb'))

    self.id_to_pos = pickle.load(open(file_path + 'id_to_pos.pkl', 'rb'))
    self.pos_to_id = {self.id_to_pos[i]: i for i in self.id_to_pos}
    self.id_to_ent = pickle.load(open(file_path + 'id_to_ent.pkl', 'rb'))
    self.ent_to_id = {self.id_to_ent[i]: i for i in self.id_to_ent}
    self.id_to_fine_pos = pickle.load(open(file_path + 'id_to_fine_pos.pkl', 'rb'))
    self.fine_pos_to_id = {self.id_to_fine_pos[i]: i for i in self.id_to_fine_pos}
    print('... finished within %.2f seconds' % (time() - start_time))
    return

  def eval_state(self, pred_batches, num_states=2000):
    """Meaning decomposition of learned latent states
    
    Args:
      bert_tokenized: [batch, max_len]
      spacy_tokenized: [batch, max_len]
    """
    out_dict = {}

    ## Gather all predictions
    latent_tags = []
    for pred in pred_batches:
      z_batch = pred['z']
      lens = pred['lens']
      for li, l in enumerate(lens):
        latent_tags.append(z_batch[li][1:l-1])
    # latent to word where words are from bert tokenization
    latent_word_dict_bert = compute_tag_word_dict(latent_tags, self.bert_tokenized_all)
    latent_word_dict_bert_repr = compute_representative_words(latent_word_dict_bert)
    out_dict['activated_states_bert'] = len(latent_word_dict_bert)
    print('%d activated states with bert tokenization' % len(latent_word_dict_bert))

    ## convert a tag seq upon bert-tokenized to spacy-tokenized
    latent_tags_spacy = []
    assert(len(self.bert_to_spacy_all) == len(latent_tags))
    for bert2spacy, tags in tqdm(zip(self.bert_to_spacy_all, latent_tags)):
      prev_spacy_idx = -1
      tags_converted = []
      assert(len(bert2spacy) == len(tags)) # make sure pre-stored data matches the dev data
      for bi, (si_, t) in enumerate(zip(bert2spacy, tags)):
        for si in si_:
          assert(si == prev_spacy_idx or si == prev_spacy_idx + 1)
          # if many consequtive BERT token correspond to the same spacy token, 
          # then only use the tag for the first bert token
          if(si == prev_spacy_idx + 1): 
            prev_spacy_idx += 1
            tags_converted.append(t)
      latent_tags_spacy.append(tags_converted)
    latent_word_dict = compute_tag_word_dict(latent_tags_spacy, self.spacy_tokenized_all)
    latent_word_dict_repr = compute_representative_words(latent_word_dict)
    out_dict['activated_states'] = len(latent_word_dict_repr)
    print('%d activated states after aligned to spacy tokenization' % len(latent_word_dict))

    ## compute v measure 
    latent_tags_spacy_ = []
    for l in latent_tags_spacy: latent_tags_spacy_.extend(l)
    # v_pos = v_measure_score(self.pos_tags_all_, latent_tags_spacy_)
    # v_fine_pos = v_measure_score(self.fine_pos_tags_all_, latent_tags_spacy_)
    # v_ent = v_measure_score(self.ent_tags_all_, latent_tags_spacy_)
    # out_dict['v_pos'] = v_pos
    # out_dict['v_fine_pos'] = v_fine_pos
    # out_dict['v_ent'] = v_ent

    ## full word occurrence
    full_occ = 0
    for l in latent_word_dict: full_occ += np.sum(list(latent_word_dict[l].values()))
    full_occ = float(full_occ)

    ## compute subwords occ
    subwords_occ = 0
    for l in latent_word_dict:
      for w in latent_word_dict[l]:
        if(w[:2] == '##'): subwords_occ += latent_word_dict[l][w]
    out_dict['subwords_occ'] = subwords_occ
    out_dict['subwords_ratio'] = subwords_occ / full_occ

    ## Compute aligned latent tags
    latent_to_pos = align_tags(latent_word_dict_repr, self.pos_word_dict_repr)
    out_dict['pos_to_latent'] = len(latent_to_pos)
    pos_covered = compute_covered_word_occ(latent_to_pos, latent_word_dict, full_occ)
    out_dict['pos_covered'] = pos_covered
    latent_to_fine_pos = align_tags(latent_word_dict_repr, self.fine_pos_word_dict_repr)
    out_dict['fine_pos_to_latent'] = len(latent_to_fine_pos)
    fine_pos_covered = compute_covered_word_occ(latent_to_fine_pos, latent_word_dict, full_occ)
    out_dict['fine_pos_covered'] = fine_pos_covered
    latent_to_ent = align_tags(latent_word_dict_repr, self.ent_word_dict_repr)
    out_dict['ent_to_latent'] = len(latent_to_ent)
    ent_covered = compute_covered_word_occ(latent_to_ent, latent_word_dict, full_occ)
    out_dict['ent_covered'] = ent_covered
    latent_to_bcn = align_tags(latent_word_dict_repr, self.bcn_word_dict_repr)
    out_dict['bcn_to_latent'] = len(latent_to_bcn)
    bcn_covered = compute_covered_word_occ(latent_to_bcn, latent_word_dict, full_occ)
    out_dict['bcn_covered'] = bcn_covered
    latent_to_ccg = align_tags(latent_word_dict_repr, self.ccg_word_dict_repr)
    out_dict['ccg_to_latent'] = len(latent_to_ccg)
    ccg_covered = compute_covered_word_occ(latent_to_ccg, latent_word_dict, full_occ)
    out_dict['ccg_covered'] = ccg_covered

    ## Compute not aligned tags
    total_aligned = set(latent_to_pos.keys())\
                      .union(set(latent_to_fine_pos.keys()))\
                      .union(set(latent_to_ent.keys()))\
                      .union(set(latent_to_bcn.keys()))\
                      .union(set(latent_to_ccg.keys()))
    num_not_aligned = num_states - len(total_aligned)
    out_dict['not_aligned'] = num_not_aligned
    total_covered = 0
    for l in total_aligned: total_covered += np.sum(list(latent_word_dict[l].values()))
    not_covered_r = 1 - total_covered / full_occ
    out_dict['not_covered_r'] = not_covered_r

    ## How many states covers 90% of the not aligned tags
    not_aligned = set(range(2000)) - total_aligned
    not_aligned_occ = []
    for l in not_aligned: 
      if(l in latent_word_dict_repr):
        not_aligned_occ.append((l, np.sum(list(latent_word_dict_repr[l].values()))))
    not_aligned_occ.sort(key=lambda x:x[1], reverse=True)
    not_align_occ_total = float(np.sum(list(x[1] for x in not_aligned_occ)))
    not_aligned_top_cnt = 0
    not_aligned_top = []
    for l, c in not_aligned_occ:
      not_aligned_top_cnt += c
      not_aligned_top.append(l)
      if(not_aligned_top_cnt / not_align_occ_total > 0.9): break
    out_dict['not_aligned_top'] = len(not_aligned_top)

    ## How many stop words occurrence are within the not aligned tags
    stop_occ = 0
    for l in not_aligned_top:
      if(l in latent_word_dict_repr):
        for w in latent_word_dict_repr[l]:
          if(w in self.stopwords): stop_occ += latent_word_dict_repr[l][w]
    out_dict['stop_occ_in_not_aligned'] = stop_occ
    out_dict['stop_occ_r_in_not_aligned'] = stop_occ / not_align_occ_total
        
    ## Compute precision and recall
    pos_prec, pos_recl = compute_prec_recl(self.pos_tags_all_, latent_tags_spacy_, 
                                  self.pos_to_id, self.id_to_pos, latent_to_pos)
    out_dict['pos_prec'] = pos_prec
    out_dict['pos_recl'] = pos_recl
    fine_pos_prec, fine_pos_recl = compute_prec_recl(self.fine_pos_tags_all_, latent_tags_spacy_, 
                                  self.fine_pos_to_id, self.id_to_fine_pos, latent_to_fine_pos)
    out_dict['fine_pos_prec'] = fine_pos_prec
    out_dict['fine_pos_recl'] = fine_pos_recl
    ent_prec, ent_recl = compute_prec_recl(self.ent_tags_all_, latent_tags_spacy_, 
                                  self.ent_to_id, self.id_to_ent, latent_to_ent)
    out_dict['ent_prec'] = ent_prec
    out_dict['ent_recl'] = ent_recl
    return out_dict, not_aligned_occ, latent_word_dict_repr

def compute_w_ent(w_dist):
  if(w_dist.sum() == 0): return 0
  w_dist = w_dist / float(w_dist.sum())
  w_ent = -(w_dist * np.log(w_dist + 1e-5)).sum()
  return w_ent

def compute_ent_stopword(w_dist, stopwords, tokenizer):
  sw_cnt = 0
  non_sw_cnt = 0
  for wid, w_cnt in enumerate(w_dist):
    w = tokenizer.convert_ids_to_tokens(wid)
    if(w in stopwords): sw_cnt += w_cnt
    else: non_sw_cnt += w_cnt
  if(sw_cnt + non_sw_cnt == 0): 
    assert(w_dist.sum() == 0)
    # import ipdb; ipdb.set_trace()
    ent = -1
  else: 
    p = sw_cnt / float(sw_cnt + non_sw_cnt)
    ent = -p * np.log(p + 1e-8) - (1 - p) * np.log(1 - p + 1e-8)
    # if(ent < 0):
    #   import pdb; pdb.set_trace()
  return ent

def compute_ent_subword(w_dist, tokenizer):
  sw_cnt = 0
  non_sw_cnt = 0
  for wid, w_cnt in enumerate(w_dist):
    w = tokenizer.convert_ids_to_tokens(wid)
    if(w[:2] == '##'): sw_cnt += w_cnt
    else: non_sw_cnt += w_cnt
  if(sw_cnt + non_sw_cnt == 0): 
    # assert(w_dist.sum() == 0)
    ent = -1
  else:
    p = sw_cnt / float(sw_cnt + non_sw_cnt)
    ent = -p * np.log(p + 1e-8) - (1 - p) * np.log(1 - p + 1e-8)
  return ent