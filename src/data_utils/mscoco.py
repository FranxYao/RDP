"""MSCOCO dataset with GPT2 tokenization"""

import json 
import torch

import numpy as np 

from tqdm import tqdm
from transformers import GPT2Tokenizer
from time import time 
from torch.utils.data import (Dataset, DataLoader, random_split)
from nltk.corpus import stopwords

STOPWORDS = stopwords.words('english')
STOPWORDS.extend(['"', "'", '.', ',', '?', '!', '-', 
      ':', '@', '/', '[', ']', '(', ')'])
STOPWORDS = set(STOPWORDS)


def mscoco_read_json(file_path, bleu_baseline=False):
  """Read the mscoco dataset
  Args:
    file_path: path to the raw data, a string
  Returns:
    sentence_sets: the sentence sets, a list of paraphrase lists
  """
  print("Reading mscoco raw data .. ")
  print("  data path: %s" % file_path)
  start_time = time()
  with open(file_path, "r") as fd:
    data = json.load(fd)

  print("%d sentences in total, %.2fsec" % 
    (len(data["annotations"]), time() - start_time))
  
  # aggregate all sentences of the same images
  image_idx = set([d["image_id"] for d in data["annotations"]])
  paraphrases = {}
  for im in image_idx: paraphrases[im] = []
  for d in tqdm(data["annotations"]):
    im = d["image_id"]
    sent = d["caption"]
    paraphrases[im].append(sent.lower())

  # filter out sentence sets size != 5 
  sentence_sets = [paraphrases[im] for im in paraphrases 
    if(len(paraphrases[im]) == 5)]

  # blue on the training set, a baseline/ upperbound
  if(bleu_baseline):
    print("calculating bleu ... ")
    hypothesis = [s[0] for s in sentence_sets]
    references = [s[1:] for s in sentence_sets]
    bleu = dict()
    bleu['1'] = corpus_bleu(
      references, hypothesis, weights=(1., 0, 0, 0))
    bleu['2'] = corpus_bleu(
      references, hypothesis, weights=(0.5, 0.5, 0, 0))
    bleu['3'] = corpus_bleu(
      references, hypothesis, weights=(0.333, 0.333, 0.333, 0))
    bleu['4'] = corpus_bleu(
      references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
    print('bleu on the training set:')
    pprint(bleu)
  return sentence_sets


def preprocess_train(tokenizer, pad_id, sentence_sets):
  """Process training set. Pad to max length"""
  print('Processing sentence')
  sentences = []
  sent_bow = []
  sent_lens = []
  bow_lens = []
  for ss in tqdm(sentence_sets):
    for si in ss:
      sent_ids = tokenizer('[BOS] ' + si)['input_ids']
      sentences.append(sent_ids)
      sent_lens.append(len(sent_ids))

      sent_back = tokenizer.convert_ids_to_tokens(sent_ids)
      s_bow = []
      for w, wi in zip(sent_back, sent_ids):
        if(w not in ['[BOS]', '[EOS]'] 
          and len(w) > 1 and w[1:].lower() not in STOPWORDS):
          s_bow.append(wi)
      sent_bow.append(s_bow)
      bow_lens.append(len(s_bow))

  max_slen = int(np.percentile(sent_lens, [95])[0])
  max_bow = int(np.percentile(bow_lens, [95])[0])

  print('Padding to max sentence length %d' % max_slen)
  sentences_ = []
  for s in sentences:
    s = s[: max_slen]
    len_ = len(s)
    for _ in range(max_slen - len_): 
      s.append(pad_id)
    assert(len(s) == max_slen)
    sentences_.append(s)
  print('Padding to max bow length %d' % max_bow)
  sent_bow_ = []
  for bow in sent_bow:
    bow = bow[: max_bow]
    len_ = len(bow)
    for _ in range(max_bow - len_): 
      bow.append(pad_id)
    assert(len(bow) == max_bow)
    sent_bow_.append(bow)
  return sentences_, sent_bow_, max_slen, max_bow

def preprocess_test(tokenizer, pad_id, sentence_sets, max_slen, max_bow):
  """"""
  refs = []
  sentences = []
  bow = []
  for ss in tqdm(sentence_sets):
    sent_ids = tokenizer('[BOS] ' + ss[0])['input_ids']
    sent_back = tokenizer.convert_ids_to_tokens(sent_ids)
    s_bow = []
    for w, wi in zip(sent_back, sent_ids):
      if(w not in ['[BOS]', '[EOS]'] 
        and len(w) > 1 and w[1:].lower() not in STOPWORDS):
        s_bow.append(wi)

    sent_ids = sent_ids[: max_slen]
    slen = len(sent_ids)
    for _ in range(max_slen - slen): sent_ids.append(pad_id)
    assert(len(sent_ids) == max_slen)
    sentences.append(sent_ids)
    
    s_bow = s_bow[: max_bow]
    blen = len(s_bow)
    for _ in range(max_bow - blen): s_bow.append(pad_id)
    assert(len(s_bow) == max_bow)
    bow.append(s_bow)

    refs.append(ss[1:])
  return sentences, bow, refs


class MSCOCOTrainDataset(Dataset):

  def __init__(self, sentences, bow):
    super(MSCOCOTrainDataset, self).__init__()
    self.sentences = torch.tensor(sentences)
    self.bow = torch.tensor(bow)
    return 

  def __len__(self):
    return len(self.sentences)

  def __getitem__(self, idx):
    instance = {}
    instance['input_ids'] = self.sentences[idx]
    instance['bow'] = self.bow[idx]
    return instance

def collate_fn(batch):
  sents = []
  bow = []
  refs = []
  for b in batch:
    sents.append(b['input_ids'])
    bow.append(b['bow'])
    refs.append(b['references'])
  batch = {'input_ids': torch.tensor(sents), 
           'bow': torch.tensor(bow), 
           'references': refs
           }
  return batch

class MSCOCOTestDataset(Dataset):

  def __init__(self, sentences, bow, references):
    super(MSCOCOTestDataset, self).__init__()
    self.sentences = sentences
    self.bow = bow
    self.references = references
    return 

  def __len__(self):
    return len(self.sentences)

  def __getitem__(self, idx):
    instance = {}
    instance['input_ids'] = self.sentences[idx]
    instance['bow'] = self.bow[idx]
    instance['references'] = self.references[idx]
    return instance

class MSCOCOData(object):
  def __init__(self, data_path='../data/mscoco/', batch_size=10, 
    test_batch_size=10, subsample=False, cache_dir=''):
    super(MSCOCOData, self).__init__()
    print('Processing mscoco data ... ')

    self.data_path = data_path
    self.batch_size = batch_size
    self.test_batch_size = test_batch_size

    # print('Processing  ... ')
    if(cache_dir != ''):
      print('reading data from %s' % cache_dir)
      self.tokenizer = GPT2Tokenizer.from_pretrained(
        'gpt2', bos_token='[BOS]', cache_dir=cache_dir)
    else:
      self.tokenizer = GPT2Tokenizer.from_pretrained(
        'gpt2', bos_token='[BOS]')
    # register pad token
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.bos_id = self.tokenizer.convert_tokens_to_ids(['[BOS]'])[0]
    self.eos_id = self.tokenizer.convert_tokens_to_ids(['<|endoftext|>'])[0]
    self.pad_id = self.tokenizer.convert_tokens_to_ids(['<|endoftext|>'])[0]

    self.word2id = self.tokenizer.get_vocab()
    self.id2word = {self.word2id[word]: word for word in self.word2id}
    self.space_token = self.id2word[1001][0]

    train_data = mscoco_read_json(data_path + 'captions_train2014.json')
    test_data = mscoco_read_json(data_path + 'captions_val2014.json')

    # Hard set dataset sizes
    if(subsample): 
      train_idx = np.load(data_path + 'train_idx.npy')
      dev_idx = np.load(data_path + 'dev_idx.npy')
      print(len(train_data))
      print(train_idx.max(), dev_idx.max())
      dev_data = [train_data[i] for i in dev_idx]
      train_data = [train_data[i] for i in train_idx]
      test_data = list(test_data)[:1000]
    else:
      # train_len = int(0.8 * len(train_data))
      # dev_len = len(train_data) - train_len
      # train_data, dev_data = random_split(train_data, [train_len, dev_len])
      print('Loading train and dev index')
      train_idx = np.load(data_path + 'train_idx_full.npy')
      dev_idx = np.load(data_path + 'dev_idx_full.npy')
      # print(len(train_data))
      # print(train_idx.max(), dev_idx.max())
      dev_data = [train_data[i] for i in dev_idx]
      train_data = [train_data[i] for i in train_idx]

    train_sentences, train_bow, max_slen, max_bow = preprocess_train(
      self.tokenizer, self.pad_id, train_data)
    self.train_dataset = MSCOCOTrainDataset(train_sentences, train_bow)
    self.max_slen = max_slen
    self.max_bow = max_bow

    dev_sentences, dev_bow, dev_references = preprocess_test(
      self.tokenizer, self.pad_id, dev_data, max_slen, max_bow)
    self.dev_dataset = MSCOCOTestDataset(
      dev_sentences, dev_bow, dev_references)

    test_sentences, test_bow, test_references = preprocess_test(
      self.tokenizer, self.pad_id, test_data, max_slen, max_bow)
    self.test_dataset = MSCOCOTestDataset(
      test_sentences, test_bow, test_references)
    return 

  def decode_sent(self, s, z=None):
    """Decode index to sentence"""
    sent = []
    if(z is None):
      for si in s:
        siw = self.id2word[si]
        if(siw[0] == self.space_token): siw = siw[1:]
        if(si == self.eos_id): break
        if(si != self.bos_id): sent.append(siw)
      sent = ' '.join(sent)
      return sent
    else:
      states = []
      for si, zi in zip(s, z):
        siw = self.id2word[si]
        if(siw[0] == self.space_token): siw = siw[1:]
        if(si == self.eos_id): break
        if(si != self.bos_id): 
          sent.append(siw)
          states.append(zi)
      sent = ' '.join(sent)
      return sent, states
  
  @property
  def num_batch_per_epoch(self):
    num_batch = len(self.train_dataset) // self.batch_size
    return num_batch

  def train_dataloader(self):
    loader = DataLoader(self.train_dataset, 
                        batch_size=self.batch_size, 
                        shuffle=True, 
                        num_workers=0, 
                        pin_memory=False, 
                        drop_last=True
                        )
    return loader

  def val_dataloader(self):
    loader = DataLoader(self.dev_dataset, 
                        batch_size=self.test_batch_size, 
                        shuffle=False, 
                        num_workers=0,
                        drop_last=False,
                        pin_memory=False,
                        collate_fn=collate_fn)
    return loader

  def test_dataloader(self):
    loader = DataLoader(self.test_dataset, 
                        batch_size=self.test_batch_size, 
                        shuffle=False, 
                        num_workers=0,
                        drop_last=False,
                        pin_memory=False,
                        collate_fn=collate_fn)
    return loader
