import json
import torch 

import numpy as np 

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer
from frtorch import torch_model_utils as tmu


def read_labels(label_path):
  """Read the labels of the dataset"""
  label2id = {'O': 0}
  id2label = {0: 'O'}
  i = 1
  with open(label_path) as fd:
    for l in fd:
      label = l[:-1]
      label2id[label] = i
      id2label[i] = label
      i += 1
  return label2id, id2label

def get_token_map(prev_token, after_token):
  """map the tokens from the bert tokenizer to the original blank tokenizer"""
  map = []
  i = 0
  # do not count [CLS] and [SEP]
  for j in range(1, len(after_token) - 1):
    if(prev_token[i] == after_token[j]):
      map.append(j)
      i += 1
      if(i == len(prev_token)): break
    else:
      if(after_token[j][:2] == '##'):
        continue
      else:
        if(after_token[j] != prev_token[i][: len(after_token[j])]):
          # print(prev_token)
          # print(after_token)
          # print(prev_token[i])
          # print(after_token[j])
          continue
        else:
          map.append(j)
          i += 1
          if(i == len(prev_token)): break
  return map

def get_longest_label(label_set):
  longest_label = label_set[0]
  len_ = len(longest_label.split('/'))
  for l in label_set[1:]:
    if(len(l.split('/')) > len_):
      len_ = len(l.split('/'))
      longest_label = l
  return longest_label

def pipeline(data, label2id, setname):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  sent_origin = []
  sent_tokens = [] 
  token_maps = []
  labels = []
  label_lens = []
  print('processing %s' % setname)
  for d in tqdm(data):
    sent = [t.lower() for t in d['tokens']]
    tokenized = tokenizer(' '.join(sent))['input_ids']
    tokenized_back = tokenizer.convert_ids_to_tokens(tokenized)
    token_map = get_token_map(sent, tokenized_back)
    label = [0] * len(sent)
    for mention in d['mentions']:
      start = mention['start']
      end = mention['end']
      l = get_longest_label(mention['labels'])
      label[start] = label2id['B-' + l]
      for i in range(start + 1, end): # to be verified
        label[i] = label2id['I-' + l]
    sent_origin.append(sent)
    sent_tokens.append(tokenized)
    token_maps.append(token_map)
    labels.append(label)
    label_lens.append(len(label))
  return sent_origin, sent_tokens, token_maps, labels, label_lens

def get_max_len(data):
  max_len = max([len(d) for d in data])
  return max_len

def pad_max_len(data, max_len, pad):
  data = [tmu.pad_or_trunc_seq(d, max_len, pad) for d in data]
  return data

def collate_fn(batch):
  batch_dict = {
    'sent_origin': [b['sent_origin'] for b in batch],
    'sent_tokens': torch.tensor([b['sent_tokens'] for b in batch]),
    'attention_masks': torch.tensor([b['attention_masks'] for b in batch]),
    'token_maps': torch.tensor([b['token_maps'] for b in batch]),
    'labels': torch.tensor([b['labels'] for b in batch]),
    'label_lens': torch.tensor([b['label_lens'] for b in batch])
    }
  return batch_dict

class BBNDataset(Dataset):
  def __init__(self, 
               sent_origin, 
               sent_tokens, 
               attention_masks, 
               token_maps, 
               labels, 
               label_lens):
    super().__init__()
    self.sent_origin = sent_origin
    self.sent_tokens = np.array(sent_tokens)
    self.attention_masks = np.array(attention_masks)
    self.token_maps = np.array(token_maps)
    self.labels = np.array(labels)
    self.label_lens = np.array(label_lens)
    return 

  def __len__(self):
    return len(self.sent_origin)

  def __getitem__(self, idx):
    instance = {'sent_origin': self.sent_origin[idx], 
                'sent_tokens': self.sent_tokens[idx],
                'attention_masks': self.attention_masks[idx], 
                'token_maps': self.token_maps[idx], 
                'labels': self.labels[idx],
                'label_lens': self.label_lens[idx]}
    return instance

class BBNData(object):
  def __init__(self, data_path, batch_size=10):
    super().__init__()
    self.data_path = data_path
    self.batch_size = batch_size
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # read data from file 
    self.label2id, self.id2label = read_labels(data_path + '/label_vocab.txt')
    train_data = [json.loads(l) for l in open(data_path + '/train.json')]
    test_data = [json.loads(l) for l in open(data_path + '/test.json')]
    train_len = int(len(train_data) * 0.8)
    dev_len = len(train_data) - train_len
    train_data, dev_data = random_split(train_data, [train_len, dev_len])

    # data processing pipeline
    (train_sent_origin, train_sent_tokens, train_token_maps, 
      train_labels, train_label_lens) =\
      pipeline(train_data, self.label2id, 'train')
    train_attention_masks = [[1] * len(d) for d in train_sent_tokens]
    (dev_sent_origin, dev_sent_tokens, dev_token_maps, 
      dev_labels, dev_label_lens) =\
      pipeline(dev_data, self.label2id, 'dev')
    dev_attention_masks = [[1] * len(d) for d in dev_sent_tokens]
    (test_sent_origin, test_sent_tokens, test_token_maps, 
      test_labels, test_label_lens) =\
      pipeline(test_data, self.label2id, 'test')
    test_attention_masks = [[1] * len(d) for d in test_sent_tokens]

    # pad to max length
    max_len_label = max([get_max_len(train_labels), 
      get_max_len(dev_labels), get_max_len(test_labels)])
    print('max label length %d' % max_len_label)
    # use 0 as pad
    train_labels = pad_max_len(train_labels, max_len_label, 0)

    max_len_sent = max([get_max_len(train_sent_tokens), 
      get_max_len(dev_sent_tokens), get_max_len(test_sent_tokens)])
    print('max sentence length %d' % max_len_sent)
    train_sent_tokens = pad_max_len(train_sent_tokens, max_len_sent, 0)
    train_attention_masks = pad_max_len(train_attention_masks, max_len_sent, 0)
    # use the last token as train token map
    train_token_maps = pad_max_len(
      train_token_maps, max_len_label, max_len_sent - 1)
    self.train_data = BBNDataset(train_sent_origin, 
                                 train_sent_tokens, 
                                 train_attention_masks, 
                                 train_token_maps, 
                                 train_labels,
                                 train_label_lens)

    dev_labels = pad_max_len(dev_labels, max_len_label, 0)
    dev_sent_tokens = pad_max_len(dev_sent_tokens, max_len_sent, 0)
    dev_attention_masks = pad_max_len(dev_attention_masks, max_len_sent, 0)
    dev_token_maps = pad_max_len(
      dev_token_maps, max_len_label, max_len_sent - 1)
    self.dev_data = BBNDataset(dev_sent_origin,
                               dev_sent_tokens,
                               dev_attention_masks, 
                               dev_token_maps,
                               dev_labels, 
                               dev_label_lens)

    test_labels = pad_max_len(test_labels, max_len_label, 0)
    test_sent_tokens = pad_max_len(test_sent_tokens, max_len_sent, 0)
    test_attention_masks = pad_max_len(test_attention_masks, max_len_sent, 0)
    test_token_maps = pad_max_len(
      test_token_maps, max_len_label, max_len_sent - 1)
    self.test_data = BBNDataset(test_sent_origin,
                                test_sent_tokens,
                                test_attention_masks, 
                                test_token_maps,
                                test_labels, 
                                test_label_lens)
    return 

  @property
  def label_size(self):
    return len(self.label2id)

  def train_dataloader(self):
    loader = DataLoader(self.train_data, 
                        batch_size=self.batch_size, 
                        shuffle=True, 
                        num_workers=0,
                        drop_last=False,
                        pin_memory=False,
                        collate_fn=collate_fn)
    return loader

  def val_dataloader(self):
    loader = DataLoader(self.dev_data, 
                        batch_size=self.batch_size, 
                        shuffle=False, 
                        num_workers=0,
                        drop_last=False,
                        pin_memory=False,
                        collate_fn=collate_fn)
    return loader

  def test_dataloader(self):
    loader = DataLoader(self.test_data, 
                        batch_size=self.batch_size, 
                        shuffle=False, 
                        num_workers=0,
                        drop_last=False,
                        pin_memory=False,
                        collate_fn=collate_fn)
    return loader