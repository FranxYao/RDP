"""Dataset class for loading 20news 

NOTE: this file also gives an example implementation of bucketing with the 
pytorch DataLoader class (because Bucketing is no longer supported by torchtext)
Currently we suggest to entirely abandon torchtext (because of its poor 
maintainance)

References:
https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler
"""

import torch 
import codecs
import numpy as np 

# from tqdm import tqdm
from time import time 
from torch.utils.data import (Dataset, DataLoader, random_split)
from transformers import BertTokenizer
from frtorch import BucketSampler
from frtorch import torch_model_utils as tmu


def collate_fn(batch):
  sent_lens = np.array([len(s['input_ids']) for s in batch])
  max_len = max(sent_lens)
  input_ids = np.array(
    [tmu.pad_or_trunc_seq(s['input_ids'], max_len) for s in batch])
  attention_mask = np.array([
    tmu.pad_or_trunc_seq(s['attention_mask'], max_len) for s in batch])
  batch_dict={'input_ids': torch.tensor(input_ids), 
              'attention_mask': torch.tensor(attention_mask),
              'sent_lens': torch.tensor(sent_lens)
              }
  return batch_dict

class News20Dataset(Dataset):
  def __init__(self, data):
    super(News20Dataset, self).__init__()
    self.data = data
    return 

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    instance = {}
    instance['input_ids'] = self.data[idx][0]
    instance['attention_mask'] = self.data[idx][1]
    return instance

class News20Data(object):
  """Processed 20News Dataset. 
  """
  def __init__(self, data_path='../data/news/', batch_size=10, is_test=False):
    super(News20Data, self).__init__()
    print('Processing dataset ...')

    self.batch_size = batch_size
    self.data_path = data_path
    print('Reading data ...')
    start_time = time()
    with codecs.open(data_path + '20news.txt', encoding='utf-8') as fd:
      data = fd.readlines()
    print('... %d seconds' % (time() - start_time))

    # TODO: set train index and dev index 
    train_idx = np.load(data_path + 'train_idx.npy')
    dev_idx = np.load(data_path + 'dev_idx.npy')
    test_idx = np.load(data_path + 'test_idx.npy')

    train_data = [data[i][:-1] for i in train_idx]
    dev_data = [data[i][:-1] for i in dev_idx]
    test_data = [data[i][:-1] for i in test_idx]

    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if(is_test == False):
      print('Tokenizing and sorting train data ...')
      start_time = time()
      train_data = self.tokenizer(train_data)
      train_data_ = []
      for s, a in zip(train_data['input_ids'], train_data['attention_mask']):
        if(len(s) > 2):
          train_data_.append((s, a))
      train_data_.sort(key=lambda x: len(x[0]))
      train_data = train_data_
      print('... %d seconds' % (time() - start_time))

    print('Tokenizing dev data ...')
    start_time = time()
    dev_data = self.tokenizer(dev_data)
    dev_data_ = []
    for s, a in zip(dev_data['input_ids'], dev_data['attention_mask']):
      if(len(s) > 2):
        dev_data_.append((s, a))
    dev_data = dev_data_
    # dev_data.sort(key=lambda x: len(x[0]))

    # Read dev data POS Tags and NER tags

    # Read dev data bert2spacy tokenization mapping
    print('... %d seconds' % (time() - start_time))

    print('Tokenizing test data ...')
    start_time = time()
    test_data = self.tokenizer(test_data)
    test_data_ = []
    for s, a in zip(test_data['input_ids'], test_data['attention_mask']):
      if(len(s) > 2):
        test_data_.append((s, a))
    test_data = test_data_
    # test_data.sort(key=lambda x: len(x[0]))
    print('... %d seconds' % (time() - start_time))

    # Test data annotation
    if(is_test == False):
      self.train_dataset = News20Dataset(train_data)
    else: self.train_dataset = None
    self.dev_dataset = News20Dataset(dev_data)
    self.test_dataset = News20Dataset(test_data)
    return 

  @property
  def num_batch_per_epoch(self):
    if(self.train_dataset is not None):
      num_batch = len(self.train_dataset) // self.batch_size
    else: num_batch = -1
    return num_batch

  def train_dataloader(self):
    sampler = BucketSampler(self.train_dataset, self.batch_size)
    loader = DataLoader(self.train_dataset, 
                        batch_sampler=sampler,
                        num_workers=0,
                        pin_memory=False,
                        collate_fn=collate_fn
                        )
    return loader

  def val_dataloader(self, shuffle=False):
    loader = DataLoader(self.dev_dataset, 
                        batch_size=self.batch_size, 
                        shuffle=shuffle, 
                        num_workers=0,
                        drop_last=False,
                        pin_memory=False,
                        collate_fn=collate_fn
                        )
    return loader

  def test_dataloader(self):
    loader = DataLoader(self.test_dataset, 
                        batch_size=self.batch_size, 
                        shuffle=False, 
                        num_workers=0,
                        drop_last=False,
                        pin_memory=False,
                        collate_fn=collate_fn
                        )
    return loader