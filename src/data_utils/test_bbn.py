# to be executed under src/ 

import json
from tqdm import tqdm 
from collections import Counter
from transformers import BertTokenizer

train_path = '../data/BBN/BBN/train.json'

train_data = [json.loads(l) for l in open(train_path)]

all_labels = []
all_atom_labels = []

for d in train_data:
  for e in d['mentions']:
    for l in e['labels']:
      all_labels.append(l)
      all_atom_labels.extend(l.split('/'))

all_labels = Counter(all_labels)
all_atom_labels = Counter(all_atom_labels)

with open('../data/BBN/BBN/label_vocab.txt', 'w') as fd:
  for l in all_labels:
    fd.write('B-' + l + '\n')
    fd.write('I-' + l + '\n')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sent_lens = []
sentences = []
for d in tqdm(train_data):
  s = tokenizer(' '.join(d['tokens']))['']
  sentences.append(s)
  sent_lens.append(len(s))
sent_lens = Counter(sent_lens)