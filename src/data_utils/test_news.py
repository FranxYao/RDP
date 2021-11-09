import json
from tqdm import tqdm 
from collections import Counter
from transformers import BertTokenizer
from data_utils import News20Data

# data_path = '../data/news/'

dataset = News20Data()

train_loader = dataset.train_dataloader()
print(dataset.num_batch_per_epoch)

for i, d in enumerate(train_loader):
  if(i % 1000 == 0): print(d['sent_lens'])
  # if(i > 2000): break

batch = next(iter(train_loader))
print(batch['sent_lens'])

# indices = list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
# sampler = BucketSampler(indices)

# for ind in sampler: 
#   print(ind)