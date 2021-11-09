# to be executed under the path './src'

from data_utils import BBNData, News20Data
from transformers import BertTokenizer

dataset = BBNData('../data/BBN/BBN/')
dataset = News20Data(batch_size=50)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_loader = dataset.train_dataloader()
batch = next(iter(train_loader))

test_loader = dataset.test_dataloader()
batch = next(iter(test_loader))
