%load_ext autoreload
%autoreload 2

from data_utils import MSCOCOData
from transformers import GPT2Tokenizer, GPT2Model

dataset = MSCOCOData(subsample=True)
dataset.train_dataset.sentences.shape

train_loader = dataset.train_dataloader()
batch = next(iter(train_loader))

val_loader = dataset.val_dataloader()
batch = next(iter(val_loader))

gpt2 = GPT2Model.from_pretrained('gpt2')
enc = gpt2(batch['input_ids'][:, 1:])