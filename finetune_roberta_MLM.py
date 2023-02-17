import pandas as pd
from transformers import (
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    DataCollatorForLanguageModeling
)
import torch
import random
from torch.utils.data import Sampler, Dataset, DataLoader
import numpy as np
from transformers import Trainer, TrainingArguments
import more_itertools

df = pd.read_csv('ILDC_multi.csv')
df['text'] = df['text'].str.lower().str.replace('\n',' ',regex=True)
text = list(df[~(df['split']=='test')]['text'])

with open('ILDC_text.txt','w') as f:
  for doc in text:
    f.write(doc)
    f.write('\n')

with open('ILDC_text.txt','r') as f:
  print(len(f.read().splitlines()))
  

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed(10)  

class SmartBatchingSampler(Sampler):
    def __init__(self, data_source, batch_size,shuffle=False):
        super(SmartBatchingSampler, self).__init__(data_source)
        self.len = len(data_source)
        sample_lengths = [len(seq) for seq in data_source]
        argsort_inds = np.argsort(sample_lengths)
        self.batches = list(more_itertools.chunked(argsort_inds, n=batch_size))
        self._backsort_inds = None
        self.shuffle = shuffle
    def __iter__(self):
        if self.batches:
            last_batch = self.batches.pop(-1)
            if self.shuffle:
              np.random.shuffle(self.batches)
            self.batches.append(last_batch)
        self._inds = list(more_itertools.flatten(self.batches))
        yield from self._inds

    def __len__(self):
        return self.len
    
    @property
    def backsort_inds(self):
        if self._backsort_inds is None:
            self._backsort_inds = np.argsort(self._inds)
        return self._backsort_inds

class MaskedLMDataset(Dataset):
    def __init__(self, file, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.lines = self.load_lines(file)
        self.ids = self.encode_lines(self.lines)
        self.sampler=None
    def load_lines(self,file):
        with open(file) as f:
            lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
        return lines
        
    def encode_lines(self,lines):
        batch_encoding = self.tokenizer(
            lines, add_special_tokens=True, truncation=True, max_length=256
        )

        return batch_encoding["input_ids"]
    
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return torch.tensor(self.ids[idx], dtype=torch.long)

    def get_dataloader(self, batch_size,shuffle=False):
        self.sampler = SmartBatchingSampler(
            data_source=self.ids,
            batch_size=batch_size,
            shuffle=shuffle
        )
        collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                     mlm=True,mlm_probability=0.15)
        dataloader = DataLoader(
            dataset=self,
            collate_fn=collate_fn
            , sampler=self.sampler
            , num_workers= 2
        )
        return dataloader

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
 
model = RobertaForMaskedLM.from_pretrained('roberta-base')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

dataset = MaskedLMDataset('ILDC_text.txt',tokenizer=tokenizer
)


training_args = TrainingArguments(
    output_dir='/roberta_ildc_pretrain',
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_gpu_train_batch_size=8,
    save_steps=1000,
    prediction_loss_only=True,
    fp16=True
)
 
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

print ('Start a trainer...')
# Start training
trainer.train()
 
# Save
trainer.save_model('/roberta_ildc_pretrain/')
print ('Finished training all...','/roberta_ildc_pretrain')