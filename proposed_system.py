from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import json
import pandas as pd
from torchcrf import CRF
from sklearn.preprocessing import MultiLabelBinarizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

from transformers import RobertaTokenizerFast


from transformers import RobertaModel

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

START_TAG = "<START>"
STOP_TAG = "<STOP>"
#Adding start and stop tags to the tag index list
tag_to_ix = {'PREAMBLE':0, 'NONE':1, 'FAC':2, 'ARG_RESPONDENT':3, 'RLC':4,
       'ARG_PETITIONER':5, 'ANALYSIS':6, 'PRE_RELIED':7, 'RATIO':8, 'RPC':9,
       'ISSUE':10, 'STA':11, 'PRE_NOT_RELIED':12,START_TAG:13,STOP_TAG:14}

max_seq_len = 256

hdp_features_train = torch.load('/legalevalRR_docs_train_hdp_submission.pt')
hdp_features_dev = torch.load('/legalevalRR_docs_dev_hdp_submission.pt')

df = pd.read_csv('legalevalRR_.csv')

documents_text = []
documents_labels = []
attention_masks_doc = []
input_ids_doc = []

df.dropna(inplace=True)

for id in df['id'].unique():
    sentences_text = df[df['id']==id]['text'].tolist()
    sentences_labels = df[df['id']==id]['label'].tolist()
    encoded = tokenizer(sentences_text,truncation=True,
                        add_special_tokens = True,
                        return_tensors = 'pt',
                        return_attention_mask=True,
                        padding ='longest',
                        max_length =  max_seq_len)
    input_ids = encoded['input_ids']
    attention_masks = encoded['attention_mask']
    input_ids_doc.append(input_ids)
    attention_masks_doc.append(attention_masks)
    tags = torch.tensor([tag_to_ix[label] for label in sentences_labels], dtype=torch.long)
    # Adding start tag to beginning of the list
    tags = torch.cat([torch.tensor([tag_to_ix[START_TAG]], dtype=torch.long), tags])
    sentences_label_ids = tags

    documents_labels.append(sentences_label_ids)
                                              


train_x = input_ids_doc
train_y = documents_labels
train_masks = attention_masks_doc


with open('dev.json','r') as f:
    data = json.loads(f.read())
documents = []
for doc in data:
    sentences = []
    sent_labels = []
    sentences.append(doc['id'])
    sents = []
    for sent in doc['annotations'][0]['result']:
        sents.append(" ".join(sent['value']['text'].lower().replace('\n',' ').replace('...',' ').replace(',',' ')[:-1].split()))
        sent_labels.append(sent['value']['labels'][0])
    sentences.append(sents)
    sentences.append(sent_labels)
    documents.append(sentences)
df_dev = pd.DataFrame(documents,columns=['id','text','label']).explode(['text','label'])

documents_text = []
documents_labels = []
#attention_masks_doc = []
input_ids_doc = []
attention_masks_doc = []
df = df_dev
for id in df['id'].unique():
    sentences_text = df[df['id']==id]['text'].tolist()
    sentences_labels = df[df['id']==id]['label'].tolist()
    encoded = tokenizer(sentences_text,truncation=True,
                                          add_special_tokens = True,
                                          return_tensors = 'pt',
                                          return_attention_mask=True,
                                              padding = 'longest',
                                            max_length =  max_seq_len)
    attention_masks_doc.append(encoded['attention_mask'])
    #attention_masks = encoded['attention_mask']
    input_ids_doc.append(encoded['input_ids'])
    #attention_masks_doc.append(attention_masks)
    tags = torch.tensor([tag_to_ix[label] for label in sentences_labels], dtype=torch.long)
    # Adding start tag to beginning of the list
    tags = torch.cat([torch.tensor([tag_to_ix[START_TAG]], dtype=torch.long), tags])
    sentences_label_ids = tags

    documents_labels.append(sentences_label_ids)
                                  

dev_x = input_ids_doc
dev_y = documents_labels
dev_masks = attention_masks_doc

data_tr = zip(train_x,train_y,train_masks,hdp_features_train)
data_tr = sorted(data_tr,key = lambda x:x[0].size()[0])
train_x,train_y,train_masks,hdp_features_train = zip(*data_tr)

data_dev = zip(dev_x,dev_y,dev_masks,hdp_features_dev)
data_dev = sorted(data_dev,key = lambda x:x[0].size()[0])
dev_x,dev_y,dev_masks,hdp_features_dev = zip(*data_dev)

EMBEDDING_DIM = 768

HIDDEN_DIM_Doc = 50

batch_size = 2

epochs = 25

learning_rate = 1e-4 # run 10 --- 2e-5, hidden dim =50


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels,masks, hdp_feats):
        'Initialization'
        self.labels = labels
        self.data = data
        self.masks = masks
        self.hdp_feats = hdp_feats

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
       
        X = self.data[index]
        y = self.labels[index]
        mask = self.masks[index]
        sent_features_hdp = self.hdp_feats[index]

        return X, y, mask, sent_features_hdp

def collate_fn(batch):

  x, y, mask,  hdp_feats = list(map(list, zip(*batch)))
  lens_x = [z.size()[1] for z in x]
  l = max(lens_x)
  
  x_padded = [torch.nn.functional.pad(z,value=tokenizer.pad_token_id,mode='constant',pad = (1,l-z.size()[1]-1)) for z in x]
  num_sent = [z.size()[0] for z in x_padded]
  max_num_sent = max(num_sent)

  x_padded_sent = [torch.cat([z,torch.full((max_num_sent-z.size()[0],l),
                                                dtype=torch.long, fill_value = tokenizer.pad_token_id)]) for z in x_padded]
  y_padded_sent = [torch.nn.functional.pad(z,value=tag_to_ix[STOP_TAG],mode='constant', pad=(0,max_num_sent+1-len(z))) for z in y]
  x_padded = [torch.nn.functional.pad(z,value=tokenizer.pad_token_id,mode='constant',pad = (1,l-z.size()[1]-1)) for z in x]
  
  x_masks_padded = [torch.nn.functional.pad(z,value=tokenizer.pad_token_id,mode='constant',pad = (1,l-z.size()[1]-1)) for z in mask]
  x_masks_padded_sent = [torch.cat([z,torch.full((max_num_sent-z.size()[0],l),
                                                dtype=torch.long, fill_value = tokenizer.pad_token_id)]) for z in x_masks_padded]
  hdp_feats = [torch.cat([z,torch.zeros((max_num_sent-z.size()[0],53), # 53 is the number of topics found, this may be changed depending on run of hdp
                                                dtype=torch.long)],0) for z in hdp_feats]
  
  
  return torch.stack(x_padded_sent),torch.stack(y_padded_sent), torch.stack(x_masks_padded_sent), y, torch.nan_to_num(torch.stack(hdp_feats)).float()

training_dataset = Dataset(train_x,train_y,train_masks,hdp_features_train)
val_dataset = Dataset(dev_x,dev_y,dev_masks,hdp_features_dev)
training_generator = torch.utils.data.DataLoader(training_dataset,
                                                 batch_size=batch_size,
                                                 collate_fn = collate_fn,
                                                 shuffle=False)
val_generator = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            collate_fn = collate_fn,
                                            shuffle=False)

class model1(nn.Module):

    def __init__(self,  tag_to_ix,  hidden_dim_doc, dropout_p = 0.1,embedding_dim=768, num_topics=53):
        super(model1, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim_doc = hidden_dim_doc
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        #self.dropout_p = dropout_p
        #self.dropout_sent = nn.Dropout(self.dropout_p)
        self.word_embeds = RobertaModel.from_pretrained('/roberta_ildc_pretrain/checkpoint-20000')
        for param in self.word_embeds.parameters():
          param.requires_grad=False
        modules = [self.word_embeds.pooler, self.word_embeds.encoder.layer[11]] 
        for module in modules:
          for param in module.parameters():
            param.requires_grad = True
        self.lstm_doc = nn.LSTM(self.embedding_dim+num_topics, hidden_dim_doc,
                                num_layers=1, bidirectional=True,batch_first=True)
        
        
        # Maps the output of the sentence LSTM into tag space.
        self.hidden2tag = nn.Linear(2*hidden_dim_doc, self.tagset_size)

        self.crf = CRF(self.tagset_size,batch_first=True)

    def init_hidden_doc(self, hidden_dim):
        return (torch.randn(2, self.batch_size, hidden_dim).to('cuda'),
                torch.randn(2, self.batch_size, hidden_dim).to('cuda'))

    def init_hidden_sent(self, hidden_dim):
        return (torch.randn(2, self.max_num_sent, hidden_dim).to('cuda'),
                torch.randn(2, self.max_num_sent, hidden_dim).to('cuda'))
    
    

    def _get_lstm_features(self, document,mask, hdp_feats):
        self.hidden_doc = self.init_hidden_doc(self.hidden_dim_doc)
        sentence_embeddings = []
        #print(document.size())
        for indx in range(document.size()[0]):
          doc = document[indx]
          #print(doc.size())
          self.max_num_sent = doc.size()[0]
          embeds = self.word_embeds(input_ids = doc,attention_mask= mask[indx]).last_hidden_state[:,0,:]
          sentence_embeddings.append(torch.cat([embeds.view(self.max_num_sent, 1, -1),
                                               hdp_feats[indx].view(self.max_num_sent, 1, -1)],2))
        sentence_embeddings = torch.stack(sentence_embeddings)
        #print(sentence_embeddings.squeeze(2).size())
        lstm_out_doc, self.hidden_doc = self.lstm_doc(sentence_embeddings.squeeze(2),self.hidden_doc)
        #print(lstm_out_doc.size())
        lstm_feats = self.hidden2tag(lstm_out_doc)
        #print(lstm_feats.size())
        return lstm_feats

    def neg_log_likelihood(self, document, tags, masks, hdp_feats):
        self.batch_size = document.size()[0]
        feats = self._get_lstm_features(document,masks, hdp_feats)
        loss = - self.crf.forward(emissions=feats,tags=tags[:,1:],mask=(torch.sum(masks,2)>=1))
        return loss

    def forward(self, document,masks, hdp_feats):  
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(document,masks, hdp_feats)

        # Find the best path, given the features.
        tag_seq = self.crf.decode(emissions=lstm_feats,mask=(torch.sum(masks,2)>=1))
        return tag_seq



model = model1(tag_to_ix,HIDDEN_DIM_Doc)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#for param in model.word_embeds.parameters():
#      param.requires_grad=False
#modules = [model.word_embeds.encoder.layer[10], model.word_embeds.encoder.layer[11]] 
#for module in modules:
#  for param in module.parameters():
#    param.requires_grad = True
model.to(device)

mlb = MultiLabelBinarizer(classes =[0,1,2,3,4,5,6,7,8,9,10,11,12])

import gc

from tqdm.notebook import tqdm
best_val = 0
for epoch in tqdm(range(epochs)):  
    for batch_doc, batch_tags, batch_masks,tags_unpadded, hdp_feats in tqdm(training_generator):
        model.zero_grad()
        loss = model.neg_log_likelihood(batch_doc.to(device), batch_tags.to(device),batch_masks.to(device), hdp_feats.to(device))
        loss.backward()
        optimizer.step()
        gc.collect()
        torch.cuda.empty_cache()
    with torch.no_grad():
      preds_val = []
      true_val = []
      for batch_doc, batch_tags, batch_masks, batch_unpadded_tags, hdp_feats in tqdm(val_generator):
        model.batch_size = batch_doc.size()[0]
        best_paths = model.forward(batch_doc.to(device),batch_masks.to(device), hdp_feats.to(device))
        preds_val.append(best_paths)
        true_val.append(batch_unpadded_tags)
      preds_val_new = [pred for preds in preds_val for pred in preds ]
      true_val_new = [list(label.cpu().detach().numpy())[1:] for labels\
                          in true_val for label in labels]
          
      labels_bin_true = mlb.fit_transform(true_val_new)
      labels_bin_pred = mlb.fit_transform(preds_val_new)
      sc = precision_recall_fscore_support(labels_bin_true,labels_bin_pred,average="weighted")

      sc = precision_recall_fscore_support(labels_bin_true,labels_bin_pred,average="micro")
      if sc[2]>best_val:
        best_val = sc[2]
        torch.save(model.state_dict(),"/model_for_submission")
        print("epoch ", epoch+1)


model = model1(tag_to_ix,HIDDEN_DIM_Doc)
model.load_state_dict(torch.load("/model_for_submission"))
model.to(device)

# Generate predictions
hdp_features_test = torch.load('/legalevalRR_docs_test_hdp_submission.pt')
with open('/RR_TEST_DATA_FS.json','r') as f:
    data = json.loads(f.read())
documents = []
for doc in data:
    sentences = []
    sent_labels = []
    sentences.append(doc['id'])
    sents = []
    for sent in doc['annotations'][0]['result']:
        sents.append(" ".join(sent['value']['text'].lower().replace('\n',' ').replace('...',' ').replace(',',' ')[:-1].split()))
        sent_labels.append(sent['id'])
    sentences.append(sents)
    sentences.append(sent_labels)
    documents.append(sentences)
df_test = pd.DataFrame(documents,columns=['id','text','sentence_id']).explode(['text','sentence_id'])
df_test['text'] = df_test['text'].str.replace('(0?[1-9]|[12][0-9]|3[01])[\/\-](0?[1-9]|1[012])[\/\-]\d{4}','N',regex=True).str.replace('\d+',"N",regex=True).str.replace('(',' ',regex=True).str.replace(')',' ',regex=True)
df_test.dropna(inplace=True)

documents_text = []
documents_labels = []
#attention_masks_doc = []
input_ids_doc = []
attention_masks_doc = []
df = df_test
for id in df['id'].unique():
    sentences_text = df[df['id']==id]['text'].tolist()
    sentences_labels = df[df['id']==id]['sentence_id'].tolist()
    encoded = tokenizer(sentences_text,truncation=True,
                                          add_special_tokens = True,
                                          return_tensors = 'pt',
                                          return_attention_mask=True,
                                              padding = 'longest',
                                            max_length =  max_seq_len)
    attention_masks_doc.append(encoded['attention_mask'])
    #attention_masks = encoded['attention_mask']
    input_ids_doc.append(encoded['input_ids'])
    #attention_masks_doc.append(attention_masks)


    documents_labels.append(sentences_labels)
                                  
test_x = input_ids_doc
test_y = documents_labels
test_masks = attention_masks_doc


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels,masks, hdp_feats):
        'Initialization'
        self.labels = labels
        self.data = data
        self.masks = masks
        self.hdp_feats = hdp_feats

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
       
        X = self.data[index]
        y = self.labels[index]
        mask = self.masks[index]
        sent_features_hdp = self.hdp_feats[index]

        return X, y, mask, sent_features_hdp

def collate_fn(batch):

  x, y, mask,  hdp_feats = list(map(list, zip(*batch)))
  lens_x = [z.size()[1] for z in x]
  l = max(lens_x)
  
  x_padded = [torch.nn.functional.pad(z,value=tokenizer.pad_token_id,mode='constant',pad = (1,l-z.size()[1]-1)) for z in x]
  num_sent = [z.size()[0] for z in x_padded]
  max_num_sent = max(num_sent)

  x_padded_sent = [torch.cat([z,torch.full((max_num_sent-z.size()[0],l),
                                                dtype=torch.long, fill_value = tokenizer.pad_token_id)]) for z in x_padded]
  x_padded = [torch.nn.functional.pad(z,value=tokenizer.pad_token_id,mode='constant',pad = (1,l-z.size()[1]-1)) for z in x]
  
  x_masks_padded = [torch.nn.functional.pad(z,value=tokenizer.pad_token_id,mode='constant',pad = (1,l-z.size()[1]-1)) for z in mask]
  x_masks_padded_sent = [torch.cat([z,torch.full((max_num_sent-z.size()[0],l),
                                                dtype=torch.long, fill_value = tokenizer.pad_token_id)]) for z in x_masks_padded]
  hdp_feats = [torch.cat([z,torch.zeros((max_num_sent-z.size()[0],53),
                                                dtype=torch.long)],0) for z in hdp_feats]
  
  
  return torch.stack(x_padded_sent),torch.stack(x_masks_padded_sent), y, torch.nan_to_num(torch.stack(hdp_feats)).float()


test_dataset = Dataset(test_x,test_y,test_masks,hdp_features_test)

test_generator = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            collate_fn = collate_fn,
                                            shuffle=False)

model.eval()
with torch.no_grad():
      preds_val = []
      sent_ids = []
      for batch_doc, batch_masks, batch_ids, hdp_feats in tqdm(test_generator):
        model.batch_size = batch_doc.size()[0]
        best_paths = model.forward(batch_doc.to(device),batch_masks.to(device), hdp_feats.to(device))
        preds_val.extend(best_paths)
        sent_ids.extend(batch_ids)
        
ix_to_tag = {0:'PREAMBLE', 1:'NONE', 2:'FAC', 3:'ARG_RESPONDENT', 4:'RLC',
       5:'ARG_PETITIONER', 6:'ANALYSIS', 7:'PRE_RELIED', 8:'RATIO', 9:'RPC',
       10:'ISSUE', 11:'STA', 12:'PRE_NOT_RELIED'}

predictions = []
for i, preds in enumerate(preds_val):
  predictions.append([ix_to_tag[pred] for pred in preds[:len(sent_ids[i])]])
  
with open('/RR_TEST_DATA_FS.json','r') as f:
    data = json.loads(f.read())
documents = []
for i,doc in enumerate(data):
    for j,sent in enumerate(doc['annotations'][0]['result']):
      data[i]['annotations'][0]['result'][j]['value']['labels'] = [predictions[i][j]]

with open('/RR_TEST_DATA_FS_preds.json','w') as f:
    f.write(json.dumps(data,indent=4))