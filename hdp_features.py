import torch
import json
import numpy as np
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import spacy
import nltk
nltk.download('stopwords')
nlp=spacy.load('en_core_web_sm',disable=['parser', 'ner'])
import tomotopy as tp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
#Adding start and stop tags to the tag index list
tag_to_ix = {'PREAMBLE':0, 'NONE':1, 'FAC':2, 'ARG_RESPONDENT':3, 'RLC':4,
       'ARG_PETITIONER':5, 'ANALYSIS':6, 'PRE_RELIED':7, 'RATIO':8, 'RPC':9,
       'ISSUE':10, 'STA':11, 'PRE_NOT_RELIED':12,START_TAG:13,STOP_TAG:14}

with open('legalevalRR_dict.json','r') as f:
    word_to_ix = f.read()
word_to_ix = json.loads(word_to_ix)

df = pd.read_csv('legalevalRR_.csv')
df.dropna(inplace=True)
texts = []
for id in df['id'].unique():
    sentences_text = df[df['id']==id]['text'].tolist()
    sentences_labels = df[df['id']==id]['label'].tolist()
    texts.append(' . '.join(sentences_text))

def tokenize(sentences):

    for sentence in sentences:

         yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))



processed_data = list(tokenize(texts))

stop_words = stopwords.words('english')
bigram = gensim.models.Phrases(processed_data , min_count=5, threshold=100) # higher threshold fewer phrases.
bigram_mod = gensim.models.phrases.Phraser(bigram)

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        text_list = []
        for token in doc:
          if token.pos_ in allowed_postags:
            text_list.append(token.lemma_) 
          elif token.pos_=='PROPN':
            text_list.append(token.text) 
        texts_out.append(text_list)
    return texts_out

data_words_nostops = remove_stopwords(processed_data)

data_words_bigrams = make_bigrams(data_words_nostops)

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

data_lemmatized = lemmatization(data_words_bigrams)


term_weight = tp.TermWeight.IDF
hdp = tp.HDPModel(tw=term_weight, min_cf=5, rm_top=0, gamma=1,
                  alpha=0.1, initial_k=10, seed=28)

for vec in data_lemmatized:
    hdp.add_doc(vec)

# Initiate sampling burn-in  (i.e. discard N first iterations)
hdp.burn_in = 50
hdp.train(0,workers=1)
print('Num docs:', len(hdp.docs), ', Vocab size:', hdp.num_vocabs,
      ', Num words:', hdp.num_words)
print('Removed top words:', hdp.removed_top_words)

# Train model
for i in range(0, 5000, 100):
    hdp.train(100,workers=1) # 100 iterations at a time
    print('Iteration: {}\tLog-likelihood: {}\tNum. of topics: {}'.format(i, hdp.ll_per_word, hdp.live_k))


hdp.purge_dead_topics()

train_features = []
for id in df['id'].unique():
    sentences_text = df[df['id']==id]['text'].tolist()
    sent_proc = tokenize(sentences_text)
    sent_nostops = remove_stopwords(sent_proc)
    sent_bgrams = make_bigrams(sent_nostops)
    sent_lem = lemmatization(sent_bgrams)
    sent_features = []
    for t in sent_lem:
      if t!=[]:
        sent_features.append(hdp.infer(hdp.make_doc(t))[0])
      elif t == []:
        sent_features.append(np.zeros(hdp.k))

    train_features.append(torch.nan_to_num(torch.tensor(np.stack(sent_features))))

hdp_train = train_features

torch.save(hdp_train,'legalevalRR_docs_train_hdp_submission.pt')

with open('dev.json','r') as f:
    data = json.loads(f.read())
documents = []
for doc in data:
    sentences = []
    sent_labels = []
    sentences.append(doc['id'])
    sents = []
    for sent in doc['annotations'][0]['result']:
        sents.append(" ".join(sent['value']['text'].lower().replace('\n',' ')[:-1].split()))
        sent_labels.append(sent['value']['labels'][0])
    sentences.append(sents)
    sentences.append(sent_labels)
    documents.append(sentences)
df_dev = pd.DataFrame(documents,columns=['id','text','label']).explode(['text','label'])


dev_features = []
for id in df_dev['id'].unique():
    sentences_text = df_dev[df_dev['id']==id]['text'].tolist()
    sent_proc = tokenize(sentences_text)
    sent_nostops = remove_stopwords(sent_proc)
    sent_bgrams = make_bigrams(sent_nostops)
    sent_lem = lemmatization(sent_bgrams)
    sent_features = []
    for t in sent_lem:
      if t!=[]:
        sent_features.append(hdp.infer(hdp.make_doc(t))[0])
      elif t==[]:
        sent_features.append(np.zeros(hdp.k))

    dev_features.append(torch.nan_to_num(torch.tensor(np.stack(sent_features))))

hdp_dev = dev_features

torch.save(hdp_dev,'legalevalRR_docs_dev_hdp_submission.pt')

with open('/RR_TEST_DATA_FS.json','r') as f:
    data = json.loads(f.read())
documents = []
for doc in data:
    sentences = []
    sent_labels = []
    sentences.append(doc['id'])
    sents = []
    for sent in doc['annotations'][0]['result']:
        sents.append(" ".join(sent['value']['text'].lower().replace('\n',' ')[:-1].split()))
        sent_labels.append(sent['value']['labels'][0])
    sentences.append(sents)
    sentences.append(sent_labels)
    documents.append(sentences)
df_test = pd.DataFrame(documents,columns=['id','text','label']).explode(['text','label'])


test_features = []
for id in df_test['id'].unique():
    sentences_text = df_test[df_dev['id']==id]['text'].tolist()
    sent_proc = tokenize(sentences_text)
    sent_nostops = remove_stopwords(sent_proc)
    sent_bgrams = make_bigrams(sent_nostops)
    sent_lem = lemmatization(sent_bgrams)
    sent_features = []
    for t in sent_lem:
      if t!=[]:
        sent_features.append(hdp.infer(hdp.make_doc(t))[0])
      elif t==[]:
        sent_features.append(np.zeros(hdp.k))

    test_features.append(torch.nan_to_num(torch.tensor(np.stack(sent_features))))

hdp_test = test_features

torch.save(hdp_test,'legalevalRR_docs_test_hdp_submission.pt')