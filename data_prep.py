import pandas as pd
import json
import flatten_json
with open('train.json','r') as f:
    data = json.loads(f.read())
dict_flattened = (flatten_json.flatten(record, '.') for record in data)
df = pd.json_normalize(data,record_path=['annotations'],meta=['id',['data','text'],	['meta','group']],max_level=4)
df = pd.DataFrame(dict_flattened)
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
    
df = pd.DataFrame(documents,columns=['id','text','label']).explode(['text','label'])
df.to_csv('legalevalRR_.csv')
df['text'] = df['text'].str.replace('(0?[1-9]|[12][0-9]|3[01])[\/\-](0?[1-9]|1[012])[\/\-]\d{4}','N',regex=True).str.replace('\d+',"N",regex=True).str.replace('(',' ',regex=True).str.replace(')',' ',regex=True)
df.dropna(inplace=True)
word_to_ix = {'<pad>':0,'<unk>':1,'N':2}
# Adding special tokens to the list. N denotes numbers
for sentence in df['text'].tolist():
    for word in sentence.split():
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

with open('legalevalRR_dict.json','w') as f:
 #   f.write(str(word_to_ix))
    json.dump(word_to_ix,f)