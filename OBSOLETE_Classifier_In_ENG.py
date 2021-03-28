# https://towardsdatascience.com/transformers-for-multilabel-classification-71a1a0daf5e1
# https://huggingface.co/transformers/custom_datasets.html
# https://chriskhanhtran.github.io/posts/named-entity-recognition-with-transformers/
from trans_conll import *
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
# import tensorflow as tf
import torch
from transformers import DistilBertTokenizerFast
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
import pickle
from transformers import *
from tqdm import tqdm, trange
from ast import literal_eval
from transformers import DistilBertForTokenClassification
# https://huggingface.co/transformers/custom_datasets.html

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

# df = pd.read_csv('./dataset_conll/train.txt', sep='\t')
path ='./dataset_conll/train.txt'

texts, tag_1_docs, tag_2_docs, is_argum = read_conll(path)
print(texts[1], tag_1_docs[1], tag_2_docs.head(), is_argum.head())
# from sklearn.model_selection import train_test_split
# texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2)

#train_texts, validate_texts, test_texts = np.split(texts, [int(len(texts)*0.8), int(len(texts)*0.9)])
#print(type(train_texts[1]))
#train_is_argum, validate_is_argum, test_is_argum = np.split(is_argum, [int(len(is_argum)*0.8), int(len(is_argum)*0.9)])
train_texts, validate_texts, train_is_argum, validate_is_argum = train_test_split(texts, is_argum, test_size=.2)


unique_tags = set(argum for doc in is_argum for argum in doc)
tag2id = {argum: id for id, argum in enumerate(unique_tags)}
id2tag = {id: argum for argum, id in tag2id.items()}


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(validate_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)



train_labels = encode_tags(train_is_argum, train_encodings,tag2id )
val_labels = encode_tags(validate_is_argum, val_encodings,tag2id )


train_encodings.pop("offset_mapping") # we don't want to pass this to the model
val_encodings.pop("offset_mapping")
train_dataset = WNUTDataset(train_encodings, train_labels)
val_dataset = WNUTDataset(val_encodings, val_labels)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=len(unique_tags))

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()

