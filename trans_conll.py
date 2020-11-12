from pathlib import Path
import re
import numpy as np
import torch

def read_conll(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_1_docs = []
    tag_2_docs = []
    is_argum = []
    for doc in raw_docs:
        tokens = []
        tags_1 = []
        tags_2 = []
        is_args = []
        sps =  []
        sents_num = []
        docs = []
        for line in doc.split('\n'):
            token, tag_1, tag_2, is_arg, sp, sent, doc  = line.split('\t')
            tokens.append(token)
            tags_1.append(tag_1)
            tags_2.append(tag_2)
            is_args.append(is_arg)
            sps.append(sp)
            sents_num.append(sent)
            docs.append(doc)
        token_docs.append(tokens)
        tag_1_docs.append(tags_1)
        tag_2_docs.append(tags_2)
        is_argum.append(is_args)

    return token_docs, tag_1_docs, tag_2_docs, is_argum

# path = 'C:/Users/athanasis/Desktop/Semester_C/Dataset/dataset_conll/train.txt'

# texts, tag_1_docs, tag_2_docs, is_argum = read_conll(path)

def percentage_split(seq, percentages):
   assert sum(percentages) == 1.0
   prv = 0
   size = len(seq)
   cum_percentage = 0
   for p in percentages:
       cum_percentage += p
       nxt = int(cum_percentage * size)
       yield seq[prv:nxt]
       prv = nxt




def encode_tags(tags, encodings, tag2id ):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels



class WNUTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
