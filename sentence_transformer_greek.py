# Importing the libraries needed
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
from sklearn.metrics import classification_report
import transformers
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, models
from transformers import DistilBertModel, DistilBertTokenizer,BertModel,BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import statistics

# https://medium.datadriveninvestor.com/deploy-your-pytorch-model-to-production-f69460192217
# Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


# df = pd.read_csv('./Final Data/binary-train_no_double_quotes.txt',sep='\t')
df = pd.read_csv('./Final Data/binary_translated.txt',sep='\t')

df = df[['Sentence','Argument']]
# print("DF Shape",df.shape)

# df['Argument'].value_counts()

ax = sns.countplot(df['Argument'])
plt.xlabel('review sentiment')
class_names = ['Non-Argument', 'Argument']
ax.set_xticklabels(class_names)

# Defining some key variables that will be used later on in the training
MAX_LEN = 70
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
LOADER_SIZE = 2048
EPOCHS = 200
LEARNING_RATE = 0.01
export_string = "epochs-" + str(EPOCHS) + "_lr-" + str(LEARNING_RATE) + "_sentence_transformer_greek.bin"


train_size = 0.50
train_dataset=df.sample(frac=train_size,random_state=200)
test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))


sentences = train_dataset['Sentence'].tolist()
labels = train_dataset['Argument'].tolist()
sentences_test = test_dataset['Sentence'].tolist()
labels_test = test_dataset['Argument'].tolist()
le = LabelEncoder()
le.fit(labels)

# encoder = SentenceTransformer('bert-base-multilingual-cased',device='cuda')

encoder = SentenceTransformer('./output/bert-mult-make-multilingual-en-el-2020-12-19_23-16-42')
# encoder = SentenceTransformer('./output/bert-mult-make-multilingual-en-el-2021-03-27_01-01-58')
train_embedding = encoder.encode(sentences, convert_to_tensor=True)
# print("MEAN:", torch.mean(train_embedding[0]))
train_embedding.to(device)
test_embedding = encoder.encode(sentences_test, convert_to_tensor=True)
test_embedding.to(device)

labels = torch.tensor(labels)
labels.to(device)
test_labels = torch.tensor(labels_test)
test_labels.to(device)


class Batcher(object):
    def __init__(self, data_x, data_y, batch_size):
        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size
        self.n_samples = data_x.shape[0]
        self.indices = torch.randperm(self.n_samples)
        self.ptr = 0
        self.upto = self.ptr + self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr >= self.n_samples:
            self.ptr = 0
            self.upto = self.ptr + self.batch_size
            raise StopIteration
        else:
            batch_indices = self.indices[self.ptr:self.upto]
            self.ptr = self.upto
            self.upto += self.batch_size
            return self.data_x[batch_indices].to(device), self.data_y[batch_indices].to(device)
            

train_loader = Batcher(train_embedding,labels,batch_size=LOADER_SIZE)



# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()

        #self.pre_classifier1 = torch.nn.Linear(768, 768)
        self.pre_classifier2 = torch.nn.Linear(768, 768)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(768, 1)
        self.classifier = torch.nn.Sigmoid()

    def forward(self, inputs):  
            #output = self.pre_classifier1(inputs)
            output = self.pre_classifier2(inputs)
            output = self.pre_classifier(inputs)
            output = self.dropout(inputs)
            output = self.linear(inputs)
            output = self.classifier(output)
            return output


# Function to calcuate the accuracy of the model
def binary_acc(y_pred, y_test):
    y_pred = y_pred.squeeze(1).float()
    # print("Pred",y_pred,"Pred",y_pred.shape[0],"Label", y_test, "Shape", y_test.shape[0])
    correct_results_sum = (y_pred == y_test).sum()
    acc = correct_results_sum/y_test.shape[0]
    acc = acc * 100
    return acc,correct_results_sum


save_path = './SavedModel/epochs-200_lr-0.01_sentence_transformer.bin'

# Evaluation
test_loader = Batcher(test_embedding,test_labels,batch_size=LOADER_SIZE)
sent = "Πρέπει να διδαχθούν οι μαθητές να ανταγωνίζονται ή να συνεργάζονται ; Λέγεται πάντα ότι ο ανταγωνισμός μπορεί να προωθήσει αποτελεσματικά την ανάπτυξη της οικονομίας ."
sen = "Από την άλλη πλευρά , η σημασία του ανταγωνισμού είναι ότι πώς να γίνετε πιο αριστεία για να κερδίσετε τη νίκη ."
# eval_embedding = encoder.encode(test_embedding, convert_to_tensor=True)
batch_num = 0
model = BERTClass() # Model class must be defined somewhere
model.load_state_dict(torch.load(save_path))
model.to(device)  
model.eval() # run if you only want to use it for inference
accu_test = []
for val in test_loader:
    batch_num += 1
    emb, lab= val
    out= model(emb)
    acc, correct = binary_acc(torch.round(out),lab)
    print("Batch Num:",batch_num," Accuracy:",acc.item())
    accu_test.append(round(acc.item(), 3))

print("Overall Accuracy",statistics.mean(accu_test))