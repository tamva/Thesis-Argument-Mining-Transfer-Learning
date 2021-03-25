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

# Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


df = pd.read_csv('./Final Data/binary-train_no_double_quotes.txt',sep='\t')

df = df[['Sentence','Argument']]
print("DF Shape",df.shape)

df['Argument'].value_counts()

ax = sns.countplot(df['Argument'])
plt.xlabel('review sentiment')
class_names = ['Non-Argument', 'Argument']
ax.set_xticklabels(class_names)

# Defining some key variables that will be used later on in the training
MAX_LEN = 70
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
LOADER_SIZE = 1024
EPOCHS = 200
LEARNING_RATE = 0.01
export_string = "epochs-" + str(EPOCHS) + "_lr-" + str(LEARNING_RATE) + "_sentence_transformer.bin"

# plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
# sns.distplot(token_lens)
# plt.xlim([0, 256])
# plt.xlabel('Token count')

train_size = 0.8
train_dataset=df.sample(frac=train_size,random_state=200)
test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))
print(train_dataset.head())
print(test_dataset.head())


sentences = train_dataset['Sentence'].tolist()
labels = train_dataset['Argument'].tolist()
sentences_test = test_dataset['Sentence'].tolist()
labels_test = test_dataset['Argument'].tolist()
le = LabelEncoder()
le.fit(labels)

encoder = SentenceTransformer('bert-base-multilingual-cased',device='cuda')

train_embedding = encoder.encode(sentences, convert_to_tensor=True)
print("MEAN:", torch.mean(train_embedding[0]))
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
        torch.nn.init.xavier_uniform(self.linear.weight)
#       maybe add one sigmoid for the binary classification    


#     def forward(self, input_ids, attention_mask):
    def forward(self, inputs):  
            #output = self.pre_classifier1(inputs)
            output = self.pre_classifier2(inputs)
            output = self.pre_classifier(inputs)
            output = self.dropout(inputs)
            output = self.linear(inputs)
            output = self.classifier(output)
            return output
            
num_samples, train_embeddings_dim = train_embedding.size()
n_labels = labels.unique().shape[0]
model = BERTClass()
print(n_labels)
model.to(device)       


# Creating the loss function and optimizer
loss_function = torch.nn.BCELoss() # is The sigmoid
# loss_function = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(params =  model.parameters(), lr=LEARNING_RATE)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)


# Function to calcuate the accuracy of the model
def binary_acc(y_pred, y_test):
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc,correct_results_sum

def train(epoch):
    # print("Epoch Number",epoch)
    epoch_loss = []
    epoch_accu = []
    correct_predictions = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    
    i = 1
    for data in train_loader:    
        embedding, label = data
        optimizer.zero_grad()
        # i+=1
        embedding.to(device)
        outputs= model(embedding)       
        label = label.unsqueeze(1).float()
        label.to(device)
        loss = loss_function(outputs, label)
        acc, correct_predictions = binary_acc(torch.round(outputs),label)
        #print("loss:", loss, "acc:", acc)
        
        correct_predictions += correct_predictions

        epoch_accu.append(acc.item())

        epoch_loss.append(loss.item())
            
        loss.backward()
        optimizer.step()

    average_loss = statistics.mean(epoch_loss)
    average_acc =  statistics.mean(epoch_accu)

    print(f'Epoch {epoch}: | Loss: {average_loss:.5f} | Acc: {average_acc:.3f}')

    return    

save_path = '/home/tamvakidis/Desktop/Tamvakidis_TH/SavedModel'
for epoch in range(EPOCHS):
    train(epoch)
# torch.save(model.state_dict(), save_path) 
torch.save(model.state_dict(), os.path.join(save_path, os.path.join(save_path, export_string)))
# torch.save(model.state_dict(), os.path.join(save_path, os.path.join(save_path, "sentence_tr_20epochs.bin")))