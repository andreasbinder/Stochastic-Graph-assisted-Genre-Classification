# LSTM 
# https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0
# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
from matplotlib import use
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import tensor
import numpy as np
import scipy

# Preliminaries

#from torchtext.data import Field, TabularDataset, BucketIterator

# Models

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Training

import torch.optim as optim

from collections import OrderedDict

# Evaluation

# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#import seaborn as sns

from sklearn.preprocessing import LabelEncoder


class MLP(nn.Module):

    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.mlp = nn.Sequential(OrderedDict([
          ('linear1', nn.Linear(self.input_size, 256)),
          ('relu1', nn.ReLU()),
          ('linear2', nn.Linear(256, 128)),
          ('relu2', nn.ReLU()),
          ('linear3', nn.Linear(128, self.output_size))
        ]))


    def forward(self, text):
        
        out = self.mlp(text)

        return out

    def construct_sparse_tensor(self, coo):
        # https://discuss.pytorch.org/t/creating-a-sparse-tensor-from-csr-matrix/13658/5
        '''import torch
        import numpy as np
        from scipy.sparse import coo_matrix'''

        # coo = coo_matrix(([3,4,5], ([0,1,1], [2,0,2])), shape=(2,3))

        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()


    def train_mlp(self, x_train, x_test, y_train, y_test):
        # https://medium.com/swlh/text-classification-using-scikit-learn-pytorch-and-tensorflow-a3350808f9f7
        optimizer = optim.Adam(self.parameters(), lr=0.0002)
        criterion = nn.CrossEntropyLoss()

        epochs = 15
        batch_size = 128
        

        #x_train = torch.tensor(scipy.sparse.csr_matrix.todense(X_train)).float()
        #x_test = torch.tensor(scipy.sparse.csr_matrix.todense(X_test)).float()

        '''x_train = self.construct_sparse_tensor(X_train.tocoo())
        x_test = self.construct_sparse_tensor(X_test.tocoo())

        y_train = LabelEncoder().fit_transform(y_train)
        y_test = LabelEncoder().fit_transform(y_test)

        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)'''

        for epoch in range(epochs):
            #print(len(desc))
            running_train_loss = 0.0
            running_train_acc = 0.0
            print(f"Epoch: {epoch}")
            c = 0
            # for description, label in zip(desc, labels):
            # for index, i in enumerate(range(0, len_desc, batch_size)):
            for index, i in enumerate(range(0, x_train.shape[0], batch_size)):
                

                #description = torch.tensor([word_to_ix[word] for word in description])
                #label = torch.tensor(label_to_idx[label]).long().unsqueeze(dim=0)
                #batch = enc_descriptions[i: i + batch_size]
                #label = enc_labels[i: i + batch_size]
                
                # print(i)

                batch = x_train[i: i + batch_size]
                label = y_train[i: i + batch_size]

                # batch = torch.tensor([torch.cat((text, torch.full(self.max_len - len(text), PAD))) for text in batch])
                # np.array([np.append(d, ['<pad>' for _ in range(max_len - len(d))]) if max_len - len(d) > 0 else d for d in desc ], dtype=object)
                output = self.forward(batch)
                
                #print(output.shape)
                #print(label.shape)

                loss = criterion(output, label)

                predictions = output.argmax(axis=1)
                running_train_acc += (predictions == label).sum() 

                optimizer.zero_grad()
                loss.backward()
                running_train_loss += loss.item()
                optimizer.step()

            running_val_loss = 0.0
            running_val_acc = 0.0
            for index, i in enumerate(range(0, x_test.shape[0], batch_size)):
                
                #batch = enc_descriptions[idx_val[i: i + batch_size]]
                # label = enc_labels[idx_val[i: i + batch_size]]

                batch = x_test[i: i + batch_size]
                label = y_test[i: i + batch_size]

                output = self.forward(batch)
                
                predictions = output.argmax(axis=1)
                running_val_acc += (predictions == label).sum() 

                loss = criterion(output, label)
                # optimizer.zero_grad()
                # loss.backward()
                running_val_loss += loss.item()
                # optimizer.step()

            print(f"Train Loss: {running_train_loss:.4f} Train Acc {running_train_acc / x_train.shape[0]:.3f} Val Loss {running_val_loss:.4f} Val Acc {running_val_acc / x_test.shape[0]:.3f}")
        
        print("Finished Training")



