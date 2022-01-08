# LSTM 
# https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0
# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
import typing
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

from tqdm import tqdm


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


    def train_mlp(self, x_train : torch.Tensor, 
                        x_val : torch.Tensor, 
                        x_test : torch.Tensor, 
                        y_train : torch.Tensor, 
                        y_val : torch.Tensor, 
                        y_test : torch.Tensor, 
                        hparams : dict):
        # https://medium.com/swlh/text-classification-using-scikit-learn-pytorch-and-tensorflow-a3350808f9f7
        lr = hparams["lr"]
        epochs = hparams["epochs"]
        batch_size = hparams["batch_size"]
        patience = hparams["patience"]
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        #x_train = torch.tensor(scipy.sparse.csr_matrix.todense(X_train)).float()
        #x_test = torch.tensor(scipy.sparse.csr_matrix.todense(X_test)).float()

        # constant for early stopping
        best_loss = np.inf
        best_acc = np.NINF
        trace_train = []
        trace_val = []

        for epoch in range(epochs):
            
            running_train_loss = 0.0
            running_train_acc = 0.0
            print(f"Epoch: {epoch}")
            
            for i in tqdm(range(0, x_train.shape[0], batch_size)):
                
                batch = x_train[i: i + batch_size]
                label = y_train[i: i + batch_size]

                output = self.forward(batch)

                loss = criterion(output, label)

                predictions = output.argmax(axis=1)
                running_train_acc += (predictions == label).sum() 

                optimizer.zero_grad()
                loss.backward()
                running_train_loss += loss.item()
                optimizer.step()

            running_val_loss = 0.0
            running_val_acc = 0.0
            for i in tqdm(range(0, x_val.shape[0], batch_size)):
                
                batch = x_val[i: i + batch_size]
                label = y_val[i: i + batch_size]

                output = self.forward(batch)
                
                predictions = output.argmax(axis=1)
                running_val_acc += (predictions == label).sum() 

                loss = criterion(output, label)

                running_val_loss += loss.item()

            print(f"Train Loss: {running_train_loss:.4f} Train Acc {running_train_acc / x_train.shape[0]:.3f} Val Loss {running_val_loss:.4f} Val Acc {running_val_acc / x_val.shape[0]:.3f}")
        
            trace_train.append(running_train_loss)
            trace_val.append(running_val_loss)

            # early stopping
            if running_val_acc > best_acc:
                best_acc = running_val_acc
                best_epoch = epoch
                best_state = {key: value.cpu() for key, value in self.state_dict().items()}
            else:
                if epoch >= best_epoch + patience:
                    break

        # load and save best model
        self.load_state_dict(best_state)
        torch.save(best_state, 'model.pt')

        # final evaluation
        predictions = self.forward(x_test).argmax(axis=1)
                
        test_acc = (predictions == y_test).sum() 

        print(f"Test Acc {test_acc / x_test.shape[0]:.3f}")  

        print("Finished Training")
        return trace_train, trace_val


