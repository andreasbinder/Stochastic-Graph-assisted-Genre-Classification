# LSTM 
# https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0
# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
from matplotlib import use
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import tensor
import numpy as np

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


class LSTM(nn.Module):

    def __init__(self, vocab_size, max_len, dimension=128, use_glove=False):
        super(LSTM, self).__init__()

        # TODO pretrained embedding
        
        self.embedding_size = 100
        self.use_glove = use_glove

        if use_glove:
            from torchtext.vocab import GloVe
            embedding_glove = GloVe(name='6B', dim=100)
            self.embedding = nn.Embedding.from_pretrained(embedding_glove.vectors)
            # for setting trainable to False
            self.embedding.weight.requires_grad=False
            
        else:
            self.embedding = nn.Embedding(vocab_size, self.embedding_size)

        self.dimension = dimension
        self.n_layers = 2
        self.max_len = max_len
        # self.hidden_dim
        self.lstm = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=dimension,
                            num_layers=self.n_layers,
                            batch_first=True)

        
        self.mlp = nn.Sequential(OrderedDict([
          ('linear1', nn.Linear(self.embedding_size, 256)),
          ('relu1', nn.ReLU()),
          ('linear2', nn.Linear(256, 128)),
          ('relu2', nn.ReLU()),
          ('linear3', nn.Linear(128, self.dimension))
        ]))

        self.single_layer = nn.Linear(self.embedding_size, self.dimension)
        self.dropout = nn.Dropout()

        #self.drop = nn.Dropout(p=0.5)

        self.dummy = nn.Linear(self.max_len * self.embedding_size, 10)

        self.fc = nn.Linear(dimension * max_len, 10)

    def forward(self, text):
        # text_emb shape: (batch_size, max_length, embedding_size)
        text_emb = self.embedding(text)

        batch_size = text.size(0)
        
        # print(text_emb.shape)

        use_lstm = True
        # hidden = self.init_hidden(batch_size)
        if use_lstm:
            out, _ = self.lstm(text_emb)
            # todo view
            out = out.reshape(batch_size, self.max_len * self.dimension) #self.dimension

            out = self.fc(out)
        else:
            # out = self.mlp(text_emb)
            '''out = self.single_layer(text_emb)
            out = self.dropout(out)
            out = out.reshape(batch_size, self.max_len * self.dimension)

            out = self.fc(out)
            '''
            out = text_emb.reshape(batch_size, self.max_len * self.embedding_size)
            # print(text_emb.shape)
            out = self.dummy(out)
        return out

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.dimension)
        return hidden

    def train_lstm(self, desc,labels, vocab, word_to_ix, label_to_idx, idx_train, idx_val, idx_test ):
        
        optimizer = optim.Adam(self.parameters(), lr=0.0002)
        criterion = nn.CrossEntropyLoss()

        epochs = 15
        batch_size = 128
        len_desc = len(desc)

        print("Encode Description")
        # enc_descriptions = torch.tensor([[word_to_ix[word] for word in d] for d in desc])
        # enc_labels = torch.tensor([label_to_idx[label] for label in labels]).long()

        PAD = word_to_ix['<pad>']

        for epoch in range(epochs):
            #print(len(desc))
            running_train_loss = 0.0
            running_train_acc = 0.0
            print(f"Epoch: {epoch}")
            c = 0
            # for description, label in zip(desc, labels):
            # for index, i in enumerate(range(0, len_desc, batch_size)):
            for index, i in enumerate(range(0, idx_train.shape[0], batch_size)):
                

                #description = torch.tensor([word_to_ix[word] for word in description])
                #label = torch.tensor(label_to_idx[label]).long().unsqueeze(dim=0)
                #batch = enc_descriptions[i: i + batch_size]
                #label = enc_labels[i: i + batch_size]
                
                # print(i)

                batch = desc[idx_train[i: i + batch_size]]
                label = labels[idx_train[i: i + batch_size]]
                # TODO padding numpy
                batch = np.array([np.pad(text, (0,self.max_len - len(text)), 'constant', constant_values=('<pad>')) for text in batch])
                batch = torch.tensor([[word_to_ix[word] for word in item] for item in batch])
                # TODO padding 
                label = torch.tensor([label_to_idx[label_] for label_ in label]).long()

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
            for index, i in enumerate(range(0, idx_val.shape[0], batch_size)):
                
                #batch = enc_descriptions[idx_val[i: i + batch_size]]
                # label = enc_labels[idx_val[i: i + batch_size]]

                batch = desc[idx_val[i: i + batch_size]]
                label = labels[idx_val[i: i + batch_size]]
                # TODO padding numpy
                batch = np.array([np.pad(text, (0,self.max_len - len(text)), 'constant', constant_values=('<pad>')) for text in batch])
                batch = torch.tensor([[word_to_ix[word] for word in item] for item in batch])
                # TODO padding 
                label = torch.tensor([label_to_idx[label_] for label_ in label]).long()

                output = self.forward(batch)
                
                predictions = output.argmax(axis=1)
                running_val_acc += (predictions == label).sum() 

                loss = criterion(output, label)
                # optimizer.zero_grad()
                # loss.backward()
                running_val_loss += loss.item()
                # optimizer.step()

            print(f"Train Loss: {running_train_loss:.4f} Train Acc {running_train_acc / idx_train.shape[0]:.3f} Val Loss {running_val_loss:.4f} Val Acc {running_val_acc / idx_val.shape[0]:.3f}")
        
        print("Finished Training")

        # batch = torch.tensor([[word_to_ix[word] for word in item] for item in desc[idx_test]])
        # TODO padding 
        # label = torch.tensor([label_to_idx[label_] for label_ in labels[idx_test]]).long()

        batch = desc[idx_test]
        label = labels[idx_test]
        # TODO padding numpy
        batch = np.array([np.pad(text, (0,self.max_len - len(text)), 'constant', constant_values=('<pad>')) for text in batch])
        batch = torch.tensor([[word_to_ix[word] for word in item] for item in batch])
        # TODO padding 
        label = torch.tensor([label_to_idx[label_] for label_ in label]).long()

        #batch = enc_descriptions[idx_test]
        #label = enc_labels[idx_test]

        output = self.forward(batch)
        
        predictions = output.argmax(axis=1)
        test_acc = (predictions == label).sum() 
        print(f"Final Test Acc: {test_acc / idx_test.shape[0]:.3f}")



class LSTM_template(nn.Module):

    def __init__(self, vocab_size, dimension=128):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, 1)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out



    def train_lstm(self, desc,labels, vocab, word_to_ix, label_to_idx ):
        
        optimizer = optim.Adam(self.parameters())
        criterion = nn.CrossEntropyLoss()

        epochs = 10

        for epoch in range(epochs):
            for description, label in zip(desc, labels):
                length = torch.tensor(len(description)).int().cpu()
                description = torch.tensor([word_to_ix[word] for word in description])
                label = torch.tensor(label_to_idx[label]).long()
                
                output = self.forward(description, length)
                
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss.item())