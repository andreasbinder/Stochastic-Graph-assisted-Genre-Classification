from pickle import FALSE
import typing
import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import Data

import torch.nn.functional as F

from torch_sparse import SparseTensor

from sklearn.preprocessing import LabelEncoder

import numpy as np

from tqdm import tqdm

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()

        self.out_dim = out_dim
        self.in_dim = in_dim

        self.dropout = dropout
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim) # TODO hidden_dim
        self.conv3 = SAGEConv(hidden_dim, out_dim)
        
    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html
        adj = SparseTensor(row=edge_index[0], col=edge_index[1])
        
        x = self.conv1(x, edge_index) #adj_t
        
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.conv2(x, data.edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        '''
        x = self.conv3(x, data.edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)'''
        return torch.log_softmax(x, dim=-1)

    def construct_sparse_tensor(self, coo):
        # https://discuss.pytorch.org/t/creating-a-sparse-tensor-from-csr-matrix/13658/5
        # coo = coo_matrix(([3,4,5], ([0,1,1], [2,0,2])), shape=(2,3))

        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

    def train_graphsage_batchwise(self, X, edge_index, labels, idx_train, idx_val, idx_test, hparams):
        # https://medium.com/swlh/text-classification-using-scikit-learn-pytorch-and-tensorflow-a3350808f9f7
        lr = hparams["lr"]
        epochs = hparams["epochs"]
        batch_size = hparams["batch_size"]
        patience = hparams["patience"]
        use_glove = hparams["use_glove"]

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()


        edge_index = edge_index.t().contiguous()
        #edge_index = sparse(edge_index)
        # print(edge_index)

        # x_sparse = self.construct_sparse_tensor(X.tocoo())
        # x_sparse = torch.ones(9892, 100)
        print("Create Data Object")
        # data = Data(x=x_sparse, edge_index=edge_index.t().contiguous(), y=labels)
        data = Data(x=X, edge_index=edge_index, y=labels)
        
        data.train_mask = idx_train
        data.val_mask = idx_val
        data.test_mask = idx_test

   
        # torch.LongTensor

        #x_train = torch.tensor(scipy.sparse.csr_matrix.todense(X_train)).float()
        #x_test = torch.tensor(scipy.sparse.csr_matrix.todense(X_test)).float()
        

        
        # https://discuss.pytorch.org/t/how-to-define-train-mask-val-mask-test-mask-in-my-own-dataset/56289/5
        # neighborloader example
        # - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py
        train_loader = NeighborLoader(
            data,
            # data.subgraph(torch.LongTensor(data.train_mask)),
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=batch_size,
            # TODO
            input_nodes=torch.LongTensor(data.train_mask)
            # input_nodes=torch.LongTensor(range(300))
        )

        val_loader = NeighborLoader(
            data,
            # data.subgraph(torch.LongTensor(data.train_mask)),
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=batch_size,
            # TODO
            input_nodes=torch.LongTensor(data.val_mask)
            # input_nodes=torch.LongTensor(range(300))
        )

        test_loader = NeighborLoader(
            data,
            # data.subgraph(torch.LongTensor(data.train_mask)),
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=batch_size,
            # TODO
            input_nodes=torch.LongTensor(data.test_mask)
            # input_nodes=torch.LongTensor(range(300))
        )

        best_loss = np.inf
        best_acc = np.NINF
        trace_train = []
        trace_val = []

        for epoch in range(1, epochs + 1):
            #print(len(desc))
            running_train_loss = 0.0
            running_train_acc = 0.0

            running_val_acc = 0.0
            running_val_loss = 0.0

            print(f"Epoch: {epoch}")
            c = 0
            self.train()
            n_train = 0
            for batch in tqdm(train_loader):
                
                n_train += batch.x.shape[0]
                # assert False, "batch_size"
                
                optimizer.zero_grad()
                
                # forward pass
                out = self.forward(batch)
                
                '''print(out.shape)
                print(batch.y.shape)
                assert False, "training"'''
                # loss caculation and backward pass
                loss = criterion(out, batch.y)
                running_train_loss += loss.item()
                predictions = out.argmax(axis=1)
                running_train_acc += (predictions == batch.y).sum() 
                # assert False, "in loop"

                loss.backward()
                optimizer.step()
            self.eval()
            n_val = 0
            for batch in tqdm(val_loader):
                # validation
                n_val += batch.x.shape[0]
                out = self.forward(batch)
                loss = criterion(out, batch.y)
                running_val_loss += loss.item()
                predictions = out.argmax(axis=1)
                running_val_acc += (predictions == batch.y).sum() 
                # print(loss.item())
            print(f"train loss: {running_train_loss:.4f} train acc: {running_train_acc / n_train:.4f} val loss: {running_val_loss:.4f} val acc: {running_val_acc / n_val:.4f}")

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
        torch.save(best_state, f'model_graphsage_{use_glove}.pt')
       
        # final evaluation over test set 
        self.eval()
        n_test = 0
        running_test_acc = 0.0
        for batch in tqdm(test_loader):
            # validation
            n_test += batch.x.shape[0]
            predictions = self.forward(batch).argmax(axis=1)
            running_test_acc += (predictions == batch.y).sum() 

        print(f"test acc: {running_test_acc / n_test:.4f}")

        print("Finished Training")

        return trace_train, trace_val
