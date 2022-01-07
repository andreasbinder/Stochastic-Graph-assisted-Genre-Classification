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
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, out_dim)
        '''self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops = False)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops = False)
        self.conv3 = GCNConv(hidden_dim, out_dim)
        '''
    def forward(self, data):
        #print(data.x)
        #print("edge")
        #print(data.edge_index)
        
        x, edge_index = data.x, data.edge_index
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html
        adj = SparseTensor(row=edge_index[0], col=edge_index[1])
        #print(x.shape)
        x = self.conv1(x, edge_index) #adj_t
        #print(x.shape)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        #print("layer 1")
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


    def train_graphsage(self, X, edge_index, labels, idx_train, idx_val, idx_test):
        # https://medium.com/swlh/text-classification-using-scikit-learn-pytorch-and-tensorflow-a3350808f9f7
        optimizer = optim.Adam(self.parameters(), lr=0.0002)
        criterion = nn.CrossEntropyLoss()

        epochs = 15
        batch_size = 128
        
        from torch_geometric.data import Data
        # print(self.in_dim)
       
        x_sparse = self.construct_sparse_tensor(X.tocoo())
        # x_sparse = torch.ones(9892, 100)
        data = Data(x=x_sparse, edge_index=edge_index.t().contiguous())
        #x_train = torch.tensor(scipy.sparse.csr_matrix.todense(X_train)).float()
        #x_test = torch.tensor(scipy.sparse.csr_matrix.todense(X_test)).float()

        
        # x_test = self.construct_sparse_tensor(X_test.tocoo())

        labels = LabelEncoder().fit_transform(labels)
        labels = torch.tensor(labels)
        # y_test = LabelEncoder().fit_transform(y_test)

        '''y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)'''

        for epoch in range(epochs):
            #print(len(desc))
            running_train_loss = 0.0
            running_train_acc = 0.0
            print(f"Epoch: {epoch}")
            c = 0
            # for description, label in zip(desc, labels):
            # for index, i in enumerate(range(0, len_desc, batch_size)):
            optimizer.zero_grad()
            out = self.forward(data)
            loss = F.nll_loss(out[idx_train], labels[idx_train])
            loss.backward()
            optimizer.step()
            print(loss.item())

            self.eval()
            pred = out.argmax(dim=1)
            correct = (pred[idx_test] == labels[idx_test]).sum()
            acc = int(correct) / int(idx_test.sum())
            print(f'Accuracy: {acc:.4f}')

            '''running_val_loss = 0.0
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
                # optimizer.step()'''

            #print(f"Train Loss: {running_train_loss:.4f} Train Acc {running_train_acc / x_train.shape[0]:.3f} Val Loss {running_val_loss:.4f} Val Acc {running_val_acc / x_test.shape[0]:.3f}")
        
        print("Finished Training")

    def train_graphsage_batchwise(self, X, edge_index, labels, idx_train, idx_val, idx_test):
        # https://medium.com/swlh/text-classification-using-scikit-learn-pytorch-and-tensorflow-a3350808f9f7
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        epochs = 10
        batch_size = 128


        '''labels = LabelEncoder().fit_transform(labels)
        labels = torch.tensor(labels)'''



        edge_index = edge_index.t().contiguous()
        #edge_index = sparse(edge_index)
        # print(edge_index)

        # x_sparse = self.construct_sparse_tensor(X.tocoo())
        # x_sparse = torch.ones(9892, 100)
        print("Create Data Object")
        # data = Data(x=x_sparse, edge_index=edge_index.t().contiguous(), y=labels)
        data = Data(x=X, edge_index=edge_index, y=labels)
        '''print(data)
        assert False, "Graph Test"'''
        
        data.train_mask = idx_train
        data.val_mask = idx_val
        data.test_mask = idx_test

   

        print(data.train_mask)
        print(type(data.train_mask))

        # torch.LongTensor

        #x_train = torch.tensor(scipy.sparse.csr_matrix.todense(X_train)).float()
        #x_test = torch.tensor(scipy.sparse.csr_matrix.todense(X_test)).float()
        

        
        # https://discuss.pytorch.org/t/how-to-define-train-mask-val-mask-test-mask-in-my-own-dataset/56289/5
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

        '''c = 0
        for batch in tqdm(train_loader):
            c+=batch.x.shape[0]
        print(c)

        c = 0
        for batch in tqdm(val_loader):
            c+=batch.x.shape[0]
        print(c)'''


        

        for epoch in range(epochs):
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
                # print(i)
                '''print(batch.train_mask.shape)
                print(batch.val_mask.shape)

                assert False'''

                # total_examples += batch.batch_size
                '''print(batch.batch_size)
                print("in between")
                print(batch.x.shape[0])'''
                n_train += batch.x.shape[0]
                # assert False, "batch_size"
                
                optimizer.zero_grad()
                
                # forward pass
                out = self.forward(batch)
                '''print(f"mask {batch.mask.shape}")
                print(out.shape)
                print(batch.y.shape)

                print(batch.train_mask.shape)'''

                # assert False, "training"
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

        '''self.eval()
        pred = self.forward(data).argmax(dim=1)
        print("pred")
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        print(f'Test Acc: {acc:.4f}')'''

        print("Finished Training")

    def train_graphsage_old(self, data, train_idx):
        lr = 1e-4 
        epochs = 50 
        hidden_dim = 75
        
        model = GraphSAGE(in_dim=data.num_node_features, 
                        hidden_dim=hidden_dim, 
                        out_dim=self.out_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, 1 + epochs):
            loss = self.train(model, data, train_idx, optimizer)
            # result = test(model, data, split_idx, evaluator)
            #logger.add_result(run, result)
        if epoch % 10 == 0:
                # train_acc, valid_acc, test_acc = result
                print(f'Epoch: {epoch}/{epochs}, '
                    f'Loss: {loss:.4f}')

    def train_(model, data, train_idx, optimizer):
        model.train()
        optimizer.zero_grad()
        out = model(data)[train_idx]
        loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test(model, data, split_idx, evaluator):
        model.eval()
        out = model(data)
        y_pred = out.argmax(dim=-1, keepdim=True)
        train_acc = evaluator.eval({
            'y_true': data.y[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': data.y[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': data.y[split_idx['test']],
            'y_pred': y_pred[split_idx['test']],
        })['acc']
        return train_acc, valid_acc, test_acc