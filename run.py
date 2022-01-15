import sys

import random 

from src.tdde13.data_handling import * #
from src.tdde13.evaluation import * 
from sklearn.feature_extraction.text import CountVectorizer # we could have use TfidfVectorizer too. From sklearn doc : "tf-idf vectors are also known to work well in practice"
from sklearn.preprocessing import LabelEncoder

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def train_mlp():
    from src.tdde13.baselines import MLP

    X_train, X_val, X_test, y_train, y_val, y_test = get_data()

    use_glove = True
    if use_glove:
        # print(X_train)

        X_train_transformed = glove_embedding(X_train)
        X_val_transformed = glove_embedding(X_val)
        X_test_transformed = glove_embedding(X_test)

        # assert False, "embedding"
    else:
        vectorizer = CountVectorizer()
        X_train_transformed = vectorizer.fit_transform(X_train)
        X_val_transformed = vectorizer.transform(X_val)
        X_test_transformed = vectorizer.transform(X_test)

        X_train_transformed = construct_sparse_tensor(X_train_transformed.tocoo())
        X_val_transformed = construct_sparse_tensor(X_val_transformed.tocoo())
        X_test_transformed = construct_sparse_tensor(X_test_transformed.tocoo())


    input_size = X_train_transformed.shape[1]
    output_size = 10

    y_train = LabelEncoder().fit_transform(y_train)
    y_val = LabelEncoder().fit_transform(y_val)
    y_test = LabelEncoder().fit_transform(y_test)

    y_train = torch.tensor(y_train)
    y_val = torch.tensor(y_val)
    y_test = torch.tensor(y_test)

    hparams = {
        "lr" : 0.0002,
        "epochs" : 35, # 15,
        "batch_size" : 128,
        "patience" : 5
    }

    mlp = MLP(input_size, output_size)

    trace_train, trace_val = mlp.train_mlp(X_train_transformed, X_val_transformed, X_test_transformed, y_train, y_val, y_test, hparams)

    plot_curves(trace_train, trace_val)

def train_graphsage():
    from src.tdde13.graphsage import GraphSAGE

    X, y, edge_index, idx_train, idx_val, idx_test = get_data_graphsage()

    use_glove = False
    if use_glove:
        X_transformed = glove_embedding(X)
    else:
        vectorizer = CountVectorizer()
        X_transformed = vectorizer.fit_transform(X)
        X_transformed = construct_sparse_tensor(X_transformed.tocoo())
        

    y = LabelEncoder().fit_transform(y)
    y = torch.tensor(y)   

    in_dim = X_transformed.shape[1] 
    hidden_dim = 128
    out_dim = 10

    hparams = {
        "lr" : 0.001,
        "epochs" : 35, # 15,
        "batch_size" : 128,
        "patience" : 5,
        "use_glove" : use_glove
    }

    graphsage = GraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)

    # graphsage.train_graphsage(X_transformed, edge_index, y, idx_train, idx_val, idx_test)
    trace_train, trace_val = graphsage.train_graphsage_batchwise(X_transformed, edge_index, y, idx_train, idx_val, idx_test, hparams)

    plot_curves(trace_train, trace_val)

if __name__ == "__main__":
    train_graphsage()