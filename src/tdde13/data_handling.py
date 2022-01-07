
from os import read
from typing import List
from numpy.lib.type_check import real_if_close
import pandas as pd
import numpy as np
import ast
import torch
from sklearn.model_selection import train_test_split
import pickle

def glove_embedding(X):
    import spacy
    # import en_core_web_sm

    nlp = spacy.load("en_core_web_sm")
    from torchtext.legacy.data import Field
    # https://stackoverflow.com/questions/41170726/add-remove-custom-stop-words-with-spacy
    nlp.Defaults.stop_words |= {".",",","'","?",";"}

    n_nodes = X.shape[0]
    text_length = 200
    embedding_dim = 100
    

    text_field = Field(
        tokenize='basic_english', 
        sequential=True,
        lower=True,
        pad_token='<pad>', 
        eos_token='<eos>',
        fix_length=text_length,
        stop_words=nlp.Defaults.stop_words,
        # postprocessing=lambda x: embedding_glove[x]
    )
    
    # sadly have to apply preprocess manually
    # TODO
    preprocessed_text = pd.Series(X).apply(lambda x: text_field.preprocess(x))
    # load fastext simple embedding with 300d
    text_field.build_vocab(
        preprocessed_text, 
        vectors='glove.6B.100d' #300d
    )
    processed = text_field.process(preprocessed_text)

    import torch.nn as nn
    from torchtext.vocab import GloVe
    embedding_glove = GloVe(name='6B', dim=embedding_dim)
    embedding = nn.Embedding.from_pretrained(embedding_glove.vectors)
    # for setting trainable to False
    embedding.weight.requires_grad=False
    em = embedding(processed.t())
    em = em.reshape(n_nodes, text_length * embedding_dim)

    #print(em.shape)
    # print(em.to_sparse().shape)

    return em
    



def construct_sparse_tensor(coo):
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

def get_data(n=10000):
    
    path = 'datasets/book_reviews/finalv2.csv'
    df = pd.read_csv(path)
    df = df[:n]
    print(len(df.index))
    df = df.dropna(subset=['description', 'genres'])
    print(len(df.index))
    X_train, X_val_and_test, y_train, y_val_and_test  = train_test_split(df.description, df.genres,
                                                    random_state=42,
                                                    test_size=0.2)

    X_val, X_test, y_val, y_test  = train_test_split(X_val_and_test, y_val_and_test,
                                                    random_state=42,
                                                    test_size=0.5)



    return X_train, X_val, X_test, y_train, y_val, y_test

def get_data_graphsage(n=10000):
    
    path = 'datasets/book_reviews/finalv2.csv'
    df = pd.read_csv(path)
    df = df[:n]
    print(len(df.index))
    df = df.dropna(subset=['description', 'genres'])
    print(len(df.index))

    df = df.reset_index()

    edge_index_path = 'edge_index_desc_10k_clean_reset.pkl'
    with open(edge_index_path, 'rb') as f:  # Python 3: open(..., 'rb')
        edge_index = pickle.load(f)
    
    idx_train, idx_val, idx_test = split(labels=df.genres.to_numpy(), train_size=0.8, val_size=0.1, test_size=0.1)

    return df.description.to_numpy(), df.genres.to_numpy(), edge_index, idx_train, idx_val, idx_test
        

def load_reviews_and_descriptions():
    import pickle
    # ../../
    df = pd.read_csv('datasets/book_reviews/finalv2.csv')
    df_subset = df[2000:8000]

    with open('preprocessed_reviews_2K-8K.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        reviews, reviews_vocab, reviews_word_to_ix = pickle.load(f)

    with open('objs10000.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        descriptions, descriptions_vocab, descriptions_word_to_ix = pickle.load(f)

    label_to_idx = create_label_lookup(df_subset)
    descriptions = descriptions[2000:8000]

    cut_length = 5000
    reviews = np.array([text[:cut_length] for text in reviews], dtype=object)
    '''print(reviews.shape)
    print(reviews[0].shape)
    print(descriptions.shape)
    print(descriptions[0].shape)'''
    print("Joining")
    joined = np.array([np.concatenate((r,d), axis=0) if len(d) > 0 else r for (r, d) in zip(reviews, descriptions)], dtype=object)
    print("Splitting")
    # joined = np.concatenate((reviews, descriptions.T), axis=1)
    idx_train, idx_val, idx_test = split(labels=df_subset.genres.to_numpy(), train_size=0.6, val_size=0.2, test_size=0.2)

    vocab = reviews_vocab | descriptions_vocab

    # add special tokens to vocabulary
    # TODO do I even need '<eos>' and '<sos>'?
    vocab.update(['<unk>', '<pad>'])

    word_to_ix = dict(zip(vocab, range(len(vocab))))


    return joined, df_subset.genres.to_numpy(), vocab, word_to_ix, label_to_idx, idx_train, idx_val, idx_test



def split(labels: np.ndarray,
          train_size: float = 0.025,
          val_size: float = 0.025,
          test_size: float = 0.95,
          random_state: int = 42) -> List[np.ndarray]:
    """Split the arrays or matrices into random train, validation and test subsets.

    Parameters
    ----------
    labels: np.ndarray [n_nodes]
        The class labels
    train_size: float
        Proportion of the dataset included in the train split.
    val_size: float
        Proportion of the dataset included in the validation split.
    test_size: float
        Proportion of the dataset included in the test split.
    random_state: int
        Random_state is the seed used by the random number generator;

    Returns
    -------
    split_train: array-like
        The indices of the training nodes
    split_val: array-like
        The indices of the validation nodes
    split_test array-like
        The indices of the test nodes

    """
    idx = np.arange(labels.shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=labels)

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=labels[idx_train_and_val])
    
    return idx_train, idx_val, idx_test


def remove_empty_descriptions(descriptions, labels):
    d = []
    l = []
    for _d, _l in zip(descriptions, labels):
        # print(_d)
        if len(_d) > 0:
            d.append(_d)
            l.append(_l)
    return np.array(d, dtype=object), np.array(l, dtype=object)

def preprocess_text(nlp, descriptions):

    # remove nan values
    desc = np.array([np.array(preprocess(nlp, des)) if isinstance(des, str) else "" for des in descriptions], dtype=object)

    # preprocessed_descriptions = [ for text in desc]
    # temp = np.matrix.flatten(desc).tolist()
    temp = np.hstack(desc.flatten()).tolist()
    
    vocab, word_to_ix = create_vocabulary(temp)
    #vocab = set(temp)

    #word_to_ix = dict(zip(vocab, range(len(vocab))))

    return desc, vocab, word_to_ix


def create_vocabulary(temp):
    vocab = set(temp)

    # add special tokens to vocabulary
    # TODO do I even need '<eos>' and '<sos>'?
    vocab.update(['<unk>', '<pad>'])

    word_to_ix = dict(zip(vocab, range(len(vocab))))

    return vocab, word_to_ix


count = 0
def preprocess(nlp, text):

    global count
    print(count)
    count += 1

    # tokenize text
    try:
        doc = nlp(text)
    except:
        print(text)

    # doc = nlp(text)

    # removing any stop words, replacing each remaining token with its lemma (base form)
    text = [token.lemma_ for token in doc if not token.is_stop]

    # discard all lemmas that contain non-alphabetical characters, and put words in lower case.
    text = [t for t in text if t.isalpha()]

    return text

def remove_unreachable_edges(df):
    ids = set(df.book_id)
    edges = df.similar_books.map(lambda links: [link for link in links if link in ids])
    return edges

def literal_string_list_to_list(df):
    return df.similar_books.map(ast.literal_eval)

def pad_sequence(desc):
    max_len = max([len(x) for x in desc])
    padded_desc = np.array([np.append(d, ['<pad>' for _ in range(max_len - len(d))]) if max_len - len(d) > 0 else d for d in desc ], dtype=object)
    return padded_desc, max_len

def create_adjacency_matrix(df):

    x = []
    y = []
    N = len(df.index)
    for i, row in enumerate(df.similar_books):
        if row != []:
            for item in row:
                '''print(item)
                print(df.index[df.book_id == item])'''
                if item in df.book_id:
                    '''print(item)
                    print(df.index[df.book_id == item])'''
                    match = df.index[df.book_id == item]
                    if len(match) > 0:

                        idx = [0]
                        x.append(i)
                        y.append(idx)

    # return torch.sparse.FloatTensor([x,y], torch.ones(len(x)).float(), (N, N))
    return x, y, N

    
def create_label_lookup(df):
    genres = set(df.genres)
    label_to_idx = dict(zip(genres, range(len(genres))))
    return label_to_idx




class DataHandler():

    def __init__(self) -> None:
        pass
    
    
    def load_data(self, data_path: str, iterative: bool = True, select_features: List = None, save: bool = False, input_format: str = 'json', output_format: str = 'json') -> pd.DataFrame:
        
        # TODO
        if iterative:

            chunksize = 10 ** 3
            # books = './datasets/book_reviews/goodreads_books.json'
            with pd.read_json(data_path, chunksize=chunksize, lines=True) as reader:
                # b = next(iter(reader))
                # TODO check how many chunks
                for i, chunk in enumerate(reader):
                    if i == 0:
                        df = chunk[select_features]
                    else:
                        df.append(chunk[select_features])

        else:
            df = pd.read_json(data_path, lines=True)

        '''if select_features is None:
            df.to_csv('out.csv',index=False)
        else:
            df[select_features].to_csv('out.csv',index=False)'''

        df.to_csv('out.csv',index=False)

    
    def create_genre(self, data_path: str = './datasets/book_reviews/goodreads_book_genres_initial.json', first_genre: bool = True) -> pd.DataFrame:

        # load 
        df_genres = pd.read_json(data_path, lines=True)
        # drop books without genre alias empty dict
        df_genres_no_na = df_genres.loc[df_genres.genres != {}]

        if first_genre:
            genres = [next(iter(row.genres)) for row in df_genres_no_na.itertuples()]
        else:
            genres = [list(row.genres.keys()) for row in df_genres_no_na.itertuples()]
            
        data = np.array([df_genres_no_na.book_id, genres])
        return pd.DataFrame(columns=df_genres_no_na.columns, data=np.swapaxes(data,0,1))
        
    def create_reviews(self, data_path: str = './datasets/book_reviews/goodreads_reviews_spoiler_raw.json', chunksize: int = 10 ** 4, features: List = ['book_id', 'review_text'], df_genre_sample: pd.DataFrame = None) -> pd.DataFrame:
        
        # all features: 'user_id', 'book_id', 'review_id', 'rating', 'review_text',
        # 'date_added', 'date_updated', 'read_at', 'started_at', 'n_votes',
        # 'n_comments'
        ids = df_genre_sample.book_id.tolist()
        data = [[] for _ in range(len(features))]
        with pd.read_json(data_path, chunksize=chunksize, lines=True) as reader:
                
            reviews = next(iter(reader))
            for row in reviews.itertuples():
                if row.book_id in ids:
                    for i in range(len(features)):
                        data[i].append(row[features[i]])

            # reviews = [next(iter(row.genres)) for row in reviews.itertuples()]
            '''for i, chunk in enumerate(reader):
                if i == 0:
                    reviews = chunk[features]
                else:
                    reviews.append(chunk[features])'''
                    
            #chunks = sum([1 for chunk in reader])

        # reviews.to_csv('reviews.csv',index=False)
        #return chunks
        return data

    def create_metadata(self, data_path: str = './datasets/book_reviews/goodreads_reviews_spoiler_raw.json', chunksize: int = 10 ** 4, features: List = ['book_id', 'review_text']) -> pd.DataFrame:
        
        # all features: 'isbn', 'text_reviews_count', 'series', 'country_code', 'language_code',
        #'popular_shelves', 'asin', 'is_ebook', 'average_rating', 'kindle_asin',
        #'similar_books', 'description', 'format', 'link', 'authors',
        #'publisher', 'num_pages', 'publication_day', 'isbn13',
        #'publication_month', 'edition_information', 'publication_year', 'url',
        #'image_url', 'book_id', 'ratings_count', 'work_id', 'title',
        #'title_without_series'
        with pd.read_json(data_path, chunksize=chunksize, lines=True) as reader:

            for i, chunk in enumerate(reader):
                if i == 0:
                    reviews = chunk[['book_id', 'review_text']]
                else:
                    reviews.append(chunk[['book_id', 'review_text']])
                    
            #chunks = sum([1 for chunk in reader])

        # reviews.to_csv('reviews.csv',index=False)
        #return chunks
        return reviews

    def create_dataset(self, n_samples: int = 10**5, n_categories: int = 10) -> pd.DataFrame:
        
        df_genres = self.create_genre()

        df_genres_sampled = df_genres.groupby('genres', group_keys=False).apply(lambda x: x.sample(n=n_samples / n_categories))

        df_reviews = self.create_reviews(df_genre_sample=df_genres_sampled)
        
        return df_reviews

    def similiarbooks2ints(self, similiarbooks):
        return [int(''.join(c for c in st if c.isdigit())) if st != '[[]]' else [] for st in similiarbooks.split()]