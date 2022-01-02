
from os import read
from typing import List
from numpy.lib.type_check import real_if_close
import pandas as pd
import numpy as np
import ast
import torch
from sklearn.model_selection import train_test_split

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