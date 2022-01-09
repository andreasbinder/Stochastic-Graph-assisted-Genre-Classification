
import pandas as pd
from src.tdde13.data_handling import *
# joint books
def filter_reviews():
    #meta_100 = pd.read_csv('meta_100k.csv')
    #meta_id = meta_100.book_id.unique()
    path = './datasets/book_reviews/goodreads_book_genres_initial.json'
    df_genres = pd.read_json(path, lines=True)
    df_genres = df_genres[:1000000]
    meta_id = df_genres.book_id.unique()
    print("Sucessfully read meta file")
    data_path = './datasets/book_reviews/goodreads_reviews_spoiler_raw.json'
    chunksize = 10 ** 5
    # ids = df_genre_sample.book_id.tolist()
    # data = [[] for _ in range(len(features))]
    ids = []
    texts = []
    with pd.read_json(data_path, chunksize=chunksize, lines=True) as reader:
        
        # chunk = next(iter(reader))
        for i, chunk in enumerate(reader):
            print(f"Chunk: {i * chunksize}")
            mask_ = chunk.book_id.isin(meta_id)
            dummy = chunk[['book_id', 'review_text']].loc[mask_]
            dummy.to_csv(f'datasets/temp/reviews_{i * chunksize}.csv',index=False)

def filter_meta():
    #meta_100 = pd.read_csv('meta_100k.csv')
    #meta_id = meta_100.book_id.unique()
    reviews_500k = pd.read_csv('reviews_500k.csv')
    review_id = reviews_500k.book_id.unique()
    print("Sucessfully read meta file")
    data_path = './datasets/book_reviews/goodreads_books.json'
    chunksize = 10 ** 4
    # ids = df_genre_sample.book_id.tolist()
    # data = [[] for _ in range(len(features))]
    ids = []
    texts = []
    with pd.read_json(data_path, chunksize=chunksize, lines=True) as reader:
        
        # chunk = next(iter(reader))
        for i, chunk in enumerate(reader):
            print(f"Chunk: {i * chunksize}")
            mask_ = chunk.book_id.isin(review_id)
            dummy = chunk[['book_id', 'description', 'similar_books']].loc[mask_]
            dummy.to_csv(f'datasets/temp/book_{i * chunksize}.csv',index=False)

def generate_books():
    #meta_100 = pd.read_csv('meta_100k.csv')
    #meta_id = meta_100.book_id.unique()

    data_path = './datasets/book_reviews/goodreads_books.json'
    chunksize = 10 ** 4
    with pd.read_json(data_path, chunksize=chunksize, lines=True) as reader:
        
        # chunk = next(iter(reader))
        for i, chunk in enumerate(reader):
            print(f"Chunk: {i * chunksize}")
            chunk[['book_id', 'description', 'similar_books']].to_csv(f'datasets/temp/book_{i * chunksize}.csv',index=False)


def preprocess_spacy():
    # next(iter(pd.read_csv('datasets/book_reviews/final.csv', chunksize=10)))
    import spacy
    import pickle
    # import en_core_web_sm


    # , exclude=["ner", "parser"]
    nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser"])
    df = pd.read_csv('datasets/book_reviews/finalv2.csv')
    df10000 = df[:10000]
    desc, vocab, word_to_ix = preprocess_text(nlp, df10000.description)
    with open('objs10000.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([desc, vocab, word_to_ix], f)

def preprocess_reviews():
    import spacy
    import pickle
    df = pd.read_csv('datasets/book_reviews/finalv2.csv')
    nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser"])
    import ast
    # type(ast.literal_eval(first.reviews)[0])
    def join_str(review):
        review = ast.literal_eval(review)
        # type(first.reviews)
        review_str = ''.join(review)
        return review_str
    

    df_subset = df[:10000]
    reviews = df_subset.reviews.map(join_str)
    desc, vocab, word_to_ix = preprocess_text(nlp, reviews)
    with open('preprocessed_reviews_10000.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([desc, vocab, word_to_ix], f)

if __name__ == "__main__":
    preprocess_reviews()