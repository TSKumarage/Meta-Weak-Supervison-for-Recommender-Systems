"""
Authors : Kumarage Tharindu & Paras Sheth
Organization : DMML, ASU
Project : Meta-Weak Supervision for Recommender Systems
Task : Word to Vector Embedding : Create w2v representation of the given data

"""

from gensim.models import KeyedVectors
import nltk
import numpy as np
from tqdm import tqdm


class Word2Vec:
    def __init__(self, pre_train_model='GoogleNews-vectors-negative300.bin.gz',
                 embedding_dim=300, aggregate_strategy='sum'):
        self.pre_train_model = pre_train_model
        self.embedding_file_path = EMBEDDING_FILE
        self.embedding_dim = embedding_dim
        self.aggregate_strategy = aggregate_strategy

    def tokenization(self, data):
        tokens = []
        for i in range(len(data)):
            tokenizer = nltk.tokenize.WhitespaceTokenizer()
            tokens.append(tokenizer.tokenize(data[i]))
        return tokens

    def fit(self, data):
        list_of_words = self.tokenization(data)
        word2vec = KeyedVectors.load_word2vec_format(self.embedding_file_path, binary=True)

        # Assumed distribution for the random embeddings
        emb_mean, emb_std = (0.004451992, 0.4081574)

        data_embedded = np.zeros(shape=(len(list_of_words), self.embedding_dim))

        idx = 0

        if self.aggregate_strategy == 'sum':
            for sentence in tqdm(list_of_words):
                sum_of_vectors = np.zeros(self.embedding_dim)
                for word in sentence:
                    try:
                        sum_of_vectors += word2vec[word]
                    except:
                        sum_of_vectors += np.random.normal(emb_mean, emb_std, self.embedding_dim)
                        pass
                data_embedded[idx] = sum_of_vectors
                idx += 1

        if self.aggregate_strategy == 'mean':
            for sentence in tqdm(list_of_words):
                sum_of_vectors = np.zeros(self.embedding_dim)
                for word in sentence:
                    try:
                        sum_of_vectors += word2vec[word]
                    except:
                        sum_of_vectors += np.random.normal(emb_mean, emb_std, self.embedding_dim)
                        pass
                data_embedded[idx] = sum_of_vectors / len(sentence)
                idx += 1

        return data_embedded