"""
Authors : Kumarage Tharindu & Paras Sheth
Organization : DMML, ASU
Project : Meta-Weak Supervision for Recommender Systems
Task : Aspect Bag of Word Embedding

"""
import numpy as np
import collections

class AspectBOW:
    def __init__(self, aspect_dict):

        self.aspect_dict = aspect_dict
        self.num_aspects = len([key for key in self.aspect_dict])
        self.num_seed_words = 0

    def fit(self, data):
        seed_words = []
        for aspect in aspect_dict:
            seed_words.extend(aspect_dict[aspect])
        self.num_seed_words = len(seed_words)

        # Create a preprocessor object
        bow_emb = []
        # bow_emb.append(seed_words)

        segments = data.values

        for segment in segments:
            if type(segment) == float:
                continue

            bow_vec = [segment.count(word) for word in seed_words]
            bow_emb.append(bow_vec)

        return bow_emb