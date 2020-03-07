"""
Authors : Kumarage Tharindu & Paras Sheth
Organization : DMML, ASU
Project : Meta-Weak Supervision for Recommender Systems
Task : Bert Embedding : Create bert representation of the given data

"""

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from keras.preprocessing.sequence import pad_sequences

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
# %matplotlib inline

import numpy as np
from tqdm import tqdm


class BertVec:
    def __init__(self, pre_train_model='bert-base-uncased',
                 embedding_dim=768, aggregate_strategy='2ndtolast'):
        self.pre_train_model = pre_train_model
        self.embedding_file_path = EMBEDDING_FILE
        self.embedding_dim = embedding_dim
        self.aggregate_strategy = aggregate_strategy
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained(pre_train_model)


    def tokenization(self, data):
        tokernized_data = []

        # Add the special tokens.
        # Split the sentence into tokens.
        # Map the token strings to their vocabulary indeces.
        for sentence in data.values:
            sentence = "[CLS] " + sentence
            sentence.replace(".", ". [SEP]")
            sentence += " [SEP]"
            tokenized_text = self.tokenizer.tokenize(sentence)
            # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokernized_data.append(tokenized_text)
            # Display the words with their indeces.
            # for tup in zip(tokenized_text, indexed_tokens):
            #     print('{:<12} {:>6,}'.format(tup[0], tup[1]))

        print(tokernized_data[0])

        return tokernized_data

    def fit(self, data, MAX_LEN = 64):
        list_of_words = self.tokenization(data)

        # Pad our input tokens
        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokernized_data],
                                  maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokernized_data]
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        tok = 1
        segments_id_list = []

        for segment in input_ids:
            segment_ids = []
            for token in segment:
                if token == 0:
                    segment_ids.append(token)
                else:
                    segment_ids.append(tok)

                if token == 1012:
                    tok = int(not tok)
            segments_id_list.append(segment_ids)

        segments_id_list = np.array(segments_id_list)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor(input_ids)
        segments_tensors = torch.tensor(segments_id_list)

        # Load pre-trained model (weights)
        model = BertModel.from_pretrained(self.pre_train_model)

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        model.eval()

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)

        bert_embed = []

        if self.aggregate_strategy == '2ndtolast':

            for sentence_id in range(len(sample_data.values)):
                # `token_vecs` is a tensor with shape [ x 768]
                token_vecs = encoded_layers[11][sentence_id]

                # Calculate the average of all token vectors.
                sentence_embedding = torch.mean(token_vecs, dim=0)
                bert_embed.append(sentence_embedding.numpy())


        return bert_embed