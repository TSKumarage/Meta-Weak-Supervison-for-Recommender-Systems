"""
Authors : Kumarage Tharindu & Paras Sheth
Organization : DMML, ASU
Project : Meta-Weak Supervision for Recommender Systems
Task : Aspect extraction Model

"""
import os
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, Model
from keras.layers import Input, Dense, LSTM, Embedding, Concatenate, RepeatVector, TimeDistributed, Bidirectional
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.losses import binary_crossentropy
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

def teacher_model():
    # Teacher params
    teacher_input_dim = num_seed_words

    # Input layer
    teacher_inp = Input(shape=(teacher_input_dim,), name='input_layer')

    # Weight the seed word importance
    # attention_probs = Dense(teacher_input_dim, activation='softmax', name='attention_probs')(teacher_inp)
    # attention_mul = layers.multiply([teacher_inp, attention_probs], name='attention_mul')

    # Output the aspect probabilities
    teacher_output = Dense(num_aspects, input_dim=teacher_input_dim,
                           activation='relu', name='teacher_output')(teacher_inp)

    # Overall teacher model
    model = Model(teacher_inp, teacher_output)
    model.summary()

    return model

def student_model():
    # Input layer
    student_inp = Input(shape=(student_input_dim,), name='input_layer')

    # Output the aspect probabilities
    student_output = Dense(num_aspects, input_dim=student_input_dim,
                           activation='softmax', name='student_output')(student_inp)

    # Overall student model
    model = Model(student_inp, student_output)
    model.summary()

    return model

def iterative_training(s_model, t_model, params):

    epochs = 10
    batch_size = 16

    # Optimizer function def
    teacher_adam = optimizers.adam(lr=000.1)
    student_adam = optimizers.adam(lr=00.1)

    t_model.compile(loss='categorical_crossentropy', optimizer=teacher_adam)
    s_model.compile(loss='categorical_crossentropy', optimizer=student_adam)

    # Early Stopping Callback with best model saving callback
    # es = EarlyStopping(monitor='val_student_output_loss', mode='min', verbose=1, patience=10)  # Monitoring the validation loss of the fake news classifier
    # mc = ModelCheckpoint('best_model.h5', monitor='val_fakenews_loss', mode='min', save_best_only=True)

    valid = np.random.dirichlet(np.ones(num_aspects), size=batch_size)

    # Select a random batch of reviews
    idx = np.random.randint(0, teacher_train.shape[0], batch_size)
    data_samples = teacher_train[idx]

    teacher_init_loss = t_model.train_on_batch(data_samples, valid)
    print("Teacher init train ....")
    for epoch in range(epochs):
        # ---------------------
        #  Train Teacher
        # ---------------------

        # Select a random batch of reviews
        idx = np.random.randint(0, teacher_train.shape[0], batch_size)
        data_samples = teacher_train[idx]

        valid = t_model.predict(data_samples)

        teacher_loss = t_model.train_on_batch(data_samples, valid)
        print("Teacher init Loss ------- ", teacher_loss)

    print("Teacher init train completed....")

    converge = 1
    limit = 20
    student_epochs = 50
    teacher_epochs = 50
    batch_size = 128
    student_losses = []
    teacher_losses = []
    best_student_loss = 15

    print("Iterative training......")
    while (converge <= limit):

        for epoch in range(student_epochs):  # Applying Teacher and training Student

            idx = np.random.randint(0, teacher_train.shape[0], batch_size)
            data_samples = teacher_train[idx]

            valid = t_model.predict(data_samples)

            data_samples = student_train[idx]
            student_loss = s_model.train_on_batch(data_samples, valid)

        s_losses.append(student_loss)

        if student_loss < best_student_loss:
            best_student_loss = student_loss
            weights = s_model.get_weights()
            # print(weights)
            s_model.save('drive/My Drive/Meta-Weak/best_student_model.h5')  # OR model.save_weights()
            print("Best model w/ val loss {} saved to {}".format(best_student_loss, 'best_student_model.h5'))

        for epoch in range(teacher_epochs):  # Applying Student and updating teacher

            idx = np.random.randint(0, student_train.shape[0], batch_size)
            data_samples = student_train[idx]

            valid = s_model.predict(data_samples)

            data_samples = teacher_train[idx]
            teacher_loss = t_model.train_on_batch(data_samples, valid)

        teacher_losses.append(teacher_loss)

        print("Iteration : ", converge, " ---Teacher Loss: ", teacher_loss, " - Student Loss: ", student_loss)
        x = list(range(1, limit + 1))
        converge += 1

    plt.plot(x, student_losses, label="Student Model")
    plt.plot(x, teacher_losses, label="Teacher Model")
    plt.legend(loc="upper right")
    plt.show()
