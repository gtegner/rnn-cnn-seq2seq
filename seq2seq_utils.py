from tensorflow.python.lib.io import file_io

from keras.preprocessing.text import text_to_word_sequence
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop
from nltk import FreqDist
import numpy as np
import os
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from keras.preprocessing.sequence import pad_sequences


def load_data2(source,dist,max_len,vocab_size):
    f = file_io.FileIO(source, 'r')
    X_data = f.read()
    f.close()
    f = file_io.FileIO(dist, 'r')
    y_data = f.read()
    f.close()

    X = [x + '\t' + y for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) > 0 and len(y) > 0 and len(x) <= max_len and len(y) <= max_len]

    thefile = open('data/eng-sv.txt','w')
    for item in X:
        thefile.write("%s\n" % item)

    return X
   # y = [y for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) > 0 and len(y) > 0 and len(x) <= max_len and len(y) <= max_len]


    return X

def load_data(source, dist, max_len, vocab_size):

    # Reading raw text from source and destination files
    f = file_io.FileIO(source, 'r')
    X_data = f.read()
    f.close()
    f = file_io.FileIO(dist, 'r')
    y_data = f.read()
    f.close()

    # Splitting raw text into array of sequences
    X = [text_to_word_sequence(x)[::-1] for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) > 0 and len(y) > 0 and len(x) <= max_len and len(y) <= max_len]
    y = [text_to_word_sequence(y) for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) > 0 and len(y) > 0 and len(x) <= max_len and len(y) <= max_len]

    # Creating the vocabulary set with the most common words
    dist = FreqDist(np.hstack(X))
    X_vocab = dist.most_common(vocab_size-1)
    dist = FreqDist(np.hstack(y))
    y_vocab = dist.most_common(vocab_size-1)

    # Creating an array of words from the vocabulary set, we will use this array as index-to-word dictionary
    X_ix_to_word = [word[0] for word in X_vocab]
    # Adding the word "ZERO" to the beginning of the array
    X_ix_to_word.insert(0, '<SOS>')


    # Adding the word 'UNK' to the end of the array (stands for UNKNOWN words)
    X_ix_to_word.append('UNK')
    #print(X_ix_to_word)

    # Creating the word-to-index dictionary from the array created above
    X_word_to_ix = {word:ix for ix, word in enumerate(X_ix_to_word)}

    # Converting each word to its index value
    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):
            if word in X_word_to_ix:
                X[i][j] = X_word_to_ix[word]
            else:
                X[i][j] = X_word_to_ix['UNK']

    y_ix_to_word = [word[0] for word in y_vocab]
    y_ix_to_word.insert(0, '<SOS>')


    y_ix_to_word.append('UNK')
    y_word_to_ix = {word:ix for ix, word in enumerate(y_ix_to_word)}
    for i, sentence in enumerate(y):
        for j, word in enumerate(sentence):
            if word in y_word_to_ix:
                y[i][j] = y_word_to_ix[word]
            else:
                y[i][j] = y_word_to_ix['UNK']
    return (X, len(X_vocab)+2, X_word_to_ix, X_ix_to_word, y, len(y_vocab)+2, y_word_to_ix, y_ix_to_word)



class Lang():
    def __init__(self, name, data, word2ix,ix2word,vocab_len):
        self.name = name
        self.data = data
        self.word2index = word2ix
        self.index2word = ix2word
        self.n_words = vocab_len

        self.variable_data = []

    def sentence2variable(self, sentence):
        print(sentence)
        result = Variable(torch.LongTensor(sentence)).view(1,-1)
        return result


    def pad_and_variable(self, max_length):

        X_max_len = max([len(sentence) for sentence in self.data])
        print(X_max_len)
        self.data = pad_sequences(self.data, maxlen=X_max_len, dtype='int32')
        variables = []
        for sentence in self.data:
            variables.append(self.sentence2variable(sentence))

        self.variable_data = variables


def combine_langs(lang1,lang2):
    return zip(lang1.variable_data, lang2.variable_data)


def prepare_data(en_en, sv_en, MAX_LENGTH, VOCAB_SIZE):
    X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word = load_data(en_en, sv_en, MAX_LENGTH, VOCAB_SIZE)

    input_lang = Lang(en_en, X,X_word_to_ix, X_ix_to_word, X_vocab_len)
    output_lang = Lang(sv_en, y, y_word_to_ix, y_ix_to_word, y_vocab_len)

    input_lang.pad_and_variable(MAX_LENGTH)
    output_lang.pad_and_variable(MAX_LENGTH)

    data = combine_langs(input_lang,output_lang)

    #Split data
    indices_length = len(data)
    indices = np.arange(indices_length)
    np.random.shuffle(indices)
    train_p = 0.7
    val_p = 0.2
    test_p = 0.1

    train_index = indices[0:train_p*indices]
    val_index = indices[train_p*indices + 1: (train_p + val_p)*indices]
    test_index = indices[(train_p + val_p)*indices + 1:]

    train = data[train_index]
    val = data[val_index]
    test = data[test_index]

    return train,val,test,input_lang,output_lang






def load_test_data(source, X_word_to_ix, max_len):
    f = open(source, 'r')
    X_data = f.read()
    f.close()

    X = [text_to_word_sequence(x)[::-1] for x in X_data.split('\n') if len(x) > 0 and len(x) <= max_len]
    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):
            if word in X_word_to_ix:
                X[i][j] = X_word_to_ix[word]
            else:
                X[i][j] = X_word_to_ix['UNK']
    return X

def create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, hidden_size, num_layers):
    model = Sequential()

    # Creating encoder network
    model.add(Embedding(X_vocab_len, 1000, input_length=X_max_len, mask_zero=True))
    model.add(LSTM(hidden_size))
    model.add(RepeatVector(y_max_len))

    # Creating decoder network
    for _ in range(num_layers):
        model.add(LSTM(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(y_vocab_len)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
    return model

def process_data(word_sentences, max_len, word_to_ix):
    # Vectorizing each element in each sequence
    sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
    for i, sentence in enumerate(word_sentences):
        for j, word in enumerate(sentence):
            sequences[i, j, word] = 1.
    return sequences

def find_checkpoint_file(folder):
    checkpoint_file = [f for f in os.listdir(folder) if 'checkpoint' in f]
    if len(checkpoint_file) == 0:
        return []
    modified_time = [os.path.getmtime(f) for f in checkpoint_file]
    return checkpoint_file[np.argmax(modified_time)]