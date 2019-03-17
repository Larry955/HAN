import json
import pickle
import logging
import random
import re
import argparse
from HAN import *

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

from nltk.tokenize import word_tokenize

from defaults import *

parser = argparse.ArgumentParser('sentiment_lstm')

parser.add_argument('--full_data_path', '-d', help='Full path of data', default=FULL_DATA_PATH)
parser.add_argument('--model-path', '-P', help='Full path of model', default=MODEL_PATH)
parser.add_argument('--processed_pickle_data_path', '-p', help='Full path of processed pickle data path',
                    default=PROCESSED_PICKLE_DATA_PATH)
parser.add_argument('--max_length', '-m', help='Max length of comment', type=int, default=COMMENT_MAX_LENGTH)
parser.add_argument('--batch_size', '-b', help='Batch size', type=int, default=BATCH_SIZE)
parser.add_argument('--seed', '-s', help='Random seed', type=int, default=SEED)
parser.add_argument('--epoch', '-e', help='Epochs', type=int, default=EPOCH)

parser.add_argument('--training_data_ready', '-t', help='Pass when training data is ready', action='store_true')
parser.add_argument('--data_model_ready', '-D', help='Pass when model is ready', action='store_true')
parser.add_argument('--verbosity', '-v', help='verbosity, stackable. 0: Error, 1: Warning, 2: Info, 3: Debug',
                    action='count')

parser.description = 'Trains a simple LSTM model on the Digikala product comment dataset for the sentiment classification task'
parser.epilog = "Larry King"

args = parser.parse_args()

from keras.layers import Dense, Embedding, LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential, load_model
from keras.preprocessing import sequence

full_data_path = args.full_data_path
batch_size = args.batch_size
random.seed(args.seed)
is_training_data_ready = args.training_data_ready
is_data_model_ready = args.data_model_ready

pickle_data_path = args.processed_pickle_data_path
model_path = args.model_path
epoch = args.epoch
max_length = args.max_length

verbosity = args.verbosity
if not verbosity:
    verbosity = 0

'''
Logging config
use one of the following print for data output:
logging.debug for code debug
logging.info for events occurring, like status monitors
logging.warn for avoidable warning
logging.warning for non-avoidable warning
logging.error, logging.exception, logging.critical for appropriate errors (there don't raise exception, you have to do that yourself)
'''
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=40 - verbosity * 10)

def create_model(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size + 1, 128))
    model.add(Bidirectional(LSTM(128, activation='tanh', dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model



comment = "this guy who i thought was really cool in the eighties " \
           "just to maybe make up my mind whether."

tokens = word_tokenize(comment)
processed_comment = dict()
processed_comment['tokens'] = tokens
processed_comment['pol'] = 1

def create_word_set(comments):
    word_set = set()
    for word in comments['tokens']:
        word_set.add(word)
    return word_set

def create_word_index(word_set):
    word_index = dict()
    i = 1
    for word in word_set:
        word_index[word] = i
        i += 1
    word_index['UNK'] = i
    return word_index

def prepare_training_data(processed_comments, word_idx):
    X = []
    y = []
    print(processed_comments)

    X.append([word_idx[word] for word in processed_comments['tokens']])
    y.append(processed_comments['pol'])
    print(X, y)
    return np.asarray(X), np.asarray(y)

w_set = create_word_set(processed_comment)
w_idx = create_word_index(w_set)
X, y = prepare_training_data(processed_comment, w_idx)
print(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('X_train: \n', X_train)
print('y_train: \n', y_train)

if is_training_data_ready:
    with open(pickle_data_path, 'rb') as f:  # default: processed_data.pickle
        X, y, word_idx = pickle.load(f)
else:
    print('Processing data...')
    all_comments = process_data(full_data_path)

    print('Create word set...')
    word_set = create_word_set(all_comments)

    print('Create word to index...')
    word_idx = create_word_index(word_set)

    print('Prepare training data...')
    X, y = prepare_training_data(all_comments, word_idx)

    with open(PROCESSED_PICKLE_DATA_PATH, 'wb') as f:
        pickle.dump((X, y, word_idx), f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


if is_data_model_ready:
    model = load_model(model_path)
else:
    model = create_model(len(word_idx))
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epoch,
              validation_data=(X_test, y_test))
    model.save(MODEL_PATH)

y_pred = model.predict(X_test, batch_size=batch_size)