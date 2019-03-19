import numpy as np
import pandas as pd
import re
import argparse
import os
import pickle

from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

from han_config import *
from nltk import tokenize

parser = argparse.ArgumentParser('HAN')

parser.add_argument('--full_data_path', '-d', help='Full path of data', default=FULL_DATA_PATH)
parser.add_argument('--embedding_path', '-s', help='The pre-trained embedding vector', default=EMBEDDING_PATH)
# parser.add_argument('--processed_pickle_data_path', '-D', help='Full path of processed pickle data',
#                     default=PROCESSED_PICKLE_DATA_PATH)
# parser.add_argument('--model_path', '-m', help='Full path of model', default=MODEL_PATH)
parser.add_argument('--epoch', '-e', help='Epochs', type=int, default=EPOCH)
parser.add_argument('--batch_size', '-b', help='Batch size', type=int, default=BATCH)
parser.add_argument('--training_data_ready', '-t', help='Pass when training data is ready', action='store_true')
parser.add_argument('--model_ready', '-M', help='Pass when model is ready', action='store_true')
parser.add_argument('--verbosity', '-v', help='verbosity, stackable. 0: Error, 1: Warning, 2: Info, 3: Debug',
                    action='count')

parser.description = 'Implementation of HAN for Sentiment Classification task'
parser.epilog = "Larry King@https://github.com/Larry955/HAN"

args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epoch


data_path = args.full_data_path
data_file_name = data_path.split('/')[-1]

pickle_path = ""
model_path = ""
if data_file_name.find(".tsv") != -1:
    pickle_path = data_file_name.replace(".tsv", ".pickle")
    model_path = data_file_name.replace(".tsv", "_model.h5")
assert (pickle_path != "" and model_path != "")

is_training_data_ready = args.training_data_ready
is_model_ready = args.model_ready

emb_file_flag = ''
embedding_dim = 0

embedding_path = args.embedding_path
if embedding_path.find('glove') != -1:
    emb_file_flag = 'glove'     # pre-trained word vector is glove
    embedding_dim = int(((embedding_path.split('/')[-1]).split('.')[2])[:-1])
elif embedding_path.find('GoogleNews-vectors-negative300.bin') != -1:
    emb_file_flag = 'google'    # pre-trained word vector is GoogleNews
    embedding_dim = 300
# print('embedding_dim: ', embedding_dim)

class_num = 0

verbosity = args.verbosity
if not verbosity:
    verbosity = 0


os.environ['KERAS_BACKEND']='tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# Move tf and keras down to prevent print Using * backend message when using -h flag
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.utils.np_utils import to_categorical
from keras.engine.topology import Layer
from keras import initializers
from keras import backend as K

from keras.layers import Dense, Input
from keras.layers import Embedding, GRU, Bidirectional,TimeDistributed
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True    # Use memory of GPU dynamically
session = tf.Session(config=config)

# Stop training if val_loss keep decreasing for 4 epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=0)
# Save the best model
save_best_model = ModelCheckpoint(filepath="checkpoints/checkpoint-{epoch:02d}e-val_loss{val_loss:.2f}.hdf5",
                                  monitor ='val_loss', verbose=0, save_best_only = False, save_weights_only = True)

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    #string = re.sub(r"n\'t", " n\'t", string)
    #string = re.sub(r"\'re", " \'re", string)
    #string = re.sub(r"\'d", " \'d", string)
    #string = re.sub(r"\'ll", " \'ll", string)
    #string = re.sub(r",", " , ", string)
    #string = re.sub(r"!", " ! ", string)
    #string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


"""
Process data(tsv format)
Output:
data: 3-dims, [total_words, max_sentences, max_words_per_sentence]
labels: multi-classification
word_index: a dict maps word into index
"""
def process_data(path):
    data_train=pd.read_csv(path, sep='\t')
    print (data_train.shape)
    reviews = []
    labels = []
    texts = []
    for idx in range(data_train.review.shape[0]):
    # for idx in range(100):
        text = clean_str(data_train.review[idx])
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)
        labels.append(int(data_train.sentiment[idx]))

    #Input shape would be [of reviews each batch,of sentences , of words in each sentences]
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    print('The len of texts: ',len(texts))
    data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int16')

    for i, sentences in enumerate(reviews):
        for j, sent in enumerate(sentences):
            if j < MAX_SENTS:
               wordTokens = text_to_word_sequence(sent)
               k = 0
               for _, word in enumerate(wordTokens):
                    if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                        data[i, j, k] = tokenizer.word_index[word]
                        k = k + 1

    word_index = tokenizer.word_index
    labels = to_categorical(np.asarray(labels))
    return data,labels,word_index

"""
1. Read word vector from pre-trained file
2. Get word vector for words we will train
3. Create embedding matrix
"""
def create_emb_mat(emb_path, word_idx, emb_dim):
    embeddings_index = {}
    if emb_file_flag == 'glove':
        f = open(os.path.join(embedding_path), encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vec
        f.close()
    elif emb_file_flag == 'google':
        wv_from_bin = KeyedVectors.load_word2vec_format(emb_path, binary=True)
        for word, vector in zip(wv_from_bin.vocab, wv_from_bin.vectors):
            vec = np.asarray(vector, dtype='float32')
            embeddings_index[word] = vec

    counter=0
    emb_matrix = np.random.random((len(word_idx) + 1, emb_dim))
    for word, i in word_idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            emb_matrix[i] = embedding_vector
        else :
            counter += 1
    print('invalid word embedding: ',counter)
    return emb_matrix

"""
Implementation of Attention Layer
"""
class AttLayer(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = {
            'attention_dim': self.attention_dim
        }
        base_config = super(AttLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

"""
Model: Hierarchical Attention Neural Network
"""
def create_model(emb_matrix):
    # Embedding layer
    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[emb_matrix],
                                mask_zero=False,
                                input_length=MAX_SENT_LENGTH,
                                trainable=True)

    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
    l_att = AttLayer(100)(l_lstm)
    sent_encoder = Model(sentence_input, l_att)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sent_encoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
    l_att_sent = AttLayer(100)(l_lstm_sent)
    preds = Dense(5, activation='softmax')(l_att_sent)
    # print('pred.shape: ', preds.shape)

    model = Model(review_input, preds)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=epochs, batch_size=batch_size,
              callbacks=[save_best_model, early_stopping])
    return model

if __name__ == '__main__':
    if is_training_data_ready:
        with open(pickle_path, 'rb') as f:
            # print('data ready')
            data, labels, word_index = pickle.load(f)
        f.close()
    else:
        data, labels, word_index = process_data(data_path)
        with open(pickle_path, 'wb') as f:
            pickle.dump((data, labels, word_index), f, protocol=4)
        f.close()
    # Generate data for training, validation and test
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
    # print('x_train.head: ', x_train[:10][0][0])
    # print('y_train.head: ', y_train[:10])

    if is_model_ready:
        # print('model ready')
        model = load_model(model_path, custom_objects={'AttLayer': AttLayer})
    else:
        # Generate embedding matrix consists of embedding vector
        embedding_matrix = create_emb_mat(embedding_path, word_index, embedding_dim)

        # Create model for training
        model = create_model(embedding_matrix)
        model.save(model_path)

    print("Evaluating...")
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size)
    print("Test score: ", score)
