import numpy as np
import pandas as pd
import re

from bs4 import BeautifulSoup

import sys
import os

from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.engine.topology import Layer, InputSpec
from keras import backend as K
os.environ['KERAS_BACKEND']='tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

print ('os imformation:',os.environ.keys())
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)

from keras.layers import GlobalAveragePooling2D, multiply, Permute, Concatenate, Add, Activation, Lambda
from keras.utils import multi_gpu_model

from keras.layers import Masking
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, Conv2D, GlobalMaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional,TimeDistributed,Flatten ,concatenate , GlobalMaxPooling2D
from keras.layers.pooling import MaxPooling1D ,MaxPool2D,AveragePooling2D
from keras.models import Sequential, Model
from keras.layers import ZeroPadding1D
from keras.layers.core import Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization

from keras import initializers, regularizers, constraints, optimizers
from nltk import tokenize, word_tokenize
from keras.regularizers import l2

from keras.models import model_from_yaml
import yaml

from keras.models import load_model

MAX_SENT_LENGTH = 150
MAX_SENTS = 18
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300


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
# data_path = '/home/psycho/LSTM/yelp_14/yelp-2014-all.tsv'
data_path = "F:/1Study information/Scut/Papers/experiments/datasets/imdb_2class_train_data.tsv"
#
earlyStopping = EarlyStopping(monitor='val_loss', patience=4, verbose=0)
saveBestModel = ModelCheckpoint(filepath="checkpoint-{epoch:02d}e-val_acc_{val_acc:.2f}.hdf5", monitor = 'val_loss', verbose=0, save_best_only = False, save_weights_only = True)


def PreProcess(path):
    data_train=pd.read_csv(path, sep='\t')
    print (data_train.shape)
    reviews = []
    labels = []
    texts = []
    # for idx in range(data_train.review.shape[0]):
    for idx in range(100):
        text = BeautifulSoup(data_train.review[idx])
        text = clean_str(text.get_text())
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)
        labels.append(int(data_train.sentiment[idx])-1)

    #Input shape would be [of reviews ench batch,of sentences , of words in each sentences]
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    print('The len of texts:',len(texts))
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
    print('Total %s unique tokens.' % len(word_index))
    labels = to_categorical(np.asarray(labels))
    #labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    return data,labels,word_index
"""
Train and validation data separation
"""
data,labels,word_index=PreProcess(data_path)

x_train = data[:900364]
y_train = labels[:900364]
indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

x_val = data[900364:1012909]
y_val = labels[900364:1012909]
indices = np.arange(x_val.shape[0])
np.random.shuffle(indices)
x_val = x_val[indices]
y_val = y_val[indices]


x_test = data[1012909:1125458]
y_test = labels[1012909:1125458]
indices = np.arange(x_test.shape[0])
np.random.shuffle(indices)
x_test = x_test[indices]
y_test = y_test[indices]

print('x_train.shape: ', x_train.shape) # should round to 900364
print('x_val.shape: ', x_val.shape)
print('x_test.shape: ', x_test)

print('Rating distribution:')
print(labels.sum(axis=0))
print('Number of positive and negative reviews in traing,validation and test set')
print (y_train.sum(axis=0))
print (y_val.sum(axis=0))
print (y_test.sum(axis=0))

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
#注意力机制
class Attention_layer(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(Attention_layer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

    def compute_output_shape(self, input_shape):
        """Shape transformation logic so Keras can infer output shape
        """
        return (input_shape[0], input_shape[-1])



class aMatrix(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(aMatrix, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(aMatrix, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)
        print('x.shape:',x.shape)
        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
        print('ait.shape:',ait.shape)
        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return weighted_input 
        #return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[-1]

    def compute_output_shape(self, input_shape):
        """Shape transformation logic so Keras can infer output shape
        """
        return (input_shape[0], input_shape[1], input_shape[-1])

"""
CNN attention,Using SE
"""
def se_block(input_feature, ratio=6):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert se_feature._keras_shape[1:] == (1,1,channel)
    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1,1,channel//ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1,1,channel)
    if K.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2))(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature

"""
CNN attention,Using CBAM 
"""
def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attention(input_feature, ratio=8):
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]
    
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    
    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7
    
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature
    
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat) 
    assert cbam_feature._keras_shape[-1] == 1
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
    return multiply([input_feature, cbam_feature])

# GLOVE_DIR = "/home/psycho/LSTM"
GLOVE_DIR = "F:/1Study information/Scut/Papers/experiments/datasets"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Total %s word vectors.' % len(embeddings_index))

counter=0
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else :
        counter+=1
print ('Length of embedding_matrix:', embedding_matrix.shape[0])
print('shape of embedding matrix:',embedding_matrix.shape)
print('无效单词嵌入:',counter)
"""
Embedding_layer
"""
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            mask_zero=False,
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)
"""
Model
"""
drop = 0.55
batch_size=100
W_reg = 0.01
#LSTM phase with Hierachical Attention 
sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='float32')
embedded_sequences = embedding_layer(sentence_input)
embedded_sequences = Dropout(0.3)(embedded_sequences)
l_lstm = Bidirectional(GRU(250,return_sequences=True,recurrent_dropout=drop,kernel_regularizer=l2(W_reg)))(embedded_sequences)
#l_lstm = Dropout(drop)(l_lstm) 

l_dense = TimeDistributed(Dense(120))(l_lstm)
l_att = Attention_layer()(l_dense)
dense_1 = Dense(120,activation='tanh')(l_att)
dense_1 = Dropout(0.3)(dense_1)
sentEncoder = Model(sentence_input, dense_1)
print('shape of dense_1:',dense_1.shape)

review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='float32')
print('shape of review_input:',review_input.shape)
review_encoder = TimeDistributed(sentEncoder)(review_input)
print('shape of review_encoder:',review_encoder.shape)
l_lstm_sent = Bidirectional(GRU(250,return_sequences=True,recurrent_dropout=drop,kernel_regularizer=l2(W_reg)))(review_encoder)
#l_lstm_sent = Dropout(0.3)(l_lstm_sent)
l_dense_2 = TimeDistributed(Dense(150))(l_lstm_sent)
sent_att = aMatrix()(l_dense_2)

#CNN phase with CBAM 
filter_sizes = [3,4,5]
num_filters = 32

reshape=Reshape((MAX_SENTS,MAX_SENT_LENGTH,1))(sent_att)

conv_0 = Conv2D(180, kernel_size=(filter_sizes[0], filter_sizes[0]), padding='valid', kernel_initializer='normal', activation='relu',W_regularizer=l2(W_reg))(reshape)
conv_0 = cbam_block(conv_0,8)
conv_1 = Conv2D(180, kernel_size=(filter_sizes[1], filter_sizes[1]), padding='valid', kernel_initializer='normal', activation='relu',W_regularizer=l2(W_reg))(reshape)
conv_1 = cbam_block(conv_1,8)
conv_2 = Conv2D(180, kernel_size=(filter_sizes[2], filter_sizes[2]), padding='valid', kernel_initializer='normal', activation='relu',W_regularizer=l2(W_reg))(reshape)
conv_2 = cbam_block(conv_2,8)


#conv_0 = Dropout(0.5)(conv_0)
#conv_1 = Dropout(0.5)(conv_1)
#conv_2 = Dropout(0.5)(conv_2)

maxpool_0 = MaxPool2D(pool_size=(4,4), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(3,3), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(2,2), strides=(1,1), padding='valid')(conv_2)


averagepool_0 = AveragePooling2D(pool_size=(4,4), strides=(1,1), padding='valid')(conv_0)
averagepool_1 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='valid')(conv_1)
averagepool_2 = AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid')(conv_2)

maxpool_0 = Dropout(0.2)(maxpool_0)
maxpool_1 = Dropout(0.2)(maxpool_1)
maxpool_2 = Dropout(0.2)(maxpool_2)
concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])

concatenated_tensor_1 = Concatenate(axis=1)([averagepool_0, averagepool_1, averagepool_2])
#features=cbam_block(concatenated_tensor,8)
dropout_0=Dropout(drop)(concatenated_tensor)
#print("shape of conv_0:",conv_0.shape)

flatten = Flatten()(dropout_0)

flatten1 = Flatten()(concatenated_tensor_1)

flatten = Concatenate(axis=0)([flatten, flatten1])
print('flatten.shape',flatten.shape)
output = Dense(units=5, activation='softmax')(flatten)
print('output.shape: ', output.shape)
model = Model(review_input, output)
sgd = optimizers.SGD(lr=0.075, decay=1e-5, momentum=0.9, nesterov=True)
#ada = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
model =  multi_gpu_model(model, 2)
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['acc'])

model.summary()
#model.load_weights('6780.hdf5')

model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=batch_size,
          callbacks=[saveBestModel,earlyStopping])


"""
Evaluating Phase
"""

print("Evaluating...")
score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)

print("Test score:",score)
print('batch_size:',batch_size)

