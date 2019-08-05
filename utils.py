import pandas as pd
from sklearn.model_selection import train_test_split
import re
from nltk import sent_tokenize
import os
import numpy as np

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical


MAX_WORDS_PER_REVIEW = 20000
MAX_NB_SENTS = 18
MAX_WORDS_PER_SENT = 150

class DataUtil:
	def __init__(self, file_path):
		self.file_path = file_path
		assert os.path.splitext(file_path)[1] == '.tsv'		# 文件后缀名为tsv
	
	def __clean_text(self, text):
		text = re.sub(r'[^@#$%^&*<>{}\\0-9]', '', text)
		text = text.strip().lower()
		return text

	# 将文档分割为句子-文档的层级结构，返回的数据类型是三维的
	def get_hie_3d_data(self):
		raw_data = pd.read_csv(self.file_path, sep='\t')
		
		reviews = []
		reviews_with_sents = []
		labels = []
		for i in range(raw_data.review.shape[0]):
			review = raw_data.review[i]
			review = self.__clean_text(review)
			reviews.append(review)
			sents = sent_tokenize(review)
			reviews_with_sents.append(sents)
			labels.append(int(raw_data.sentiment[i]))

		tokenizer = Tokenizer(num_words=MAX_WORDS_PER_REVIEW)
		tokenizer.fit_on_texts(reviews)
		word_idx = tokenizer.word_index
		data = np.zeros((len(reviews), MAX_NB_SENTS, MAX_WORDS_PER_SENT), dtype='int16')
		for i, sents in enumerate(reviews_with_sents):
			for j, sent in enumerate(sents):
				if j < MAX_NB_SENTS:
					word_tokens = text_to_word_sequence(sent)
					k = 0
					for _, word in enumerate(word_tokens):
						if k < MAX_WORDS_PER_SENT and word_idx[word] < MAX_WORDS_PER_REVIEW:
							data[i, j, k] = word_idx[word]
							k += 1
		labels = to_categorical(np.asarray(labels))
		return data, labels, word_idx

		
	# 将文档当成一个长序列处理，返回的数据类型是二维的
	def get_doc_2d_data(self):
		raw_data = pd.read_csv(self.file_path, sep='\t')
		reviews = []
		labels = []
		for i in range(raw_data.review.shape[0]):
			review = raw_data.review[i]
			review = self.__clean_text(review)
			reviews.append(review)
			labels.append(int(raw_data.sentiment[i]))
		
		tokenizer = Tokenizer(num_words=MAX_WORDS_PER_REVIEW)
		tokenizer.fit_on_texts(reviews)
		word_idx = tokenizer.word_index
		data = np.zeros((len(reviews), MAX_WORDS_PER_REVIEW), dtype='int16')
		for i, review in enumerate(reviews):
			word_tokens = text_to_word_sequence(review)
			for j, word in enumerate(word_tokens):
				if j < MAX_WORDS_PER_REVIEW:
					data[i, j] = word_idx[word]
		labels = to_categorical(np.asarray(labels))
		return data, labels, word_idx

	
	# 划分数据集，如果with_val为true，则将数据集划分为train/test/val
	# 否则，划分为train/test
	def split_data(self, data, labels, with_val, split=0.1):
		assert split < 0.3	# 数据分割比split必须小于0.3，否则test数据集过大
		x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=split, random_state=1)
		if with_val:
			x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=split, random_state=1)
			return x_train, x_test, x_val, y_train, y_test, y_val
		else:
			return x_train, x_test, y_train, y_test


if __name__ == '__main__':
	dd = DataUtil("F:/1Study information/Scut/Papers/experiments/datasets/imdb_2class_train_data.tsv")
	hie_data, labels, word_index = dd.get_hie_3d_data()
	doc_data, labels, word_index = dd.get_doc_2d_data()
