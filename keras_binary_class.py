from keras.models import Sequential
from keras.layers import Dense, Activation, Bidirectional, GRU,Reshape
from keras.utils.np_utils import to_categorical

import numpy as np

x_train = np.random.random((1000, 20))  # 生成1000个数据，每个数据含20个0-1之间的数
labels = np.random.randint(2, size=(1000, 1))   # 生成1000个数据，每个数据取值为0或1
labels = to_categorical(np.asarray(labels))
x_test = np.random.random((100, 20))
test_labels = np.random.randint(2, size=(100, 1))
test_labels = to_categorical(np.asarray(test_labels))

print('x_train.shape: ', x_train.shape) # should round to 900364
print('x_test.shape: ', x_test.shape)

print('Rating distribution:')
print(labels.sum(axis=0))

model = Sequential()
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='sigmoid'))       # 最后一层网络，输出的是一维（数据集的维度）的就为1，否则会报错，二分类问题的输出维度是1
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
model.fit(x_train, labels, epochs=10, batch_size=32)
model.summary()

score = model.evaluate(x_test,test_labels, batch_size=32)
print('score: ', score)
