from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np

def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1
    return results

#初始化数据
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)

#print(train_data[0],train_labels[0])

x_train=vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)

y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')

#构建网络
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(1,activation='sigmoid'))
#优化器、损失函数、指标
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#训练集和验证集
'''
x_val=x_train[:10000]
partial_x_train=x_train[10000:]
y_val=y_train[:10000]
partial_y_train=y_train[10000:]
'''
#训练模型
model.fit(x_train,y_train,epochs=4,batch_size=512)
#results=model.evaluate(x_test,y_test)

#测试模型
print(model.predict(x_test))
