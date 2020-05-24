#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import SimpleRNN
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

#Loading training data
f = open("training_label.txt",'r',encoding = 'utf8')
lines=f.readlines()

x_train = []
y_train = []

for line in lines :
    y_train.append(int(line.split('+++$+++')[0]))
    x_train.append(line.split('+++$+++')[1].strip())
    
y_train = y_train[:10000]
x_train = x_train[:10000]

#Remove stop words
from nltk.corpus import stopwords

def filter_stop_words(train_sentences, stop_words):
    for i, sentence in enumerate(train_sentences):
        new_sent = [word for word in sentence.split() if word not in stop_words]
        train_sentences[i] = ' '.join(new_sent)
    return train_sentences

stop_words = set(stopwords.words("english"))
x_train = filter_stop_words(x_train, stop_words)

#Loading testing data
f = open("testing_label.txt",'r',encoding = 'utf8')
lines=f.readlines()

x_test = []
y_test = []

for line in lines :
    if len(line) == 1:
        continue
    y_test.append(int(line.split('#####')[0]))
    x_test.append(line.split('#####')[1].strip())	


token = Tokenizer(num_words = 5000) 
#使用Tokenizer模組建立token，建立一個4000字的字典
token.fit_on_texts(x_train)  
#讀取所有訓練資料影評，依照每個英文字在訓練資料出現的次數進行排序，
#前4000名的英文單字會加進字典中
token.word_index
#可以看到它將英文字轉為數字的結果，例如:the轉換成1

x_train_seq = token.texts_to_sequences(x_train)
x_test_seq = token.texts_to_sequences(x_test)
#透過texts_to_sequences可以將訓練和測試集資料中的影評文字轉換為數字list

#截長補短
x_train = sequence.pad_sequences(x_train_seq, maxlen = 500)
x_test = sequence.pad_sequences(x_test_seq, maxlen = 500)

#RNN
modelRNN = Sequential()
modelRNN.add(Embedding(output_dim = 32,   #輸出的維度是32，希望將數字list轉換為32維度的向量
     input_dim=5000,  #輸入的維度是5000，也就是我們之前建立的字典是5000字
     input_length=500)) #數字list截長補短後都是500個數字

#隨機捨棄20%的神經元，避免overfitting
#modelRNN.add(Dropout(0.7))

modelRNN.add(SimpleRNN(units=16))
#建立16個神經元的RNN層

modelRNN.add(Dense(units=256,activation='relu')) 
#建立256個神經元的隱藏層
#ReLU激活函數
#modelRNN.add(Dropout(0.35))

modelRNN.add(Dense(units=1,activation='sigmoid'))
#建立一個神經元的輸出層
#Sigmoid激活函數

modelRNN.summary()
modelRNN.compile(loss='binary_crossentropy',
     optimizer='adam',
     metrics=['accuracy']) 

train_history = modelRNN.fit(x_train,y_train, 
         epochs=10, 
         batch_size=100,
         verbose=2,
         validation_split=0.2)

#Plot accuracy
show_train_history(train_history,'accuracy','val_accuracy')
#Plot loss
show_train_history(train_history,'loss','val_loss')

scores = modelRNN.evaluate(x_test, y_test,verbose=1)
scores[1]

#LSTM
modelLSTM = Sequential() #建立模型
modelLSTM.add(Embedding(output_dim=32, input_dim = 5000, input_length = 500))

modelLSTM.add(LSTM(32)) 
#建立32個神經元的LSTM層

modelLSTM.add(Dense(units=256,activation='relu')) 
#建立256個神經元的隱藏層
#modelLSTM.add(Dropout(0.2))

modelLSTM.add(Dense(units=1,activation='sigmoid'))
#建立一個神經元的輸出層
    
STM .summary()
#查看模型摘要

modelLSTM.compile(loss='binary_crossentropy',
     optimizer='adam',
     metrics=['accuracy']) 

train_history = modelLSTM.fit(x_train,y_train, 
         epochs=10, 
         batch_size=100,
         verbose=2,
         validation_split=0.2)

#Plot accuracy
show_train_history(train_history,'accuracy','val_accuracy')
#Plot loss
show_train_history(train_history,'loss','val_loss')

scores = modelLSTM.evaluate(x_test, y_test, verbose=1)
scores[1]
#評估模型準確率
