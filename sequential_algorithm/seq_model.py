# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 08:11:12 2020

@author: vamshikrishna Banari
"""

from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding
from keras.layers.core import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras import metrics
from keras.preprocessing import sequence
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#Load cleaned dataset
data = pd.read_csv('Cleaned_JobDescs.csv', header = 0, names = ['Query', 'Description'])
#data = pd.read_csv('Cleaned_JobsNonIT.csv', header = 0, names = ['Query', 'Description'])
    
#Split the dataset to Training and Test subsets (90/10)
train, test = train_test_split(data, test_size = 0.1, random_state = 17) #random_state = None
    
train_descs = train['Description']
train_labels = train['Query']
     
test_descs = test['Description']
test_labels = test['Query']
    
# Model Parameters
vocab_size = 1000
    
sequences_length = 1200
    
embedding_dimensionality = 64 #possibly low??
max_features = 2000 #equal to vocab_size
    
num_labels = len(train_labels.unique())
batch_size = 32
nb_epoch = 2
    
nof_filters = 200
kernel_size = 16
    
hidden_dims = 512
    
# Convert Texts to Numeric Vectors for Input
tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(train_descs)
    
x_train = tokenizer.texts_to_sequences(train_descs)
x_test = tokenizer.texts_to_sequences(test_descs)
    
x_train = sequence.pad_sequences(x_train, maxlen = sequences_length, padding = 'post')
x_test = sequence.pad_sequences(x_test, maxlen = sequences_length, padding = 'post')
    
encoder = LabelBinarizer()
encoder.fit(train_labels)
y_train = encoder.transform(train_labels)
y_test = encoder.transform(test_labels)
    

def model_init():
    
    model = Sequential()
    model.add(Embedding(max_features, embedding_dimensionality, input_length = 1200))
    #model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    
    model.add(Conv1D(nof_filters, kernel_size, padding='valid', activation='relu', strides = 1))
    model.add(GlobalMaxPooling1D())
    
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', #'sgd', 'adam', 'RMSprop', 'Adagrad'
                       metrics = [metrics.categorical_accuracy])
    
    history = model.fit(x_train, y_train,
                        batch_size = batch_size,
                        epochs = nb_epoch,
                        verbose = True,
                        validation_split = 0.2)
    
    return model
    

def model_pred(model, job_des = x_test, job_label = y_train):
    job_des = tokenizer.texts_to_sequences(job_des)
    job_des = sequence.pad_sequences(job_des, maxlen = sequences_length, padding = 'post')
    #job_label = encoder.transform(job_label)
    
    ynew = model.predict_classes(job_des)
    ylabels = encoder.inverse_transform(job_label)
    
    return (ynew, ylabels)

    

