
# coding: utf-8

# In[1]:


from data import *
x,y = get_raw_data()
x = lower_case(x)
x = chunk(x,150)
x = preproces(x)
x = steamit(x)
x = stopit_dude(x)


# In[2]:


df = labels_to_dataframe(x,y)
ration_data, freq_list = keep_labels(df, keep_ratio = 0.25, freq_Value = False)


# In[15]:


ration_data_text = list(ration_data['text'])
labels           = np.array(ration_data.drop('text', 1))


# In[30]:


import re
import xml.sax.saxutils as saxutils
from keras import backend as K


from bs4 import BeautifulSoup

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from pandas import DataFrame

from random import random
import numpy as np


# In[31]:


max_vocab_size = 30000
input_tokenizer = Tokenizer(max_vocab_size)
input_tokenizer.fit_on_texts(ration_data_text)
input_vocab_size = len(input_tokenizer.word_index) + 1
print("input_vocab_size:",input_vocab_size)
totalX = np.array(pad_sequences(input_tokenizer.texts_to_sequences(ration_data_text), maxlen=150))


# In[32]:


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[33]:


embedding_dim = 256
model = Sequential()
model.add(Embedding(input_vocab_size, embedding_dim,input_length = 150))
model.add(GRU(256, dropout=0.9, return_sequences=True))
model.add(GRU(256, dropout=0.9))
model.add(Dense(22, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1_m,precision_m, recall_m])


# In[35]:


history = model.fit(totalX, labels, validation_split=0.2, batch_size=128, epochs=100)

with open('result.txt','w') as f:
    f.write(str(history.history))

