#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import csv
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import string
import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Activation
import keras.utils as kutils
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import string



# In[7]:


df1 = pd.read_csv('SW_EpisodeIV.txt', sep =' ', header=0, escapechar='\\')
df2 = pd.read_csv('SW_EpisodeV.txt', sep =' ', header=0, escapechar='\\')
df3 = pd.read_csv('SW_EpisodeVI.txt', sep =' ', header=0, escapechar='\\')


# In[13]:


all_dialogues=list(pd.concat([df1,df2,df3]).dialogue.values)
print('Tamanho: ', len(all_dialogues))
print(all_dialogues[:10])


# In[21]:


from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

import string


# In[23]:


all_sents=[[w.lower() for w in word_tokenize(sen) if not w in string.punctuation] for sen in all_dialogues]

x=[]
y=[]
for sen in all_sents:
    for i in range(1,len(sen)):
        x.append(sen[:i])
        y.append(sen[i])


# In[24]:


from sklearn.model_selection import train_test_split
import numpy as np


# In[25]:


all_text=[c for sen in x for c in sen]
all_text+=[c for c in y]

all_text.append('UNK')


# In[26]:


words=list(set(all_text))


# In[27]:


word_indexes={word: index for index, word in enumerate(words)}

max_features=len(word_indexes)


# In[28]:


x=[[word_indexes[c] for c in sen] for sen in x]
y=[word_indexes[c] for c in y]


# In[29]:


y=kutils.to_categorical(y, num_classes=max_features)


# In[31]:


maxlen=max([len(sen) for sen in x])
x=pad_sequences(x, maxlen=maxlen)


# In[33]:


embedding_size=10
model=Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(LSTM((100)))
model.add(Dropout(0.1))
model.add(Dense(max_features, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[34]:


model.summary()


# In[35]:


model.fit(x,y,epochs=10,verbose=5)


# In[36]:


import pickle
print('saving model')
model.save('star_wars_bot.h5')

with open('star_wars_bot-dict.pkl','wb') as handle:
    pickle.dump(word_indexes, handle)
with open('star_wars_bot-maxlen.pkl','wb') as handle:
    pickle.dump(maxlen,handle)
    


# In[ ]:


nltk.download()


# In[ ]:




