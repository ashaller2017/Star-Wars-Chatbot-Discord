import keras
import discord
import os
import json
import requests
import random
import giphy_client
from discord.ext.commands import Bot
from giphy_client.rest import ApiException
import pickle
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import tensorflow as tf
discord_token = 'ODIwNDE3MzE1NDcxNDkxMTQz.YE03Kg._DBfRiHiRWxclILNqDGbFV9n7Ds'
giphy_token = 'CW3xcemJ0ama1GdQRL4hSu89EzyoEp40'
df1 = pd.read_csv('SW_EpisodeIV.txt', sep =' ', header=0, escapechar='\\')
df2 = pd.read_csv('SW_EpisodeV.txt', sep =' ', header=0, escapechar='\\')
df3 = pd.read_csv('SW_EpisodeVI.txt', sep =' ', header=0, escapechar='\\')
all_dialogues=list(pd.concat([df1,df2,df3]).dialogue.values)
model=keras.models.load_model('star_wars_bot.h5')
maxlen=pickle.load(open('star_wars_bot-maxlen.pkl','rb'))
word_indexes=pickle.load(open('star_wars_bot-dict.pkl','rb'))
stopwords_list=stopwords.words('english')
lemmatizer=WordNetLemmatizer()

all_sents = [[w.lower() for w in word_tokenize(sen) if not w in string.punctuation]\
            for sen in all_dialogues]

x = []
y = []
for sen in all_sents:
    for i in range(1, len(sen)):
        x.append(sen[:i])
        y.append(sen[i])
all_text = [c for sen in x for c in sen]
all_text += [c for c in y]

all_text.append('UNK')
words = list(set(all_text))
word_indexes = {word: index for index, word in enumerate(words)}

max_features = len(word_indexes)



def get_word_by_index(index, word_indexes):
    for w, i in word_indexes.items():
        if index == i:
            return w
        
    return None


def my_tokenizer(doc):
    words=word_tokenize(doc)
    pos_tags=pos_tag(words)
    non_stopwords=[w for w in pos_tags if not w[0].lower() in stopwords_list]
    non_punctuation=[w for w in non_stopwords if not w[0] in string.punctuation]
    lemmas=[]
    for w in non_punctuation:
        if w[1].startswith('J'):
            pos=wordnet.ADJ
        elif w[1].startswith('V'):
            pos=wordnet.VERB
        elif w[1].startswith('N'):
            pos=wordnet.NOUN
        elif w[1].startswith('R'):
            pos=wordnet.ADV
        else:
            pos=wordnet.NOUN
        lemmas.append(lemmatizer.lemmatize(w[0],pos))
    return lemmas

tfidf_vectorizer=TfidfVectorizer(tokenizer=my_tokenizer)
tfidf_matrix=tfidf_vectorizer.fit_transform(tuple(all_dialogues))

def find_closest_reponse(question):
    query_vect=tfidf_vectorizer.transform([question])
    similarity=cosine_similarity(query_vect,tfidf_matrix)
    max_similarity=np.argmax(similarity, axis=None)
    return ' '.join(all_dialogues[max_similarity+1].split(' ')[:20])

def get_response(input):
    sample_seed = find_closest_reponse(input)
    sample_seed_vect = np.array([[word_indexes[c] if c in word_indexes.keys() else word_indexes['UNK']  for c in word_tokenize(sample_seed)]])

    test = tf.keras.preprocessing.sequence.pad_sequences(sample_seed_vect, \
                        maxlen=maxlen, \
                        padding='pre')
    
    predicted = []
    i = 0
    while i < 20:
        predicted = model.predict_classes(
                                tf.keras.preprocessing.sequence.pad_sequences(sample_seed_vect, \
                                               maxlen=maxlen, \
                                               padding='pre'),\
                                verbose=0)
        new_word = get_word_by_index(predicted[0], word_indexes)
        sample_seed += ' ' + new_word

        sample_seed_vect = np.array([[word_indexes[c] \
                                  if c in word_indexes.keys() else \
                                 word_indexes['UNK'] \
                                 for c in word_tokenize(sample_seed)]])
        i += 1
        
    gen_text = ''
    for index in sample_seed_vect[0][:20]:
        gen_text += get_word_by_index(index, word_indexes) + ' '
        
    output = ''
    for i, gen in enumerate(gen_text.split(' ')):
        if gen == 'UNK':
            output += sample_seed.split(' ')[i] + ' '
        else:
            output += gen + ' ' 
    return output

bot = Bot(command_prefix='$')
api_instance = giphy_client.DefaultApi()
client = discord.Client()

async def search_gifs(query):
    try:
        response = api_instance.gifs_search_get(giphy_token, 
            query, limit=3, rating='g')
        lst = list(response.data)
        gif = random.choices(lst)

        return gif[0].url

    except ApiException as e:
        return "Exception when calling DefaultApi->gifs_search_get: %s\n" % e

    


@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    elif message.content.startswith('$hello'):
        await message.channel.send('Hello!')
    elif message.content.startswith('$cat'):

        gif=await search_gifs('cats')
        await message.channel.send('Gif URL: '+gif)
    elif message.content.startswith('$dog'):

        gif=await search_gifs('dog')
        await message.channel.send('Gif URL: '+gif)

    elif message.content.startswith('$penis'):
        await message.channel.send('PENIS')
    else:
        message_response= await client.wait_for('message')
        input=message_response.content
        output=get_response(input)
        await message.channel.send(output)


client.run('your token here')
