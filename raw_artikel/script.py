# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:00:42 2021

@author: troja
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

##################### Functions #####################
# Sätze tokenisieren
def tokenizeSent(corpus):
    sentences = []
    for line in corpus: # iteriere durch DF
        sent_tok = sent_tokenize(line)
        sentences.append(sent_tok)
    return sentences

# Satzzeichen und Zahlen entfernen sowie alles kleinschreiben
def removePunct(corpus):
    sentences_clean = []
    for line in corpus: # iteriere durch DF      
        sentences_clean.append(re.sub(r'[^(a-zA-Z0-9öüäÖÜÄ)\s]', '', line).lower())
    return sentences_clean

# Wörter tokenisieren
def tokenizeWords(corpus):
    words = []
    for line in corpus: # iteriere durch DF
        words.append(word_tokenize(line))
    return words

# Stoppwörter entfernen
def removeStopwords(corpus):
    words_clean = []
    words_clean_joined = []
    for line in corpus: # iteriere durch DF
        words_tok = []
        for word in line: # iteriere durch jedes Wort pro Zeile und vergleiche, ob Stoppwort oder nicht
            if not word in stop_words_DE and not word in stop_words_IT and not word in stop_words_FR:
                words_tok.append(word)
        words_clean.append(words_tok)
        words_clean_joined.append(" ".join(words_tok))
    return words_clean, words_clean_joined


##################### Files laden #####################
# Stopwords laden
nltk.download('stopwords')
stop_words_DE = list(set(stopwords.words('german')))
stop_words_FR = list(set(stopwords.words('french')))
stop_words_IT = list(set(stopwords.words('italian')))

#new_words = ["(", ")", "us", "im"]
#stop_words = stop_words + new_words

# Data-File laden
raw_data = pd.read_csv('data/raw_data.csv', sep = ';', usecols=['PROD_EXTERNER_NAME_DE'], encoding='utf-8')
raw_data.rename(columns={"PROD_EXTERNER_NAME_DE": "PRODUKT"}, inplace=True)
# leere Zeilen dropen
raw_data.dropna(subset = ['PRODUKT'], inplace=True)
# nur Unique-Werte behalten
raw_data.drop_duplicates(inplace=True)

##################### Tokenisieren #####################

# Sätze tokenisieren
#raw_data_lst_tok = tokenizeSent(raw_data_lst)

# Satzzeichen entfernen
raw_data['PRODUKT_CLEAN'] = removePunct(raw_data['PRODUKT'])

# Wörter tokenisieren
raw_data['PRODUKT_WORDS'] = tokenizeWords(raw_data['PRODUKT_CLEAN'])

# Stoppwörter entfernen
raw_data['PRODUKT_WORDS_CLEAN'], raw_data['PRODUKT_WORDS_CLEAN_LIST'] = removeStopwords(raw_data['PRODUKT_WORDS'])

##################### Analytics #####################

# most common words
freq = pd.Series(' '.join(raw_data['PRODUKT_WORDS_CLEAN_LIST']).split()).value_counts()[:10]
freq

# least common words
freq = pd.Series(' '.join(raw_data['PRODUKT_WORDS_CLEAN_LIST']).split()).value_counts()[-10:]
freq

# Wordcloud
wordcloud = WordCloud(
        background_color = 'black',
        stopwords = stop_words_DE,
        max_words = 50,
        random_state = 42,
        relative_scaling = 0.5,
        collocations = False,
        normalize_plurals = True,
        max_font_size = 80
        ).generate_from_text(str(raw_data['PRODUKT_WORDS_CLEAN_LIST']))
        #generate_from_text(str(dat_comments.loc[dat_comments['createDate_month'] == m, 'words_clean_joined']))
        #generate(str(dat_comments['words_clean_joined']))

print(wordcloud)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud)
plt.axis('off')
#plt.savefig('wordcloud_'+str(m)+'.png', bbox_inches='tight')
plt.show()

#### Most frequently occuring Bi-grams ####
def get_top_n2_words(corpus, n=None):
    #vec1 = CountVectorizer(ngram_range=(2,2), max_features=2000).fit(corpus)
    vec1 = CountVectorizer(ngram_range=(2,2)).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# Convert most freq words to dataframe for plotting bar plot
#top2_words = get_top_n2_words(dat_comments['words_clean_joined'], n=10)
top2_words = get_top_n2_words(raw_data['PRODUKT_WORDS_CLEAN_LIST'], n=10)
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
print(top2_df)

# Barplot of most freq Bi-grams
fix, ax = plt.subplots(figsize=(13,6.5))
ax.bar(x = "Bi-gram", height = "Freq", data = top2_df)
ax.set(title = "Bi-Grame der Produkte",
       xlabel = "Bi-gram",
       ylabel = "Anzahl")
plt.setp(ax.get_xticklabels(), rotation = 45)
#plt.savefig('bigram_'+str(m)+'.png', bbox_inches='tight')
plt.show()


