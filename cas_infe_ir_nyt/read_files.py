# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:48:46 2019

@author: u224208
"""
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

import re
import nltk
import copy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from datetime import datetime

from wordcloud import WordCloud #, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

##################### Files laden #####################
# Stopwords laden
nltk.download('stopwords')
stop_words = list(set(stopwords.words('english')))
stop_words_clean =  []
for line in stop_words:
    stop_words_clean.append(re.sub("'", '', line))
stop_words = copy.deepcopy(stop_words_clean)

new_words = ["(", ")", "us", "im"]
stop_words = stop_words + new_words

# Articles-File laden
#filename = r'C:\Users\u224208\OneDrive - SBB\02_CAS\Spyder\ir_nyt\nyt-comments\ArticlesApril2018.csv'
#dat_articles = pd.read_csv(filename)

# Comments-File laden
filename = [r'C:\Users\u224208\OneDrive - SBB\02_CAS\Spyder\ir_nyt\nyt-comments\CommentsJan2018.csv', r'C:\Users\u224208\OneDrive - SBB\02_CAS\Spyder\ir_nyt\nyt-comments\CommentsFeb2018.csv', r'C:\Users\u224208\OneDrive - SBB\02_CAS\Spyder\ir_nyt\nyt-comments\CommentsMarch2018.csv', r'C:\Users\u224208\OneDrive - SBB\02_CAS\Spyder\ir_nyt\nyt-comments\CommentsApril2018.csv']
dat_comments = pd.DataFrame()
for f in filename:
    print(f)
    dat_comments_tmp = pd.read_csv(f)
    frames = [dat_comments, dat_comments_tmp]
    dat_comments = pd.concat(frames)

dat_comments = dat_comments[['userID', 'createDate', 'commentBody', 'articleID']]



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
        sent_cleaned = []
        for sentence in line: # iteriere durch jeden Satz pro Zeile und entferne Satzzeichen und Zahlen
            sent_cleaned.append(re.sub(r'[^(a-zA-Z)\s]', '', sentence).lower())
        sentences_clean.append(sent_cleaned)
    return sentences_clean

# Wörter tokenisieren
def tokenizeWords(corpus):
    words = []
    for line in corpus: # iteriere durch DF
        word_tok = []
        for sentence in line: # iteriere durch jeden Satz pro Zeile und tokenisiere Wörter
            word_tok.append(word_tokenize(sentence))
        words.append(word_tok)
    return words

# Wortliste entschachteln
def orderSentList(corpus):
    words_list = []
    for line in corpus: # iteriere durch DF
        words_sent = []
        for sentence in line: # iteriere durch jeden Satz
            for word in sentence: # iteriere durch jedes Wort und bild eine List pro Zeile
                words_sent.append(word)
        words_list.append(words_sent)
    return words_list

# Stoppwörter entfernen
def removeStopwords(corpus):
    words_clean = []
    words_clean_joined = []
    for line in corpus: # iteriere durch DF
        words_tok = []
        for word in line: # iteriere durch jedes Wort pro Zeile und vergleiche, ob Stoppwort oder nicht
            if not word in stop_words:
                words_tok.append(word)
        words_clean.append(words_tok)
        words_clean_joined.append(" ".join(words_tok))
    return words_clean, words_clean_joined

# print top10 Werte von 2 Spalten
def printDf(df, columns):
    for i in range(0,10):
        print('############ Spalte 1 ############')
        print(df[columns[0]][i])
        print('------------ Spalte 2 ------------')
        print(df[columns[1]][i])

# Wörter zählen
def printCountWords(df, columns):
    y = 0
    z = 0
    for line in df[columns[0]]:
        for word in line:
            y = y + 1
    for line in df[columns[1]]:
        for word in line:
            z = z + 1
    print('Anzahl Wörter mit Stoppwörter:', y)
    print('Anzahl Wörter ohne Stoppwörter:', z)
    print('Wörter entfernt:', y - z)

# Stemming
#def porterStemming(corpus):
#    ps = PorterStemmer()
#    words_stemmed = []
#    for line in corpus: # iteriere durch DF
#        words_stem = []
#        for word in line: # iteriere durch jedes Wort pro Zeile und vergleiche, ob Stoppwort oder nicht
#            words_stem.append(ps.stem(word))
#        words_stemmed.append(words_stem)
#    return words_stemmed

# Anz. Wörter pro Eintrag zählen
def countWordsPerLine(corpus):
    word_count = []
    for line in corpus:
        z = 0
        for words in line:
            z = z + 1
        word_count.append(z)
    return word_count

# Datum formatieren
def formatCreateDate(date_column):
    createDate_date = []
    createDate_month = []
    for line in date_column:
        ts = int(line)
        createDate_date.append(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d'))
        createDate_month.append(datetime.utcfromtimestamp(ts).strftime('%b-%Y'))
    return createDate_date, createDate_month

##################### Tokenisieren #####################
# Sätze tokenisieren
dat_comments['sentences'] = tokenizeSent(dat_comments['commentBody'])
#printDf(dat_comments, ['commentBody', 'sentences'])

# Satzzeichen entfernen
dat_comments['sentences_clean'] = removePunct(dat_comments['sentences'])
#printDf(dat_comments, ['sentences', 'sentences_clean'])

# Wörter tokenisieren
dat_comments['words_lists'] = tokenizeWords(dat_comments['sentences_clean'])
#printDf(dat_comments, ['sentences_clean', 'words_lists'])

# Wortliste entschachteln
dat_comments['words'] = orderSentList(dat_comments['words_lists'])
#printDf(dat_comments, ['words_lists', 'words'])

# Stoppwörter entfernen
dat_comments['words_clean'], dat_comments['words_clean_joined'] = removeStopwords(dat_comments['words'])
#printDf(dat_comments, ['words_clean', 'words_clean_joined'])

# Wörter stemmen
#dat_comments['words_stemm'] = porterStemming(dat_comments['words_clean'])
#printDf(dat_comments, ['words_clean', 'words_stemm'])

##################### generiere weitere Spalten #####################
# Anz. Wörter pro Kommentar
dat_comments['word_count'] = countWordsPerLine(dat_comments['words_clean'])
dat_comments.sort_values(by=['word_count'], ascending = False)['word_count'][:10]
dat_comments.word_count.describe()

# Datum formatieren
dat_comments['createDate_date'], dat_comments['createDate_month'] = formatCreateDate(dat_comments['createDate'])

months = ['Jan-2018', 'Feb-2018', 'Mar-2018', 'Apr-2018', 'May-2018']

#lösche überflüssige Spalten
dat_comments = dat_comments.drop(columns=['sentences', 'sentences_clean', 'words_lists'])

dat_comments_jan = dat_comments[dat_comments['createDate_month'] == 'Jan-2018']
dat_comments_feb = dat_comments[dat_comments['createDate_month'] == 'Feb-2018']
dat_comments_mar = dat_comments[dat_comments['createDate_month'] == 'Mar-2018']
dat_comments_apr = dat_comments[dat_comments['createDate_month'] == 'Apr-2018']

##################### Analytics #####################
# Wörter zählen
printCountWords(dat_comments, ['words', 'words_clean'])

# Most common words Top100
for m in months:
    print(m)
    print("###############")
    freq = pd.Series(' '.join(dat_comments.loc[dat_comments['createDate_month'] == m, 'words_clean_joined']).split()).value_counts()[:10]
    print(freq)

# least common words
freq = pd.Series(' '.join(dat_comments['words_clean_joined']).split()).value_counts()[-10:]
freq

# Wordcloud
for m in months:
    wordcloud = WordCloud(
            background_color = 'black',
            stopwords = stop_words,
            max_words = 50,
            random_state = 42,
            relative_scaling = 0.5,
            collocations = False,
            normalize_plurals = True,
            max_font_size = 80
            ).generate_from_text(str(dat_comments['words_clean_joined']))
            #generate_from_text(str(dat_comments.loc[dat_comments['createDate_month'] == m, 'words_clean_joined']))
            #generate(str(dat_comments['words_clean_joined']))

    print(wordcloud)
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('wordcloud_'+str(m)+'.png', bbox_inches='tight')
    plt.show()


#### Most frequently occuring words ####
def get_top_n_words(corpus, n = None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]

# Convert most freq words to dataframe for plotting bar plot
    for m in months:
        top_words = get_top_n_words(dat_comments.loc[dat_comments['createDate_month'] == m, 'words_clean_joined'], n=10)
        #top_words = get_top_n_words(dat_comments['words_clean_joined'], n=10)
        top_df = pd.DataFrame(top_words)
        top_df.columns=["Word", "Freq"]
        
        # Barplot of most freq words
        fix, ax = plt.subplots(figsize=(13,6.5))
        ax.bar(x = "Word", height = "Freq", data = top_df)
        ax.set(title = m,
               xlabel = "Words",
               ylabel = "Anzahl")
        plt.setp(ax.get_xticklabels(), rotation = 45)
        plt.savefig('unigram_'+str(m)+'.png', bbox_inches='tight')
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
for m in months:
    #top2_words = get_top_n2_words(dat_comments['words_clean_joined'], n=10)
    top2_words = get_top_n2_words(dat_comments.loc[dat_comments['createDate_month'] == m, 'words_clean_joined'], n=10)
    top2_df = pd.DataFrame(top2_words)
    top2_df.columns=["Bi-gram", "Freq"]
    print(m)
    print("#########")
    print(top2_df)
    
    # Barplot of most freq Bi-grams
    fix, ax = plt.subplots(figsize=(13,6.5))
    ax.bar(x = "Bi-gram", height = "Freq", data = top2_df)
    ax.set(title = m,
           xlabel = "Bi-gram",
           ylabel = "Anzahl")
    plt.setp(ax.get_xticklabels(), rotation = 45)
    plt.savefig('bigram_'+str(m)+'.png', bbox_inches='tight')
    plt.show()

#################################
dat_comments_apr = copy.deepcopy(dat_comments)




 

sns.boxplot(x="createDate_month", y="word_count", data=dat_comments)
plt.show()

dat_comments.boxplot(by="createDate_month", column="word_count")


