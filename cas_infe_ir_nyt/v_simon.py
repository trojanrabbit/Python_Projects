# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 11:12:16 2019

@author: u224208
"""

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import copy


for col in dat_comments.columns: 
    print(col)

##################### Functions #####################
# Wortliste entschachteln
def orderWordsList(corpus):
    words_clean_grouped = []
    words_clean_grouped_joined = []
    for line in corpus:
        words_clean_grouped_tmp = []
        for item in line:
            for word in item:
                words_clean_grouped_tmp.append(word)
        words_clean_grouped.append(words_clean_grouped_tmp)
        words_clean_grouped_joined.append(" ".join(words_clean_grouped_tmp))
    return words_clean_grouped, words_clean_grouped_joined

# neues Dataframe mit Häufigkeit Vorkommen eines Suchwortes und des Datum generieren
def searchTerm(df, search_term):
    z = 0
    df_search_term = pd.DataFrame(columns = ['search_term', 'word_count', 'createDate_date'])
    for line in df['words_clean']:
        print('Linie:', z, 'von', len(df))
        word_count = 0
        for word in line:
            if word == search_term:
                word_count = word_count + 1
        createDate_date = df['createDate_date'][z]
        df_search_term.loc[len(df_search_term)] = [search_term, word_count, createDate_date]
        z = z + 1
    return df_search_term

def searchTerm2(df, search_terms):
    z = 0
    df_search_term = pd.DataFrame(columns = ['search_term', 'word_count', 'createDate_date'])
    for line in df['words_clean']:
        print('Linie:', z, 'von', len(df))
        word_count = 0
        for word in line:
            if word in search_terms:
                word_count = word_count + 1
                createDate_date = df['createDate_date'][z]
                df_search_term.loc[len(df_search_term)] = [search_term, 1, createDate_date]
        z = z + 1
    return df_search_term

##################### spezifische Dataframes #####################
# Anzahl Kommentare und Anzahl Wörter pro User
grouped = dat_comments.groupby('userID').agg({'userID': np.count_nonzero,
                                         'word_count': np.sum})
df_grouped = grouped.add_suffix('_').reset_index()
df_grouped = df_grouped.rename(columns={'userID': 'userID', 'userID_': 'comment_count', 'word_count_': 'word_count'})

# Top10 User nach Anzahl Wörter
df_top10_user_word_count = df_grouped.sort_values(by = 'word_count', ascending = False)[:10]
# von Top10 User nach Anz. Wörter alle Zeilen von CSV
#df_top10_user_words = copy.deepcopy(dat_comments[dat_comments['userID'].isin(list(df_top10_user_word_count['userID']))])

# Gruppiere 'words_clean' nach User
#df_grouped2 = dat_comments.groupby('userID')['words_clean'].apply(list).reset_index() # von allen User
df_grouped2 = dat_comments[dat_comments['userID'].isin(list(df_top10_user_word_count['userID']))].groupby('userID')['words_clean'].apply(list).reset_index() # nur Top10 User nach Anz. Wörter

# Dataframe mit Häufigkeit Vorkommen und Datum nach einem Suchwort generieren
search = 'trump'
z = 0
df_search = pd.DataFrame(columns = ['words_clean'])

z = 0
df_search_term = pd.DataFrame(columns = ['search_term', 'word_count', 'createDate_date'])
for line in df_search['words_clean']:
    res = [x for x in line if x == 'trump']
    createDate_date = df_search['createDate_date'][z]
    df_search_term.loc[len(df_search_term)] = ['term', len(res), createDate_date]
    z = z + 1

df_search = copy.deepcopy(dat_comments[dat_comments['words_clean'].apply(lambda x: search in x)]) # exportiere Zeilen vom ursprünglichen Dataframe mit Suchwort
df_search = df_search.reset_index()
df_search_term = searchTerm(df_search, 'trump')

df_grouped3 = df_search_term.groupby(['createDate_date'])['word_count'].sum().reset_index()

##################### generiere weitere Spalten #####################
# Wortliste entschachteln von Top10 User nach Anz. Wörter
df_grouped2['words_clean_grouped'], df_grouped2['words_clean_grouped_joined'] = orderWordsList(df_grouped2['words_clean'])


##################### Histogramme Anzahl Kommentare + Wörter pro User #####################
# Plot Histogram Anzahl Kommentare
df_grouped.sort_values(by = 'comment_count', ascending = False)[:10]

fix, ax = plt.subplots(figsize=(5,3))
ax.hist(df_grouped["comment_count"], bins=100)
ax.set(title = "Anz. Kommentare pro User",
       xlabel = "Anz. Kommentare",
       ylabel = "Häufigkeit")
plt.savefig('hist_user_count.png', bbox_inches='tight')
plt.show()

# Plot Histogram Anzahl Wörter nach User
df_grouped.sort_values(by = 'word_count', ascending = False)[:10]

fig, ax = plt.subplots()
ax.hist(df_grouped["word_count"], bins=100)
ax.set(title = "Histogramm Anzahl Wörter (ohne Stoppwörter)",
       xlabel = "Anz. Wörter",
       ylabel = "Häufigkeit")
plt.show()

# Plot Histogram Anzahl Wörter gesamt
fix, ax = plt.subplots(figsize=(13,6.5))
ax.hist(dat_comments["word_count"], bins=100)
ax.set(title = "Anz. Wörter pro Kommentar (ohne Stoppwörter)",
       xlabel = "Anz. Wörter",
       ylabel = "Häufigkeit")
plt.savefig('hist_word_count_gross.png', bbox_inches='tight')
plt.show()


##################### Wordcloud für jeden Top10 User nach Anz. Wörter #####################
# Wordcloud für jeden Top10 User
z = 0
for line in df_grouped2['userID']:
    wordcloud = WordCloud(
            background_color = 'black',
            stopwords = stop_words,
            max_words = 50,
            random_state = 42
            ).generate(str(df_grouped2['words_clean_grouped_joined'][z]))
    
    print(wordcloud)
    plt.figure(figsize=(5,2.5))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('user_'+str(z)+'.png', bbox_inches='tight')
    plt.show()
    z = z + 1

##################### Vorkommen Suchwort #####################
df_search_term.sort_values(by = 'word_count')
fix, ax = plt.subplots(figsize=(13,6.5))
ax.plot(df_grouped3['createDate_date'], df_grouped3['word_count'])
ax.set(title = "Häufigkeit von 'trump'",
       xlabel = "Tage Jan-Mai",
       ylabel = "Anzahl")
plt.setp(ax.get_xticklabels(), rotation = 45)
plt.savefig('vorkommen_trumo.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df_grouped3['createDate_date'], df_grouped3['word_count'])
plt.xticks(rotation=30)
plt.show()
