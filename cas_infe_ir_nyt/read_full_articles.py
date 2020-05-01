# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:48:46 2019

@author: u224208
"""
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt


# Articles-File laden
# Data columns (total 4 columns):
# index        1324 non-null int64
# articleID    1324 non-null object
# webURL       1324 non-null object
# article      1324 non-null object

filename_articles = r'/Users/sis/Documents/ZHAW/03_Information_Retrieval/Project/ir_nyt/FullArticlesApril2018.csv'
df_articles = pd.read_csv(filename_articles, encoding='utf-8', low_memory=False)

# Comments-File laden
# approveDate              264924 non-null int64
# articleID                264924 non-null object
# articleWordCount         264924 non-null float64
# commentBody              264924 non-null object
# commentID                264924 non-null float64
# commentSequence          264924 non-null float64
# commentTitle             264911 non-null object
# commentType              264924 non-null object
# createDate               264924 non-null int64
# depth                    264924 non-null float64
# editorsSelection         264924 non-null bool
# inReplyTo                264924 non-null float64
# newDesk                  264924 non-null object
# parentID                 264924 non-null float64
# parentUserDisplayName    83875 non-null object
# permID                   264904 non-null object
# picURL                   264924 non-null object
# printPage                264924 non-null float64
# recommendations          264924 non-null int64
# recommendedFlag          0 non-null float64
# replyCount               264924 non-null int64
# reportAbuseFlag          0#  non-null float64
# sectionName              264924 non-null object
# sharing                  264924 non-null int64
# status                   264924 non-null object
# timespeople              264924 non-null int64
# trusted                  264924 non-null int64
# typeOfMaterial           264924 non-null object
# updateDate               264924 non-null int64
# userDisplayName          264824 non-null object
# userID                   264924 non-null float64
# userLocation             264786 non-null object
# userTitle                100 non-null object
# userURL                  0 non-null float64

filename_comments = r'/Users/sis/Documents/ZHAW/03_Information_Retrieval/Project/ir_nyt/nyt-comments/CommentsApril2018.csv'
df_comments = pd.read_csv(filename_comments, encoding='utf-8', low_memory=False)


### TEMP MAKE PROCESSING FAST ###

#df_articles = df_articles.iloc[0:100]
#df_comments = df_comments.iloc[0:100]

### FUNCTIONS ###

def compute_sentiment(articles):
    
    sentiments = []
    for art in articles:
        blob = TextBlob(art)
        sentiments.append(blob.sentiment.polarity)
    
    return sentiments

### MAIN ###
    
df_articles['sentiment_article'] = compute_sentiment(df_articles['article'])
df_comments['sentiment_comment'] = compute_sentiment(df_comments['commentBody'])

df = pd.merge(df_articles, df_comments, on='articleID')

df = df[['articleID', 'sentiment_article', 'sentiment_comment', 'userID']]

#print(df.to_string())

#plt.scatter(df['articleID'],df['sentiment_comment'])
#plt.scatter(df['articleID'],df['sentiment_article'])
#plt.show()

df = df.groupby('userID').count()

print(df)

print("Artikel Analyse")
print(df_articles['sentiment_article'].describe())


print("Kommentare Analyse")
print(df_comments['sentiment_comment'].describe())
