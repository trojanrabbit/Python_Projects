#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:55:19 2019

@author: sis
"""

import requests
from lxml import html
import pandas as pd
import csv


def fetch_article(url):

 #   URL = "https://www.nytimes.com/2018/04/24/sports/football/nfl-cheerleaders.html"
    CAFILE = "/Users/sis/.config/ca.crt"

    page = requests.get(url = url, verify = CAFILE)
    tree = html.fromstring(page.content)

    # <section name="articleBody" itemProp="articleBody" class="meteredContent css-1r7ky0e">
    # //*[@id="story"]/section
    
    #print(type(tree))
    
    article = tree.xpath('//article[@id="story"]/section[@name="articleBody"]//text()')
    #article = tree.xpath('//article')
    
    return " ".join(article)

ARTICLES = "ArticlesMay2017"
INPUT_FILE = "/Users/sis/Documents/ZHAW/03_Information_Retrieval/Project/ir_nyt/nyt-comments/" + ARTICLES + ".csv"

article_df = pd.read_csv(INPUT_FILE)
article_list = []

article_headers = ("index", "articleID", "webURL", "article")
article_list.append(article_headers)

article_df = article_df.iloc[0:]

for index, row in article_df.iterrows():
    # access data using column names
     
    article = fetch_article(row['webURL'])
    
    print(index)
     
    row = (index, row['articleID'], row['webURL'], article)
    article_list.append(row)


with open("Full" + ARTICLES + ".csv", 'w') as f:
    writer = csv.writer(f , lineterminator='\n')
    for row in article_list:
        writer.writerow(row)
