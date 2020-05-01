# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:55:43 2020

@author: Simon Würsten
"""
# Constraints: 
"""
    b_titleType = 'movie' AND 
    b_genres IS NOT NULL AND
    b_startYear BETWEEN 1970 AND 2020 AND 
    r_averageRating > 0
    --> Anz. Datensätze: 188'527
"""

##############---- import packages ----##############
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile, rmtree
from pathlib import Path


##############---- Functions ----############## 
def selectSampleImgs(df, df_grouped, s_size, del_row, col_name):
    df_grouped = df_grouped.iloc[:del_row]
    df = df[df[col_name].isin(df_grouped[col_name])]
    df_sample = df.sample(s_size)
    return df_sample


def selectSampleImgsStrat(df, df_grouped, min_mov, col_name):
    df_grouped = df_grouped[df_grouped.b_tconst >= min_mov]
    df = df[df[col_name].isin(df_grouped[col_name])]
    df_sample = df.groupby(col_name).apply(lambda x: x.sample(min_mov))
    df_sample = df_sample.reset_index(level = 0, drop = True)
    return df_sample


def splitData(df, t_size, v_size, tr_size):
    train, test = train_test_split(df, train_size = tr_size)
    test, val = train_test_split(test, train_size = t_size / (t_size +v_size))
    return train, val, test


def splitDataStrat(df, t_size, v_size, tr_size, col_name):
    train, test = train_test_split(df, train_size = tr_size, stratify= df[col_name])
    test, val = train_test_split(test, train_size = t_size / (t_size +v_size), stratify = test[col_name])
    return train, val, test


def moveImgs(d_set, mov, s_dir, d_dir, classification):
    #---- movie img for genres
    if classification == 'b_genres_lst':
        Path(d_dir + "/" + d_set + "/" + mov[1]).mkdir(parents=True, exist_ok=True)
        src = s_dir + "/" + mov[0] + ".jpg"
        dst = d_dir  + "/" + d_set + "/" + mov[1] + "/" + mov[0] + ".jpg"
        copyfile(src, dst)
        try:
            copyfile(src, dst)
        except:
            pass
    #---- movie images for decade    
    if classification == 'decade':    
        Path(d_dir + "/" + d_set + "/" + mov[2]).mkdir(parents=True, exist_ok=True)
        src = s_dir + "/" + mov[0] + ".jpg"
        dst = d_dir  + "/" + d_set + "/" + mov[2] + "/" + mov[0] + ".jpg"
        try:
            copyfile(src, dst)
        except:
            pass        


##############---- Main ----##############
#---- load data
df = pd.read_pickle('data/imdb_data_long_v2') # imdb_data_long_v2: genre-classification, imdb_data_v2: decade-classification
        
#---- define inputs
class_col = 'b_genres_lst' # 'decade', 'b_genres_lst'
train_size = 0.6
validation_size = 0.2
test_size = 0.2
sample_size = 80000 # wird nur bei nicht stratifiziertem Sample angewendet
sample_type = 'stratified' # 'stratified', None

# Folder to read pictures from
source_dir ='data/dl_imdb_pics/pics'
# Folder to write pictures to
dest_dir ='data/dl_imdb_pics/pics_genres_strat1k'

#---- filter movies only with Poster-URL
df = df[df.url != 'None']
#---- set decade
df['decade'] = [str(year)[2] + "0" for year in df['b_startYear']]

#---- check count of movies per class
df_grp = df.groupby(class_col ).count()[['b_tconst']].sort_values('b_tconst', ascending = False).reset_index()
print(df_grp)

if sample_type == 'stratified':
    #---- stratified samples
    # Anz. Sample pro Gruppe/Klasse, basierend auf visueller Inspektion des obigen print(df_genre)
    min_movies = 1000 # <------------------------------------------------- anpassen bei Bedarf
    
    df_sample = selectSampleImgsStrat(df, df_grp, min_movies, class_col)
    train, val, test = splitDataStrat(df_sample, test_size, validation_size, train_size, class_col)
else:
    #---- nicht stratifiziertes sample
    # vgl. print(df_genre) oben; wie viele Genres sollen entfernt werden (von unten). None falls keine, sonst zB. -3 für letzte drei Genres
    del_rows = None # <------------------------------------------------- anpassen bei Bedarf
    
    df_sample = selectSampleImgs(df, df_grp, sample_size, del_rows, class_col)
    train, val, test = splitData(df_sample, test_size, validation_size, train_size)

#---- inspect sampling
# sample
print("gesamtes Sample")
print(df_sample.groupby(class_col).count()[['b_tconst']].sort_values('b_tconst', ascending = False).reset_index())
# train set
print("train set")
print(train.groupby(class_col).count()[['b_tconst']].sort_values('b_tconst', ascending = False).reset_index())
# validation set
print("validation set")
print(val.groupby(class_col).count()[['b_tconst']].sort_values('b_tconst', ascending = False).reset_index())
# test set
print("test set")
print(test.groupby(class_col).count()[['b_tconst']].sort_values('b_tconst', ascending = False).reset_index())

#---- move images
# delete if destination dir exists
rmtree(dest_dir)

train = train[['b_tconst','b_genres_lst','decade']].values.tolist()
val = val[['b_tconst','b_genres_lst','decade']].values.tolist()
test = test[['b_tconst','b_genres_lst','decade']].values.tolist()

print("move for train...")
[moveImgs("train", movie, source_dir, dest_dir, class_col) for movie in train]
print("move for validate...")
[moveImgs("validate", movie, source_dir, dest_dir, class_col) for movie in val]
print("move for test...")
[moveImgs("test", movie, source_dir, dest_dir, class_col) for movie in test]






