# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:52:43 2020

@author: Simon Würsten
"""
##############---- import packages ----##############
import pyodbc
import pandas as pd

##############---- Functions ----##############
def getSQLData(sql_query):
    """
    Constraints on _v_titlesBasicsRatings:
    b_titleType = 'movie' AND 
    b_genres IS NOT NULL
    --> Anz. Datensätze: 474'634
    """
    conn = pyodbc.connect('Driver={SQL Server};'
                          'Server=dbi-sql-server.database.secure.windows.net;'
                          'Database=dbi-sql-db;'
                          'UID=simon;'
                          'PWD=rub-ZZ9-3rq-eEc')
    result = pd.read_sql_query(sql_query, conn)
    return result

##############---- Main ----##############
#---- get data from SQL DB View
sql_query = 'SELECT * FROM _v_titlesBasicsRatings WHERE b_startYear BETWEEN 1970 AND 2020 AND r_averageRating > 0' # Anz. Datensätze: 188'527
df_imdb_movies = getSQLData(sql_query)
df = df_imdb_movies.copy()

#---- genres to list
df['b_genres_lst'] = df.b_genres.str.split(',')
#---- list to multiple rows
df_long = df.explode('b_genres_lst')

#---- save data frames
df.to_pickle('imdb_data')
df_long.to_pickle('imdb_data_long')