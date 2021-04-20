# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np

#%%
#---- read csv
df = pd.read_csv('data/adult.data', sep=",", header=None, names=['age', 
                                                                 'workclass', 
                                                                 'fnlwgt', 
                                                                 'education', 
                                                                 'education-num', 
                                                                 'marital-status', 
                                                                 'occupation', 
                                                                 'relationship', 
                                                                 'race', 
                                                                 'sex', 
                                                                 'capital-gain', 
                                                                 'capital-loss', 
                                                                 'hours-per-week', 
                                                                 'native-country',
                                                                 'income-class'])

print(df.columns)
print(df.head(5))

#%%
df['id'] = df.index


df['income-class'] = df['income-class'].str.strip()

df.loc[df['income-class'] == '>50K', 'label'] = 1
df.loc[df['income-class'] == '<=50K', 'label'] = 0

df.groupby('label').count()

df = df.drop(columns=['income-class'])

