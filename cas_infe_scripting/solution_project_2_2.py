# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 07:29:21 2019

@author: u224208
"""
import urllib
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# cache timeout (based on code by Simon Stäheli)
CACHE_TIMEOUT = 3600 * 6 # 1h

###########################---- functions ----###########################
# data parsing
def parse_json_data(column_list, plaindata):
    dict_data = json.loads(plaindata)
    df = pd.DataFrame(index = range(len(dict_data["records"])))

    for z in column_list:
        lst = []
        print("verarbeitete Spalte: ", z)
        print("-------------------------------")
        for i in range(len(dict_data["records"])):
            try:
                lst.append(dict_data["records"][i]["fields"][z])
            except Exception:
                lst.append(None)
        df[z] = pd.DataFrame(lst)
    return df

# linear regression
def get_linear_regression(xi,y):
    A = np.array([ xi, np.ones(len(xi))])
    a, b = np.linalg.lstsq(A.T,y)[0] # obtaining the parameters
    return a, b # return as a tuple

# calculate opening hours
def calculate_opening_hours(df, column_iter):
    df[["von"+str(column_iter), "bis"+str(column_iter)]] = df[["von"+str(column_iter), "bis"+str(column_iter)]].replace(np.nan, 0)
    df["hours"+str(column_iter)] = pd.to_datetime(df.bis1) - pd.to_datetime(df.von1)
    df["in_seconds"+str(column_iter)] = pd.to_numeric(df.hours1) / 1000000000
    df["in_hours"+str(column_iter)] = df.in_seconds1 / 3600
    return df

# read or download data (concept based on code by Simon Stäheli)
def open_file(filename, url):
    plaintext = []
    if os.path.isfile(filename):
        timedifference = time.time() - CACHE_TIMEOUT
        ts_file = os.path.getmtime(filename)
        if timedifference < ts_file:
            print("File ist aktuell. Lade File...")
            print("----------------------------------------------")
            with open(filename, 'r') as input:
                plaintext.append(json.load(input))
        else:
            print("File ist nicht mehr aktuell. Downloade File...")
            print("----------------------------------------------")
            plaintext.append(download_file(filename, url))
    else:
        print("File existiert nicht. Downloade File...")
        print("----------------------------------------------")
        plaintext.append(download_file(filename, url))
    return plaintext

# download data
def download_file(filename, url):
    plaintext = []
    print("URL: ", url)
    print("----------------------------------------------")
    try:
        response = urllib.request.urlopen(url)
        plaintext.append(response.read().decode("utf-8"))
        #with open(filename, 'w') as output:
            #json.dump(plaintext, output)
    except Exception as e:
        print("Error! Fehlermeldung beim Download oder Speichern:", e)
        print("----------------------------------------------")
    return plaintext

###########################---- API call ----###########################
# get data
url = {
       "haltestelle_oeffnungszeiten.txt":"https://data.sbb.ch/api/records/1.0/search/?dataset=haltestelle-offnungszeiten&rows=3000",
       "passenger_freq.txt": "https://data.sbb.ch/api/records/1.0/search/?dataset=passagierfrequenz&rows=999&sort=bahnhof_haltestelle"
       }
plaintext = []

### Variante 1 ###
for f, u in url.items():
    print("#####################################")
    print(f)
    print("----------------------------------------------")
    plaintext.append(open_file(f, u))

### Variante 2 (ohne Cache und Methoden) ####
for f, u in url.items():
    print("URL: ", u)
    print("-------------------------------")
    try:
        response = urllib.request.urlopen(u)
        plaintext_tmp = response.read().decode("utf-8")
        plaintext.append(plaintext_tmp)
    except Exception as e:
        print("Konnte nicht zum API-Service verbinden. Fehlermeldung:", e)

###########################---- frequency data ----###########################
# column descriptions:
# dtv: durchschnittlicher täglicher verkehr (Mo-So)
# dwv: durchschnittlicher werktäglicher verkehr (Mo-Fr)
# dnwv: durchschnittlicher nicht werktäglicher verkehr (Sa, So, Feiertage)
        
# parse passenger frequency json-data
columns_from_json = ["bahnhof_haltestelle", "kanton", "dtv", "dwv", "dnwv"]
df_pax = parse_json_data(columns_from_json, plaintext[1]) # <------------ führt zu Fehler mit Variante 1 (siehe oben). Irgendwie scheint

df_pax_kanton = df_pax.groupby("kanton").sum()

# plot histograms for each measure
plt.figure(figsize=(9, 3))
plt.subplot(1,3,1)
plt.title("dtv")
plt.hist(df_pax_kanton["dtv"])
plt.subplot(1,3,2)
plt.title("dwv")
plt.hist(df_pax_kanton["dwv"])
plt.subplot(1,3,3)
plt.title("dnwv")
plt.hist(df_pax_kanton["dnwv"])
plt.suptitle('histograms nach kantone')
plt.show()

###########################---- opening hours ----###########################
# parse opening hours json-data
columns_from_json = ["stationsbezeichnung", "service", "von1", "bis1", "von2", "bis2", "mo", "di", "mi", "do", "fr", "sa", "so"]
df_opening = parse_json_data(columns_from_json, plaintext[0])

# calculate opening hours 1
df_opening = calculate_opening_hours(df_opening, 1)

# calculate opening ghours 2
df_opening = calculate_opening_hours(df_opening, 2)

# calculate total opening hours
df_opening["total_hours"] = df_opening.in_hours1 + df_opening.in_hours2

# group by station name
df_opening_station = df_opening.groupby("stationsbezeichnung").sum()
df_opening_station = df_opening_station.reset_index()
df_opening_station.head()

###########################---- merge df ----###########################
# merge dataframes and keep relevant columns only
df_open_pax = pd.merge(df_opening_station, df_pax, left_on = "stationsbezeichnung", right_on = "bahnhof_haltestelle", how = "inner")
df_open_pax = df_open_pax.drop(["stationsbezeichnung", "in_seconds1", "in_hours1", "in_hours2", "kanton"], 1)

###########################---- correlation ----###########################
# correlation total opening hours <-> durchschnittlicher täglicher verkehr
np.corrcoef(df_open_pax["total_hours"], df_open_pax["dtv"])

###########################---- linear regression ----###########################
# haben Bahnhöfe mit mehr Personenaufkommen längere Öffnungszeiten der verfügbaren Geschäfte und Services? Bzw. führen längere Öffnungszeiten zu höherem Personenaufkommen?
# credits: Jose Spillner
w = get_linear_regression(df_open_pax["total_hours"], df_open_pax["dtv"])
print("regression\t","dtv = ",round(w[0],2),"* total_hours + ",round(w[1],1))

# scatter plot with regression line
line = w[0]*df_open_pax["total_hours"] + w[1]
plt.plot(df_open_pax["total_hours"],df_open_pax["dtv"],'o',df_open_pax["total_hours"],line)
plt.title('total opening hours vs. dtv')
plt.xlabel("total opening hours")
plt.ylabel("dtv")
plt.show()

# --> Korrelation ist tief, evtl. aufgrund der Ausreisser (Bahnhof Zürich + Bern), abgesehen von denen passt die Regressionsgerade jedoch nicht schlecht.


