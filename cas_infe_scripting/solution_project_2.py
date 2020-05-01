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

def get_linear_regression(xi,y):
    A = np.array([ xi, np.ones(len(xi))])
    a, b = np.linalg.lstsq(A.T,y)[0] # obtaining the parameters
    return a, b # return as a tuple

###########################---- API call ----###########################
# get data
url = ["https://data.sbb.ch/api/records/1.0/search/?dataset=haltestelle-offnungszeiten&rows=3000", "https://data.sbb.ch/api/records/1.0/search/?dataset=passagierfrequenz&rows=999&sort=bahnhof_haltestelle"]
plaintext = []

for i in url:
    print("URL: ", i)
    print("-------------------------------")
    try:
        response = urllib.request.urlopen(i)
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
dict_pax_data = json.loads(plaintext[1])
df_pax = pd.DataFrame(index = range(len(dict_pax_data["records"])))
columns_from_json = ["bahnhof_haltestelle", "kanton", "dtv", "dwv", "dnwv"]

for z in columns_from_json:
    lst = []
    print(z)
    print("-------------------------------")
    for i in range(len(dict_pax_data["records"])):
        #print(dict_pax_data["records"][i]["fields"][z])
        try:
            lst.append(dict_pax_data["records"][i]["fields"][z])
        except Exception:
            lst.append(None)
    df_pax[z] = pd.DataFrame(lst)

df_pax_kanton = df_pax.groupby("kanton").sum()

# parsing with function (somehow not working)
# columns_from_json = ["bahnhof_haltestelle", "kanton", "dtv", "dwv", "dnwv"]
# df_pax = parse_json_data(columns_from_json, plaintext[1])

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
dict_opening_data = json.loads(plaintext[0])
df_opening = pd.DataFrame(index = range(len(dict_opening_data["records"])))
columns_from_json = ["stationsbezeichnung", "service", "von1", "bis1", "von2", "bis2", "mo", "di", "mi", "do", "fr", "sa", "so"]

for z in columns_from_json:
    lst = []
    print(z)
    print("-------------------------------")
    for i in range(len(dict_opening_data["records"])):
        #print(dict_pax_data["records"][i]["fields"][z])
        try:
            lst.append(dict_opening_data["records"][i]["fields"][z])
        except Exception:
            lst.append(None)
    df_opening[z] = pd.DataFrame(lst)

# parsing with function (somehow not working)
# columns_from_json = ["stationsbezeichnung", "service", "von1", "bis1", "von2", "bis2", "mo", "di", "mi", "do", "fr", "sa", "so"]
# df_opening = parse_json_data(columns_from_json, plaintext[0])

# calculate opening hours 1
df_opening[["von1", "bis1"]] = df_opening[["von1", "bis1"]].replace(np.nan, 0)
df_opening["hours1"] = pd.to_datetime(df_opening.bis1) - pd.to_datetime(df_opening.von1)
df_opening["in_seconds1"] = pd.to_numeric(df_opening.hours1) / 1000000000
df_opening["in_hours1"] = df_opening.in_seconds1 / 3600
df_opening[["von1", "bis1", "hours1", "in_seconds1"]]

# calculate openin ghours 2
df_opening[["von2", "bis2"]] = df_opening[["von2", "bis2"]].replace(np.nan, 0)
df_opening["hours2"] = pd.to_datetime(df_opening.bis2) - pd.to_datetime(df_opening.von2)
df_opening["in_seconds2"] = pd.to_numeric(df_opening.hours2) / 1000000000
df_opening["in_hours2"] = df_opening.in_seconds2 / 3600
df_opening[["von2", "bis2", "hours2", "in_seconds2"]]

# calculations above could be improved with function

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

# --> Korrelation ist tief, evtl. aufgrund der Ausreisser (Bahnhof Zürich + Bern), abgesehen von denen passt die Regressionsgerade jedoch nicht schlecht.