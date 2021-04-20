import pandas as pd
import numpy as np

import feat_functions as ff


### MAIN ###
stichtag = "2020-12-31"
stichtag = np.datetime64(stichtag)

# Daten laden
df_kk = pd.read_csv("data/KUKONDATEN_ZU_SPL_FOR_APRIORIS_2021-03-21.csv", parse_dates=["VERKAUFSDATUM"])
df_kk.dtypes
df_spl = pd.read_csv("data/SPL_FOR_APRIORIS.csv", sep=";")
df_spl.dtypes
# Daten transformieren
df_kk = df_kk.rename(columns={"SORTIMENTS_CLUSTER_NOVA_TXT": "SORTIMENTE"})
df_kk.KLASSE = df_kk.KLASSE.astype(str)

# Nr Unique-Kunden ausgeben
df_kk["TKID"].nunique()
df_spl["TKID"].nunique()
# Uniques ausgeben
df_kk["SORTIMENTE"].unique()
# Min- + Max-Date
df_kk["VERKAUFSDATUM"].min()
df_kk["VERKAUFSDATUM"].max()

## einzelne Features berechnen
df_single = df_kk.groupby("TKID").agg(
    MIN_DATE = ("VERKAUFSDATUM", "min"),
    MAX_DATE = ("VERKAUFSDATUM", "max"),
    ABSATZ_TOTAL = ("TKID", "size"))
df_single["KUNDE_SEIT_TAGE"] = stichtag - df_single["MIN_DATE"]
df_single["KUNDE_SEIT_MONATE"] = np.ceil((stichtag - df_single["MIN_DATE"]) / np.timedelta64(1, 'M')).astype(int)

## Gruppe von Features berechnen
last_x_m = [1, 2, 3, 6, 9, 12]
dfs = []
for x in last_x_m:
    df_small = ff.last_x_month_sortiment(df_kk, x, stichtag)
    # TKID als Index verwenden
    df_small.set_index("TKID", inplace=True)
    # DFs sammeln
    dfs.append(df_small)   
    print("-----------")

# gesammelte DFs zusammenführen
df_large = pd.concat(dfs, axis=1)

## Feature-DFs zusammenführen
df_grp = pd.concat([df_large, df_single], axis=1)

## Zero und NaN setzen
dfs = []
for x in last_x_m:
    df_grp = ff.set_zero_nan(df_grp, x)  
    print("-----------")


df_grp.to_csv("save/kunden_features.csv")
