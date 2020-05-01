# -*- coding: utf-8 -*-
"""
Created End of Septembe 2019

@author: Simon Würsten
"""
# import modules/packages
import urllib
import json
from geojson import MultiLineString
import geojsonio

print("*****************************************")
print("Hinweis: nur Streckennetz SBB gemäss https://data.sbb.ch/explore/dataset/linie/information/")
print("*****************************************")

while True:
    # ask for departure station and get data
    bpk_begin = input("Bitte Abgangsort eingeben (zB. Aigle, Basel SBB, Bassersdorf,...): ")
    url = "https://data.sbb.ch/api/records/1.0/search/?dataset=linie&facet=linie&facet=bpk_ende&refine.bpk_anfang=" + bpk_begin
    
    try:
        response = urllib.request.urlopen(url)
        plaintext = response.read().decode("utf-8")
    except Exception as e:
        print("Konnte nicht zum API-Service verbinden. Fehlermeldung:", e)
        continue;
    
    # parse json-data
    dict_data = json.loads(plaintext)
    dict_keys = dict_data.keys()
    
    # print lines and line numbers
    try:
        dict_data["facet_groups"][1]["facets"]
    except Exception as e:
        print("----------------------------------")
        print("Bahnhof nicht gefunden oder keine Linie ab gewähltem Startpunkt. Fehlermeldung:", e)
        
        continue;
    else:
        print("Linien mit Start", bpk_begin, "haben folgenden Endbahnhof:")
        for i in range(len(dict_data["facet_groups"][1]["facets"])):
            print("----------------------------------")
            for y in dict_data["facet_groups"][1]["facets"][i]:
                if y == "name":
                    print("Linie:", dict_data["facet_groups"][1]["facets"][i][y])
                    print("Nummer:",dict_data["facet_groups"][2]["facets"][i][y])
        print("----------------------------------")
    
    user_input = input("Als nächstes werden die Endstationen auf einer Karte dargestellt. Dazu wird ein Browserfenster geöffnet. Fortfahren (y/n)? ")
    
    # create multi linestring and plot geojson-data
    if user_input == "y":
        concat_geojson = []
        for i in range(len(dict_data["records"])):
            #print(dict_data["records"][i]["fields"]["tst"]["coordinates"])
            concat_geojson.append(dict_data["records"][i]["fields"]["tst"]["coordinates"])
        geojson_multi_ls = json.dumps(MultiLineString(concat_geojson))
        geojsonio.display(geojson_multi_ls)
        break
    break
