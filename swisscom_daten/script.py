# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 16:49:28 2021

@author: u224208
"""

from pymongo import MongoClient
from ssl import CERT_NONE
from json import load

# Config file
db_config_file = 'config_p100_miamongo.json'

# Create mongo client
with open(db_config_file) as json_file:
    cfg = load(json_file)
con_str = f"mongodb://{cfg['db_user']}:{cfg['db_password']}@{cfg['db_host']}:{cfg['db_port']}/?authSource=admin"
client = MongoClient(con_str, ssl=True, ssl_cert_reqs=CERT_NONE)

# Test connection
print(client.server_info())
print(client.list_database_names())

# MIP database
#mip_db = client["mip"]
db = client["MIP_db"]
db

############################
import pandas as pd


# Collections auflisten
collections = []
for coll in db.collection_names():
    print(coll)
    collections.append(coll)

# eine Collection auswählen
col = db["PLZ_geojson_swisscom"]
col_res = list(col.find())


# List of Dict to DF
df = pd.DataFrame(col_res)



import geopandas
from shapely import wkt
df["geometrygft"] = geopandas.GeoSeries.from_wkt(df['geometry'])

import pandas as pd

gdf['geometry']=df['geometry'].apply(lambda x: shapely.geometry.shape(x))


