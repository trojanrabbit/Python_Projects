# CAS Machine Intelligence - Deep Learning Project
Klassifikation von IMDb Filmposter nach Genre und Jahrzent

#### Links
- https://www.imdb.com/interfaces/
  - title.basics.tsv.gz
  - title.ratings.tsv.gz

Inspiriert von:
- https://www.kaggle.com/neha1703/movie-genre-from-its-poster
- https://towardsdatascience.com/movie-posters-81af5707e69a?gi=45f6c37d4a34
- https://jinglescode.github.io/datascience/2019/11/17/predict-movie-earnings-with-posters/

#### get_data.py
Datensets von IMDB von SQL Datenbank laden und als Pickle speichern.

#### data_prep.py
Daten von Pickle laden und transformieren

#### data.zip
die gespeicherten Pickles
- imdb_data: pro Zeile ein Film
- imdb_data_long: pro Genre ein Film (> zB. ein Film mit drei Genres wird auf drei Zeilen aufgeteilt mit jeweils einem Genre)

#### Image data
- Download link: https://drive.google.com/open?id=1HwAMnGyQm_63yOYBrH1a9n3wezgZlDoz
- Meta data about each IMDB record: data/result.csv
