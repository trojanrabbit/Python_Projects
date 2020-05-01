# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:56:39 2020

@author: Trojan Rabbit
"""



def downloadPoster(df_alg, image_folder):
    for df_set, df in df_alg.items():
        print("------------------------------------------------------")
        print(f"Starting with downloading files for {df_set}...\n")
        already_downloaded = 0
        count = 1
        http_errors = []
        for index, movie in df.iterrows():
            poster_url = movie[9]
            decade = str(movie[10])
            movie_id = str(movie[0])
            movie_name = movie[1]
            file_name = movie_id + ".jpg"
            file_path = "\\".join([image_folder, df_set, decade, file_name])
            print(f"downloading movie poster for {movie_name} _ (movie-nr {count})")
            if os.path.isfile(file_path):
                already_downloaded += 1
                count += 1
            else:
                try:
                    urllib.request.urlretrieve(poster_url, file_path)
                    count += 1
                except HTTPError:
                    http_errors.append(movie_id)
                    count += 1
        print(f"{len(http_errors)} posters had an HTTPError.")
        print(f"{already_downloaded} posters were downloaded before.\n")
              
def addFilepath(df_alg, image_folder):
    for df_set, df in df_alg.items():
        file_path = []
        for index, movie in df.iterrows():
            decade = str(movie[10])
            movie_id = str(movie[0])
            file_name = movie_id + ".jpg"
            file_path.append("\\".join([image_folder, df_set, decade, file_name]))
        df_alg[df_set]['file_path'] = file_path

def downloadPoster2(entry):
    path, uri = entry
    if not os.path.exists(path):
        urllib.request.urlretrieve(uri, path)
        time.sleep(10)
    return path

def createUrlPathLst(df_split):
    split_lst = []
    for df_set, df in df_split.items():
        for index, row in df.iterrows():
            col_list = [row.file_path, row.url]
            split_lst.append(col_list)
    return split_lst



###################################################################################♦

