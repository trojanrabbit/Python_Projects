import requests
import urllib.request
from bs4 import BeautifulSoup
import pickle
from concurrent.futures import ThreadPoolExecutor as PoolExecutor

#urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Example data: data = ['0015724', '0016906', '0023331', '0031458', '0035423', '0036606', '0038687',  '0039442', '0054724', '0056166', '9913056', '9913084', '9913936', '9914286', '9914642', '9914644', '9914942', '9915790', '9916160', '9916428']
movie_db = pickle.load( open("/Users/sis/Documents/ZHAW/CAS MAIN 2020.4/02_Deep_Learning/dl_project_imdb/data/imdb_data", "rb" ) )

data = movie_db['b_tconst']
data = data[0:1000]


s = requests.Session()
f = open("results.csv", "w+")


def getPoster(imdbid):
    url = 'https://www.imdb.com/title/tt' + imdbid + '/reference'
    response = s.get(url, verify=False)

    soup = BeautifulSoup(response.text, 'html.parser')
    parsed_result = soup.findAll('img', {'class' : 'titlereference-primary-image'})

    result = ""

    if(len(parsed_result) != 0):
        img_url = parsed_result[0]['src']
        result = imdbid + ";" + str(response.status_code) + ";" + img_url
        urllib.request.urlretrieve(img_url, "pics/" + imdbid + ".jpg")
    else:
        result = imdbid + ";" + str(response.status_code) + ";None"

    f.write(result + "\r\n")

#for movie in data:
#    getPoster(movie)

# create a thread pool of 4 threads
with PoolExecutor(max_workers=4) as executor:

    # distribute the 1000 URLs among 4 threads in the pool
    # _ is the body of each page that I'm ignoring right now
    for _ in executor.map(getPoster, data):
        pass

f.close()

