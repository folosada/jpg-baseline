import requests
import csv
import urllib
import os


# Limit the quantity of files you want to download
LIMIT_FILES = 10
BASE_FOLDER = 'Trabalho Final/jpg-baseline/dataset_files_download'
DATASET_FOLDER = BASE_FOLDER + '/dataset'

if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

links = []
i = 0
with open(BASE_FOLDER + '/data.csv', mode='r') as infile:
    readCSV = csv.reader(infile, delimiter=';')
    for row in readCSV:
        if i > LIMIT_FILES:
            break
        
        i += 1
        nef = row[1]
        if (nef == 'NEF'):
            continue
        links.append(nef)

i = 0
for link in links:
    filename = DATASET_FOLDER + "/" + link.split("/")[-1]
    print(filename)
    r = requests.get(link)
    with open(filename, 'wb') as f:
        f.write(r.content)