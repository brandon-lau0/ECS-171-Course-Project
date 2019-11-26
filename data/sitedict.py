# tab file contents:
# 4 cols: name, id, lat (y val), long (x val)
# dictionary of dictionaries:
#   outer dictionary is id -> dict
#   inner dictionary is name -> name, lat -> lat, long -> long

# note that this is going to be imported into the main file so the info filepath
# includes the data directory

import os
import json

SITEDICT = {}

sitedictpath = os.path.join(os.getcwd(), "sitedict.json")

# If the json file already exists, load it
if os.path.exists(sitedictpath):
    with open(sitedictpath, 'r') as fin:
        SITEDICT = json.load(fin)
else:
    # Convert text file to dictionary of dictionaries
    with open(os.path.join(os.getcwd(), "data", "info.txt")) as f:
        for line in f:
            (name, id, lat, long) = line.split("\t")
            innerdict = {}
            innerdict["name"] = name
            innerdict["lat"] = lat
            innerdict["long"] = long
            SITEDICT[str(id)] = innerdict
    # Save dictionaries to json
    with open(sitedictpath, 'w') as fout:
        json.dump(SITEDICT, fout)
