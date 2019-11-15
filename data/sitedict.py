# tab file contents:
# 4 cols: name, id, lat (y val), long (x val)
# dictionary of dictionaries:
#   outer dictionary is id -> dict
#   inner dictionary is name -> name, lat -> lat, long -> long

# note that this is going to be imported into the main file so the info filepath
# includes the data directory

import os

SITEDICT = {}
with open(os.path.join(os.getcwd(), "data", "info.txt")) as f:
    for line in f:
        (name, id, lat, long) = line.split("\t")
        innerdict = {}
        innerdict["name"] = name
        innerdict["lat"] = lat
        innerdict["long"] = long
        SITEDICT[str(id)] = innerdict
