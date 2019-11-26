
import requests
import csv
import os
import sys
import time
import pandas as pd
from io import StringIO

sys.path.insert(0, './data')
import sitedict
from sitedict import *

sites = list(SITEDICT.keys())
for site in sites:
    df = pd.DataFrame()
    lat = SITEDICT[site]["lat"]
    long = SITEDICT[site]["long"]
    for year in range(1980, 2020):

        # Parameters for web request
        url = "https://wcc.sc.egov.usda.gov/nwcc/view"

        payload = {
            'intervalType': ' View Historic ',
            'report': 'ALL',
            'timeseries': 'Daily',
            'format': 'copy',
            'sitenum': site,
            'interval': 'YEAR',
            'year': year,
            'month': 'CY'
        }

        # Do post for this year and site
        with requests.Session() as s:
            with open(os.path.join(os.getcwd(), "temp.csv"), 'wb') as f:
                with s.post(url, params=payload, stream=True) as r:
                    for line in r.iter_lines():
                        f.write(line + b"\n") # write to file

        # Read csv into df
        df = df.append(pd.read_csv(os.path.join(os.getcwd(), "temp.csv"),
                skiprows=4), ignore_index=True, sort=False)

    # Add in latitude and longitude
    df.insert(1, "Latitude", lat)
    df.insert(2, "Longitude", long)


    # Save completed site to pickle just in case this crashes
    # It also might be useful to have the pkls though anyway
    print(f"Completed site {site}: {SITEDICT[site]['name']}", flush=True)
    pklpath = os.path.join(os.getcwd(), "data", f"site{site}.pkl")
    df.to_pickle(pklpath)

    # I'm sleeping in hopes that they don't notice I'm sending a thousand
    # requests to their server so they don't ban me forever
    # This also is why I'm expecting the script to fail partway through
    # Actually, it didn't get rejected at 10 seconds
    # (I tried 5 seconds and it failed partway through)
    time.sleep(10)

# I saved each site to a pickle and now I'm loading them and merging
# Note: I did this because I expected the loop above to crash in the middle
# and I'd be able to restart it for only some sites
merged = pd.DataFrame()
for site in sites:
    pklpath = os.path.join(os.getcwd(), "data", f"site{site}.pkl")
    df = pd.read_pickle(pklpath)
    merged = merged.append(df, ignore_index=True, sort=False)


csvpath = os.path.join(os.getcwd(), "data", "merged.csv")
pklpath = os.path.join(os.getcwd(), "data", "data.pkl")

# For some reason the csv has a bunch of exmpty columns so I'm going to manually
# delete those and actually overwrite the pickle with a new pickle later
merged.to_csv(path_or_buf=csvpath, index=False)
merged.to_pickle(pklpath)
