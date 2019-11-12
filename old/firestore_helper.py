# Helper functions for using firestore
# Feel free to request more functions (update, delete, special queries, etc.)

import pandas as pd
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
# from pandas import compat

# Use the application default credentials
# We're pretending I didn't just stick a private key in a json file
cred = credentials.Certificate("./ecs171-f19.json")
firebase_admin.initialize_app(cred, {
  'projectId': 'ecs171-f19',
})

db = firestore.client()

# Should be used with caution
def firestore_add(data, collection_name=u'data'):
    # if data is a string, assume it's a file name to a csv
    # csv must have headers
    # otherwise data must be a dataframe
    if isinstance(data, str):
        data = pd.read_csv(os.path.join(os.getcwd(), data), sep=",")

    data_dict = data.to_dict('index')

    # A batch lets you do a bunch of things without writing each one
    batch = db.batch()

    # Add each row
    for i, row in enumerate(data_dict):
        if i >= 8800 and data_dict[row]["Date"] != "2004-01-11" and data_dict[row]["Date"] != "2004-01-10":

            # Remove NaN values
            for item in list(data_dict[row]):
                if  data_dict[row][item] != data_dict[row][item]:
                    del data_dict[row][item]

            # creates a blank reference document with a firestore-generated uid
            fs_ref = db.collection(collection_name).document()

            # "set" is the firestore INSERT
            batch.set(fs_ref, data_dict[row])

            if i % 400 == 0:
                batch.commit()
                batch = db.batch()

    batch.commit()

# Reads all documents (rows) in a collection (table)
def firestore_read(collection_name=u'data'):
    data_dict = {}

    # Stream gets all documents
    docs = db.collection(collection_name).stream()
    for doc in docs:
        data_dict[doc.id] = doc.to_dict()

    # Create pandas df from collection data
    df = pd.DataFrame.from_dict(data_dict, orient='index')

    # Drop off uid row names
    df.reset_index(drop=True, inplace=True)
    return df
