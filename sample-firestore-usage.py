# pip install --upgrade firestore-admin

# Sample code for using the firestore_helper file

import firestore_helper
from firestore_helper import *
import pandas as pd


def sample_add_data():
    # data as dataframe
    df = pd.DataFrame({'col1': [1, 2], 'col2': [0.5, 0.75]})

    #data as csv - just a filename - csv must have headers
    fname = "sample_data.csv"

    # avoid adding data unless you know what you're doing
    firestore_add(df)
    firestore_add(fname)

    # if you want to specify a collection (default is 'data'), add a parameter:
    firestore_add(df, "new_collection")

def sample_read_data():
    # returns data as a dataframe
    df = firestore_read()

    # if you want to specify a collection (default is 'data'), add a parameter:
    df = firestore_read("new_collection")
