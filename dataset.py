import os
import pandas as pd

# Class that will read, clean, and bin data
class DataSet():
    def __init__(self, remove_outliers=False):
        # data is in a folder called data
        csvpath = os.path.join(os.getcwd(), "data", "merged.csv")
        pklpath = os.path.join(os.getcwd(), "data", "data.pkl")

        if os.path.exists(pklpath):
            self.df = pd.read_pickle(pklpath)
        else:
            self.df = pd.read_csv(csvpath, sep=",")
            self.df.to_pickle(pklpath)

        # self.orig_df = pd.read_csv(os.path.join(os.getcwd(), "data", data),
                                # sep="\s+", names=colnames, header=None)
        if remove_outliers:
            self.remove_outliers()

    def remove_outliers(self):
        # TODO
        self.df = self.df


    def split_data(self, train_prop):
        # TODO
        print("TODO")

    def get_colnames(self):
        return list(self.df.columns)
