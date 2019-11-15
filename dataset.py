import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

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

    # Returns a cleaned dataframe with the columns of the features removed
    # features: an array of strings of feature names
    # ex. ['mpg', 'car name']
    # df (optional): dataframe, if not specified will use self.df
    def remove_features(self, features, df=None):
        if df is None:
            df = self.df
        self.cleaned_df = df.drop(features, axis=1)
        return self.cleaned_df

    def remove_outliers(self):
        # TODO
        self.df = self.df

    # Returns a tuple of the training and testing dataframes
    # train_prop: proportion of data to be trained
    # df (optional): dataframe, by default will use self.df
    # random (optional): whether to randomize the rows, by default will not randomize
    def split_data(self, train_prop, df=None, random=False):
        if df is None:
            df = self.df
        train, test = train_test_split(df, train_size=train_prop, shuffle=random)
        self.train = train
        self.test = test
        return train, test

    # Returns a tuple of the X and Y dataframes
    # y_colname: name of the Y column
    # df (optional): dataframe, by default will use self.df
    def split_X_Y(self, y_colname, df=None):
        if df is None:
            df = self.df
        self.df_X = df.drop(columns=[y_colname])
        self.df_Y = df[[y_colname]]
        return self.df_X, self.df_Y

    def get_colnames(self):
        return list(self.df.columns)

    # Returns the dataframe
    def get_df(self):
        return self.df

    def get_cleaned_df(self):
        return self.cleaned_df
