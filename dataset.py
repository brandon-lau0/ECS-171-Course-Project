import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from statistics import mean

sys.path.insert(0, './data')
import sitedict
from sitedict import *

# Builder for DataSet class
class DataSet_Builder():
    def __init__(self):
        self.xcols = []
        self.ycols = []

        # data is in a folder called data
        csvpath = os.path.join(os.getcwd(), "data", "merged.csv")
        pklpath = os.path.join(os.getcwd(), "data", "data.pkl")

        if os.path.exists(pklpath):
            self.df = pd.read_pickle(pklpath)
        else:
            self.df = pd.read_csv(csvpath, sep=",")
            self.df.to_pickle(pklpath)

    def remove_outliers(self):
        #optional
        self.df = self.df

    def set_xcols(self, xcols):
        self.xcols = xcols

    def set_ycols(self, ycols):
        self.ycols = ycols

    def scale_data(self):
        # optional
        self.df = self.df

    def set_locality(self, proportion):
        # proportion is a number 0-1 where 1 is all sites, 0 is none
        # 0.5 contains approximately half the sites centered at the CoM
        self.df = _get_df_of_radius(proportion)

    def build_dataset(self):
        return _DataSet(self.df, self.xcols, self.ycols)


    # Returns a df of sites within a specified radius proportion
    def _get_df_of_radius(self, proportion, df=None):
        # Notes on proportion: this is not the proportion of sites to keep
        # Proportion is (proportion of area to keep around center)^2
        #   That means a proportion of 0.5 keeps a quarter of the total area

        # The main problem with this method is you lose the min & max for both
        # latitude and longitude until the proportion is exactly 1

        # proportion        % remaining
        # 0.1               8.8
        # 0.25              13.1
        # 0.5               44.8
        # 0.75              71.4
        # 0.9               78.6
        # 0.95 - 0.9999     89.2

        # center of mass of points
        long_center, lat_center = self._get_center_coordinates()

        # min and max of latitude and longitude
        long_min, long_max = self._get_min_max("long")
        lat_min, lat_max = self._get_min_max("lat")

        # acceptable range is a box that grows with proportion away from the
        # center of mass
        long_range = [long_center - (long_center - long_min) * proportion,
                        long_center + (long_max - long_center) * proportion]
        lat_range = [lat_center - (lat_center - lat_min) * proportion,
                        lat_center + (lat_max - lat_center) * proportion]

        # sites within the acceptable range
        valid_sites = []
        for site in SITEDICT.keys():
            if (float(SITEDICT[site]["lat"]) >= lat_range[0] and
                float(SITEDICT[site]["lat"]) <= lat_range[1] and
                float(SITEDICT[site]["long"]) >= long_range[0] and
                float(SITEDICT[site]["long"]) <= long_range[1]):
                valid_sites.append(site)

        if df is None:
            df = self.df

        return df[df["Site Id"].isin(valid_sites)]


    def _get_center_coordinates(self):
        sites = list(SITEDICT.keys())
        longitude_center = mean(float(SITEDICT[site]["long"]) for site in sites)
        latitude_center = mean(float(SITEDICT[site]["lat"]) for site in sites)
        return longitude_center, latitude_center

    def _get_min_max(self, key_str):
        minv = min(float(SITEDICT[site][key_str]) for site in SITEDICT.keys())
        maxv = max(float(SITEDICT[site][key_str]) for site in SITEDICT.keys())
        return minv, maxv


    # probably deleting

    # # Returns a cleaned dataframe with the columns of the features removed
    # # features: an array of strings of feature names
    # # ex. ['mpg', 'car name']
    # # df (optional): dataframe, if not specified will use self.df
    # def remove_features(self, features, df=None):
    #     if df is None:
    #         df = self.df
    #     self.cleaned_df = df.drop(features, axis=1)
    #     return self.cleaned_df

# Class that will read, clean, and bin data
class _DataSet():
    def __init__(self, df, xcols, ycols):
        self.df = df
        self.xcols = xcols
        self.ycols = ycols


    def set_split(self, test_prop):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                self.df[self.xcols], self.df[self.ycols],
                test_size=test_prop, shuffle = True)

    def get_Xtrain(self):
        return self.X_train

    def get_Xtest(self):
        return self.X_test

    def get_Ytrain(self):
        return self.Y_train

    def get_Ytest(self):
        return self.Y_test

    def get_colnames(self):
        return list(self.df.columns)

    # Returns the dataframe
    def get_df(self):
        return self.df



    # # Returns a tuple of the training and testing dataframes
    # # train_prop: proportion of data to be trained
    # # df (optional): dataframe, by default will use self.df
    # # random (optional): whether to randomize the rows, by default will not randomize
    # def split_data(self, train_prop, df=None, random=False):
    #     if df is None:
    #         df = self.df
    #     train, test = train_test_split(df, train_size=train_prop, shuffle=random)
    #     self.train = train
    #     self.test = test
    #     return train, test
    #
    # # Returns a tuple of the X and Y dataframes
    # # y_colname: name of the Y column
    # # df (optional): dataframe, by default will use self.df
    # def split_X_Y(self, y_colname, df=None):
    #     if df is None:
    #         df = self.df
    #     self.df_X = df.drop(columns=[y_colname])
    #     self.df_Y = df[[y_colname]]
    #     return self.df_X, self.df_Y


    # def get_cleaned_df(self):
    #     return self.cleaned_df
