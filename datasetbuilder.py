import os
import sys
import pandas as pd
import enum
import dataset
from dataset import *
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from statistics import mean

sys.path.insert(0, './data')
import sitedict
from sitedict import *

class Timestep(enum.Enum):
    daily = 1
    weekly = 2
    biweekly = 3

# Builder for DataSet class
class DataSet_Builder():
    def __init__(self):
        # defaults
        self.xcols = []
        self.ycols = []
        self.timestep = Timestep.daily

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
        # TODO
        self.df = self.df

    def set_xcols(self, xcols):
        self.xcols = xcols

    def set_ycols(self, ycols):
        self.ycols = ycols

    def set_timestep(self, timestep):
        # timestep is a string
        # acceptable strs: daily, weekly, biweekly
        self.timestep = Timestep[timestep]

    def use_pca(self):
        # TODO: if using pca, change df, xcols, and ycols appropriately
        self.df = self.df

    def scale_data(self):
        # TODO: scaling on data: min/max
        # what do we do for dates?
        # optional
        # self.df = self.df
        # DOESN'T WORK YET
        scaler = MinMaxScaler()
        scaler.fit(self.df)
        self.df = scaler.transform(self.df)

    def use_rect_radius(self, proportion):
        # proportion is a number 0-1 where 1 is all sites, 0 is none
        # 0.5 contains approximately half the sites centered at the CoM
        self.df = self._get_df_of_radius(proportion)

    def build_dataset(self):
        self._clean_df()
        self._use_timestep()
        return DataSet(self.df, self.xcols, self.ycols)

    def _clean_df(self):
        # TODO: drop off rows with blank values for x or y col
        # DOESN'T WORK YET
        for col in self.df.columns:
            self.df.drop(self.df.index[self.df[col] == -99.9], inplace = True)
        self.df = self.df

    def _use_timestep(self):
        # TODO: average or remove rows based on the timestep
        # Note that some days will already be lost either by outliers or
        #   if they were cleaned, and some weeks won't have any data left

        # use self.timestep to read the timestep

        self.df = self.df

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
