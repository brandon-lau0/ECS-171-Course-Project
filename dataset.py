import pandas as pd
import numpy as np
import TSNN
from TSNN import *
import OLS
from OLS import *
from sklearn.model_selection import train_test_split

# used in format_date() and get_past_dates()
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import joblib

# Class built by DataSet_Builder
# Do not construct a DataSet object yourself
class DataSet():
    def __init__(self, df, xcols, ycols, pca, origxcols, scaler, scale_data):
        self.df = df
        self.xcols = xcols
        self.ycols = ycols
        self.pca = pca
        self.origxcols = origxcols
        self.scaler = scaler
        self.scale_data = scale_data
        self.pred_input = []

        # set default split of 0.3
        self.set_split(0.3)


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

    def get_df(self):
        return self.df

    def get_data_with_date(self):
        fullcols = ["Date"] + self.xcols + self.ycols
        return self.df.loc[:,fullcols]

    def run_ANN(self, params):
        # retrieve loss and prediction
        # question: what are we actually predicting?
        return "TODO"

    def run_OLS(self, filename):
        # retrieve loss and prediction
        # set default split of 0.3
        self.set_split(0.3)
        return run_OLS_predictor(self.get_data_with_date(), self.get_Xtrain(), self.get_Ytrain(), self.get_Xtest(), self.get_Ytest(), self.pred_input, filename)

    def run_TSNN(self):
        # will change name when I figure out what it's called
        self._averaged_sites()
        # dataset = read_csv('test.csv', header=None)
        fullcols = self.xcols + self.ycols
        models = get_future_models(self.df.loc[:,fullcols], num_epochs=20, back_steps=2, future_steps=2)

        input_data = np.array([[list(range(0,len(self.pred_input))), self.pred_input]])
        input_data = input_data.reshape(1,2,len(self.pred_input))
        preds = get_predictions(input_data, models)
        print(preds)

    def _averaged_sites(self):
        # average the sites for each week
        self.df = self.df.groupby(['Date'])[self.df.loc[self.xcols[0]:].columns].mean().reset_index()


    def impute_inputs(self, future_date, time_step):





        rows = self._get_past_dates(future_date, time_step)

        inputs = []

        date_obj = datetime.strptime(future_date, '%Y-%m-%d')
        day_of_year = date_obj.timetuple().tm_yday



        for col in self.origxcols:
            inputs.append(rows[col].mean())

        inputs.append(np.sin(2 * np.pi * day_of_year/365))
        inputs.append(np.cos(2 * np.pi * day_of_year/365))


        cols =  self.origxcols +  ["day_of_year_sin","day_of_year_cos"]
        if "day_of_year_sin" not in self.xcols: # PCA
            self.pca = joblib.load("pca.save")
            singledf = pd.DataFrame(columns=cols)
            singledf.loc[0] = inputs

            vals = self.pca.transform(singledf)
            inputdf = pd.DataFrame(data=vals)
            inputs = list(inputdf.iloc[0])
            cols = list(inputdf.columns)



        if self.scale_data == 1:
            self.scaler = joblib.load("scaler.save")
            singledf = pd.DataFrame(columns = cols)
            singledf.loc[0] = inputs
            vals = self.scaler.transform(singledf)
            inputdf = pd.DataFrame(data=vals)

            # only need to scale last two cols
            inputs = inputs[:len(cols)-2] + list(inputdf.iloc[0])[len(cols)-2:]

        self.pred_input = inputs


    # given a date ('year-month-day' string) in the future, returns a list that contains data from past dates at the same time in the year
    # if timestep is weekly, round supplied date to the date at beginning of week
    def _get_past_dates(self, future_date, time_step):
        date_format_string = '%Y-%m-%d'

        # find the first year in the dataset
        first_date = str(self.df["Date"].iloc[0].date())
        first_date_object = datetime.strptime(first_date, date_format_string)
        first_year = first_date_object.year

        if time_step == 'daily':
            future_date_object = datetime.strptime(future_date, date_format_string).date()
        else: # if time_step == 'weekly'
            date_obj = datetime.strptime(future_date, date_format_string).date()
            future_date_object = date_obj - timedelta(days=date_obj.weekday())

        # start at the first year with the matching day
        past_date = datetime(first_year, future_date_object.month, future_date_object.day).date()

        list_of_dates = []
        while past_date < future_date_object:
            yesterday = past_date - timedelta(days=1)
            tomorrow = past_date + timedelta(days=1)
            list_of_dates.append(yesterday.strftime(date_format_string))
            list_of_dates.append(past_date.strftime(date_format_string))
            list_of_dates.append(tomorrow.strftime(date_format_string))

            past_date = past_date + relativedelta(years = 1)

        past_dates = self.df[self.df['Date'].isin(list_of_dates)] # note: might be inefficient

        # while past_date < future_date_object:
        #     past_dates.append(self.df.loc[df['Date'] == past_date.strftime(date_format_string)]) # FIXME how efficient is this retrieval??
        #     past_date = past_date + relativedelta(years = 1) # go to same date, 1 year in the future

        # if self.df['Date'] == past_date.strftime(date_format_string):
        #     past_dates.append(row)

        return past_dates
        # return []




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
