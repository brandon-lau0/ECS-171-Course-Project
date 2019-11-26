import pandas as pd
import numpy as np
import TSNN
from TSNN import *
import OLS
from OLS import *
import ANN
from ANN import *
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

    def run_ANN(self, params, filename):
        # retrieve loss and prediction for ANN
        fpath = os.path.join(os.getcwd(), "results", filename)
        print(self.pred_input)
        (mse, pred) = train(self.df, self.xcols, self.ycols, params, fpath, self.pred_input)
        return mse, pred

    def run_OLS(self, filename):
        # retrieve loss and prediction for OLS
        # set split of 0.3
        self.set_split(0.3)
        return run_OLS_predictor(self.get_data_with_date(), self.get_Xtrain(), self.get_Ytrain(), self.get_Xtest(), self.get_Ytest(), self.pred_input, filename)

    def run_TSNN(self, filename, params):
        # retrieve loss and predictions for TSNN
        self._averaged_sites()
        weeks_past = 4
        fullcols = self.xcols + self.ycols
        x = np.array(self.df.loc[:,fullcols].tail(weeks_past))
        input_data = np.expand_dims(x, axis=0)
        fpath = os.path.join(os.getcwd(), "results", filename)
        models = get_future_models(self.df.loc[:,fullcols], nodes_per_layer=params["numneuron"],
                hidden_layers=params["hiddenlayer"], activation_func=params["activation"],
                loss_func =params["loss"], opt=params["optimizer"],num_epochs=20, back_steps=weeks_past, future_steps=4, graph_path=fpath)
        error = models[1]
        input_data = input_data.reshape(1,weeks_past,len(input_data[0][0]))
        preds = get_predictions(input_data, models[0])
        return error, preds

    def _averaged_sites(self):
        # average the sites for each week
        self.df = self.df.groupby(['Date'])[self.df.loc[self.xcols[0]:].columns].mean().reset_index()


    def impute_inputs(self, future_date, time_step):
        # Makes a list of inputs for OLS and ANN for the days we want to predict
        rows = self._get_past_dates(future_date, time_step)

        inputs = []

        date_obj = datetime.strptime(future_date, '%Y-%m-%d')
        day_of_year = date_obj.timetuple().tm_yday


        # take the average of past dates
        for col in self.origxcols:
            inputs.append(rows[col].mean())

        # sin and cos of dates
        inputs.append(np.sin(2 * np.pi * day_of_year/365))
        inputs.append(np.cos(2 * np.pi * day_of_year/365))


        cols =  self.origxcols +  ["day_of_year_sin","day_of_year_cos"]

        # PCA was selected so do the PCA on these cols
        if "day_of_year_sin" not in self.xcols:
            self.pca = joblib.load("pca.save")
            singledf = pd.DataFrame(columns=cols)
            singledf.loc[0] = inputs

            vals = self.pca.transform(singledf)
            inputdf = pd.DataFrame(data=vals)
            inputs = list(inputdf.iloc[0])
            cols = list(inputdf.columns)

        # Scale data was selected so scale data
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
            # we want that day, the day before, and the day after
            yesterday = past_date - timedelta(days=1)
            tomorrow = past_date + timedelta(days=1)
            list_of_dates.append(yesterday.strftime(date_format_string))
            list_of_dates.append(past_date.strftime(date_format_string))
            list_of_dates.append(tomorrow.strftime(date_format_string))

            past_date = past_date + relativedelta(years = 1)

        past_dates = self.df[self.df['Date'].isin(list_of_dates)] # note: might be inefficient


        return past_dates
