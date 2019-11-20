import pandas as pd
from sklearn.model_selection import train_test_split

# used in format_date() and get_past_dates()
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta

# Class built by DataSet_Builder
# Do not construct a DataSet object yourself
class DataSet():
    def __init__(self, df, xcols, ycols):
        self.df = df
        self.xcols = xcols
        self.ycols = ycols

        print(self.df)

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

    def run_ANN(self, params):
        # retrieve loss and prediction
        # question: what are we actually predicting?
        return "TODO"

    def run_OLS(self):
        # retrieve loss and prediction
        return "TODO"

    def run_Time(self):
        # will change name when I figure out what it's called
        self._averaged_sites()
        return "TODO"

    def _averaged_sites(self):
        # average the sites for each week
        self.df = self.df.groupby(['Date'])[self.df.loc[self.xcols[0]:].columns].mean().reset_index()


    def impute_inputs(self, future_date, time_step):
        rows = self._get_past_dates(future_date, time_step)
        print(rows)

        inputs = []
        for col in self.xcols:
            inputs.append(rows[col].mean())

        return inputs


    # given a date ('year-month-day' string) in the future, returns a list that contains data from past dates at the same time in the year
    # if timestep is weekly, round supplied date to the date at beginning of week
    def _get_past_dates(self, future_date, time_step):
        date_format_string = '%Y-%m-%d'

        # find the first year in the dataset
        first_date = str(self.df["Date"].iloc[0].date())
        print(first_date)
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
