import pandas as pd
from sklearn.model_selection import train_test_split

# Class built by DataSet_Builder
# Do not construct a DataSet object yourself
class DataSet():
    def __init__(self, df, xcols, ycols):
        self.df = df
        self.xcols = xcols
        self.ycols = ycols

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
