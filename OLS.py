import numpy as np
import pandas as pd
import os
# import matplotlib.pyplot as matplot
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# #    Data = [x_train, y_train, x_test, y_test]
# def get_OLS_error (x_train, y_train, x_test, y_test):
#
#     # (x_train,y_train) in fit()
#     reg = LinearRegression().fit(x_train, y_train)
#
#     # Get prediction based on linear model above
#     y_pred = reg.predict(x_test)
#
#     # Get error of the prediction in comparsion to actual data from testing set
#     #    We define error = mean_squared_error
#     #          accuracy = 1 - mean_squared_error
#     MSE = mean_squared_error(y_test, y_pred)
#     return(MSE)

def run_OLS_predictor(data, x_train, y_train, x_test, y_test, pred_input, filename):

    # data = data.drop(0, axis=0)
    print(data)

    # We might want to observe the snow depth pattern throughout the year,
    #    In other word, we want a plot that have 'Date' on x-axis and snow-depth
    #        on y-axis
    #    In addition, we want to plot prediction and actual data of snow-depth
    #        both on the same graph for comparsion

    # In order for matplot to handle graphing Date, need to remove '/'s in 'Date'
    #    This might not be necessary, since, if matplot cannot interpret a data as
    #        number, it will just try to plot data in order of array indices
    # for index, row in data.iterrows():
    #     data.at[index, 'Date'] = data.at[index, 'Date'].replace('-', '')

    # Setting up data for graphing
    for index, row in y_test.iterrows():
        y_test.at[index, 0] = float(y_test.at[index, 0])
    for index, row in y_train.iterrows():
        y_test.at[index, 0] = float(y_test.at[index, 0])

    whole_x = pd.concat([x_train, x_test], ignore_index=True)
    whole_y = pd.concat([y_train, y_test], ignore_index=True)

    for index, row in whole_y.iterrows():
        whole_y.at[index, 0] = float(whole_y.at[index, 0])

    reg = LinearRegression().fit(x_train, y_train)

    # Get prediction based on linear model above
    #    Note that the features uses for this linear models are PCA, which cannot
    #        be used for graphing
    y_pred = reg.predict(x_test)
    y_pred_whole = reg.predict(whole_x)

    output_pred = reg.predict(np.asarray(pred_input).reshape(1, -1))
    print(output_pred)

    # fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
    # auc_keras = auc(fpr_keras, tpr_keras)

    MSE = mean_squared_error(y_test, y_pred)
    MSE_whole = mean_squared_error(whole_y, y_pred_whole)

    # Resulted MSE = 36.84 & MSE_whole = 30.26
    #    This means prediction using regression model have ~65 accuracy!
    #        (Accuracy defined as 1-MSE)
    # print(MSE)
    # print(MSE_whole)

    # Here is the "simple" plot that takes ~30 seconds to plot
    #    The blue lines represents actual snow depth, while red lines represents
    #        predictions
    #    If you lack patients, please read the attacted screenshot
    # plt.plot(data['Date'], data["SNWD.I-1 (in) "] ,color='blue')
    # plt.plot(data['Date'], y_pred_whole, color='red')
    print(data)
    print(data["Date"])
    print( data["SNWD.I-1 (in) "])
    plt.plot(data['Date'],  data["SNWD.I-1 (in) "]-y_pred_whole, color='green')
    plt.legend
    plt.title('Actual - Predicted')
    plt.ylabel('Inches')
    plt.xlabel('Date')
    plt.savefig(os.path.join(os.getcwd(), "results", filename))
    # print(output_pred[0][0])
    return MSE,output_pred[0][0]

