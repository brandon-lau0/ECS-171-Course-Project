import numpy as np
import pandas as pd
import os
# import matplotlib.pyplot as matplot
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import PolynomialFeatures

def run_OLS_predictor(data, x_train, y_train, x_test, y_test, pred_input, filename):

    # data = data.drop(0, axis=0)
    print(data)

    # Setting up data for graphing
    for index, row in y_train.iterrows():
        y_train.at[index, 'SNWD.I-1 (in) '] = float(y_train.at[index, 'SNWD.I-1 (in) '])
    for index, row in y_test.iterrows():
        y_test.at[index, 'SNWD.I-1 (in) '] = float(y_test.at[index, 'SNWD.I-1 (in) '])

    whole_x = pd.concat([x_train, x_test], ignore_index=True)
    whole_y = pd.concat([y_train, y_test], ignore_index=True)

    polynomial_features = PolynomialFeatures(degree=1)
    x_poly = polynomial_features.fit_transform(x_train)
    x_test = polynomial_features.transform(x_test)
    whole_x = polynomial_features.transform(whole_x)
    # y_poly = polynomial_features.transform(y_train)

    reg = LinearRegression().fit(x_poly, y_train)
    # Get prediction based on linear model above
    #    Note that the features uses for this linear models are PCA, which cannot
    #        be used for graphing
    y_pred = reg.predict(x_test)
    y_pred_whole = reg.predict(whole_x)

    # print(y_pred)
    pred_input = polynomial_features.transform([pred_input])

    output_pred = reg.predict(np.asarray(pred_input).reshape(1, -1))
    print(output_pred)

    # fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
    # auc_keras = auc(fpr_keras, tpr_keras)
    MSE = mean_squared_error(y_test, y_pred)

    print(MSE)

    y_actual = np.array(data["SNWD.I-1 (in) "])
    y_pred_whole = y_pred_whole.flatten()
    date = data['Date']
    plt.scatter(date, y_actual-y_pred_whole, color='green')
    plt.legend
    plt.title('Actual - Predicted')
    plt.ylabel('Inches')
    plt.xlabel('Date')
    plt.savefig(os.path.join(os.getcwd(), "results", filename))
    print(output_pred[0][0])
    return MSE,output_pred[0][0]
