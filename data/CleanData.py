import keras
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

import warnings
warnings.filterwarnings("ignore")

def CleanData ():

    ColumnName = [ 'Battery1','Battery2','Date','Prec','RDC2',
                    'RDC20','RDC8','Salt2','Salt20','Salt8',
                    'SoilMois2','SoilMois20','Soil8','SNWD',
                    'SoilTemp2','SoilTemp20','SoilTemp8','TempAvg',
                    'TempMax','TempMin','TempObs','Time','WTEQ']

    CleanColumn = ['Date','Prec','RDC2','RDC20','RDC8','Salt2','Salt20',
                    'Salt8','SoilMois2','SoilMois20','Soil8','SNWD','SoilTemp2',
                    'SoilTemp20','SoilTemp8','TempAvg','TempMax','TempMin',
                    'TempObs','WTEQ']


    CleanColumnX = ['Prec','RDC2','RDC20','RDC8','Salt2','Salt20','Salt8',
                    'SoilMois2','SoilMois20','Soil8','SoilTemp2','SoilTemp20',
                    'SoilTemp8','TempAvg','TempMax','TempMin','TempObs','WTEQ']
    # #NOTE: CHANGE THE FILE DIRECTORY HERE FOR YOUR DATA
    data = pd.read_csv("merged.csv", names = ColumnName )

    dClean = data[CleanColumn]
    dCl = dClean.loc[1:, CleanColumnX]

    clf = IsolationForest()
    test = clf.fit_predict(dCl)

    count = 0

    indexes = []
    #getting the indexes to remove the outliers and count the outliers
    for i in range (len(test)):
        if test[i] == -1:
            indexes.append(i)
            count = count + 1
    print("This is the outliers in the isolation trees", count)

    dClean = dClean.drop(index = indexes)

    X_train, X_test, y_train, y_test = train_test_split(dClean[CleanColumnX], dClean['SNWD'], test_size=0.30, shuffle = False)

    scaler = StandardScaler()

    # print(X_train)
    # Separating out the features
    x = X_train.loc[1:, CleanColumnX].values
    x = scaler.fit_transform(x)
    xTest = scaler.transform(X_test.loc[1:, CleanColumnX].values)
    # print(xTest)
    pca = PCA(.95)
    principalComponents = pca.fit_transform(x)
    principalComponents2 = pca.transform(xTest)
    principalDf = pd.DataFrame(data = principalComponents)
    principalDf2 = pd.DataFrame(data = principalComponents2)

    y = y_train.loc[1:,].reset_index()
    y = y.drop(columns= ['index'])
    yTest = y_test.loc[1:,].reset_index()
    yTest = yTest.drop(columns=['index'])

    finalDf = pd.concat([principalDf, y], axis = 1)
    finalDf2 = pd.concat([principalDf2, yTest], axis = 1)

    print(finalDf)
    print(finalDf2)
    return principalDf, y, principalDf2, yTest

a, b, c, d = CleanData()
