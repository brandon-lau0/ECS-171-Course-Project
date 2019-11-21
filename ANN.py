import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras import optimizers

import warnings
warnings.filterwarnings("ignore")

filePath = "data/merged.csv"

def create_ANN_model(input_dimension, num_hidden_layers, num_neurons, activation_func, optimizer, loss):
  '''
  Creates and returns the compiled ANN model object.

  Parameters:
  input_dimension (int) - number of inputs the ANN model takes in
  num_hidden_layers (int) - number of hidden layers for the ANN model
  num_neurons (int) - number of neurons for each of the hidden layers
  activation_func (str) - the activation function used in the layers
  optimizer (str) - the optimizer function (i.e. 'sgd')
  loss (str) - the loss function (i.e. 'acc')

  Returns:
  The compiled ANN model object
  '''

  model = Sequential()

  for i in range(0, num_hidden_layers):
    if i == 0:
      model.add(Dense(num_neurons, input_dim=input_dimension, activation=activation_func, bias_initializer='ones'))
    else:
      model.add(Dense(num_neurons, activation=activation_func, bias_initializer='ones'))

  model.add(Dense(1, activation='relu', bias_initializer='ones'))

  # sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
  # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  # model.compile(optimizer=sgd, loss=loss, metrics=[metric])

  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  return model

def train(dataframe, XCols, YCol, params, filePathToSaveGraph, pred_input):
  '''
  Trains and model using provided data and outputs the accuracy graph.

  Parameters:
  dataframe - dataframe containing the data read from CSV file
  XCols (list of strings) - contains the relevant feature names for the X values
  YCol (list of strings) - contains the feature name of the Y value you're predicting

  params (dict) - contains the hyperparameters
  filePathToSaveGraph (str) - file path of where to save the graph
  Returns:
  Tuple containing: (model error, predicted snow depth)
  '''
  input_dimension = len(XCols)
  num_hidden_layers = params["hiddenlayer"]
  num_neurons = params["numneuron"]
  activation_func = params["activation"]
  optimizer = params["optimizer"]
  loss = params["loss"]
  model = create_ANN_model(input_dimension, num_hidden_layers, num_neurons, activation_func, optimizer, loss)
  # if params["hiddenlayer"] == 1:
  #     history = model.fit(dataframe[XCols], dataframe[YCol], epochs=20, batch_size=1, validation_split=0.34)
  # else:
  history = model.fit(dataframe[XCols], dataframe[YCol], epochs=10, batch_size=1, validation_split=0.34)
  print(history)
  print(pred_input)
  result = model.predict(np.array([pred_input]))
  plt.figure()
  plt.plot(1 - np.array(history.history['accuracy']))
  plt.plot(1 - np.array(history.history['val_accuracy']))
  plt.title('Model Error for Training vs. Testing Data')
  plt.ylabel('error')
  plt.xlabel('epoch')
  plt.legend(['training', 'testing'], loc='upper right')
  plt.savefig(filePathToSaveGraph)
  print(list(1 - np.array(history.history['accuracy'])))
  print(result[0])
  error = []
  for e in np.nditer(1 - np.array(history.history['accuracy'])):
    error.append(str(e))
  return error, str(result[0])

