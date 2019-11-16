import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore")

filePath = "data/merged.csv"

def create_ANN_model(input_dimension, num_layers, num_neurons, output_dimension, activation_func, optimizer, loss):
  '''
  Creates the ANN model

  Parameters:
  input_dimension (int) - number of inputs the ANN model takes in
  num_layers (int) - number of layers for the ANN model
  num_neurons (int) - number of neurons for each of the hidden layers
  output_dimension (int) - number of outputs the ANN model outputs
  activation_func (str) - the activation function used in the layers
  optimizer (str) - the optimizer function (i.e. 'sgd')
  loss (str) - the loss function (i.e. 'acc')

  Returns:
  The compiled ANN model object
  '''

  model = Sequential()

  for i in range(0, num_layers):
    if i == 0:
      model.add(Dense(num_neurons, input_dim=input_dimension, activation=activation_func, bias_initializer='ones'))
    else:
      model.add(Dense(num_neurons, activation=activation_func, bias_initializer='ones'))

  model.add(Dense(output_dimension, activation=activation_func, bias_initializer='ones'))

  model.compile(optimizer=optimizer, loss=loss, metrics=[loss])

  return model
