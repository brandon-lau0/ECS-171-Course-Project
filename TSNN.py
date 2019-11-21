
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Flatten

# EXAMPLE OF HOW TO USE THIS CODE
# dataset = read_csv('test.csv', header=None)
# models = get_future_models(dataset, num_epochs=20, back_steps=2, future_steps=2)
# input_data = np.array([[[0,1,2,3,4,5], [1,2,3,4,5,6]]])
# input_data = input_data.reshape(1,2,6)
# preds = get_predictions(input_data, models)
# print(preds)

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def transform_3D(dataset, look_back, pred_forward):
    num_cols = dataset.shape[1]
    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # frame as supervised learning
    timescale = look_back + pred_forward
    reframed = series_to_supervised(values, look_back, pred_forward)
    vals = reframed.values
    y_data = vals[:,(look_back*num_cols):]
    x_data = vals[:,:(look_back*num_cols)]
    if pred_forward > 1:
        new_y_data = []
        for i in range(pred_forward):
            newCol = []
            for j in range(y_data.shape[0]):
                newCol.append(y_data[j,i*num_cols:(i+1)*num_cols])
            new_y_data.append(newCol)
        y_data = new_y_data
    x_data = x_data.reshape(x_data.shape[0], look_back, x_data.shape[1]//look_back)
    return (x_data, y_data)


def get_model(train_x, train_y, nodes_per_layer=5, hidden_layers=1, activation_func="relu", output_activation=None, loss_func="mean_squared_error", opt="SGD", num_epochs=1):
    model = Sequential()
    model.add(LSTM(nodes_per_layer, activation=activation_func, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Flatten())
    for i in range(0, hidden_layers-1):
        model.add(Dense(nodes_per_layer, activation=activation_func))

    model.add(Dense(len(train_y[0]), output_activation))
    model.compile(loss=loss_func, optimizer=opt, metrics=['accuracy'])
    # loss = []
    # error = []
    for i in range(num_epochs):
        history = model.fit(train_x, train_y, epochs=1)
        # loss.append(history.history[0])
        # error.append(history.history[1])
        model.reset_states()
    return model

def get_future_models(sequential_data, nodes_per_layer=5, hidden_layers=1, activation_func="relu", output_activation=None,
                      loss_func="mean_squared_error", opt="SGD", num_epochs=1, back_steps=1, future_steps=1):
    trainVec = transform_3D(sequential_data, back_steps, future_steps)
    train_x = trainVec[0]
    train_y = trainVec[1]

    if future_steps == 1:
        return get_model(train_x, train_y, nodes_per_layer, hidden_layers, activation_func, output_activation, loss_func, opt, num_epochs)

    models = []
    model = None

    for i in range(future_steps):
        print(train_x.shape)
        model = get_model(train_x, np.array(train_y[i]), nodes_per_layer, hidden_layers, activation_func, output_activation, loss_func, opt, num_epochs)
        models.append(model)
        new_x_data = []
        for j in range(train_x.shape[0]):
            new_train_x = np.append(train_x[j],train_y[i][j])
            new_train_x = new_train_x.reshape(train_x[j].shape[0]+1, train_x[j].shape[1])
            new_x_data.append(new_train_x)
        train_x = np.array(new_x_data)

    return models

def get_predictions(input_data, models):
    predictions = []

    for model in models:
        print(input_data)
        shape = input_data.shape
        pred = model.predict(input_data)
        predictions.append(pred)
        new_input_data = np.append(input_data, pred)
        new_input_data = new_input_data.reshape(shape[0], shape[1]+1, shape[2])
        input_data = new_input_data

    return predictions
