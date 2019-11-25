
import numpy as np
from numpy import concatenate
from matplotlib import pyplot as plt
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
# error = models[1]
# input_data = np.array([[[0,1,2,3,4,5], [1,2,3,4,5,6]]])
# input_data = input_data.reshape(1,2,6)
# preds = get_predictions(input_data, models[0])
# print(preds)

def series_to_supervised(data, look_back=1, look_forward=1, dropnan=True):
    '''
    Modifies a series dataset to supervised learning, with input and output data

    Parameters:
    data (NumPy DataFrame or 2 dimensional list) - series dataset to modify
    look_back (int) - number of time steps back to set the data
    look_forward (int) - number of time steps forward to set the data
    dropnan (bool) - boolean to indicate whether or not to drop NaN values

    Returns:
    The dataset formatted for supervised learning
    '''
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()

    for i in range(look_back, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, look_forward):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # Concatenate the different columns
    total_data = concat(cols, axis=1)
    total_data.columns = names
    # drop rows with NaN values
    if dropnan:
        total_data.dropna(inplace=True)
    return total_data

def transform_3D(dataset, look_back, pred_forward):
    '''
    Transforms a time series dataset to a 3D DataFrame, where the depth dimension
    is the number of time steps.

    Parameters:
    dataset (NumPy DataFrame or 2 dimensional list) - series dataset to modify
    look_back (int) - number of time steps back to set the data
    pred_forward (int) - number of time steps forward to set the data

    Returns:
    A 3D DataFrame for the x and y values with samples, features, and time steps
    as the dimensions
    '''
    num_cols = dataset.shape[1]
    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # frame as supervised learning
    timescale = look_back + pred_forward
    reframed = series_to_supervised(values, look_back, pred_forward)
    vals = reframed.values
    # selecting the x data and the y data
    y_data = vals[:,(look_back*num_cols):]
    x_data = vals[:,:(look_back*num_cols)]

    #rearranging the y data to the right dimensions
    if pred_forward > 1:
        new_y_data = []
        for i in range(pred_forward):
            newCol = []
            for j in range(y_data.shape[0]):
                newCol.append(y_data[j,i*num_cols:(i+1)*num_cols])
            new_y_data.append(newCol)
        y_data = new_y_data
        # rearranging the x data to the right dimensions
    x_data = x_data.reshape(x_data.shape[0], look_back, x_data.shape[1]//look_back)
    return (x_data, y_data)


def get_model(train_x, train_y, nodes_per_layer=5, hidden_layers=1, activation_func="relu", output_activation=None, loss_func="mean_squared_error", opt="SGD", num_epochs=1, graph_path="metric_graph.jpg"):
    '''
    Creates and returns a model with the given parameters, input data, and output data.
    Also returns the history of the training error.
    '''

    model = Sequential()

    # Building the layers of the model
    model.add(LSTM(nodes_per_layer, activation=activation_func, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Flatten()) # Flattening the model to go from 2D to 1D for easier output reading
    for i in range(0, hidden_layers-1):
        model.add(Dense(nodes_per_layer, activation=activation_func))
    model.add(Dense(len(train_y[0]), output_activation))

    # Compiling the model according to the parameters
    model.compile(loss=loss_func, optimizer=opt, metrics=['mean_absolute_percentage_error'])
    history_list = [] # Generating the history for graphing
    for i in range(num_epochs):
        history_list.append(model.fit(train_x, train_y, epochs=1,
        validation_split=0.33))
        model.reset_states()

    x_vals = list(range(num_epochs))
    history = [[],[],[],[]]
    for hist in history_list:
        history[0] += hist.history["loss"]
        history[1] += hist.history["mean_absolute_percentage_error"]
        history[2] += hist.history["val_loss"]
        history[3] += hist.history["val_mean_absolute_percentage_error"]

    # Plotting the history of the error
    plt.figure()
    plt.plot(x_vals, history[0], label="Training")
    plt.plot(x_vals, history[2], label="Testing")
    plt.title('Error')
    plt.legend()
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Epoch")
    plt.xticks(np.arange(0,num_epochs, step=1))
#    axs[1].plot(x_vals, history[0], label="Training")
#    axs[1].plot(x_vals, history[2], label="Testing")
#    axs[1].set_title('Error')
#    axs[1].legend()
#    axs[1].set(ylabel="Error",xlabel="Epoch")
    plt.savefig(graph_path)

    error = []

    for e in np.nditer(1 - history[2][len(history[0])-1]):
        error.append(str(e))

    return (model, error)

def get_future_models(sequential_data, nodes_per_layer=5, hidden_layers=1, activation_func="relu", output_activation=None,
                      loss_func="mean_squared_error", opt="SGD", num_epochs=1, back_steps=1, future_steps=1, graph_path="oops.png"):
    '''
    Gets a number of models equal to the number of weeks ahead to predict.
    The each model is trained on all the successive weeks, up to the back_steps parameter.
    '''
    # Transforming the data and splitting it into y data and x data
    trainVec = transform_3D(sequential_data, back_steps, future_steps)
    train_x = trainVec[0]
    train_y = trainVec[1]

    # Only need one model if future_steps = 1
    if future_steps == 1:
        return get_model(train_x, train_y, nodes_per_layer, hidden_layers, activation_func, output_activation, loss_func, opt, num_epochs, graph_path)

    # initializing variables
    models = []
    model = None

    # getting each model, and appending the real value of each new week for successive models
    for i in range(future_steps):
        print(train_x.shape)
        model = get_model(train_x, np.array(train_y[i]), nodes_per_layer, hidden_layers, activation_func, output_activation, loss_func, opt, num_epochs, graph_path)
        models.append(model)
        new_x_data = []
        for j in range(train_x.shape[0]):
            new_train_x = np.append(train_x[j],train_y[i][j])
            new_train_x = new_train_x.reshape(train_x[j].shape[0]+1, train_x[j].shape[1])
            new_x_data.append(new_train_x)
        train_x = np.array(new_x_data)

    # changing the format to be easier to parse for history or the actual model
    ret_models = [[],[]]
    for model in models:
        ret_models[0].append(model[0])
        ret_models[1].append(model[1])

    return ret_models

def get_predictions(input_data, models):
    '''
    Gets the predicted value for the input data, for a number of timesteps ahead
    equal to the number of models passed in as a parameter

    To predict more than one time step in the future, a prediction is made for
    one time step in the future, and that prediction is used as an input into
    the next week.
    '''
    predictions = []

    # getting the predictions
    for model in models:
        print(input_data)
        shape = input_data.shape
        # getting predicted values
        pred = model.predict(input_data)
        predictions.append(pred)
        # appending the predicted values to the input data for the next step
        new_input_data = np.append(input_data, pred)
        new_input_data = new_input_data.reshape(shape[0], shape[1]+1, shape[2])
        input_data = new_input_data

    # Casting the data into strings for easier printing and storage
    important_preds = []
    n = len(predictions)
    for pred in predictions:
        important_preds.append(str(pred[0][n-1]))

    return important_preds
