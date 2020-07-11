"""
Stock prediction models

This module has functions that generate predictive stock models
and their corresponding visualizations based on user-inputted
hyperparameter values. The current model list includes
    -> Moving Average
    -> Linear Regression
    -> K-Nearest Neighbors (KNN)
    -> Long Short Term Memory (LSTM)
    -> Autoregressive Integrated Moving Average (Auto-ARIMA)
"""

from datetime import datetime
from datetime import timezone
from datetime import date
import json

import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go

from .stock_preprocess import fig_layout


def historic_model(df):
    """
    Returns a plotly visualization of the historic stock price
    """

    fig = go.Figure(
        data=[go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"])],
        layout=fig_layout
    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


def moving_average_model(df, window=225, split=977, new_predictions=False, new_dates=None):
    """
    Generates a moving average model, its corresponding visualization,
    and a forecast based on user-tunable parameters to optimize
    the model
    """

    # Setting the dates to be the index
    df["Date"] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df.index = df["Date"]

    # Creating a dataframe with date and close price
    data = df[["Date", "Close"]].copy()

    # Appending forecast dates if any
    data = data.append(new_dates, ignore_index=True)

    # Splitting data into training and validation sets
    train = []
    valid = []

    # If forecast, use all the data to train the model, else the
    #   user-inputted train : valid ratio
    if(new_predictions):
        train = data[:len(df)]
        valid = data[len(df):]
    else:
        train = data[:split]
        valid = data[split:]

    predictions = []

    for i in range(0, valid.shape[0]):
        total = train['Close'][len(train) - window +
                               i:].sum() + sum(predictions)
        moving_average = total/window

        predictions.append(moving_average)

    # Create a predictions column in the validation set
    valid['Predictions'] = predictions

    rmse = 777.77  # filler error value

    # Calculate the error appropriately if it's not a forecast
    if(new_predictions == False):
        rmse = np.sqrt(
            np.mean(np.power((np.array(valid['Close']) - predictions), 2)))

    # Moving Average Plot
    fig_ma = go.Figure(layout=fig_layout)

    fig_ma.add_trace(go.Scatter(
        x=train["Date"],
        y=train["Close"],
        mode="lines",
        name="Training"
    ))

    # If forecast, plot the forecast values and not the validation data
    if(new_predictions):
        fig_ma.add_trace(go.Scatter(
            x=valid["Date"],
            y=valid["Predictions"],
            mode="lines",
            name="Forecast"
        ))
    else:
        fig_ma.add_trace(go.Scatter(
            x=df["Date"][split:],
            y=valid["Close"],
            mode="lines",
            name="Validation"
        ))

    if(new_predictions == False):
        fig_ma.add_trace(go.Scatter(
            x=df["Date"][split:],
            y=valid["Predictions"],
            mode="lines",
            name="Predictions"
        ))

    graphJSON = json.dumps(fig_ma, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON, round(rmse, 2)


def linear_regression_model(df, split=977):
    """
    Generates a linear regression model, its corresponding visualization,
    and a forecast based on the selected train : test split ratio
    """

    # Setting the dates to be the index
    df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df.index = df['Date']

    # Creating a dataframe with the dates and close prices
    intermediate = df.sort_index(ascending=True, axis=0)
    data = intermediate[["Date", "Close"]].copy()

    # Changing the date to ordinal format so it can be passed into
    #   the linear model
    data["Date"] = pd.to_datetime(data["Date"])
    data["Date"] = data["Date"].map(datetime.toordinal)

    # Splitting the data into training and validation sets
    train = data[:split]
    valid = data[split:]

    predictions = []

    x_train = train.drop("Close", axis=1)
    y_train = train["Close"]
    x_valid = valid.drop("Close", axis=1)
    y_valid = valid["Close"]

    # Defining the linear model and making predictions with it
    from sklearn.linear_model import LinearRegression

    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    predictions = linear_model.predict(x_valid)

    # Error calculation
    rmse = np.sqrt(
        np.mean(np.power((np.array(y_valid) - np.array(predictions)), 2)))

    valid['Predictions'] = 0
    valid['Predictions'] = predictions
    valid.index = data[split:].index
    train.index = data[:split].index

    # Linear Regression plot
    fig_lm = go.Figure(layout=fig_layout)

    fig_lm.add_trace(go.Scatter(
        x=df["Date"],
        y=train["Close"],
        mode="lines",
        name="Training"
    ))

    fig_lm.add_trace(go.Scatter(
        x=df["Date"][split:],
        y=valid["Close"],
        mode="lines",
        name="Validation"
    ))

    fig_lm.add_trace(go.Scatter(
        x=df["Date"][split:],
        y=valid["Predictions"],
        mode="lines",
        name="Predictions"
    ))

    graphJSON = json.dumps(fig_lm, cls=plotly.utils.PlotlyJSONEncoder)

    return linear_model, fig_lm, graphJSON, round(rmse, 2)


def knn_model(df, split=977, n_neighbors=2, weights="distance", p=2,
              new_predictions=False, ordinal_prediction_dates=None,
              original_prediction_dates=None):
    """
    Generates a K-Nearest Neighbors model, its corresponding visualization,
    and a forecast based on user-tunable hyperparameters
    """

    # Setting the dates to be the index
    df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df.index = df['Date']

    # Creating a dataframe with the dates and close prices
    intermediate = df.sort_index(ascending=True, axis=0)
    data = intermediate[["Date", "Close"]].copy()

    # Changing the date to ordinal format so it can be passed into
    #   the linear model
    data["Date"] = pd.to_datetime(data["Date"])
    data["Date"] = data["Date"].map(datetime.toordinal)

    # Splitting the data into training and validation sets
    train = data[:split]
    valid = data[split:]

    predictions = []

    x_train = train.drop("Close", axis=1)
    y_train = train["Close"]
    x_valid = valid.drop("Close", axis=1)

    # If forecast, add the forecast dates to the validation set
    if(new_predictions == True):
        x_valid = x_valid.append(ordinal_prediction_dates, ignore_index=True)

    y_valid = valid["Close"]

    # Defining the KNN model and making predictions with it
    from sklearn import neighbors
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))

    # Scaling data
    x_train_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train_scaled)
    x_valid_scaled = scaler.fit_transform(x_valid)
    x_valid = pd.DataFrame(x_valid_scaled)

    # Used gridsearch on the follwing parameter values to find optimal
    #   parameters for the initial model

    # params = {
    #     "n_neighbors": [2, 3, 4, 5, 6, 7, 8, 9],
    #     "weights": ["uniform", "distance"],
    #     "p": [2, 3, 4, 5]
    # }

    knn_model = neighbors.KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        p=p
    )

    # knn_model = GridSearchCV(knn, params, cv=5)

    knn_model.fit(x_train, y_train)

    # Gridsearch results for original model were -
    #     n_neighbors = 2
    #     weights = distance
    #     minkowski metric (p) = 2

    predictions = knn_model.predict(x_valid)
    rmse = 777.77  # filler error value

    # Calculate the error if forecast
    if(new_predictions == False):
        rmse = np.sqrt(
            np.mean(np.power((np.array(y_valid) - np.array(predictions)), 2)))

    # K-Nearest Neighbors plot
    fig_knn = go.Figure(layout=fig_layout)

    fig_knn.add_trace(go.Scatter(
        x=df["Date"],
        y=train["Close"],
        mode="lines",
        name="Training"
    ))

    fig_knn.add_trace(go.Scatter(
        x=df["Date"][split:],
        y=valid["Close"],
        mode="lines",
        name="Validation"
    ))

    # If forecast, append the forecast dates
    if(new_predictions == True):
        all_prediction_dates = df["Date"][split:]
        all_prediction_dates = all_prediction_dates.append(
            original_prediction_dates["Date"], ignore_index=True)

        fig_knn.add_trace(go.Scatter(
            x=all_prediction_dates,
            y=predictions,
            mode="lines",
            name="Predictions"
        ))
    else:
        fig_knn.add_trace(go.Scatter(
            x=df["Date"][split:],
            y=predictions,
            mode="lines",
            name="Predictions"
        ))

    graphJSON = json.dumps(fig_knn, cls=plotly.utils.PlotlyJSONEncoder)

    return knn_model, fig_knn, graphJSON, round(rmse, 2)


def auto_arima_model(df, split=977, start_p=1, max_p=3, start_q=1, max_q=3,
                     d=1, D=1, new_predictions=False, new_dates=None):
    """
    Generates an Auto-ARIMA model, its corresponding visualization, and
    forecast based on user-tunable hyperparameters
    """

    # Creating a dataframe with the dates and closing price
    df = df.sort_index(ascending=True, axis=0)
    data = df[["Date", "Close"]].copy()

    # If forecast, append forecast dates to data
    if(new_predictions):
        data = data.append(new_dates, ignore_index=True)

    # Splitting data into training and validation sets
    train = []
    valid = []

    # If forecast, the entirety of data is trained, else the split
    #   is determined by the selected train : valid ratio
    if(new_predictions):
        train = data[:len(df)]
        valid = data[len(df):]
    else:
        train = data[:split]
        valid = data[split:]

    training = train['Close']
    validation = valid['Close']

    # Defining the arima model and making predictions with it
    import pmdarima as pm

    arima_model = pm.arima.auto_arima(
        training, start_p=start_p, max_p=max_p,
        start_q=start_q, max_q=max_q, m=12, start_P=0,
        seasonal=True, d=d, D=D, trace=True,
        error_action='ignore', suppress_warnings=True)

    arima_model.fit(training)

    arima_forecast = arima_model.predict(n_periods=len(valid))
    arima_forecast = pd.DataFrame(
        arima_forecast, index=valid.index, columns=['Prediction'])
    rmse = 777.77  # filler error value

    # If forecast, calculate the error value
    if(new_predictions == False):
        rmse = np.sqrt(np.mean(
            np.power((np.array(valid['Close']) -
                      np.array(arima_forecast['Prediction'])), 2)))

    # Auto-ARIMA plot
    fig_arima = go.Figure(layout=fig_layout)

    fig_arima.add_trace(go.Scatter(
        x=train["Date"],
        y=train["Close"],
        mode="lines",
        name="Training"
    ))

    if(new_predictions):
        fig_arima.add_trace(go.Scatter(
            x=valid["Date"],
            y=arima_forecast["Prediction"],
            mode="lines",
            name="Predictions"
        ))
    else:
        fig_arima.add_trace(go.Scatter(
            x=df["Date"][split:],
            y=valid["Close"],
            mode="lines",
            name="Validation"
        ))

    if(new_predictions == False):
        fig_arima.add_trace(go.Scatter(
            x=df["Date"][split:],
            y=arima_forecast["Prediction"],
            mode="lines",
            name="Predictions"
        ))

    graphJSON = json.dumps(fig_arima, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON, round(rmse, 2)


def lstm_model(df, split=977, units=50, epochs=1, new_predictions=False,
               original_predictions=None):
    """
    Generates an LSTM model, its corresponding visualization, and forecast
    based on user-tunable hyperparameters
    """

    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM

    # Creating dataframe with the dates and close prices
    df = df.sort_index(ascending=True, axis=0)
    data = df[["Date", "Close"]].copy()

    # If forecast, add forecast dates to the dataset
    if(new_predictions):
        data = data.append(original_predictions, ignore_index=True)

    # # Set the index to be the dates
    data.index = data.Date

    data.drop('Date', axis=1, inplace=True)

    # Splitting the dataset into training and validation sets
    dataset = data.values  # 2D array containing values in singletons

    train = []
    valid = []

    # If forecast, train on the entire dataset, else train based on the
    #   train : valid ratio
    if(new_predictions):
        train = dataset[:len(df), :]
        valid = dataset[len(df):, :]
    else:
        train = dataset[0:split, :]
        valid = dataset[split:, :]

    # Scaling the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Building out the training sets in the right dimensions
    x_train, y_train = [], []

    for i in range(77, len(train)):  # arbitrary number of training data points
        x_train.append(scaled_data[i-77:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Defining the LSTM model and fitting it
    model = Sequential()
    model.add(LSTM(
        units=units,
        return_sequences=True,
        input_shape=(x_train.shape[1], 1))
    )
    model.add(LSTM(units=units))
    model.add(Dense(1))  # Dimensionality of the output
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=2)

    # Making predictions starting with the last 77 training points
    inputs = data[len(data) - len(valid) - 77:].values
    inputs = inputs.reshape(-1, 1)  # 2D array
    inputs = scaler.transform(inputs)
    actual_inputs = inputs[0:77]  # 2D array
    closing_price = []
    X_test = []

    # Moving averages with a window of 77
    for i in range(77, inputs.shape[0]):
        X_test = []

        X_test.append(actual_inputs[i-77:i, 0])

        X_test = np.array(X_test)  # 2D array
        X_test = np.reshape(
            X_test, (X_test.shape[0], X_test.shape[1], 1))  # 3D array
        predicted_price = model.predict(X_test[0:1])  # 2D array
        actual_inputs = np.vstack([actual_inputs, predicted_price[0]])
        predicted_price = scaler.inverse_transform(predicted_price)

        closing_price.append(predicted_price[0, 0])

    rmse = 777.77  # filler error value

    # Calculate error value if not forecast
    if(new_predictions == False):
        rmse = np.sqrt(np.mean(np.power((valid - closing_price), 2)))

    # Re-assign training and validation sets based on forecast requirement
    if(new_predictions):
        train = data[:len(df)]
        valid = data[len(df):]
    else:
        train = data[:split]
        valid = data[split:]

    valid['Predictions'] = closing_price

    # LSTM plot
    fig_lstm = go.Figure(layout=fig_layout)

    fig_lstm.add_trace(go.Scatter(
        x=df["Date"],
        y=train["Close"],
        mode="lines",
        name="Training"
    ))

    if(new_predictions):
        fig_lstm.add_trace(go.Scatter(
            x=data.index[len(df) - 77:],
            y=valid["Predictions"],
            mode="lines",
            name="Forecast"
        ))
    else:
        fig_lstm.add_trace(go.Scatter(
            x=df["Date"][split:],
            y=valid["Close"],
            mode="lines",
            name="Validation"
        ))

    if(new_predictions == False):
        fig_lstm.add_trace(go.Scatter(
            x=df["Date"][split:],
            y=valid["Predictions"],
            mode="lines",
            name="Predictions"
        ))

    graphJSON = json.dumps(fig_lstm, cls=plotly.utils.PlotlyJSONEncoder)

    return model, fig_lstm, graphJSON, round(rmse, 2)
