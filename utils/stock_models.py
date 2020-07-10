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

import pandas as pd
import numpy as np

import plotly
import plotly.graph_objects as go
import json

from datetime import datetime
from datetime import timezone
from datetime import date


def historic_model(df):
    """
    Returns a plotly visualization of the historic stock price
    """

    data = [go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"])]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


def moving_average_model(df, window=225, split=977, new_predictions=False, new_dates=None):
    """
    Generates a moving average model, its corresponding visualization,
    and a future forecast based on user-tunable parameters
    """

    # Setting the date to be the index
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
        total = train['Close'][len(train)-window+i:].sum() + sum(predictions)
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
    fig_ma = go.Figure()

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
    df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df.index = df['Date']

    # creating dataframe with date and the target variable
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

    for i in range(0, len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]

    new_data["Date"] = pd.to_datetime(new_data["Date"])
    new_data["Date"] = new_data["Date"].map(datetime.toordinal)

    train = new_data[:split]
    valid = new_data[split:]
    preds = []

    x_train = train.drop("Close", axis=1)
    y_train = train["Close"]
    x_valid = valid.drop("Close", axis=1)
    y_valid = valid["Close"]

    from sklearn.linear_model import LinearRegression

    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    preds = linear_model.predict(x_valid)
    rmse = np.sqrt(np.mean(np.power((np.array(y_valid) - np.array(preds)), 2)))

    valid['Predictions'] = 0
    valid['Predictions'] = preds
    valid.index = new_data[split:].index
    train.index = new_data[:split].index
    fig_lm = go.Figure()

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


def knn_model(df, split=977, n_neighbors=2, weights="distance", p=2, new_predictions=False, ordinal_prediction_dates=None, original_prediction_dates=None):
    df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df.index = df['Date']

    # creating dataframe with date and the target variable
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

    for i in range(0, len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]

    new_data["Date"] = pd.to_datetime(new_data["Date"])
    new_data["Date"] = new_data["Date"].map(datetime.toordinal)
    train = new_data[:split]
    valid = new_data[split:]
    preds = []
    x_train = train.drop("Close", axis=1)
    y_train = train["Close"]
    x_valid = valid.drop("Close", axis=1)

    if(new_predictions == True):
        x_valid = x_valid.append(ordinal_prediction_dates, ignore_index=True)

    y_valid = valid["Close"]

    from sklearn import neighbors
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))

    # scaling data
    x_train_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train_scaled)
    x_valid_scaled = scaler.fit_transform(x_valid)
    x_valid = pd.DataFrame(x_valid_scaled)
    print("x_valid before predict")
    print(x_valid)

    # using gridsearch to find the best parameter for initial model generation

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

    # fit the model and make predictions
    knn_model.fit(x_train, y_train)

    # gridsearch results for original model were -
    #     n_neighbors = 2
    #     weights = distance
    #     minkowski metric (p) = 2

    preds = knn_model.predict(x_valid)
    rmse = 0

    if(new_predictions == False):
        rmse = np.sqrt(
            np.mean(np.power((np.array(y_valid) - np.array(preds)), 2)))

    print("Predictions and length")
    print(len(preds))
    print(preds)

    # valid["Predictions"] = 0
    # valid["Predictions"] = preds
    fig_knn = go.Figure()

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

    if(new_predictions == True):
        all_prediction_dates = df["Date"][split:]
        all_prediction_dates = all_prediction_dates.append(
            original_prediction_dates["Date"], ignore_index=True)

        fig_knn.add_trace(go.Scatter(
            x=all_prediction_dates,
            y=preds,
            mode="lines",
            name="Predictions"
        ))
    else:
        fig_knn.add_trace(go.Scatter(
            x=df["Date"][split:],
            y=preds,
            mode="lines",
            name="Predictions"
        ))

    graphJSON = json.dumps(fig_knn, cls=plotly.utils.PlotlyJSONEncoder)

    return knn_model, fig_knn, graphJSON, round(rmse, 2)


def auto_arima_model(df, split=977, start_p=1, max_p=3, start_q=1, max_q=3, d=1, D=1, new_predictions=False, new_dates=None):
    import pmdarima as pm

    df = df.sort_index(ascending=True, axis=0)
    new_data = df[["Date", "Close"]]

    if(new_predictions):
        new_data = new_data.append(new_dates, ignore_index=True)

    # Training-validation splits
    train = []
    valid = []

    if(new_predictions):
        train = new_data[:len(df)]
        valid = new_data[len(df):]
    else:
        train = new_data[:split]
        valid = new_data[split:]

    training = train['Close']
    validation = valid['Close']

    arima_model = pm.arima.auto_arima(training, start_p=start_p, max_p=max_p, start_q=start_q, max_q=max_q, m=12, start_P=0,
                                      seasonal=True, d=d, D=D, trace=True, error_action='ignore', suppress_warnings=True)

    arima_model.fit(training)

    arima_forecast = arima_model.predict(n_periods=len(valid))
    # arima_forecast = arima_model.predict(
    #    n_periods = 1259 - split)
    arima_forecast = pd.DataFrame(
        arima_forecast, index=valid.index, columns=['Prediction'])
    rmse = 777.77  # filler error value

    if(new_predictions == False):
        rmse = np.sqrt(np.mean(
            np.power((np.array(valid['Close']) - np.array(arima_forecast['Prediction'])), 2)))

    fig_arima = go.Figure()

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


def lstm_model(df, split=977, units=50, epochs=1, new_predictions=False, original_predictions=None):
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM

    # creating dataframe
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)),
                            columns=['Date', 'Close'])

    for i in range(0, len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]

    # vertically stack new_data and predictions df
    if(new_predictions == True):
        new_data = new_data.append(original_predictions, ignore_index=True)

    # setting index
    new_data.index = new_data.Date

    new_data.drop('Date', axis=1, inplace=True)

    # creating train and test sets
    dataset = new_data.values  # array of arrays containing one value [[value1]
    # [value2]...]

    train = []
    valid = []

    if(new_predictions):
        train = dataset[:len(df), :]
        valid = dataset[len(df):, :]
    else:
        train = dataset[0:split, :]
        valid = dataset[split:, :]

    # converting dataset into x_train and y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []

    for i in range(60, len(train)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(
        units=units,
        return_sequences=True,
        input_shape=(x_train.shape[1], 1))
    )
    model.add(LSTM(units=units))
    model.add(Dense(1))  # dimensionality of the output
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=2)

    # Starting from the last 60 training data points
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)  # 2D array
    inputs = scaler.transform(inputs)
    actual_inputs = inputs[0:60]  # 2D array
    closing_price = []
    X_test = []

    # print(inputs[0:60, 0])  # 1D array

    for i in range(60, inputs.shape[0]):
        X_test = []

        X_test.append(actual_inputs[i-60:i, 0])

        X_test = np.array(X_test)  # 2D array
        X_test = np.reshape(
            X_test, (X_test.shape[0], X_test.shape[1], 1))  # 3D array
        predicted_price = model.predict(X_test[0:1])  # 2D array
        actual_inputs = np.vstack([actual_inputs, predicted_price[0]])
        predicted_price = scaler.inverse_transform(predicted_price)

        closing_price.append(predicted_price[0, 0])

    rmse = 777.77

    if(new_predictions == False):
        rmse = np.sqrt(np.mean(np.power((valid - closing_price), 2)))

    # plotting LSTM
    if(new_predictions):
        train = new_data[:len(df)]
        valid = new_data[len(df):]
    else:
        train = new_data[:split]
        valid = new_data[split:]

    valid['Predictions'] = closing_price
    fig_lstm = go.Figure()

    fig_lstm.add_trace(go.Scatter(
        x=df["Date"],
        y=train["Close"],
        mode="lines",
        name="Training"
    ))

    if(new_predictions):
        fig_lstm.add_trace(go.Scatter(
            x=new_data.index[len(df) - 60:],
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
