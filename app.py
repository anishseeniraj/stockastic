from flask import Flask, render_template, url_for, request, redirect
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import json
import datetime as dt

app = Flask(__name__, template_folder="templates")


@app.route("/")
def root():
    return render_template("index.html")


@app.route("/ticker", methods=["POST"])
def ticker():
    ticker = request.form["ticker"]

    return redirect("/" + ticker + "/models")


def read_historic_data(ticker):
    # Reading in stock data from Yahoo Finance
    csv_url = "https://query1.finance.yahoo.com/v7/finance/download/" + ticker + \
        "?period1=1434326400&period2=1592179200&interval=1d&events=history"
    df = pd.read_csv(csv_url)

    return df


def historic_model(df):
    # Original Trend
    data = [go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"])]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


def moving_average_model(df, window=225, split=977):
    # Moving Average
    df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df.index = df['Date']

    # creating dataframe with date and the target variable
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

    for i in range(0, len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]

    # splitting into train and validation
    train = new_data[:split]
    valid = new_data[split:]
    preds = []

    for i in range(0, valid.shape[0]):
        a = train['Close'][len(train)-window+i:].sum() + sum(preds)
        b = a/window
        preds.append(b)

    valid['Predictions'] = 0
    valid['Predictions'] = preds

    # Moving Average Plot
    fig_ma = go.Figure()

    fig_ma.add_trace(go.Scatter(
        x=df["Date"],
        y=train["Close"],
        mode="lines",
        name="Training"
    ))

    fig_ma.add_trace(go.Scatter(
        x=df["Date"][split:],
        y=valid["Close"],
        mode="lines",
        name="Validation"
    ))

    fig_ma.add_trace(go.Scatter(
        x=df["Date"][split:],
        y=valid["Predictions"],
        mode="lines",
        name="Predictions"
    ))

    graphJSON = json.dumps(fig_ma, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


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
    new_data["Date"] = new_data["Date"].map(dt.datetime.toordinal)
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

    return graphJSON


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
    new_data["Date"] = new_data["Date"].map(dt.datetime.toordinal)
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

    return graphJSON


@app.route("/<ticker>/models")
def index(ticker):
    # Reading stock data
    df = read_historic_data(ticker)

    # Generating historic and predictve plots
    historic_plot = historic_model(df)
    moving_average_plot = moving_average_model(df)
    linear_regression_plot = linear_regression_model(df)

    # k-Nearest Neighbors
    from sklearn import neighbors
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))

    # scaling data
    x_train_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train_scaled)
    x_valid_scaled = scaler.fit_transform(x_valid)
    x_valid = pd.DataFrame(x_valid_scaled)

    # using gridsearch to find the best parameter
    params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
    knn = neighbors.KNeighborsRegressor()
    knn_model = GridSearchCV(knn, params, cv=5)

    # fit the model and make predictions
    knn_model.fit(x_train, y_train)
    preds = knn_model.predict(x_valid)
    valid["Predictions"] = 0
    valid["Predictions"] = preds
    fig_knn = go.Figure()

    fig_knn.add_trace(go.Scatter(
        x=df["Date"],
        y=train["Close"],
        mode="lines",
        name="Training"
    ))

    fig_knn.add_trace(go.Scatter(
        x=df["Date"][987:],
        y=valid["Close"],
        mode="lines",
        name="Validation"
    ))

    fig_knn.add_trace(go.Scatter(
        x=df["Date"][987:],
        y=valid["Predictions"],
        mode="lines",
        name="Predictions"
    ))

    graphJSON4 = json.dumps(fig_knn, cls=plotly.utils.PlotlyJSONEncoder)

    # # Auto-ARIMA
    # import pmdarima as pm

    # data = df.sort_index(ascending=True, axis=0)

    # train = data[:987]
    # valid = data[987:]

    # training = train['Close']
    # validation = valid['Close']

    # arima_model = pm.arima.auto_arima(training, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0,
    #                                   seasonal=True, d=1, D=1, trace=True, error_action='ignore', suppress_warnings=True)

    # arima_model.fit(training)

    # arima_forecast = arima_model.predict(n_periods=272)
    # arima_forecast = pd.DataFrame(
    #     arima_forecast, index=valid.index, columns=['Prediction'])
    # fig_arima = go.Figure()

    # fig_arima.add_trace(go.Scatter(
    #     x=df["Date"],
    #     y=train["Close"],
    #     mode="lines",
    #     name="Training"
    # ))

    # fig_arima.add_trace(go.Scatter(
    #     x=df["Date"][987:],
    #     y=valid["Close"],
    #     mode="lines",
    #     name="Validation"
    # ))

    # fig_arima.add_trace(go.Scatter(
    #     x=df["Date"][987:],
    #     y=arima_forecast["Prediction"],
    #     mode="lines",
    #     name="Predictions"
    # ))

    # graphJSON5 = json.dumps(fig_arima, cls=plotly.utils.PlotlyJSONEncoder)

    # LSTM
    # from sklearn.preprocessing import MinMaxScaler
    # from keras.models import Sequential
    # from keras.layers import Dense, Dropout, LSTM

    # # creating dataframe
    # data = df.sort_index(ascending=True, axis=0)
    # new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

    # for i in range(0, len(data)):
    #     new_data['Date'][i] = data['Date'][i]
    #     new_data['Close'][i] = data['Close'][i]

    # # setting index
    # new_data.index = new_data.Date

    # new_data.drop('Date', axis=1, inplace=True)

    # # creating train and test sets
    # dataset = new_data.values

    # train = dataset[0:987, :]
    # valid = dataset[987:, :]

    # # converting dataset into x_train and y_train
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_data = scaler.fit_transform(dataset)

    # x_train, y_train = [], []

    # for i in range(60, len(train)):
    #     x_train.append(scaled_data[i-60:i, 0])
    #     y_train.append(scaled_data[i, 0])

    # x_train, y_train = np.array(x_train), np.array(y_train)
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # # create and fit the LSTM network
    # model = Sequential()
    # model.add(LSTM(units=50, return_sequences=True,
    #                input_shape=(x_train.shape[1], 1)))
    # model.add(LSTM(units=50))
    # model.add(Dense(1))

    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

    # # predicting 246 values, using past 60 from the train data
    # inputs = new_data[len(new_data) - len(valid) - 60:].values
    # inputs = inputs.reshape(-1, 1)
    # inputs = scaler.transform(inputs)

    # X_test = []

    # for i in range(60, inputs.shape[0]):
    #     X_test.append(inputs[i-60:i, 0])

    # X_test = np.array(X_test)

    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # closing_price = model.predict(X_test)
    # closing_price = scaler.inverse_transform(closing_price)

    # # plotting LSTM
    # train = new_data[:987]
    # valid = new_data[987:]
    # valid['Predictions'] = closing_price

    # fig_lstm = go.Figure()

    # fig_lstm.add_trace(go.Scatter(
    #     x=df["Date"],
    #     y=train["Close"],
    #     mode="lines",
    #     name="Training"
    # ))

    # fig_lstm.add_trace(go.Scatter(
    #     x=df["Date"][987:],
    #     y=valid["Close"],
    #     mode="lines",
    #     name="Validation"
    # ))

    # fig_lstm.add_trace(go.Scatter(
    #     x=df["Date"][987:],
    #     y=valid["Predictions"],
    #     mode="lines",
    #     name="Predictions"
    # ))

    # graphJSON6 = json.dumps(fig_lstm, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "models.html.jinja",
        ticker=ticker,
        historic_plot=historic_plot,
        moving_average_plot=moving_average_plot,
        linear_regression_plot=linear_regression_plot
    )


@app.route("/<ticker>/ma/customize/<window>/<split>")
def ma_customize_input(ticker, window, split):
    df = read_historic_data(ticker)
    moving_average_plot = moving_average_model(df, int(window), int(split))

    return render_template(
        "ma_customize.html.jinja",
        ticker=ticker,
        moving_average_plot=moving_average_plot,
        window=window,
        split=split
    )


@app.route("/ma/customize", methods=["POST"])
def ma_customize_output():
    window = request.form["window"]
    ticker = request.form["ticker"]
    split = request.form["split"]

    return redirect("/" + ticker + "/ma/customize/" + window + "/" + split)


@app.route("/<ticker>/lr/customize/<split>")
def lr_customize_input(ticker, split):
    df = read_historic_data(ticker)
    linear_regression_plot = linear_regression_model(df, int(split))

    return render_template(
        "lr_customize.html.jinja",
        ticker=ticker,
        linear_regression_plot=linear_regression_plot,
        split=split
    )


@app.route("/lr/customize", methods=["POST"])
def lr_customize_output():
    ticker = request.form["ticker"]
    split = request.form["split"]

    return redirect("/" + ticker + "/lr/customize/" + split)


@app.route("/<ticker>/model/<model_name>")
def show(ticker, model_name):
    return "Descriptive view of the model"


@app.route("/<ticker>/model/<model_name>/train")
def train(ticker, model_name):
    return "Train the MODEL_NAME model by tuning the following hyperparameters"


@app.route("/<ticker>/model/<model_name>/predict")
def predict(ticker, model_name):
    return "Predict price of TICKER at a custom date"


if __name__ == "__main__":
    app.run(debug=True)
