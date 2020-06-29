from flask import Flask, render_template, url_for, request, redirect
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import json
from datetime import datetime
from datetime import timezone
from datetime import date
from dateutil.relativedelta import relativedelta

app = Flask(__name__, template_folder="templates")


@app.route("/")
def root():
    return render_template("index.html")


@app.route("/ticker", methods=["POST"])
def ticker():
    ticker = request.form["ticker"]

    return redirect("/" + ticker + "/models")


def read_historic_data(ticker):
    # Unix timestamp calculation for today's date and five years ago to obtain Yahoo Finance data
    date_today = datetime.today().strftime("%Y-%m-%d")
    dtc = date_today.split("-")
    date_five_years_ago = (
        datetime.today() - relativedelta(years=5)).strftime("%Y-%m-%d")
    dfyc = date_five_years_ago.split("-")
    timestamp_today = int(datetime(int(dtc[0]), int(dtc[1]), int(
        dtc[2]), 0, 0).replace(tzinfo=timezone.utc).timestamp())
    timestamp_five_years_ago = int((datetime(int(dfyc[0]), int(dfyc[1]), int(
        dfyc[2]), 0, 0)).replace(tzinfo=timezone.utc).timestamp())

    # Reading in stock data from Yahoo Finance in the above timestamps' range
    csv_url = "https://query1.finance.yahoo.com/v7/finance/download/" + ticker + \
        "?period1=" + str(timestamp_five_years_ago) + "&period2=" + \
        str(timestamp_today) + "&interval=1d&events=history"
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
    rmse = np.sqrt(np.mean(np.power((np.array(valid['Close']) - preds), 2)))

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

    print(new_data["Date"])

    new_data["Date"] = pd.to_datetime(new_data["Date"])

    print(new_data["Date"])

    new_data["Date"] = new_data["Date"].map(datetime.toordinal)

    print(new_data["Date"])

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

    print(x_valid.shape)
    print(x_valid)

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

    print("Predictions and length")
    print(len(preds))
    print(preds)
    # rmse = np.sqrt(np.mean(np.power((np.array(y_valid) - np.array(preds)), 2)))
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

        print("All prediction dates and length and length")
        print(len(all_prediction_dates))
        print(type(all_prediction_dates))
        print(all_prediction_dates)
        print("Type of original prediction dates")
        print(type(original_prediction_dates["Date"]))
        print(original_prediction_dates)

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

    return knn_model, fig_knn, graphJSON, round(777.77, 2)


def auto_arima_model(df, split=977, start_p=1, max_p=3, start_q=1, max_q=3, d=1, D=1):
    import pmdarima as pm

    data = df.sort_index(ascending=True, axis=0)

    train = data[:split]
    valid = data[split:]

    training = train['Close']
    validation = valid['Close']

    arima_model = pm.arima.auto_arima(training, start_p=start_p, max_p=max_p, start_q=start_q, max_q=max_q, m=12, start_P=0,
                                      seasonal=True, d=d, D=D, trace=True, error_action='ignore', suppress_warnings=True)

    arima_model.fit(training)

    arima_forecast = arima_model.predict(
        n_periods=1260 - split - 1)
    arima_forecast = pd.DataFrame(
        arima_forecast, index=valid.index, columns=['Prediction'])
    rmse = np.sqrt(np.mean(
        np.power((np.array(valid['Close']) - np.array(forecast['Prediction'])), 2)))
    fig_arima = go.Figure()

    fig_arima.add_trace(go.Scatter(
        x=df["Date"],
        y=train["Close"],
        mode="lines",
        name="Training"
    ))

    fig_arima.add_trace(go.Scatter(
        x=df["Date"][split:],
        y=valid["Close"],
        mode="lines",
        name="Validation"
    ))

    fig_arima.add_trace(go.Scatter(
        x=df["Date"][split:],
        y=arima_forecast["Prediction"],
        mode="lines",
        name="Predictions"
    ))

    graphJSON = json.dumps(fig_arima, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON, round(rmse, 2)


def lstm_model(df, split=977, units=50, epochs=1):
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM

    # creating dataframe
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

    for i in range(0, len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]

    # setting index
    new_data.index = new_data.Date

    new_data.drop('Date', axis=1, inplace=True)

    # creating train and test sets
    dataset = new_data.values

    print(dataset)

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

    inputs = new_data[len(new_data) - len(valid) - 60:].values

    # print(inputs)

    inputs = inputs.reshape(-1, 1)

    # print(inputs)

    inputs = scaler.transform(inputs)

    # print(inputs)

    X_test = []

    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i-60:i, 0])

    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # print(X_test)

    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    rmse = np.sqrt(np.mean(np.power((valid - closing_price), 2)))

    # plotting LSTM
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

    fig_lstm.add_trace(go.Scatter(
        x=df["Date"][split:],
        y=valid["Close"],
        mode="lines",
        name="Validation"
    ))

    fig_lstm.add_trace(go.Scatter(
        x=df["Date"][split:],
        y=valid["Predictions"],
        mode="lines",
        name="Predictions"
    ))

    graphJSON = json.dumps(fig_lstm, cls=plotly.utils.PlotlyJSONEncoder)

    return model, fig_lstm, graphJSON, round(rmse, 2)


@app.route("/<ticker>/models")
def index(ticker):
    # Reading stock data
    df = read_historic_data(ticker)

    # Generating plots
    historic_plot = historic_model(df)
    moving_average_plot, ma_rmse = moving_average_model(df)
    linear_model, linear_fig, linear_regression_plot, lr_rmse = linear_regression_model(
        df)
    k_model, knn_fig, knn_plot, knn_rmse = knn_model(df)
    # lstm, lstm_fig, lstm_plot, lstm_rmse = lstm_model(df)
    # auto_arima_plot, arima_rmse = auto_arima_model(df)

    return render_template(
        "models.html.jinja",
        ticker=ticker,
        historic_plot=historic_plot,
        moving_average_plot=moving_average_plot,
        linear_regression_plot=linear_regression_plot,
        knn_plot=knn_plot
        # lstm_plot=lstm_plot
        # auto_arima_plot=auto_arima_plot
    )


@app.route("/<ticker>/ma/customize/<window>/<split>")
def ma_customize_input(ticker, window, split):
    df = read_historic_data(ticker)
    moving_average_plot, rmse = moving_average_model(
        df, int(window), int(split))

    return render_template(
        "ma_customize.html.jinja",
        ticker=ticker,
        moving_average_plot=moving_average_plot,
        rmse=rmse,
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
    linear_model, linear_fig, linear_regression_plot, rmse = linear_regression_model(
        df, int(split))

    return render_template(
        "lr_customize.html.jinja",
        ticker=ticker,
        linear_regression_plot=linear_regression_plot,
        rmse=rmse,
        split=split
    )


@app.route("/lr/customize", methods=["POST"])
def lr_customize_output():
    ticker = request.form["ticker"]
    split = request.form["split"]

    return redirect("/" + ticker + "/lr/customize/" + split)


@app.route("/<ticker>/lr/predict/<split>")
def lr_predict_input(ticker, split):
    df = read_historic_data(ticker)
    linear_model, linear_fig, linear_regression_plot, rmse = linear_regression_model(
        df, int(split))

    return render_template(
        "lr_predict.html.jinja",
        ticker=ticker,
        split=split,
        linear_regression_plot=linear_regression_plot
    )


@app.route("/lr/predict", methods=["POST"])
def lr_predict_output():
    # Form submission values
    year = request.form["year"]
    month = request.form["month"]
    day = request.form["day"]
    ticker = request.form["ticker"]
    split = request.form["split"]

    # Generating the linear model
    df = read_historic_data(ticker)
    linear_model, linear_fig, linear_regression_plot, rmse = linear_regression_model(
        df, int(split))

    # Generating the date column for predictions
    start_date = date.today()
    end_date = date(int(year), int(month), int(day))
    # Range of prediction dates
    predict_data = {"Date": pd.date_range(start=start_date, end=end_date)}
    # DataFrame with original prediction dates
    predict_dates = pd.DataFrame(data=predict_data)
    to_predict_df = pd.DataFrame(data=predict_data)
    to_predict_df["Date"] = to_predict_df["Date"].map(
        datetime.toordinal)  # DataFrame with ordinal prediction dates

    # Predicting prices on new dates
    new_predictions = linear_model.predict(to_predict_df)

    print(new_predictions)

    # Plotting predicted prices
    linear_fig.add_trace(go.Scatter(
        x=predict_dates["Date"],
        y=new_predictions,
        mode="lines",
        name="Forecast"
    ))

    linear_regression_plot = json.dumps(
        linear_fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "lr_predict.html.jinja",
        ticker=ticker,
        split=split,
        linear_regression_plot=linear_regression_plot
    )


@app.route("/<ticker>/knn/customize/<split>/<neighbors>/<weights>/<power>")
def knn_customize_input(ticker, split, neighbors, weights, power):
    df = read_historic_data(ticker)
    k_model, knn_fig, knn_plot, rmse = knn_model(
        df, int(split), int(neighbors), weights, int(power))

    return render_template(
        "knn_customize.html.jinja",
        ticker=ticker,
        knn_plot=knn_plot,
        rmse=rmse,
        split=split,
        neighbors=neighbors,
        weights=weights,
        power=power
    )


@app.route("/knn/customize", methods=["POST"])
def knn_customize_output():
    ticker = request.form["ticker"]
    split = request.form["split"]
    neighbors = request.form["neighbors"]
    weights = request.form["weights"]
    power = request.form["power"]

    return redirect("/" + ticker + "/knn/customize/" + split + "/" + neighbors + "/" + weights + "/" + power)


@app.route("/<ticker>/knn/predict/<split>/<neighbors>/<weights>/<power>")
def knn_predict_input(ticker, split, neighbors, weights, power):
    df = read_historic_data(ticker)
    k_model, knn_fig, knn_plot, rmse = knn_model(
        df, int(split), int(neighbors), weights, int(power))

    return render_template(
        "knn_predict.html.jinja",
        ticker=ticker,
        split=split,
        neighbors=neighbors,
        weights=weights,
        power=power,
        knn_plot=knn_plot
    )


@app.route("/knn/predict", methods=["POST"])
def knn_predict_output():
    # Form submission values
    year = request.form["year"]
    month = request.form["month"]
    day = request.form["day"]
    ticker = request.form["ticker"]
    split = request.form["split"]
    neighbors = request.form["neighbors"]
    weights = request.form["weights"]
    power = request.form["power"]

    # Generating the linear model
    df = read_historic_data(ticker)

    # Generating the date column for predictions
    start_date = date.today()
    end_date = date(int(year), int(month), int(day))
    # Range of prediction dates
    predict_data = {"Date": pd.date_range(start=start_date, end=end_date)}
    # DataFrame with original prediction dates
    predict_dates = pd.DataFrame(data=predict_data)
    to_predict_df = pd.DataFrame(data=predict_data)
    to_predict_df["Date"] = to_predict_df["Date"].map(
        datetime.toordinal)  # DataFrame with ordinal prediction dates
    k_model, knn_fig, knn_plot, rmse = knn_model(
        df, int(split), int(neighbors), weights, int(power), new_predictions=True, ordinal_prediction_dates=to_predict_df, original_prediction_dates=predict_dates)

    # Predicting prices on new dates
    new_predictions = k_model.predict(to_predict_df)

    # print("Predictions from returned model")
    # print(new_predictions)

    # Plotting predicted prices
    # knn_fig.add_trace(go.Scatter(
    #     x=predict_dates["Date"],
    #     y=new_predictions,
    #     mode="lines",
    #     name="Forecast"
    # ))

    # knn_plot = json.dumps(
    #     knn_fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "knn_predict.html.jinja",
        ticker=ticker,
        split=split,
        neighbors=neighbors,
        weights=weights,
        power=power,
        knn_plot=knn_plot
    )


@app.route("/<ticker>/lstm/customize/<split>/<units>/<epochs>")
def lstm_customize_input(ticker, split, units, epochs):
    df = read_historic_data(ticker)
    lstm, lstm_fig, lstm_plot, rmse = lstm_model(
        df, int(split), int(units), int(epochs))

    return render_template(
        "lstm_customize.html.jinja",
        ticker=ticker,
        lstm_plot=lstm_plot,
        rmse=rmse,
        split=split,
        units=units,
        epochs=epochs,
    )


@app.route("/lstm/customize", methods=["POST"])
def lstm_customize_output():
    ticker = request.form["ticker"]
    split = request.form["split"]
    units = request.form["units"]
    epochs = request.form["epochs"]

    return redirect("/" + ticker + "/lstm/customize/" + split + "/" + units + "/" + epochs)


@app.route("/<ticker>/lstm/predict/<split>/<units>/<epochs>/")
def lstm_predict_input(ticker, split, units, epochs):
    df = read_historic_data(ticker)
    lstm, lstm_fig, lstm_plot, rmse = lstm_model(
        df, int(split), int(units), int(epochs))

    return render_template(
        "lstm_predict.html.jinja",
        ticker=ticker,
        split=split,
        units=units,
        epochs=epochs,
        lstm_plot=lstm_plot
    )


@app.route("/lstm/predict", methods=["POST"])
def lstm_predict_output():
    # Form submission values
    year = request.form["year"]
    month = request.form["month"]
    day = request.form["day"]
    split = request.form["split"]
    ticker = request.form["ticker"]
    units = request.form["units"]
    epochs = request.form["epochs"]

    # Generating the LSTM model
    df = read_historic_data(ticker)
    lstm, lstm_fig, lstm_plot, rmse = lstm_model(
        df, int(split), int(units), int(epochs))

    # Generating the date column for predictions
    start_date = date.today()
    end_date = date(int(year), int(month), int(day))
    # Range of prediction dates
    predict_data = {"Date": pd.date_range(start=start_date, end=end_date)}
    # DataFrame with original prediction dates
    predict_dates = pd.DataFrame(data=predict_data)
    to_predict_df = pd.DataFrame(data=predict_data)
    to_predict_df["Date"] = to_predict_df["Date"].map(
        datetime.toordinal)  # DataFrame with ordinal prediction dates

    # Predicting prices on new dates
    new_predictions = lstm.predict(to_predict_df)

    print(new_predictions)

    # Plotting predicted prices
    lstm_fig.add_trace(go.Scatter(
        x=predict_dates["Date"],
        y=new_predictions,
        mode="lines",
        name="Forecast"
    ))

    lstm_plot = json.dumps(
        lstm_fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "lstm_predict.html.jinja",
        ticker=ticker,
        split=split,
        units=units,
        epochs=epochs,
        lstm_plot=lstm_plot
    )


@app.route("/<ticker>/arima/customize/<split>/<start_p>/<max_p>/<start_q>/<max_q>/<d>/<D>")
def arima_customize_input(ticker, split, start_p, max_p, start_q, max_q, d, D):
    df = read_historic_data(ticker)
    auto_arima_plot, rmse = auto_arima_model(df, int(split), int(
        start_p), int(max_p), int(start_q), int(max_q), int(d), int(D))

    return render_template(
        "arima_customize.html.jinja",
        ticker=ticker,
        auto_arima_plot=auto_arima_plot,
        rmse=rmse,
        split=split,
        start_p=start_p,
        max_p=max_p,
        start_q=start_q,
        max_q=max_q,
        d=d,
        D=D
    )


@app.route("/arima/customize", methods=["POST"])
def arima_customize_output():
    ticker = request.form["ticker"]
    split = request.form["split"]
    start_p = request.form["start_p"]
    max_p = request.form["max_p"]
    start_q = request.form["start_q"]
    max_q = request.form["max_q"]
    d = request.form["d"]
    D = request.form["D"]

    return redirect("/" + ticker + "/arima/customize/" + split + "/" + start_p + "/" + max_p + "/" + start_q + "/" + max_q + "/" + d + "/" + D)


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
    app.run(debug=True, threaded=False)
