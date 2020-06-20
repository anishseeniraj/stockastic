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


@app.route("/<ticker>/models")
def index(ticker):
    csv_url = "https://query1.finance.yahoo.com/v7/finance/download/" + ticker + \
        "?period1=1434326400&period2=1592179200&interval=1d&events=history"
    df = pd.read_csv(csv_url)

    # Original Trend
    data = [go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"])]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

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
    train = new_data[:987]
    valid = new_data[987:]
    preds = []

    for i in range(0, valid.shape[0]):
        a = train['Close'][len(train)-248+i:].sum() + sum(preds)
        b = a/248
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
        x=df["Date"][987:],
        y=valid["Close"],
        mode="lines",
        name="Validation"
    ))

    fig_ma.add_trace(go.Scatter(
        x=df["Date"][987:],
        y=valid["Predictions"],
        mode="lines",
        name="Predictions"
    ))

    graphJSON2 = json.dumps(fig_ma, cls=plotly.utils.PlotlyJSONEncoder)

    # Linear Regression
    new_data["Date"] = pd.to_datetime(new_data["Date"])
    new_data["Date"] = new_data["Date"].map(dt.datetime.toordinal)
    train = new_data[:987]
    valid = new_data[987:]
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

    valid.index = new_data[987:].index
    train.index = new_data[:987].index
    fig_lm = go.Figure()

    fig_lm.add_trace(go.Scatter(
        x=df["Date"],
        y=train["Close"],
        mode="lines",
        name="Training"
    ))

    fig_lm.add_trace(go.Scatter(
        x=df["Date"][987:],
        y=valid["Close"],
        mode="lines",
        name="Validation"
    ))

    fig_lm.add_trace(go.Scatter(
        x=df["Date"][987:],
        y=valid["Predictions"],
        mode="lines",
        name="Predictions"
    ))

    graphJSON3 = json.dumps(fig_lm, cls=plotly.utils.PlotlyJSONEncoder)

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

    return render_template(
        "models.html.jinja",
        ticker=ticker,
        plot=graphJSON,
        plot2=graphJSON2,
        plot3=graphJSON3,
        plot4=graphJSON4
    )


@app.route("/<ticker>/model/<model_name>")
def show(ticker, model_name):
    return "Descriptive look at model"


@app.route("/<ticker>/model/<model_name>/train")
def train(ticker, model_name):
    return "Train the MODEL_NAME model by tuning the following hyperparameters"


@app.route("/<ticker>/model/<model_name>/predict")
def predict(ticker, model_name):
    return "Predict price of TICKER at a custom date"


if __name__ == "__main__":
    app.run(debug=True)
