from flask import Flask, render_template, url_for, request, redirect
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import json

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

    # Plot for the original trend
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

    # note: While splitting the data into train and validation set, we cannot use random splitting since that will destroy the time component. So here we have set the last year’s data into validation and the 4 years’ data before that into train set.

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
        name="Training Set Close Prices"
    ))

    fig_ma.add_trace(go.Scatter(
        x=df["Date"][987:],
        y=valid["Close"],
        mode="lines",
        name="Validation Set Close Prices"
    ))

    fig_ma.add_trace(go.Scatter(
        x=df["Date"][987:],
        y=valid["Predictions"],
        mode="lines",
        name="Predictions"
    ))

    # fig = [go.Scatter(
    #     x=df["Date"],
    #     y=train["Close"]
    # )]
    graphJSON2 = json.dumps(fig_ma, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "models.html.jinja",
        ticker=ticker,
        plot=graphJSON,
        plot2=graphJSON2)


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
