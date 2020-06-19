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
    data = [go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"])]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    fig = [go.Scatter(
        x=df["Date"],
        y=df["Close"]
    )]
    graphJSON2 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

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
