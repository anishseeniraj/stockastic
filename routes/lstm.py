from flask import Blueprint, render_template, Flask, url_for, request, redirect
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import json
from utils.stock_preprocess import *
from utils.stock_models import *
# import keras.backend.tensorflow_backend as tb

# tb._SYMBOLIC_SCOPE.value = True

lstm = Blueprint("lstm", __name__)


@lstm.route("/<ticker>/lstm/customize/<split>/<units>/<epochs>")
def lstm_customize_input(ticker, split, units, epochs):
    df = read_historic_data(ticker)
    lstm, lstm_fig, lstm_plot, rmse = lstm_model(
        df, int(split), int(units), int(epochs))

    return render_template(
        "lstm/lstm_customize.html.jinja",
        ticker=ticker,
        lstm_plot=lstm_plot,
        rmse=rmse,
        split=split,
        units=units,
        epochs=epochs,
    )


@lstm.route("/lstm/customize", methods=["POST"])
def lstm_customize_output():
    ticker = request.form["ticker"]
    split = request.form["split"]
    units = request.form["units"]
    epochs = request.form["epochs"]

    return redirect("/" + ticker + "/lstm/customize/" + split + "/" + units + "/" + epochs)


@lstm.route("/<ticker>/lstm/predict/<split>/<units>/<epochs>/")
def lstm_predict_input(ticker, split, units, epochs):
    df = read_historic_data(ticker)
    lstm, lstm_fig, lstm_plot, rmse = lstm_model(
        df, int(split), int(units), int(epochs))

    return render_template(
        "lstm/lstm_predict.html.jinja",
        ticker=ticker,
        split=split,
        units=units,
        epochs=epochs,
        lstm_plot=lstm_plot
    )


@lstm.route("/lstm/predict", methods=["POST"])
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

    # Generating the date column for predictions
    # df with original prediction dates and dummy close prices
    to_predict = generate_dates_until(int(year), int(month), int(day))
    to_predict["Close"] = np.nan

    lstm, lstm_fig, lstm_plot, rmse = lstm_model(
        df, int(split), int(units), int(epochs), new_predictions=True, original_predictions=to_predict)

    return render_template(
        "lstm/lstm_predict.html.jinja",
        ticker=ticker,
        split=split,
        units=units,
        epochs=epochs,
        lstm_plot=lstm_plot
    )
