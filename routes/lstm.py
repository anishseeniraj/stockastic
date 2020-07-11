"""
Long Short Term Memory (LSTM) router module

This module contains all the back-end routes that deal with the LSTM
predictive model (routes that deal with model-building as well
as forecasting).

It requires utils.stock_preprocess and utils.stock_models to 
    -> Preprocess raw stock data
    -> Execute models and present visualizations
"""

import json

import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go

from flask import Blueprint, render_template, Flask, url_for, request, redirect
from utils.stock_preprocess import *
from utils.stock_models import *

# To be registered in the main router file (app.py)
lstm = Blueprint("lstm", __name__)


@lstm.route("/<ticker>/lstm/customize/<split>/<units>/<epochs>")
def lstm_customize_input(ticker, split, units, epochs):
    """
    Generates an LSTM plot based on user-inputted hyperparameter values 
    and allows the user to further customize the hyperparameter values
    """

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
    """
    Reads user-inputted hyperparameter values submitted on the front-end
    and redirects to the model-generation route
    """

    # Form submission values - model hyperparameters
    ticker = request.form["ticker"]
    split = request.form["split"]
    units = request.form["units"]
    epochs = request.form["epochs"]

    return redirect("/" + ticker + "/lstm/customize/" + split + "/"
                    + units + "/" + epochs)


@lstm.route("/<ticker>/lstm/predict/<split>/<units>/<epochs>/")
def lstm_predict_input(ticker, split, units, epochs):
    """ 
    Generates an LSTM plot based on user-inputted hyperparameter
    values and allows a forecast date to be entered on the front-end
    """

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
    """
    Reads the required forecast date and makes predictions with the
    selected model hyperparameters on the specified forecast date
    """

    # Form submission values - forecast date and model hyperparameters
    year = request.form["year"]
    month = request.form["month"]
    day = request.form["day"]
    split = request.form["split"]
    ticker = request.form["ticker"]
    units = request.form["units"]
    epochs = request.form["epochs"]

    df = read_historic_data(ticker)

    # Generate dates for forecast and populate values with NaN
    to_predict = generate_dates_until(int(year), int(month), int(day))
    to_predict["Close"] = np.nan

    lstm, lstm_fig, lstm_plot, rmse = lstm_model(
        df, int(split), int(units), int(epochs), new_predictions=True,
        original_predictions=to_predict)

    return render_template(
        "lstm/lstm_predict.html.jinja",
        ticker=ticker,
        split=split,
        units=units,
        epochs=epochs,
        lstm_plot=lstm_plot
    )
