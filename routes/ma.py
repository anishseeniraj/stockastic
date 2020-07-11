"""
Moving Average router module

This module contains all the back-end routes that deal with the Moving
Average predictive model (routes that deal with model-building as well
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
ma = Blueprint("ma", __name__)


@ma.route("/<ticker>/ma/customize/<window>/<split>")
def ma_customize_input(ticker, window, split):
    """
    Generates a moving average plot based on the user-inputted parameter
    values and allows the user to further customize the parameter values
    """

    df = read_historic_data(ticker)
    moving_average_plot, rmse = moving_average_model(
        df, int(window), int(split))

    return render_template(
        "ma/ma_customize.html.jinja",
        ticker=ticker,
        moving_average_plot=moving_average_plot,
        rmse=rmse,
        window=window,
        split=split
    )


@ma.route("/ma/customize", methods=["POST"])
def ma_customize_output():
    """
    Reads the user-inputted parameter values submitted on the front-end
    and redirects to the model-generation route
    """

    # Form submission - model parameters
    window = request.form["window"]
    ticker = request.form["ticker"]
    split = request.form["split"]

    return redirect("/" + ticker + "/ma/customize/" + window + "/" + split)


@ma.route("/<ticker>/ma/predict/<window>/<split>")
def ma_predict_input(ticker, window, split):
    """ 
    Generates a moving average plot based on the user-inputted parameter
    values and allows a forecast date to be entered on the front-end
    """

    df = read_historic_data(ticker)
    ma_plot, ma_rmse = moving_average_model(
        df, int(window), int(split))

    return render_template(
        "ma/ma_predict.html.jinja",
        ticker=ticker,
        window=window,
        split=split,
        ma_plot=ma_plot
    )


@ma.route("/ma/predict", methods=["POST"])
def ma_predict_output():
    """ 
    Reads the required forecast date and makes predictions with the
    selected model on the specified forecast date
    """

    # Form submission - forecast date and model parameters
    year = request.form["year"]
    month = request.form["month"]
    day = request.form["day"]
    ticker = request.form["ticker"]
    window = request.form["window"]
    split = request.form["split"]

    df = read_historic_data(ticker)

    # Generate dates until the forecast date and populate pre-model predictions
    #   with NaN values
    to_predict = generate_dates_until(int(year), int(month), int(day))
    to_predict["Close"] = np.nan

    ma_plot, ma_rmse = moving_average_model(
        df, int(window), int(split), new_predictions=True, new_dates=to_predict)

    return render_template(
        "ma/ma_predict.html.jinja",
        ticker=ticker,
        window=window,
        split=split,
        ma_rmse=ma_rmse,
        ma_plot=ma_plot
    )
