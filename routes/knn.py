"""
K-Nearest Neighbors (KNN) router module

This module contains all the back-end routes that deal with the KNN
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
knn = Blueprint("knn", __name__)


@knn.route("/<ticker>/knn/customize/<split>/<neighbors>/<weights>/<power>")
def knn_customize_input(ticker, split, neighbors, weights, power):
    """
    Generates a KNN plot based on user-inputted hyperparameter values 
    and allows the user to further customize the hyperparameter values
    """

    df = read_historic_data(ticker)
    k_model, knn_fig, knn_plot, rmse = knn_model(
        df, int(split), int(neighbors), weights, int(power))

    return render_template(
        "knn/knn_customize.html.jinja",
        ticker=ticker,
        knn_plot=knn_plot,
        rmse=rmse,
        split=split,
        neighbors=neighbors,
        weights=weights,
        power=power
    )


@knn.route("/knn/customize", methods=["POST"])
def knn_customize_output():
    """
    Reads user-inputted hyperparameter values submitted on the front-end
    and redirects to the model-generation route
    """

    # Form submission - model hyperparameters
    ticker = request.form["ticker"]
    split = request.form["split"]
    neighbors = request.form["neighbors"]
    weights = request.form["weights"]
    power = request.form["power"]

    return redirect("/" + ticker + "/knn/customize/" + split + "/"
                    + neighbors + "/" + weights + "/" + power)


@knn.route("/<ticker>/knn/predict/<split>/<neighbors>/<weights>/<power>")
def knn_predict_input(ticker, split, neighbors, weights, power):
    """ 
    Generates a KNN plot based on user-inputted hyperparameter
    values and allows a forecast date to be entered on the front-end
    """

    df = read_historic_data(ticker)
    k_model, knn_fig, knn_plot, rmse = knn_model(
        df, int(split), int(neighbors), weights, int(power))

    return render_template(
        "knn/knn_predict.html.jinja",
        ticker=ticker,
        split=split,
        neighbors=neighbors,
        weights=weights,
        power=power,
        knn_plot=knn_plot
    )


@knn.route("/knn/predict", methods=["POST"])
def knn_predict_output():
    """
    Reads the required forecast date and makes predictions with the
    selected model hyperparameters on the specified forecast date
    """

    # Form submission values - forecast date and model hyperparameters
    year = request.form["year"]
    month = request.form["month"]
    day = request.form["day"]
    ticker = request.form["ticker"]
    split = request.form["split"]
    neighbors = request.form["neighbors"]
    weights = request.form["weights"]
    power = request.form["power"]

    df = read_historic_data(ticker)

    # Generating dates for forecast and changing them to ordinal format
    predict_dates = generate_dates_until(int(year), int(month), int(day))
    to_predict_df = generate_dates_until(int(year), int(month), int(day))
    to_predict_df["Date"] = to_predict_df["Date"].map(
        datetime.toordinal)

    k_model, knn_fig, knn_plot, rmse = knn_model(
        df, int(split), int(neighbors), weights, int(power),
        new_predictions=True, ordinal_prediction_dates=to_predict_df,
        original_prediction_dates=predict_dates)
    new_predictions = k_model.predict(to_predict_df)

    return render_template(
        "knn/knn_predict.html.jinja",
        ticker=ticker,
        split=split,
        neighbors=neighbors,
        weights=weights,
        power=power,
        knn_plot=knn_plot
    )
