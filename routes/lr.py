"""
Moving Average router module

This module contains all the back-end routes that deal with the Linear
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
lr = Blueprint("lr", __name__)


@lr.route("/<ticker>/lr/customize/<split>")
def lr_customize_input(ticker, split):
    """
    Generates a linear regression plot based on the user-inputted parameter
    value and allows the user to further customize the parameter value
    """

    df = read_historic_data(ticker)
    linear_model, linear_fig, linear_regression_plot, rmse = linear_regression_model(
        df, int(split))

    return render_template(
        "lr/lr_customize.html.jinja",
        ticker=ticker,
        linear_regression_plot=linear_regression_plot,
        rmse=rmse,
        split=split
    )


@lr.route("/lr/customize", methods=["POST"])
def lr_customize_output():
    """
    Reads the user-inputted parameter value submitted on the front-end
    and redirects to the model-generation route
    """

    # Form submission - model parameter
    ticker = request.form["ticker"]
    split = request.form["split"]

    return redirect("/" + ticker + "/lr/customize/" + split)


@lr.route("/<ticker>/lr/predict/<split>")
def lr_predict_input(ticker, split):
    """
    Generates a linear model plot based on the user-inputted parameter
    value and allows a forecast date to be entered on the front-end
    """

    df = read_historic_data(ticker)
    linear_model, linear_fig, linear_regression_plot, rmse = linear_regression_model(
        df, int(split))

    return render_template(
        "lr/lr_predict.html.jinja",
        ticker=ticker,
        split=split,
        linear_regression_plot=linear_regression_plot
    )


@lr.route("/lr/predict", methods=["POST"])
def lr_predict_output():
    """ 
    Reads the required forecast date and makes predictions with the
    selected model parameter on the specified forecast date
    """

    # Form submission values - forecast date and model parameter
    year = request.form["year"]
    month = request.form["month"]
    day = request.form["day"]
    ticker = request.form["ticker"]
    split = request.form["split"]

    df = read_historic_data(ticker)
    linear_model, linear_fig, linear_regression_plot, rmse = linear_regression_model(
        df, int(split))

    # Generate dates for forecast and chaneg them to ordinal format
    predict_dates = generate_dates_until(int(year), int(month), int(day))
    to_predict_df = generate_dates_until(int(year), int(month), int(day))
    to_predict_df["Date"] = to_predict_df["Date"].map(
        datetime.toordinal)

    new_predictions = linear_model.predict(to_predict_df)

    # Forecast plot
    linear_fig.add_trace(go.Scatter(
        x=predict_dates["Date"],
        y=new_predictions,
        mode="lines",
        name="Forecast"
    ))

    linear_regression_plot = json.dumps(
        linear_fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "lr/lr_predict.html.jinja",
        ticker=ticker,
        split=split,
        linear_regression_plot=linear_regression_plot
    )
