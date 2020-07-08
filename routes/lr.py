from flask import Blueprint, render_template, Flask, url_for, request, redirect
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import json
from utils.stock_preprocess import *
from utils.stock_models import *

lr = Blueprint("lr", __name__)


@lr.route("/<ticker>/lr/customize/<split>")
def lr_customize_input(ticker, split):
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
    ticker = request.form["ticker"]
    split = request.form["split"]

    return redirect("/" + ticker + "/lr/customize/" + split)


@lr.route("/<ticker>/lr/predict/<split>")
def lr_predict_input(ticker, split):
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
    # DataFrame with original prediction dates
    predict_dates = generate_dates_until(int(year), int(month), int(day))
    to_predict_df = generate_dates_until(int(year), int(month), int(day))
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
        "lr/lr_predict.html.jinja",
        ticker=ticker,
        split=split,
        linear_regression_plot=linear_regression_plot
    )
