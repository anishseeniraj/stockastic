from flask import Blueprint, render_template, Flask, url_for, request, redirect
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import json
# from datetime import datetime
# from datetime import timezone
# from datetime import date
# from dateutil.relativedelta import relativedelta
from utils.stock_preprocess import *
from utils.stock_models import *

ma = Blueprint("ma", __name__)


@ma.route("/<ticker>/ma/customize/<window>/<split>")
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


@ma.route("/ma/customize", methods=["POST"])
def ma_customize_output():
    window = request.form["window"]
    ticker = request.form["ticker"]
    split = request.form["split"]

    return redirect("/" + ticker + "/ma/customize/" + window + "/" + split)


@ma.route("/<ticker>/ma/predict/<window>/<split>")
def ma_predict_input(ticker, window, split):
    df = read_historic_data(ticker)
    ma_plot, ma_rmse = moving_average_model(
        df, int(window), int(split))

    return render_template(
        "ma_predict.html.jinja",
        ticker=ticker,
        window=window,
        split=split,
        ma_plot=ma_plot
    )


@ma.route("/ma/predict", methods=["POST"])
def ma_predict_output():
    # Form submission values
    year = request.form["year"]
    month = request.form["month"]
    day = request.form["day"]
    ticker = request.form["ticker"]
    window = request.form["window"]
    split = request.form["split"]

    # Generating the linear model
    df = read_historic_data(ticker)

    # Generating the date column for predictions
    # start_date = date.today()
    # end_date = date(int(year), int(month), int(day))
    # Range of prediction dates
    # predict_data = {"Date": pd.date_range(
    #     start=start_date, end=end_date)}
    # df with original prediction dates and dummy close prices
    to_predict = generate_dates_until(int(year), int(month), int(day))
    to_predict["Close"] = np.nan

    # Generating the model
    ma_plot, ma_rmse = moving_average_model(
        df, int(window), int(split), new_predictions=True, new_dates=to_predict)

    return render_template(
        "ma_predict.html.jinja",
        ticker=ticker,
        window=window,
        split=split,
        ma_rmse=ma_rmse,
        ma_plot=ma_plot
    )
