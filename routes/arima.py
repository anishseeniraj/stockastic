from flask import Blueprint, render_template, Flask, url_for, request, redirect
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import json
from utils.stock_preprocess import *
from utils.stock_models import *

arima = Blueprint("arima", __name__)


@arima.route("/<ticker>/arima/customize/<split>/<start_p>/<max_p>/<start_q>/<max_q>/<d>/<D>")
def arima_customize_input(ticker, split, start_p, max_p, start_q, max_q, d, D):
    df = read_historic_data(ticker)
    auto_arima_plot, rmse = auto_arima_model(df, int(split), int(
        start_p), int(max_p), int(start_q), int(max_q), int(d), int(D))

    return render_template(
        "arima/arima_customize.html.jinja",
        ticker=ticker,
        auto_arima_plot=auto_arima_plot,
        rmse=rmse,
        split=split,
        start_p=start_p,
        max_p=max_p,
        start_q=start_q,
        max_q=max_q,
        d=d,
        D=D
    )


@arima.route("/arima/customize", methods=["POST"])
def arima_customize_output():
    ticker = request.form["ticker"]
    split = request.form["split"]
    start_p = request.form["start_p"]
    max_p = request.form["max_p"]
    start_q = request.form["start_q"]
    max_q = request.form["max_q"]
    d = request.form["d"]
    D = request.form["D"]

    return redirect("/" + ticker + "/arima/customize/" + split + "/" + start_p + "/" + max_p + "/" + start_q + "/" + max_q + "/" + d + "/" + D)


@arima.route("/<ticker>/arima/predict/<split>/<start_p>/<max_p>/<start_q>/<max_q>/<d>/<D>")
def arima_predict_input(ticker, split, start_p, max_p, start_q, max_q, d, D):
    df = read_historic_data(ticker)
    arima_plot, rmse = auto_arima_model(
        df, int(split), int(start_p), int(max_p), int(start_q), int(max_q), int(d), int(D))

    return render_template(
        "arima/arima_predict.html.jinja",
        ticker=ticker,
        split=split,
        start_p=start_p,
        max_p=max_p,
        start_q=start_q,
        max_q=max_q,
        d=d,
        D=D,
        arima_plot=arima_plot
    )


@arima.route("/arima/predict", methods=["POST"])
def arima_predict_output():
    # Form submission values
    year = request.form["year"]
    month = request.form["month"]
    day = request.form["day"]
    split = request.form["split"]
    ticker = request.form["ticker"]
    start_p = request.form["start_p"]
    max_p = request.form["max_p"]
    start_q = request.form["start_q"]
    max_q = request.form["max_q"]
    d = request.form["d"]
    D = request.form["D"]

    # Reading stock data
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

    arima_plot, rmse = auto_arima_model(
        df, int(split), int(start_p), int(max_p), int(start_q), int(max_q), int(d), int(D), new_predictions=True, new_dates=to_predict)

    return render_template(
        "arima/arima_predict.html.jinja",
        ticker=ticker,
        split=split,
        start_p=start_p,
        max_p=max_p,
        start_q=start_q,
        max_q=max_q,
        d=d,
        D=D,
        arima_plot=arima_plot
    )
