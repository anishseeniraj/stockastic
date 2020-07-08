from flask import Blueprint, render_template, Flask, url_for, request, redirect
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import json
from utils.stock_preprocess import *
from utils.stock_models import *

knn = Blueprint("knn", __name__)


@knn.route("/<ticker>/knn/customize/<split>/<neighbors>/<weights>/<power>")
def knn_customize_input(ticker, split, neighbors, weights, power):
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
    ticker = request.form["ticker"]
    split = request.form["split"]
    neighbors = request.form["neighbors"]
    weights = request.form["weights"]
    power = request.form["power"]

    return redirect("/" + ticker + "/knn/customize/" + split + "/" + neighbors + "/" + weights + "/" + power)


@knn.route("/<ticker>/knn/predict/<split>/<neighbors>/<weights>/<power>")
def knn_predict_input(ticker, split, neighbors, weights, power):
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
    # Form submission values
    year = request.form["year"]
    month = request.form["month"]
    day = request.form["day"]
    ticker = request.form["ticker"]
    split = request.form["split"]
    neighbors = request.form["neighbors"]
    weights = request.form["weights"]
    power = request.form["power"]

    # Generating the linear model
    df = read_historic_data(ticker)

    # Generating the date column for predictions
    # start_date = date.today()
    # end_date = date(int(year), int(month), int(day))
    # Range of prediction dates
    # predict_data = {"Date": pd.date_range(start=start_date, end=end_date)}
    # DataFrame with original prediction dates
    predict_dates = generate_dates_until(int(year), int(month), int(day))
    to_predict_df = generate_dates_until(int(year), int(month), int(day))
    to_predict_df["Date"] = to_predict_df["Date"].map(
        datetime.toordinal)  # DataFrame with ordinal prediction dates
    k_model, knn_fig, knn_plot, rmse = knn_model(
        df, int(split), int(neighbors), weights, int(power), new_predictions=True, ordinal_prediction_dates=to_predict_df, original_prediction_dates=predict_dates)

    # Predicting prices on new dates
    new_predictions = k_model.predict(to_predict_df)

    # print("Predictions from returned model")
    # print(new_predictions)

    # Plotting predicted prices
    # knn_fig.add_trace(go.Scatter(
    #     x=predict_dates["Date"],
    #     y=new_predictions,
    #     mode="lines",
    #     name="Forecast"
    # ))

    # knn_plot = json.dumps(
    #     knn_fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "knn/knn_predict.html.jinja",
        ticker=ticker,
        split=split,
        neighbors=neighbors,
        weights=weights,
        power=power,
        knn_plot=knn_plot
    )
