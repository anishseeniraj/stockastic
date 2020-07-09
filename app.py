from flask import Flask, render_template, url_for, request, redirect
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import json
from datetime import datetime
from datetime import timezone
from datetime import date
from dateutil.relativedelta import relativedelta
from utils.stock_preprocess import *
from utils.stock_models import *
from routes.ma import ma
from routes.lr import lr
from routes.knn import knn
from routes.arima import arima
from routes.lstm import lstm

app = Flask(__name__, template_folder="templates")

app.register_blueprint(ma)
app.register_blueprint(lr)
app.register_blueprint(knn)
app.register_blueprint(arima)
app.register_blueprint(lstm)


@app.route("/")
def root():
    return render_template("index.html")


@app.route("/ticker", methods=["POST"])
def ticker():
    ticker = request.form["ticker"]
    df = read_historic_data(ticker)
    historic_plot = historic_model(df)
    plots = []

    for model in ["ma", "lr", "knn", "lstm", "arima"]:
        value = request.form.get(model)

        if value:
            if value == "ma":
                moving_average_plot, ma_rmse = moving_average_model(df)

                plots.append(moving_average_plot)
            elif value == "lr":
                linear_model, linear_fig, linear_regression_plot, lr_rmse = linear_regression_model(
                    df)

                plots.append(linear_regression_plot)
            elif value == "knn":
                k_model, knn_fig, knn_plot, knn_rmse = knn_model(df)

                plots.append(knn_plot)
            elif value == "lstm":
                lstm, lstm_fig, lstm_plot, lstm_rmse = lstm_model(df)

                plots.append(lstm_plot)
            elif value == "arima":
                auto_arima_plot, arima_rmse = auto_arima_model(df)

                plots.append(auto_arima_plot)
        else:
            plots.append("")

    return render_template(
        "models.html.jinja",
        ticker=ticker,
        historic_plot=historic_plot,
        moving_average_plot=plots[0],
        linear_regression_plot=plots[1],
        knn_plot=plots[2],
        lstm_plot=plots[3],
        auto_arima_plot=plots[4]
    )


if __name__ == "__main__":
    app.run(debug=True, threaded=False)
