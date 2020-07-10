"""
Main application router file

This script contains the homepage routes for the application (serves the 
homepage and reads in initial user-input).

It requires utils.stock_preprocess and utils.stock_models to 
    -> Preprocess raw stock data
    -> Execute models and present visualizations
Addtionally, it requires all the custom router files (routes for each 
machine learning model) to be registered.
"""

from flask import Flask, render_template, url_for, request, redirect
from utils.stock_preprocess import *
from utils.stock_models import *
from routes.ma import ma
from routes.lr import lr
from routes.knn import knn
from routes.arima import arima
from routes.lstm import lstm

app = Flask(__name__, template_folder="templates")

# Register ML models routes
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
    """
    Reads stock data based on the input ticker and generates plots based 
    on the selected models
    """

    ticker = request.form["ticker"]
    df = read_historic_data(ticker)  # read past 5y stock performance data
    historic_plot = historic_model(df)  # plot historic price
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
            plots.append("")  # model not selected

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
