from flask import Flask, render_template, url_for, request, redirect

app = Flask(__name__, template_folder="templates")


@app.route("/")
def root():
    return render_template("index.html")


@app.route("/ticker", methods=["POST"])
def ticker():
    ticker = request.form["ticker"]

    return redirect("/" + ticker + "/models")


@app.route("/<ticker>/models")
def index(ticker):
    return render_template("models.html", ticker=ticker)


@app.route("/<ticker>/model/<model_name>")
def show(ticker, model_name):
    return "Descriptive look at model"


@app.route("/<ticker>/model/<model_name>/train")
def train(ticker, model_name):
    return "Train the MODEL_NAME model by tuning the following hyperparameters"


@app.route("/<ticker>/model/<model_name>/predict")
def predict(ticker, model_name):
    return "Predict price of TICKER at a custom date"


if __name__ == "__main__":
    app.run(debug=True)
