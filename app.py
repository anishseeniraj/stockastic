from flask import Flask, render_template, url_for

app = Flask(__name__)


@app.route("/")
def root():
    return "Welcome to Stockastic!"


@app.route("/TICKER/models")
def index():
    return "Requested prediction models"


@app.route("/TICKER/model/MODEL_NAME")
def show():
    return "Descriptive look at model"


@app.route("/TICKER/model/MODEL_NAME/train")
def train():
    return "Train the MODEL_NAME model by tuning the following hyperparameters"


@app.route("/TICKER/model/MODEL_NAME/predict")
def predict():
    return "Predict price of TICKER at a custom date"


if __name__ == "__main__":
    app.run(debug=True)
