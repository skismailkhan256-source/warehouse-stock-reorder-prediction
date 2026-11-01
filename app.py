# Load the trained ML model

from flask import Flask
import joblib

app = Flask(__name__)

model = joblib.load("models/demand_model.pkl")


@app.route("/")
def home():
    return "Warehouse Stock Reorder Prediction System"


if __name__ == "__main__":
    app.run(debug=True)