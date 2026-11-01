# Connect the prediction form with the ML model

from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("models/demand_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():

    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()

    numeric_columns = [
        "Inventory Level",
        "Units Ordered",
        "Price",
        "Discount",
        "Holiday/Promotion",
        "Competitor Pricing"
    ]

    for column in numeric_columns:
        data[column] = float(data[column])

    input_data = pd.DataFrame([data])

    input_encoded = pd.get_dummies(input_data, drop_first=True)

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(input_encoded)[0]

    if request.is_json:
        return jsonify({
            "predicted_demand": round(float(prediction), 2)
        })

    return render_template(
        "home.html",
        prediction=round(float(prediction), 2)
    )


if __name__ == "__main__":
    app.run(debug=True)