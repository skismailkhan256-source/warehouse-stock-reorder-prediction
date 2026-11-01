
# ============================================================
# WAREHOUSE STOCK REORDER PREDICTION SYSTEM
# Flask Backend Application
# ============================================================

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import os
import io


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "retail_store_inventory.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "demand_model.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_columns.pkl"
)


# ============================================================
# LOAD DATA, MODEL AND FEATURES
# ============================================================

try:

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Convert Date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

    # Load trained ML model
    model = joblib.load(MODEL_PATH)

    # Load feature columns used during training
    feature_columns = joblib.load(FEATURES_PATH)

    print("Dataset loaded successfully.")
    print("Model loaded successfully.")
    print("Feature columns loaded successfully.")

except Exception as e:

    print("ERROR while loading project files:")
    print(e)

    df = pd.DataFrame()
    model = None
    feature_columns = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_latest_inventory():
    """
    Returns the latest inventory record for every
    Store + Product combination.
    """

    if df.empty:
        return pd.DataFrame()

    latest = (
        df.sort_values("Date")
        .groupby(
            ["Store ID", "Product ID"],
            as_index=False
        )
        .tail(1)
        .copy()
    )

    return latest


def get_dashboard_data():
    """
    Calculates dashboard KPI and chart data.
    """

    latest_inventory = get_latest_inventory()

    if latest_inventory.empty:
        return {
            "total_inventory": 0,
            "total_units_sold": 0,
            "total_products": 0,
            "low_stock_items": 0,
            "reorder_required": 0,
            "stockout_risk": 0,
            "category_sales": {},
            "category_percentages": {},
            "region_inventory": {},
            "region_percentages": {}
        }

    # --------------------------------------------------------
    # KPI DATA
    # --------------------------------------------------------

    total_inventory = int(
        latest_inventory["Inventory Level"].sum()
    )

    total_units_sold = int(
        df["Units Sold"].sum()
    )

    total_products = int(
        df["Product ID"].nunique()
    )

    # Low stock
    low_stock_items = int(
        (latest_inventory["Inventory Level"] < 100).sum()
    )

    # Reorder required
    reorder_required = int(
        (latest_inventory["Inventory Level"] < 100).sum()
    )

    # Stockout risk
    stockout_risk = int(
        (latest_inventory["Inventory Level"] < 50).sum()
    )

    # --------------------------------------------------------
    # CATEGORY SALES
    # --------------------------------------------------------

    category_sales_series = (
        df.groupby("Category")["Units Sold"]
        .sum()
        .sort_values(ascending=False)
    )

    category_sales = {
        str(k): int(v)
        for k, v in category_sales_series.items()
    }

    total_category_sales = sum(
        category_sales.values()
    )

    if total_category_sales > 0:

        category_percentages = {
            k: round(
                (v / total_category_sales) * 100,
                2
            )
            for k, v in category_sales.items()
        }

    else:

        category_percentages = {}

    # --------------------------------------------------------
    # REGION INVENTORY
    # --------------------------------------------------------

    region_inventory_series = (
        latest_inventory
        .groupby("Region")["Inventory Level"]
        .sum()
        .sort_values(ascending=False)
    )

    region_inventory = {
        str(k): int(v)
        for k, v in region_inventory_series.items()
    }

    total_region_inventory = sum(
        region_inventory.values()
    )

    if total_region_inventory > 0:

        region_percentages = {
            k: round(
                (v / total_region_inventory) * 100,
                2
            )
            for k, v in region_inventory.items()
        }

    else:

        region_percentages = {}

    return {
        "total_inventory": total_inventory,
        "total_units_sold": total_units_sold,
        "total_products": total_products,
        "low_stock_items": low_stock_items,
        "reorder_required": reorder_required,
        "stockout_risk": stockout_risk,
        "category_sales": category_sales,
        "category_percentages": category_percentages,
        "region_inventory": region_inventory,
        "region_percentages": region_percentages
    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    dashboard_data = get_dashboard_data()

    return render_template(
        "dashboard.html",

        total_inventory=dashboard_data[
            "total_inventory"
        ],

        total_units_sold=dashboard_data[
            "total_units_sold"
        ],

        total_products=dashboard_data[
            "total_products"
        ],

        low_stock_items=dashboard_data[
            "low_stock_items"
        ],

        reorder_required=dashboard_data[
            "reorder_required"
        ],

        stockout_risk=dashboard_data[
            "stockout_risk"
        ],

        category_sales=dashboard_data[
            "category_sales"
        ],

        category_percentages=dashboard_data[
            "category_percentages"
        ],

        region_inventory=dashboard_data[
            "region_inventory"
        ],

        region_percentages=dashboard_data[
            "region_percentages"
        ]
    )


# ============================================================
# INVENTORY
# ============================================================

@app.route("/inventory")
def inventory():

    latest_inventory = get_latest_inventory()

    if latest_inventory.empty:

        inventory_data = []

        total_products = 0
        total_inventory = 0
        low_stock_items = 0
        reorder_required = 0

    else:

        # Sort product records
        latest_inventory = latest_inventory.sort_values(
            ["Product ID", "Store ID"]
        )

        # ----------------------------------------------------
        # Inventory Status
        # ----------------------------------------------------

        def get_status(inventory):

            if inventory < 50:
                return "Critical"

            elif inventory < 100:
                return "Low"

            else:
                return "Good"

        latest_inventory["Status"] = (
            latest_inventory["Inventory Level"]
            .apply(get_status)
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        total_products = int(
            latest_inventory["Product ID"].nunique()
        )

        total_inventory = int(
            latest_inventory["Inventory Level"].sum()
        )

        low_stock_items = int(
            (latest_inventory["Inventory Level"] < 100).sum()
        )

        reorder_required = int(
            (latest_inventory["Inventory Level"] < 100).sum()
        )

        # ----------------------------------------------------
        # Convert to records
        # ----------------------------------------------------

        inventory_data = (
            latest_inventory[
                [
                    "Product ID",
                    "Store ID",
                    "Category",
                    "Region",
                    "Inventory Level",
                    "Units Sold",
                    "Units Ordered",
                    "Price",
                    "Discount",
                    "Status"
                ]
            ]
            .to_dict(orient="records")
        )

    return render_template(
        "inventory.html",

        inventory_data=inventory_data,

        total_products=total_products,

        total_inventory=total_inventory,

        low_stock_items=low_stock_items,

        reorder_required=reorder_required
    )


# ============================================================
# FORECASTING PAGE
# ============================================================

@app.route("/forecasting")
def forecasting():

    return render_template(
        "forecasting.html"
    )


# ============================================================
# DEMAND PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Accept both JSON and HTML form data
        # ----------------------------------------------------

        if request.is_json:

            data = request.get_json()

        else:

            data = request.form.to_dict()

        # ----------------------------------------------------
        # Validate model
        # ----------------------------------------------------

        if model is None:

            error_message = (
                "Machine learning model could not be loaded."
            )

            if request.is_json:

                return jsonify({
                    "error": error_message
                }), 500

            return render_template(
                "forecasting.html",
                error=error_message
            )

        # ----------------------------------------------------
        # Prepare input data
        # ----------------------------------------------------

        input_data = {

            "Store ID":
                data.get("Store ID", "S001"),

            "Product ID":
                data.get("Product ID", "P0001"),

            "Category":
                data.get("Category", "Groceries"),

            "Region":
                data.get("Region", "North"),

            "Inventory Level":
                float(
                    data.get(
                        "Inventory Level",
                        100
                    )
                ),

            "Units Ordered":
                float(
                    data.get(
                        "Units Ordered",
                        100
                    )
                ),

            "Price":
                float(
                    data.get(
                        "Price",
                        50
                    )
                ),

            "Discount":
                float(
                    data.get(
                        "Discount",
                        0
                    )
                ),

            "Weather Condition":
                data.get(
                    "Weather Condition",
                    "Sunny"
                ),

            "Holiday/Promotion":
                data.get(
                    "Holiday/Promotion",
                    "No"
                ),

            "Competitor Pricing":
                float(
                    data.get(
                        "Competitor Pricing",
                        50
                    )
                ),

            "Seasonality":
                data.get(
                    "Seasonality",
                    "Summer"
                )
        }

        # ----------------------------------------------------
        # Create DataFrame
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [input_data]
        )

        # ----------------------------------------------------
        # One-hot encoding
        # ----------------------------------------------------

        input_encoded = pd.get_dummies(
            input_df
        )

        # ----------------------------------------------------
        # Match training features exactly
        # ----------------------------------------------------

        input_encoded = input_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_encoded
        )[0]

        prediction = max(
            0,
            float(prediction)
        )

        prediction = round(
            prediction,
            2
        )

        # ----------------------------------------------------
        # JSON response
        # ----------------------------------------------------

        if request.is_json:

            return jsonify({
                "predicted_demand": prediction
            })

        # ----------------------------------------------------
        # HTML response
        # ----------------------------------------------------

        return render_template(
            "forecasting.html",
            prediction=prediction,
            form_data=input_data
        )

    except Exception as e:

        print(
            "Prediction Error:",
            e
        )

        if request.is_json:

            return jsonify({
                "error": str(e)
            }), 400

        return render_template(
            "forecasting.html",
            error=(
                "Unable to generate prediction. "
                "Please check the input values."
            )
        )


# ============================================================
# REORDER PAGE
# ============================================================

@app.route("/reorder")
def reorder():

    return render_template(
        "reorder.html"
    )


# ============================================================
# REORDER CALCULATION
# ============================================================

@app.route(
    "/calculate-reorder",
    methods=["POST"]
)
def calculate_reorder():

    try:

        # ----------------------------------------------------
        # Get form data
        # ----------------------------------------------------

        product_id = request.form.get(
            "product_id",
            "P0001"
        )

        inventory = float(
            request.form.get(
                "inventory",
                0
            )
        )

        demand = float(
            request.form.get(
                "demand",
                0
            )
        )

        lead_time = float(
            request.form.get(
                "lead_time",
                1
            )
        )

        safety_factor = float(
            request.form.get(
                "safety_factor",
                0.20
            )
        )

        # ----------------------------------------------------
        # Calculations
        # ----------------------------------------------------

        lead_time_demand = (
            demand * lead_time
        )

        safety_stock = (
            demand * safety_factor
        )

        reorder_point = (
            lead_time_demand
            + safety_stock
        )

        order_quantity = max(
            0,
            int(
                np.ceil(
                    reorder_point
                    - inventory
                )
            )
        )

        # ----------------------------------------------------
        # Reorder decision
        # ----------------------------------------------------

        reorder_required = (
            inventory < reorder_point
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        reorder_result = {

            "reorder":
                reorder_required,

            "product_id":
                product_id,

            "inventory":
                round(
                    inventory,
                    2
                ),

            "demand":
                round(
                    demand,
                    2
                ),

            "lead_time":
                round(
                    lead_time,
                    2
                ),

            "safety_stock":
                round(
                    safety_stock,
                    2
                ),

            "reorder_point":
                round(
                    reorder_point,
                    2
                ),

            "order_quantity":
                order_quantity
        }

        return render_template(
            "reorder.html",
            reorder_result=reorder_result
        )

    except Exception as e:

        print(
            "Reorder Calculation Error:",
            e
        )

        return render_template(
            "reorder.html",
            error=(
                "Please enter valid numeric values."
            )
        )


# ============================================================
# REPORTS PAGE
# ============================================================

@app.route("/reports")
def reports():

    dashboard_data = get_dashboard_data()

    return render_template(

        "reports.html",

        total_inventory=dashboard_data[
            "total_inventory"
        ],

        total_units_sold=dashboard_data[
            "total_units_sold"
        ],

        total_products=dashboard_data[
            "total_products"
        ],

        low_stock_items=dashboard_data[
            "low_stock_items"
        ],

        category_sales=dashboard_data[
            "category_sales"
        ],

        category_percentages=dashboard_data[
            "category_percentages"
        ],

        region_inventory=dashboard_data[
            "region_inventory"
        ],

        region_percentages=dashboard_data[
            "region_percentages"
        ]
    )


# ============================================================
# DOWNLOAD INVENTORY REPORT
# ============================================================

@app.route("/download/inventory")
def download_inventory():

    latest_inventory = get_latest_inventory()

    if latest_inventory.empty:

        report_df = pd.DataFrame()

    else:

        def get_status(inventory):

            if inventory < 50:
                return "Critical"

            elif inventory < 100:
                return "Low"

            return "Good"

        latest_inventory["Status"] = (
            latest_inventory["Inventory Level"]
            .apply(get_status)
        )

        report_df = latest_inventory[
            [
                "Product ID",
                "Store ID",
                "Category",
                "Region",
                "Inventory Level",
                "Units Sold",
                "Units Ordered",
                "Price",
                "Discount",
                "Status"
            ]
        ].copy()

    # Convert DataFrame to CSV
    csv_data = report_df.to_csv(
        index=False
    )

    file_data = io.BytesIO(
        csv_data.encode("utf-8")
    )

    file_data.seek(0)

    return send_file(
        file_data,
        mimetype="text/csv",
        as_attachment=True,
        download_name="inventory_report.csv"
    )


# ============================================================
# DOWNLOAD LOW STOCK REPORT
# ============================================================

@app.route("/download/low-stock")
def download_low_stock():

    latest_inventory = get_latest_inventory()

    if latest_inventory.empty:

        report_df = pd.DataFrame()

    else:

        report_df = latest_inventory[
            latest_inventory["Inventory Level"] < 100
        ].copy()

    csv_data = report_df.to_csv(
        index=False
    )

    file_data = io.BytesIO(
        csv_data.encode("utf-8")
    )

    file_data.seek(0)

    return send_file(
        file_data,
        mimetype="text/csv",
        as_attachment=True,
        download_name="low_stock_report.csv"
    )


# ============================================================
# DOWNLOAD SALES REPORT
# ============================================================

@app.route("/download/sales")
def download_sales():

    # Category + Region sales report

    sales_report = (
        df.groupby(
            ["Category", "Region"]
        )["Units Sold"]
        .sum()
        .reset_index()
    )

    sales_report = sales_report.rename(
        columns={
            "Units Sold":
                "Total Units Sold"
        }
    )

    sales_report = sales_report.sort_values(
        "Total Units Sold",
        ascending=False
    )

    csv_data = sales_report.to_csv(
        index=False
    )

    file_data = io.BytesIO(
        csv_data.encode("utf-8")
    )

    file_data.seek(0)

    return send_file(
        file_data,
        mimetype="text/csv",
        as_attachment=True,
        download_name="sales_report.csv"
    )


# ============================================================
# HEALTH CHECK / API STATUS
# ============================================================

@app.route("/api/status")
def api_status():

    return jsonify({

        "status": "running",

        "application":
            "Warehouse Stock Reorder Prediction System",

        "model_loaded":
            model is not None,

        "dataset_loaded":
            not df.empty,

        "total_records":
            int(len(df))
    })


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(
        "home.html"
    ), 404


@app.errorhandler(500)
def internal_server_error(error):

    return """
    <h1>Internal Server Error</h1>
    <p>Please check the Flask terminal for details.</p>
    """, 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print(
        "Warehouse Stock Reorder Prediction System"
    )
    print("=" * 60)

    print(
        "Dataset:",
        "Loaded" if not df.empty else "Not Loaded"
    )

    print(
        "ML Model:",
        "Loaded" if model is not None else "Not Loaded"
    )

    print(
        "Records:",
        len(df)
    )

    print("=" * 60)

    app.run(
        debug=True
    )
