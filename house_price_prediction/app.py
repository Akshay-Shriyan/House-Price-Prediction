from flask import Flask, request, jsonify, render_template
import json
import joblib
import numpy as np
import os
import traceback

app = Flask(__name__, template_folder="templates")

# ---------- Helper Functions ----------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().lower().split())

def sanitize_token(s: str) -> str:
    if s is None:
        return ""
    t = normalize(s)
    return (
        t.replace(" ", "_")
        .replace("-", "_")
        .replace(".", "")
        .replace("/", "_")
    )

def find_categorical_column(columns, feature_name, value):
    """
    Tries to map a categorical value to its exact one-hot encoded column.
    Returns index or -1.
    """
    if not value:
        return -1

    value_norm = normalize(value)
    value_token = sanitize_token(value)
    feature_token = sanitize_token(feature_name)

    variants = [
        value_norm,
        value_token,
        f"{feature_name}_{value_token}",
        f"{feature_token}_{value_token}",
        f"{feature_name} {value_norm}",
    ]

    for i, c in enumerate(columns):
        c_norm = normalize(c)
        c_token = sanitize_token(c)

        # direct: "location_whitefield"
        if c_norm == value_norm or c_token == value_token:
            return i

        # match variants
        for v in variants:
            if c_norm == normalize(v) or c_token == sanitize_token(v):
                return i

    # last fallback: substring match (location only)
    for i, c in enumerate(columns):
        if value_norm in normalize(c):
            return i

    return -1


# ---------- Load Model and Columns ----------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "bangalore_home_price_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "columns.json")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file missing! Train and save the model first.")

if not os.path.exists(COLUMNS_PATH):
    raise FileNotFoundError("columns.json missing! Generate it first.")

model = joblib.load(MODEL_PATH)
raw = load_json(COLUMNS_PATH)

data_columns = raw.get("data_columns", [])
if not data_columns:
    raise ValueError("columns.json has no valid 'data_columns'.")

normalized_cols = [normalize(c) for c in data_columns]

# ---------- EXTRACT ONLY LOCATION COLUMNS ----------

locations = []

for col in data_columns:
    col_norm = normalize(col)

    if col_norm.startswith("location_"):
        clean = col_norm.replace("location_", "")
        clean = clean.replace("_", " ").strip()
        clean = " ".join([w.capitalize() for w in clean.split()])
        locations.append(clean)

# Remove dupes & sort
locations = sorted(list(dict.fromkeys(locations)))


# ---------- Routes ----------

@app.route("/")
def home():
    try:
        return render_template("index.html", locations=locations)
    except:
        return "<h3>index.html missing inside templates folder.</h3>"


@app.route("/get_location_names", methods=["GET"])
def get_location_names():
    return jsonify({"locations": locations})


@app.route("/predict_home_price", methods=["POST"])
def predict_home_price():
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Invalid JSON"}), 400

        # Extract inputs
        total_sqft = payload.get("total_sqft")
        bath = payload.get("bath")
        bhk = payload.get("bhk")
        location = payload.get("location")
        age_group = payload.get("age_group")
        facing = payload.get("facing")
        parking = payload.get("parking")

        # Validate required fields
        if total_sqft is None or bath is None or bhk is None:
            return jsonify({"error": "Missing total_sqft, bath, or bhk"}), 400

        # Convert numeric
        try:
            total_sqft = float(total_sqft)
            bath = float(bath)
            bhk = float(bhk)
        except:
            return jsonify({"error": "Numeric fields invalid"}), 400

        # Initialize input vector
        x = np.zeros(len(data_columns), dtype=float)

        # Fill numeric columns
        def set_num(feat, val):
            norm_feat = normalize(feat)
            if norm_feat in normalized_cols:
                idx = normalized_cols.index(norm_feat)
                x[idx] = val

        set_num("total_sqft", total_sqft)
        set_num("bath", bath)
        set_num("bhk", bhk)

        # derived feature
        if "sqft_per_bhk" in normalized_cols:
            idx = normalized_cols.index("sqft_per_bhk")
            x[idx] = total_sqft / bhk if bhk else 0

        # parking
        if parking is not None:
            try:
                set_num("parking", float(parking))
            except:
                pass

        # categorical: location
        if location:
            idx = find_categorical_column(data_columns, "location", location)
            if idx >= 0:
                x[idx] = 1

        # categorical: age_group
        if age_group:
            idx = find_categorical_column(data_columns, "age_group", age_group)
            if idx >= 0:
                x[idx] = 1

        # categorical: facing
        if facing:
            idx = find_categorical_column(data_columns, "facing", facing)
            if idx >= 0:
                x[idx] = 1

        # Predict
        pred = model.predict([x])[0]
        price = float(round(pred, 2))

        return jsonify({"estimated_price": price})

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
