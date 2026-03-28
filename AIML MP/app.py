from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "model" / "artifacts.joblib"

app = Flask(__name__, static_folder="static", template_folder="templates")

_artifacts = None


def load_artifacts():
    global _artifacts
    if _artifacts is None:
        if not ARTIFACTS.is_file():
            raise FileNotFoundError(
                f"Model not found at {ARTIFACTS}. Run: python train_model.py"
            )
        _artifacts = joblib.load(ARTIFACTS)
    return _artifacts


FEATURE_KEYS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    if not data:
        return jsonify({"error": "Expected JSON object with feature fields"}), 400

    values = []
    for key in FEATURE_KEYS:
        if key not in data:
            return jsonify({"error": f"Missing field: {key}"}), 400
        try:
            v = float(data[key])
        except (TypeError, ValueError):
            return jsonify({"error": f"Invalid number for {key}"}), 400
        values.append(v)

    art = load_artifacts()
    scaler = art["scaler"]
    classifier = art["classifier"]

    frame = pd.DataFrame([values], columns=FEATURE_KEYS)
    std_data = scaler.transform(frame)
    prediction = classifier.predict(std_data)
    label = int(prediction[0])

    return jsonify(
        {
            "outcome": label,
            "label": "diabetic" if label == 1 else "not_diabetic",
            "message": (
                "The model predicts: diabetic"
                if label == 1
                else "The model predicts: not diabetic"
            ),
        }
    )


@app.route("/api/health")
def health():
    try:
        load_artifacts()
        return jsonify({"status": "ok", "model_loaded": True})
    except FileNotFoundError as e:
        return jsonify({"status": "error", "detail": str(e)}), 503


if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True, host="127.0.0.1", port=5000)
