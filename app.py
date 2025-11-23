from flask import Flask, request, render_template, redirect, url_for, flash, send_file
import pickle
import pandas as pd
import io
import os

app = Flask(__name__)
app.secret_key = "titanic-secret-key"  # needed for flash messages

# Load the trained pipeline model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Columns that must be present in uploaded CSV
REQUIRED_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Path where we will store the latest prediction file
RESULTS_DIR = "results"
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "titanic_predictions.csv")

# Make sure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction_table=None, summary=None, download_available=False)

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if "file" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("home"))

    file = request.files["file"]

    if file.filename == "":
        flash("No CSV file selected.")
        return redirect(url_for("home"))

    try:
        data = file.read()
        df = pd.read_csv(io.BytesIO(data))
    except Exception as e:
        flash(f"Error reading CSV file. Please check the format. Details: {e}")
        return redirect(url_for("home"))

    # Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        flash(f"Missing required columns: {', '.join(missing_cols)}")
        return redirect(url_for("home"))

    # Create family features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Build feature DataFrame for model
    X = df[["Pclass", "Sex", "Age", "Fare", "Embarked", "IsAlone"]].copy()

    # Predict for all rows
    try:
        preds = model.predict(X)

        # Try probability if classifier supports it
        try:
            probs = model.predict_proba(X)[:, 1]
            df["Survival_Probability"] = probs
        except Exception:
            df["Survival_Probability"] = None

    except Exception as e:
        flash(f"Error during prediction. Details: {e}")
        return redirect(url_for("home"))

    # Attach predictions
    df["PredictedSurvived"] = preds
    df["PredictedLabel"] = df["PredictedSurvived"].apply(
        lambda x: "Survived" if x == 1 else "Did not survive"
    )

    # Build summary for UI
    total = len(df)
    survived_count = int((df["PredictedSurvived"] == 1).sum())
    not_survived_count = total - survived_count
    survived_pct = (survived_count / total) * 100 if total > 0 else 0
    not_survived_pct = 100 - survived_pct

    summary = {
        "total": total,
        "survived_count": survived_count,
        "not_survived_count": not_survived_count,
        "survived_pct": round(survived_pct, 2),
        "not_survived_pct": round(not_survived_pct, 2),
    }

    # Save full results to CSV for download
    df.to_csv(PREDICTIONS_FILE, index=False)

    # Create preview table
    preview_cols = [
        col for col in df.columns
        if col in [
            "PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch",
            "Fare", "Embarked", "PredictedLabel", "Survival_Probability"
        ]
        and col in df.columns
    ]
    preview_df = df[preview_cols].head(20)
    prediction_table = preview_df.to_html(
        classes="data-table",
        index=False,
        float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else x
    )

    return render_template(
        "index.html",
        prediction_table=prediction_table,
        summary=summary,
        download_available=True
    )

@app.route("/download_predictions", methods=["GET"])
def download_predictions():
    if not os.path.exists(PREDICTIONS_FILE):
        flash("No predictions available to download. Please upload a CSV first.")
        return redirect(url_for("home"))

    return send_file(
        PREDICTIONS_FILE,
        as_attachment=True,
        download_name="titanic_predictions.csv",
        mimetype="text/csv"
    )

if __name__ == "__main__":
    app.run(debug=True)
