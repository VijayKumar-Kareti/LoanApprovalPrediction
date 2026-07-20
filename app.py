from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model.pkl"

EXPECTED_FEATURES = [
    "Age",
    "Gender",
    "MaritalStatus",
    "Education",
    "EmploymentType",
    "MonthlyIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "LoanTerm",
    "CreditHistory",
    "PropertyArea",
    "Dependents",
    "SelfEmployed",
]

try:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
except Exception as exc:
    model = None
    model_error = str(exc)


def safe_render(template_name, **context):
    try:
        return render_template(template_name, **context)
    except Exception:
        return f"""
        <html>
            <body style="font-family: Arial; padding: 30px;">
                <h2>{template_name}</h2>
                <p>The template file is not created yet. We will build the UI in the next step.</p>
            </body>
        </html>
        """


@app.route("/")
def home():
    return safe_render("index.html")


@app.route("/about")
def about():
    return safe_render("about.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return safe_render("predict.html")

    if model is None:
        return safe_render(
            "result.html",
            result={
                "loan_status": "Error",
                "probability": 0.0,
                "confidence_score": 0.0,
                "reason": f"Model failed to load: {model_error}",
            },
        )

    form_data = request.form.to_dict()

    # Build a row matching the trained feature columns
    row = {}
    for col in EXPECTED_FEATURES:
        row[col] = form_data.get(col, "")

    df = pd.DataFrame([row], columns=EXPECTED_FEATURES)

    # Convert obvious numeric fields
    numeric_cols = [
        "Age",
        "MonthlyIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "LoanTerm",
        "CreditHistory",
        "Dependents",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Fill the rest as categorical text
    for col in EXPECTED_FEATURES:
        if col not in numeric_cols:
            df[col] = df[col].fillna("Unknown").astype(str)

    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]

    predicted_label = 1 if prediction == 1 else 0
    probability = float(probabilities[predicted_label])
    confidence_score = float(max(probabilities))

    if prediction == 1:
        loan_status = "Approved"
        reason = explain_prediction(df.iloc[0], approved=True)
    else:
        loan_status = "Rejected"
        reason = explain_prediction(df.iloc[0], approved=False)

    result = {
        "loan_status": loan_status,
        "probability": round(probability, 4),
        "confidence_score": round(confidence_score, 4),
        "reason": reason,
    }

    return safe_render("result.html", result=result)


def explain_prediction(row: pd.Series, approved: bool) -> str:
    monthly_income = float(row.get("MonthlyIncome", 0) or 0)
    loan_amount = float(row.get("LoanAmount", 0) or 0)
    credit_history = float(row.get("CreditHistory", 0) or 0)
    employment = str(row.get("EmploymentType", "") or "").lower()

    if approved:
        reasons = []
        if monthly_income > 50000:
            reasons.append("strong monthly income")
        if credit_history >= 1:
            reasons.append("good credit history")
        if loan_amount < 150000:
            reasons.append("manageable loan amount")
        if "salaried" in employment or "self" in employment:
            reasons.append("stable employment")
        if reasons:
            return "Loan Approved because of " + ", ".join(reasons) + "."
        return "Loan Approved based on the provided financial profile."
    else:
        reasons = []
        if monthly_income < 30000:
            reasons.append("low monthly income")
        if credit_history < 1:
            reasons.append("weak credit history")
        if loan_amount > 200000:
            reasons.append("high loan amount")
        if reasons:
            return "Loan Rejected because of " + ", ".join(reasons) + "."
        return "Loan Rejected based on the provided financial profile."


if __name__ == "__main__":
    app.run(debug=True)