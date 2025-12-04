from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)


# ---------- Helper: core analysis logic ----------

def analyze_transactions(df: pd.DataFrame) -> dict:
    """
    Takes a dataframe of subsidy transactions and returns:
    - dealer_scores: risk metrics per dealer
    - suspicious_dealers: top risky dealers
    - suspicious_farmers: farmer-level anomalies
    """

    # Standardise expected column names (rename if needed)
    col_map = {
        'DealerID': 'dealer_id',
        'Dealer_Id': 'dealer_id',
        'FarmerID': 'farmer_id',
        'Farmer_Id': 'farmer_id',
        'Quantity': 'fertilizer_qty',
        'qty': 'fertilizer_qty',
        'LandArea': 'land_area',
        'Land_Area': 'land_area',
    }
    df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})

    # Fill missing numeric values with 0
    for col in ['fertilizer_qty', 'land_area']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # ------------ Dealer-level metrics ------------

    if 'dealer_id' not in df.columns:
        raise ValueError("Input data must contain a 'dealer_id' column")

    dealer_grp = df.groupby('dealer_id')

    dealer_stats = dealer_grp.agg(
        total_transactions=('dealer_id', 'count'),
        total_fertilizer=('fertilizer_qty', 'sum') if 'fertilizer_qty' in df.columns else ('dealer_id', 'count'),
        unique_farmers=('farmer_id', 'nunique') if 'farmer_id' in df.columns else ('dealer_id', 'count'),
        avg_land_area=('land_area', 'mean') if 'land_area' in df.columns else ('dealer_id', 'count')
    ).reset_index()

    # Derived metrics
    dealer_stats['fertilizer_per_farmer'] = dealer_stats['total_fertilizer'] / dealer_stats['unique_farmers'].replace(0, np.nan)

    # Z-score based anomaly for total_fertilizer
    def zscore(series):
        if series.std() == 0:
            return pd.Series([0] * len(series), index=series.index)
        return (series - series.mean()) / series.std()

    dealer_stats['fertilizer_zscore'] = zscore(dealer_stats['total_fertilizer'])

    # Simple rule-based flags
    dealer_stats['flag_high_volume'] = dealer_stats['fertilizer_zscore'] > 2  # > 2 std dev above mean
    dealer_stats['flag_many_farmers'] = dealer_stats['unique_farmers'] > (dealer_stats['unique_farmers'].mean() + 2 * dealer_stats['unique_farmers'].std())

    # Risk score (0–100 simple formula)
    dealer_stats['risk_score'] = (
        dealer_stats['fertilizer_zscore'].clip(lower=0) * 20 +
        dealer_stats['flag_high_volume'].astype(int) * 30 +
        dealer_stats['flag_many_farmers'].astype(int) * 30
    )

    # Normalise risk_score
    max_risk = dealer_stats['risk_score'].max()
    if max_risk > 0:
        dealer_stats['risk_score'] = (dealer_stats['risk_score'] / max_risk * 100).round(1)
    else:
        dealer_stats['risk_score'] = 0

    # ------------ Farmer-level anomalies ------------

    farmer_anomalies = []
    if {'farmer_id', 'fertilizer_qty', 'land_area'}.issubset(df.columns):
        # Fertilizer > 2x land area (very rough rule)
        cond = df['fertilizer_qty'] > (2 * df['land_area'])
        sus = df[cond].copy()
        for _, row in sus.iterrows():
            farmer_anomalies.append({
                "farmer_id": str(row['farmer_id']),
                "dealer_id": str(row['dealer_id']),
                "reason": "Fertilizer quantity unusually high for land area",
                "fertilizer_qty": float(row['fertilizer_qty']),
                "land_area": float(row['land_area'])
            })

    # ------------ Prepare JSON-friendly output ------------

    dealer_scores = dealer_stats.sort_values('risk_score', ascending=False)

    suspicious_dealers = dealer_scores[dealer_scores['risk_score'] > 50].head(20)
    suspicious_dealers_json = suspicious_dealers.to_dict(orient='records')
    dealer_scores_json = dealer_scores.to_dict(orient='records')

    return {
        "summary": {
            "total_dealers": int(dealer_scores['dealer_id'].nunique()),
            "suspicious_dealers_count": int(len(suspicious_dealers)),
            "total_transactions": int(len(df))
        },
        "dealer_scores": dealer_scores_json,
        "suspicious_dealers": suspicious_dealers_json,
        "suspicious_farmers": farmer_anomalies
    }


# ---------- Flask routes ----------

@app.route("/")
def home():
    return jsonify({"message": "Subsidy Fraud Detection Backend is running 👋"})


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Expect: multipart/form-data with 'file' = CSV
    Returns: JSON with risk scores & anomalies
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded. Use form-data with key 'file'."}), 400

    file = request.files['file']

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400

    try:
        result = analyze_transactions(df)
    except Exception as e:
        return jsonify({"error": f"Analysis error: {str(e)}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    # For local dev; on production use gunicorn/uvicorn
    app.run(host="0.0.0.0", port=5000, debug=True)