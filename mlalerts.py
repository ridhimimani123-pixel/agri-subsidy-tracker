import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_anomalies_for_dealer(df_dealer: pd.DataFrame, dealer_code: str):
    """
    df_dealer: dataframe filtered for a single dealer
               columns: date, unique_farmers_24h, total_transactions_24h
    dealer_code: string, e.g. "D123"

    returns: dataframe with extra columns:
             prediction (1 normal, -1 anomaly), status, anomaly_score
    """
    if df_dealer.empty:
        return df_dealer

    features = df_dealer[["unique_farmers_24h", "total_transactions_24h"]].values

    # If there are too few points, just mark all as normal
    if len(df_dealer) < 5:
        df_dealer["prediction"] = 1
        df_dealer["status"] = "normal"
        df_dealer["anomaly_score"] = 0.0
        return df_dealer

    model = IsolationForest(
        n_estimators=100,
        contamination=0.20,  # ~20% of days can be anomalies
        random_state=42,
    )

    model.fit(features)

    preds = model.predict(features)           # 1 = normal, -1 = anomaly
    scores = model.decision_function(features)

    df_dealer["prediction"] = preds
    df_dealer["anomaly_score"] = np.round(scores, 3)
    df_dealer["status"] = df_dealer["prediction"].apply(
        lambda x: "suspicious" if x == -1 else "normal"
    )

    return df_dealer


def run_ml_alerts(csv_path: str = "dealer_stats.csv"):
    """
    Reads the CSV containing daily stats for all dealers,
    runs anomaly detection per dealer,
    prints suspicious days and also returns a combined dataframe.
    """
    df = pd.read_csv(csv_path)

    required_cols = {
        "dealer_code",
        "date",
        "unique_farmers_24h",
        "total_transactions_24h",
    }
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {', '.join(required_cols)}"
        )

    # Ensure correct dtypes
    df["unique_farmers_24h"] = df["unique_farmers_24h"].astype(int)
    df["total_transactions_24h"] = df["total_transactions_24h"].astype(int)

    results = []

    for dealer_code, group in df.groupby("dealer_code"):
        group_sorted = group.sort_values("date").reset_index(drop=True)
        out = detect_anomalies_for_dealer(group_sorted, dealer_code)
        results.append(out)

    combined = pd.concat(results, ignore_index=True)

    # Print suspicious days
    suspicious = combined[combined["status"] == "suspicious"]

    print("=== ML Anomaly Detection: Suspicious Days ===")
    if suspicious.empty:
        print("No suspicious patterns detected.")
    else:
        for _, row in suspicious.iterrows():
            print(
                f"Dealer {row['dealer_code']} | Date {row['date']} | "
                f"Farmers={row['unique_farmers_24h']} | "
                f"Tx={row['total_transactions_24h']} | "
                f"Score={row['anomaly_score']}"
            )

    # You can also save output to CSV for reports
    combined.to_csv("dealer_stats_with_alerts.csv", index=False)
    print("\nFull results saved to dealer_stats_with_alerts.csv")

    return combined


if __name__ == "__main__":
    run_ml_alerts()