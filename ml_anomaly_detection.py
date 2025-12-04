import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. Load data
df = pd.read_csv("dealer_stats.csv")

# Optional: filter by dealer_id if you want per-dealer model
dealer_id = "D123"
dealer_df = df[df["dealer_id"] == dealer_id].copy()

# 2. Features for ML model
features = dealer_df[["unique_farmers_24h", "total_sales_24h"]]

# 3. Create and train IsolationForest
# contamination ≈ expected percentage of anomalies (e.g., 0.15 = 15%)
model = IsolationForest(
    n_estimators=100,
    contamination=0.15,
    random_state=42
)

model.fit(features)

# 4. Predict anomalies
# prediction: 1 = normal, -1 = anomaly
dealer_df["prediction"] = model.predict(features)
dealer_df["anomaly_score"] = model.decision_function(features)

# 5. Mark rows as suspicious / normal
def label(row):
    return "suspicious" if row["prediction"] == -1 else "normal"

dealer_df["status"] = dealer_df.apply(label, axis=1)

# 6. Show results
print("=== Anomaly detection results for dealer:", dealer_id, "===\n")
print(dealer_df[["date", "unique_farmers_24h", "total_sales_24h", "status", "anomaly_score"]])

print("\n=== Suspicious days only ===\n")
print(dealer_df[dealer_df["status"] == "suspicious"][[
    "date",
    "unique_farmers_24h",
    "total_sales_24h",
    "status",
    "anomaly_score"
]])