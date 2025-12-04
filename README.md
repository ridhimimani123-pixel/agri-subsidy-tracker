# agri-subsidy-tracker
# 🚜 AgriSubsidyGuard  
### Auto‑Generated Fraud Investigation for Agricultural Subsidies

Agricultural subsidies for fertilizers, seeds, and equipment are intended to support genuine farmers, yet a significant portion is lost due to fraudulent practices. Dealers may inflate beneficiary lists with ghost farmers, divert subsidized goods to the open market, or submit fake invoices for products never delivered.  

**AgriSubsidyGuard** is a lightweight analytics tool that ingests subsidy transaction data, detects anomalous patterns, and generates investigation‑ready insights to help auditors identify high‑risk dealers and protect real farmers.

---

## ✅ Problem Statement

The current agricultural subsidy distribution process is vulnerable to large‑scale fraud due to the absence of real‑time monitoring and anomaly detection.  
There is a need for a scalable, data‑driven system that can automatically analyse transaction data, flag suspicious dealers and beneficiaries, and provide structured fraud investigation reports for government stakeholders.

---

## 🚀 Proposed Solution

We build a Python–Flask based backend with an HTML/JS dashboard that:

- Accepts subsidy transaction data as a CSV file  
- Runs anomaly & rule‑based analysis on dealers and farmers  
- Computes risk scores and flags suspicious entities  
- Displays insights in a clean web dashboard  
- (Extensible) Can auto‑generate PDF fraud investigation reports

---

## 🔑 Key Features

- **Automated anomaly detection**  
  Detects unusual patterns such as high fertilizer volume, abnormal farmer counts, and land‑to‑fertilizer mismatches.

- **Dealer risk scoring**  
  Assigns each dealer a 0–100 risk score based on statistical outliers and rule‑based fraud indicators.

- **Farmer‑level anomaly flags**  
  Identifies potential ghost or suspicious beneficiaries (e.g., fertilizer quantity too high for land size).

- **Interactive web dashboard**  
  View total dealers, suspicious dealers, and detailed tables for high‑risk dealers and farmer anomalies.

- **Simple CSV‑based workflow**  
  Works with basic CSV input, making it easy to test with synthetic or real data.

- **Extensible architecture**  
  Ready to plug in PDF report generation, ML models (IsolationForest), and integration with government MIS systems.

---

## 🧰 Tech Stack

### Frontend
- **HTML5, CSS3** – UI layout and styling  
- **Vanilla JavaScript (Fetch API)** – Sending CSV file to backend, rendering results  
- Custom tables for listing suspicious dealers & farmer anomalies  

### Backend
- **Python** – Core data processing  
- **Flask** – REST API for `/analyze` endpoint  
- **Pandas, NumPy** – Data cleaning, aggregation, anomaly metrics  

### (Optional Extensions)
- **Scikit‑learn** – Advanced anomaly detection  
- **ReportLab / pdfkit** – Auto‑generated PDF fraud reports  
- **Plotly / Matplotlib** – Visual charts for insights  

---

## 📂 Project Structure

```bash
project-root/
├── app.py          # Flask backend (analysis logic + API)
├── index.html      # Frontend dashboard UI
├── requirements.txt (optional)  # Python dependencies
└── sample_data.csv # Example transactions CSV (optional)

