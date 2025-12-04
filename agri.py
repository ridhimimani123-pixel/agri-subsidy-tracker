from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
import io

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, ForeignKey, func
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# ==========================
# DATABASE SETUP (SQLite)
# ==========================

DATABASE_URL = "sqlite:///./subsidy.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Dealer(Base):
    __tablename__ = "dealers"

    id = Column(Integer, primary_key=True, index=True)
    dealer_code = Column(String, unique=True, index=True)  # e.g. "D123"
    name = Column(String)
    lat = Column(Float)
    lon = Column(Float)

    transactions = relationship("Transaction", back_populates="dealer")


class Farmer(Base):
    __tablename__ = "farmers"

    id = Column(Integer, primary_key=True, index=True)
    farmer_code = Column(String, unique=True, index=True)  # e.g. "F101"
    name = Column(String)
    lat = Column(Float)
    lon = Column(Float)

    transactions = relationship("Transaction", back_populates="farmer")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(String, index=True)
    status = Column(String, default="pending")  # pending / delivered / alert
    timestamp = Column(DateTime, default=datetime.utcnow)

    dealer_id = Column(Integer, ForeignKey("dealers.id"))
    farmer_id = Column(Integer, ForeignKey("farmers.id"))

    dealer = relationship("Dealer", back_populates="transactions")
    farmer = relationship("Farmer", back_populates="transactions")


Base.metadata.create_all(bind=engine)

# ==========================
# UTILS
# ==========================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance in km between two lat/lon points."""
    R = 6371.0
    lat1_r, lon1_r, lat2_r, lon2_r = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# ==========================
# Pydantic Schemas
# ==========================

class TransactionCreate(BaseModel):
    item_id: str
    dealer_code: str
    farmer_code: str
    status: str = "pending"  # optional: pending/delivered/alert


class TransactionOut(BaseModel):
    id: int
    item_id: str
    dealer_code: str
    farmer_code: str
    status: str
    timestamp: datetime

    class Config:
        orm_mode = True


class SummaryStats(BaseModel):
    total_scans: int
    delivered: int
    pending: int
    alerts: int


class FarmerRadiusStats(BaseModel):
    dealer_code: str
    radius_km: float
    unique_farmers_last_24h: int
    unique_farmers_last_7d: int
    avg_per_day: float
    is_spike: bool
    is_low: bool


class MLDayAnomaly(BaseModel):
    date: str
    unique_farmers_24h: int
    total_transactions_24h: int
    status: str  # normal / suspicious
    anomaly_score: float


class MLAnomalyResponse(BaseModel):
    dealer_code: str
    days: List[MLDayAnomaly]


# ==========================
# FASTAPI APP
# ==========================

app = FastAPI(title="Agri Subsidy Tracker Backend")

# Allow frontend at localhost / any origin (for hackathon, it's fine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# SEED SAMPLE DATA (optional)
# ==========================

def seed_if_empty(db: Session):
    if db.query(Dealer).first():
        return  # already seeded

    # Some dealers
    d1 = Dealer(dealer_code="D123", name="Green Fert Dealer", lat=23.03, lon=72.58)
    d2 = Dealer(dealer_code="D345", name="Shakti Agro", lat=22.57, lon=88.36)

    # Some farmers
    f1 = Farmer(farmer_code="F101", name="Farmer A", lat=23.05, lon=72.60)
    f2 = Farmer(farmer_code="F202", name="Farmer B", lat=23.08, lon=72.62)
    f3 = Farmer(farmer_code="F303", name="Farmer C", lat=22.58, lon=88.40)

    db.add_all([d1, d2, f1, f2, f3])
    db.commit()

    # Some transactions (past 10 days)
    now = datetime.utcnow()
    txs = []
    for i in range(10):
        day = now - timedelta(days=10 - i)
        txs.append(Transaction(
            item_id=f"ITEM-{i+1:03d}",
            status="delivered" if i % 3 != 0 else "pending",
            timestamp=day,
            dealer=d1 if i % 2 == 0 else d2,
            farmer=f1 if i % 3 == 0 else (f2 if i % 3 == 1 else f3),
        ))
    db.add_all(txs)
    db.commit()


@app.on_event("startup")
def on_startup():
    db = SessionLocal()
    seed_if_empty(db)
    db.close()

# ==========================
# ROUTES
# ==========================

@app.get("/", tags=["root"])
def root():
    return {"message": "Agri Subsidy Tracker Backend is running"}


# ---- Create a transaction ----

@app.post("/api/transactions", response_model=TransactionOut, tags=["transactions"])
def create_transaction(payload: TransactionCreate, db: Session = Depends(get_db)):
    dealer = db.query(Dealer).filter(Dealer.dealer_code == payload.dealer_code).first()
    if not dealer:
        raise HTTPException(status_code=404, detail="Dealer not found")

    farmer = db.query(Farmer).filter(Farmer.farmer_code == payload.farmer_code).first()
    if not farmer:
        raise HTTPException(status_code=404, detail="Farmer not found")

    tx = Transaction(
        item_id=payload.item_id,
        status=payload.status,
        dealer=dealer,
        farmer=farmer,
        timestamp=datetime.utcnow(),
    )
    db.add(tx)
    db.commit()
    db.refresh(tx)

    return TransactionOut(
        id=tx.id,
        item_id=tx.item_id,
        dealer_code=dealer.dealer_code,
        farmer_code=farmer.farmer_code,
        status=tx.status,
        timestamp=tx.timestamp,
    )


# ---- Recent transactions ----

@app.get("/api/transactions/recent", response_model=List[TransactionOut], tags=["transactions"])
def recent_transactions(limit: int = 20, db: Session = Depends(get_db)):
    txs = (
        db.query(Transaction, Dealer.dealer_code, Farmer.farmer_code)
        .join(Dealer, Transaction.dealer_id == Dealer.id)
        .join(Farmer, Transaction.farmer_id == Farmer.id)
        .order_by(Transaction.timestamp.desc())
        .limit(limit)
        .all()
    )

    result = []
    for tx, dealer_code, farmer_code in txs:
        result.append(
            TransactionOut(
                id=tx.id,
                item_id=tx.item_id,
                dealer_code=dealer_code,
                farmer_code=farmer_code,
                status=tx.status,
                timestamp=tx.timestamp,
            )
        )
    return result


# ---- Summary stats for dashboard ----

@app.get("/api/stats/summary", response_model=SummaryStats, tags=["dashboard"])
def summary_stats(db: Session = Depends(get_db)):
    total_scans = db.query(func.count(Transaction.id)).scalar() or 0
    delivered = (
        db.query(func.count(Transaction.id))
        .filter(Transaction.status == "delivered")
        .scalar()
        or 0
    )
    pending = (
        db.query(func.count(Transaction.id))
        .filter(Transaction.status == "pending")
        .scalar()
        or 0
    )
    alerts = (
        db.query(func.count(Transaction.id))
        .filter(Transaction.status == "alert")
        .scalar()
        or 0
    )

    return SummaryStats(
        total_scans=total_scans,
        delivered=delivered,
        pending=pending,
        alerts=alerts,
    )


# ---- Farmers in radius + spike/low detection ----

@app.get(
    "/api/dealers/{dealer_code}/farmer-stats",
    response_model=FarmerRadiusStats,
    tags=["analytics"],
)
def farmer_radius_stats(
    dealer_code: str,
    radius_km: float = Query(10.0, gt=0),
    db: Session = Depends(get_db),
):
    dealer = db.query(Dealer).filter(Dealer.dealer_code == dealer_code).first()
    if not dealer:
        raise HTTPException(status_code=404, detail="Dealer not found")

    now = datetime.utcnow()
    last_24h = now - timedelta(hours=24)
    last_7d = now - timedelta(days=7)

    # Get all transactions in last 7 days for this dealer
    txs = (
        db.query(Transaction, Farmer)
        .join(Farmer, Transaction.farmer_id == Farmer.id)
        .filter(
            Transaction.dealer_id == dealer.id,
            Transaction.timestamp >= last_7d,
        )
        .all()
    )

    farmers_24h = set()
    farmers_7d = set()

    for tx, farmer in txs:
        if None in (farmer.lat, farmer.lon, dealer.lat, dealer.lon):
            continue
        dist = haversine_km(dealer.lat, dealer.lon, farmer.lat, farmer.lon)
        if dist <= radius_km:
            farmers_7d.add(farmer.id)
            if tx.timestamp >= last_24h:
                farmers_24h.add(farmer.id)

    count_24h = len(farmers_24h)
    count_7d = len(farmers_7d)
    avg_per_day = count_7d / 7 if count_7d > 0 else 0.0

    is_spike = False
    is_low = False
    if avg_per_day > 0:
        is_spike = count_24h > avg_per_day * 2
        is_low = count_24h < avg_per_day * 0.5

    return FarmerRadiusStats(
        dealer_code=dealer_code,
        radius_km=radius_km,
        unique_farmers_last_24h=count_24h,
        unique_farmers_last_7d=count_7d,
        avg_per_day=round(avg_per_day, 2),
        is_spike=is_spike,
        is_low=is_low,
    )


# ---- ML anomaly detection (daily) ----

@app.get(
    "/api/ml/dealers/{dealer_code}/daily-anomalies",
    response_model=MLAnomalyResponse,
    tags=["ml"],
)
def ml_daily_anomalies(
    dealer_code: str,
    db: Session = Depends(get_db),
):
    dealer = db.query(Dealer).filter(Dealer.dealer_code == dealer_code).first()
    if not dealer:
        raise HTTPException(status_code=404, detail="Dealer not found")

    # Group transactions by date
    txs = (
        db.query(
            func.date(Transaction.timestamp).label("day"),
            func.count(Transaction.id).label("total_tx"),
            func.count(func.distinct(Transaction.farmer_id)).label("unique_farmers"),
        )
        .filter(Transaction.dealer_id == dealer.id)
        .group_by(func.date(Transaction.timestamp))
        .order_by(func.date(Transaction.timestamp))
        .all()
    )

    if not txs:
        return MLAnomalyResponse(dealer_code=dealer_code, days=[])

    df = pd.DataFrame(
        [
            {
                "date": str(day),
                "total_tx": total_tx,
                "unique_farmers": unique_farmers,
            }
            for day, total_tx, unique_farmers in txs
        ]
    )

    features = df[["unique_farmers", "total_tx"]].values

    if len(df) < 5:
        # Not enough data for ML, mark all as normal
        days = [
            MLDayAnomaly(
                date=row["date"],
                unique_farmers_24h=int(row["unique_farmers"]),
                total_transactions_24h=int(row["total_tx"]),
                status="normal",
                anomaly_score=0.0,
            )
            for _, row in df.iterrows()
        ]
        return MLAnomalyResponse(dealer_code=dealer_code, days=days)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.2,  # 20% of days can be anomalies
        random_state=42,
    )
    model.fit(features)

    preds = model.predict(features)  # 1 = normal, -1 = anomaly
    scores = model.decision_function(features)

    days = []
    for i, row in df.iterrows():
        status = "suspicious" if preds[i] == -1 else "normal"
        days.append(
            MLDayAnomaly(
                date=row["date"],
                unique_farmers_24h=int(row["unique_farmers"]),
                total_transactions_24h=int(row["total_tx"]),
                status=status,
                anomaly_score=float(round(scores[i], 3)),
            )
        )

    return MLAnomalyResponse(dealer_code=dealer_code, days=days)


# ---- Graph: daily unique farmers vs date (PNG) ----

@app.get(
    "/api/plots/dealers/{dealer_code}/daily.png",
    tags=["plots"],
    responses={200: {"content": {"image/png": {}}}},
)
def daily_plot(
    dealer_code: str,
    db: Session = Depends(get_db),
):
    dealer = db.query(Dealer).filter(Dealer.dealer_code == dealer_code).first()
    if not dealer:
        raise HTTPException(status_code=404, detail="Dealer not found")

    txs = (
        db.query(
            func.date(Transaction.timestamp).label("day"),
            func.count(func.distinct(Transaction.farmer_id)).label("unique_farmers"),
        )
        .filter(Transaction.dealer_id == dealer.id)
        .group_by(func.date(Transaction.timestamp))
        .order_by(func.date(Transaction.timestamp))
        .all()
    )

    if not txs:
        raise HTTPException(status_code=404, detail="No data for dealer")

    dates = [str(day) for day, _ in txs]
    counts = [int(c) for _, c in txs]

    # Plot
    plt.figure()
    plt.plot(dates, counts, marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Daily Unique Farmers - {dealer_code}")
    plt.xlabel("Date")
    plt.ylabel("Unique Farmers")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")