from app import SessionLocal, Dealer, Farmer, Transaction
from datetime import datetime, timedelta

db = SessionLocal()

# create one dealer + farmers if not exist
d = Dealer(dealer_code="D123", name="Demo Dealer", lat=23.03, lon=72.58)
f1 = Farmer(farmer_code="F101", name="Demo Farmer 1", lat=23.05, lon=72.60)
f2 = Farmer(farmer_code="F202", name="Demo Farmer 2", lat=23.08, lon=72.62)

db.add_all([d, f1, f2])
db.commit()

now = datetime.utcnow()
for i in range(7):
    day = now - timedelta(days=i)
    tx = Transaction(
        item_id=f"ITEM-{i+1:03d}",
        status="delivered",
        timestamp=day,
        dealer=d,
        farmer=f1 if i % 2 == 0 else f2,
    )
    db.add(tx)

db.commit()
db.close()
print("Seeded demo data ✅")