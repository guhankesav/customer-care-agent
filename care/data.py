"""Seeded SQLite data standing in for the CRM, billing and ticketing systems."""
import random
import sqlite3

SCHEMA = """
CREATE TABLE customers (
  id TEXT PRIMARY KEY, name TEXT, phone TEXT, email TEXT, zip TEXT,
  plan TEXT, status TEXT
);
CREATE TABLE invoices (
  id TEXT PRIMARY KEY, customer_id TEXT, period TEXT, total_cents INTEGER
);
CREATE TABLE line_items (
  invoice_id TEXT, description TEXT, amount_cents INTEGER
);
CREATE TABLE credits (
  id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT, amount_cents INTEGER,
  reason TEXT
);
CREATE TABLE tickets (
  id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT, category TEXT,
  summary TEXT, status TEXT
);
"""

PLANS = {"Essentials": 6000, "Magenta": 7500, "Magenta MAX": 9000}
FIRST = ["Ana", "Ben", "Chen", "Dara", "Eli", "Fatima", "Gus", "Hana", "Ivan", "Jo"]
LAST = ["Lopez", "Smith", "Wang", "Patel", "Okafor", "Kim", "Novak", "Rossi"]

# Extra charges that explain a higher bill; one may be added to the latest invoice.
EXTRAS = [
    ("International roaming - Canada", 2500),
    ("Late payment fee", 700),
    ("Device installment", 3300),
    ("Premium streaming add-on", 1000),
]


def build(n_customers=200, seed=7):
    """Return an in-memory database with n customers, two invoices each."""
    rng = random.Random(seed)
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    for i in range(n_customers):
        cid = f"C{i:04d}"
        plan = rng.choice(list(PLANS))
        phone = f"206555{i:04d}"
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        email = f"{name.split()[0].lower()}{i}@example.com"
        db.execute(
            "INSERT INTO customers VALUES (?,?,?,?,?,?,?)",
            (cid, name, phone, email, f"98{i % 1000:03d}", plan, "active"),
        )
        for period in ("2026-08", "2026-09"):
            items = [(f"{plan} plan", PLANS[plan]), ("Taxes and fees", 850)]
            if period == "2026-09" and rng.random() < 0.6:
                items.append(rng.choice(EXTRAS))
            if period == "2026-08" and rng.random() < 0.15:
                items.append(EXTRAS[1])  # earlier late fee: blocks a second waiver
            inv = f"INV-{cid}-{period}"
            db.execute(
                "INSERT INTO invoices VALUES (?,?,?,?)",
                (inv, cid, period, sum(a for _, a in items)),
            )
            db.executemany(
                "INSERT INTO line_items VALUES (?,?,?)",
                [(inv, d, a) for d, a in items],
            )
    db.commit()
    return db
