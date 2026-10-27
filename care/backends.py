"""Thin service layer over the seeded database: CRM, billing and ticketing.

Business rules live here, not in the prompt, so the model cannot talk its way
past them.
"""

LATE_FEE = "Late payment fee"
MAX_CREDIT_CENTS = 5000


class PolicyError(Exception):
    pass


class Backends:
    def __init__(self, db):
        self.db = db

    # CRM
    def verify_customer(self, phone, zip_code):
        row = self.db.execute(
            "SELECT id, name, plan, status FROM customers WHERE phone=? AND zip=?",
            (phone, zip_code),
        ).fetchone()
        return dict(row) if row else None

    # Billing
    def invoices(self, customer_id):
        rows = self.db.execute(
            "SELECT id, period, total_cents FROM invoices WHERE customer_id=? ORDER BY period",
            (customer_id,),
        ).fetchall()
        out = []
        for r in rows:
            items = self.db.execute(
                "SELECT description, amount_cents FROM line_items WHERE invoice_id=?",
                (r["id"],),
            ).fetchall()
            out.append({**dict(r), "line_items": [dict(i) for i in items]})
        return out

    def late_fee_waivable(self, customer_id):
        """First late fee in the last two cycles can be waived once."""
        invs = self.invoices(customer_id)
        fees = [
            (inv["period"], i["amount_cents"])
            for inv in invs
            for i in inv["line_items"]
            if i["description"] == LATE_FEE
        ]
        already = self.db.execute(
            "SELECT COUNT(*) FROM credits WHERE customer_id=? AND reason='late_fee_waiver'",
            (customer_id,),
        ).fetchone()[0]
        latest = invs[-1]["period"] if invs else None
        if already or not fees or fees[-1][0] != latest or len(fees) > 1:
            return None
        return fees[-1][1]

    def issue_credit(self, customer_id, amount_cents, reason):
        if reason == "late_fee_waiver":
            allowed = self.late_fee_waivable(customer_id)
            if allowed is None:
                raise PolicyError("late fee is not eligible for a waiver")
            if amount_cents != allowed:
                raise PolicyError(f"waiver must equal the fee ({allowed} cents)")
        if not 0 < amount_cents <= MAX_CREDIT_CENTS:
            raise PolicyError(f"credits must be between 1 and {MAX_CREDIT_CENTS} cents")
        cur = self.db.execute(
            "INSERT INTO credits (customer_id, amount_cents, reason) VALUES (?,?,?)",
            (customer_id, amount_cents, reason),
        )
        return cur.lastrowid

    def credits(self, customer_id):
        rows = self.db.execute(
            "SELECT amount_cents, reason FROM credits WHERE customer_id=?", (customer_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # Ticketing
    def create_ticket(self, customer_id, category, summary):
        cur = self.db.execute(
            "INSERT INTO tickets (customer_id, category, summary, status) VALUES (?,?,?,'open')",
            (customer_id, category, summary),
        )
        return cur.lastrowid

    def tickets(self, customer_id):
        rows = self.db.execute(
            "SELECT category, summary FROM tickets WHERE customer_id=?", (customer_id,)
        ).fetchall()
        return [dict(r) for r in rows]
