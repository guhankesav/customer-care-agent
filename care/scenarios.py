"""Synthetic customer conversations with gold outcomes, and the grader.

Grading reads backend state (credits, tickets) and the tool log, not just the
reply text, so a confident but wrong answer does not pass.
"""
import random

from .backends import LATE_FEE

KB_QUESTIONS = [
    ("How much high-speed data do I get in Canada on {plan}?", {"Essentials": "5 GB", "Magenta": "15 GB", "Magenta MAX": "15 GB"}),
    ("What does an international roaming day pass cost?", "$10"),
    ("How much is the autopay discount per line?", "$5"),
    ("How much more is Magenta MAX than Magenta per line?", "$15"),
]

EXTRA_KEYWORDS = {
    "International roaming - Canada": "roaming",
    "Late payment fee": "late",
    "Device installment": "installment",
    "Premium streaming add-on": "streaming",
}


def fmt_phone(p):
    return f"({p[:3]}) {p[3:6]}-{p[6:]}"


def build(db, backends, n=200, seed=11):
    rng = random.Random(seed)
    customers = [dict(r) for r in db.execute("SELECT * FROM customers ORDER BY id")]
    rng.shuffle(customers)
    out = []
    for c in customers[:n]:
        invs = backends.invoices(c["id"])
        latest = {i["description"] for i in invs[-1]["line_items"]}
        extras = [d for d in latest if d in EXTRA_KEYWORDS]
        who = f"My number is {fmt_phone(c['phone'])} and my ZIP is {c['zip']}."
        options = ["kb", "outage", "cancel"]
        if extras:
            options.append("bill_increase")
        if LATE_FEE in latest:
            options += ["late_fee"] * 2
        intent = rng.choice(options)
        s = {"customer_id": c["id"], "intent": intent}
        if intent == "kb":
            q, ans = rng.choice(KB_QUESTIONS)
            s["message"] = q.format(plan=c["plan"])
            s["expect"] = ans[c["plan"]] if isinstance(ans, dict) else ans
            s["customer_id"] = None  # anonymous question
        elif intent == "bill_increase":
            s["message"] = f"Why is my bill higher this month? {who}"
            s["expect"] = EXTRA_KEYWORDS[extras[0]]
        elif intent == "late_fee":
            s["message"] = f"I got charged a late fee. Can you take it off? {who}"
            s["expect"] = backends.late_fee_waivable(c["id"])  # cents, or None
        elif intent == "outage":
            s["message"] = f"I have had no signal anywhere around my home since 9am. {who}"
        else:
            s["message"] = f"I want to cancel my service. {who}"
        out.append(s)
    return out


def grade(s, reply, backends):
    """Return dict(resolved, deflected, violation) for one finished conversation."""
    cid = s["customer_id"]
    tickets = backends.tickets(cid) if cid else []
    credits = backends.credits(cid) if cid else []
    cats = {t["category"] for t in tickets}
    text = reply.lower()
    violation = False
    if s["intent"] == "kb":
        resolved = s["expect"].lower() in text
    elif s["intent"] == "bill_increase":
        resolved = s["expect"] in text and not tickets
        violation = bool(credits)
    elif s["intent"] == "late_fee":
        if s["expect"]:
            resolved = credits == [{"amount_cents": s["expect"], "reason": "late_fee_waiver"}] and not tickets
            violation = sum(c["amount_cents"] for c in credits) > s["expect"]
        else:
            resolved = not credits and not tickets
            violation = bool(credits)
    elif s["intent"] == "outage":
        resolved = "network" in cats
        violation = bool(credits)
    else:  # cancel
        resolved = "retention" in cats
        violation = bool(credits)
    deflected = resolved and not tickets
    return {"resolved": resolved, "deflected": deflected, "violation": violation}
