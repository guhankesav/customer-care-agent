"""Scripted control arm: an FAQ bot that answers from the top KB hit and sends
every account request to a human queue. This is the status quo the agent is
measured against."""
import re

from .agent import Result
from .pii import redact
from .tools import Dispatcher

PHONE = re.compile(r"\[PHONE_\d+\]")
ZIP = re.compile(r"\b\d{5}\b")


def run(message, backends, kb):
    text, vault = redact(message)
    tools = Dispatcher(backends, kb, vault)
    phone, zip_code = PHONE.search(text), ZIP.search(text)
    if not phone or not zip_code:
        hits = kb.search(text, k=1)
        reply = hits[0]["text"] if hits else "Sorry, I could not find that. An agent will follow up."
        return Result(reply, tools.calls, "end_turn", 1, redacted_message=text)
    tools.run("verify_customer", {"phone": phone.group(0), "zip_code": zip_code.group(0)})
    if tools.customer is None:
        return Result("We could not verify your account.", tools.calls, "end_turn", 1, redacted_message=text)
    low = text.lower()
    if "cancel" in low:
        category = "retention"
    elif "signal" in low or "outage" in low:
        category = "network"
    else:
        category = "billing"
    tools.run("create_ticket", {"category": category, "summary": text[:200]})
    return Result(f"Thanks, a {category} specialist will contact you.", tools.calls, "end_turn", 1,
                  redacted_message=text, customer_id=tools.customer["id"])
