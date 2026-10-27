"""Tool schemas exposed to the model and the dispatcher that runs them.

The dispatcher holds per-conversation state: the verified customer and the PII
vault. Account tools refuse to run until verify_customer succeeds, and the model
only ever sees PII tokens such as [PHONE_1], which are resolved here.
"""
import json

from .backends import PolicyError
from .pii import digits

TOOLS = [
    {
        "name": "verify_customer",
        "description": "Verify the caller with the account phone number and billing ZIP code. "
        "Required before any account, billing or ticket tool. Pass PII tokens such as [PHONE_1] as given.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"phone": {"type": "string"}, "zip_code": {"type": "string"}},
            "required": ["phone", "zip_code"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_invoices",
        "description": "Return the verified customer's last two invoices with line items (amounts in cents).",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "name": "issue_credit",
        "description": "Apply a bill credit. Use reason 'late_fee_waiver' only when policy allows a waiver; "
        "the backend enforces eligibility and amount limits.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "amount_cents": {"type": "integer"},
                "reason": {"type": "string", "enum": ["late_fee_waiver", "goodwill"]},
            },
            "required": ["amount_cents", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "create_ticket",
        "description": "Open a ticket for a human team. Use this to escalate anything you cannot resolve.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["network", "device", "billing", "retention", "other"]},
                "summary": {"type": "string"},
            },
            "required": ["category", "summary"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_kb",
        "description": "Search the policy knowledge base. Answer policy questions only from what this returns.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]

ACCOUNT_TOOLS = {"get_invoices", "issue_credit", "create_ticket"}


class Dispatcher:
    def __init__(self, backends, kb, vault=None):
        self.backends = backends
        self.kb = kb
        self.vault = vault or {}
        self.customer = None
        self.calls = []  # (name, input, ok) for grading and audit

    def run(self, name, args):
        """Execute one tool call. Returns (content_str, is_error)."""
        try:
            result = self._run(name, args)
            self.calls.append((name, args, True))
            return json.dumps(result), False
        except (PolicyError, PermissionError, ValueError, KeyError) as e:
            self.calls.append((name, args, False))
            return f"{type(e).__name__}: {e}", True

    def _run(self, name, args):
        if name in ACCOUNT_TOOLS and self.customer is None:
            raise PermissionError("verify the customer first")
        if name == "verify_customer":
            phone = digits(self.vault.get(args["phone"], args["phone"]))
            zip_code = self.vault.get(args["zip_code"], args["zip_code"]).strip()
            found = self.backends.verify_customer(phone, zip_code)
            if not found:
                raise ValueError("no account matches that phone number and ZIP code")
            self.customer = found
            return {"verified": True, "name": found["name"].split()[0], "plan": found["plan"]}
        if name == "get_invoices":
            return self.backends.invoices(self.customer["id"])
        if name == "issue_credit":
            cid = self.backends.issue_credit(self.customer["id"], args["amount_cents"], args["reason"])
            return {"credit_id": cid, "amount_cents": args["amount_cents"]}
        if name == "create_ticket":
            tid = self.backends.create_ticket(self.customer["id"], args["category"], args["summary"])
            return {"ticket_id": tid, "category": args["category"]}
        if name == "search_kb":
            return [{"doc": h["doc"], "text": h["text"]} for h in self.kb.search(args["query"])]
        raise KeyError(f"unknown tool {name}")
