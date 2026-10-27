import json
import re
from types import SimpleNamespace as NS

import pytest

from care import agent, baseline, data, experiment, scenarios
from care.backends import Backends, PolicyError
from care.pii import PATTERNS, redact
from care.rag import BM25, load_chunks
from care.tools import Dispatcher


@pytest.fixture
def env():
    db = data.build(n_customers=60)
    return db, Backends(db), BM25(load_chunks())


def customer_with_waivable_fee(b, db):
    for (cid,) in db.execute("SELECT id FROM customers"):
        if b.late_fee_waivable(cid):
            return dict(db.execute("SELECT * FROM customers WHERE id=?", (cid,)).fetchone())
    raise AssertionError("seed has no waivable fee")


class FakeClient:
    """Plays back scripted turns and records every request sent to the model."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw["messages"], default=lambda o: o.__dict__)))
        blocks, stop = self.turns.pop(0)
        return NS(content=blocks, stop_reason=stop, usage=NS(input_tokens=100, output_tokens=20))


def tool(name, args, i):
    return NS(type="tool_use", name=name, input=args, id=f"tu_{i}")


def text(t):
    return NS(type="text", text=t)


# PII

def test_redact_replaces_phone_email_card_ssn():
    out, vault = redact("Call (206) 555-0012, mail ana1@example.com, card 4111 1111 1111 1111, ssn 123-45-6789")
    assert not any(p.search(out) for _, p in PATTERNS)
    assert set(vault.values()) == {"(206) 555-0012", "ana1@example.com", "4111 1111 1111 1111", "123-45-6789"}


def test_zip_code_is_not_redacted():
    out, _ = redact("my zip is 98012")
    assert "98012" in out


# RAG

def test_bm25_finds_roaming_policy(env):
    _, _, kb = env
    top = kb.search("how much data in Canada", k=1)[0]
    assert top["doc"] == "roaming" and "15 GB" in top["text"]


def test_bm25_finds_late_fee_policy(env):
    _, _, kb = env
    assert kb.search("late fee waived")[0]["doc"] == "billing"


# Tools and policy

def test_account_tools_require_verification(env):
    _, b, kb = env
    d = Dispatcher(b, kb)
    out, err = d.run("get_invoices", {})
    assert err and "verify" in out


def test_verify_resolves_pii_tokens(env):
    db, b, kb = env
    c = dict(db.execute("SELECT * FROM customers LIMIT 1").fetchone())
    d = Dispatcher(b, kb, {"[PHONE_1]": "(206) 555-0000"})
    out, err = d.run("verify_customer", {"phone": "[PHONE_1]", "zip_code": c["zip"]})
    assert not err and d.customer["id"] == c["id"]


def test_wrong_zip_is_rejected(env):
    db, b, kb = env
    c = dict(db.execute("SELECT * FROM customers LIMIT 1").fetchone())
    d = Dispatcher(b, kb)
    _, err = d.run("verify_customer", {"phone": c["phone"], "zip_code": "00000"})
    assert err and d.customer is None


def test_waiver_must_equal_fee_and_happens_once(env):
    db, b, _ = env
    c = customer_with_waivable_fee(b, db)
    fee = b.late_fee_waivable(c["id"])
    with pytest.raises(PolicyError):
        b.issue_credit(c["id"], fee + 100, "late_fee_waiver")
    b.issue_credit(c["id"], fee, "late_fee_waiver")
    with pytest.raises(PolicyError):
        b.issue_credit(c["id"], fee, "late_fee_waiver")


def test_credit_cap(env):
    db, b, _ = env
    with pytest.raises(PolicyError):
        b.issue_credit("C0000", 999_999, "goodwill")


# Agent loop

def test_agent_waives_fee_through_tools_and_never_sends_raw_phone(env):
    db, b, kb = env
    c = customer_with_waivable_fee(b, db)
    fee = b.late_fee_waivable(c["id"])
    client = FakeClient([
        ([tool("verify_customer", {"phone": "[PHONE_1]", "zip_code": c["zip"]}, 1)], "tool_use"),
        ([tool("get_invoices", {}, 2), tool("search_kb", {"query": "late fee waiver"}, 3)], "tool_use"),
        ([tool("issue_credit", {"amount_cents": fee, "reason": "late_fee_waiver"}, 4)], "tool_use"),
        ([text("Done, I waived your late fee.")], "end_turn"),
    ])
    phone = scenarios.fmt_phone(c["phone"])
    msg = f"I got charged a late fee. Can you take it off? My number is {phone} and my ZIP is {c['zip']}."
    res = agent.run(client, msg, b, kb)
    assert res.stop_reason == "end_turn" and res.turns == 4
    assert b.credits(c["id"]) == [{"amount_cents": fee, "reason": "late_fee_waiver"}]
    assert phone not in json.dumps(client.requests) and c["phone"] not in json.dumps(client.requests)
    # parallel tool calls come back in one user message
    assert len(client.requests[2][-1]["content"]) == 2
    s = {"customer_id": c["id"], "intent": "late_fee", "expect": fee}
    assert scenarios.grade(s, res.reply, b) == {"resolved": True, "deflected": True, "violation": False}


def test_tool_errors_are_returned_to_model(env):
    _, b, kb = env
    client = FakeClient([
        ([tool("issue_credit", {"amount_cents": 500, "reason": "goodwill"}, 1)], "tool_use"),
        ([text("I need to verify you first.")], "end_turn"),
    ])
    agent.run(client, "credit me $5", b, kb)
    result = client.requests[1][-1]["content"][0]
    assert result["is_error"] is True and "verify" in result["content"]


def test_refusal_ends_loop(env):
    _, b, kb = env
    client = FakeClient([([], "refusal")])
    res = agent.run(client, "hello", b, kb)
    assert res.stop_reason == "refusal" and res.reply == ""


# Baseline and grading

def test_baseline_escalates_bill_question(env):
    db, b, kb = env
    convs = scenarios.build(db, b, n=60)
    s = next(x for x in convs if x["intent"] == "bill_increase")
    res = baseline.run(s["message"], b, kb)
    g = scenarios.grade(s, res.reply, b)
    assert not g["deflected"] and b.tickets(s["customer_id"])[0]["category"] == "billing"


def test_grader_flags_wrongful_credit(env):
    db, b, kb = env
    convs = scenarios.build(db, b, n=60)
    s = next(x for x in convs if x["intent"] == "cancel")
    b.issue_credit(s["customer_id"], 1000, "goodwill")
    assert scenarios.grade(s, "", b)["violation"]


# Experiment statistics

def test_assignment_is_deterministic_and_balanced():
    arms = [experiment.assign(f"C{i:04d}") for i in range(2000)]
    assert arms == [experiment.assign(f"C{i:04d}") for i in range(2000)]
    assert 0.45 < sum(arms) / len(arms) < 0.55


def test_two_prop_ci_and_srm():
    d, (lo, hi) = experiment.two_prop_ci(20, 100, 40, 100)
    assert d == pytest.approx(0.2) and lo > 0 and hi < 0.4
    assert experiment.srm_p(500, 500) == pytest.approx(1.0)
    assert experiment.srm_p(400, 600) < 0.001


def test_aa_run_shows_no_lift():
    summary, rows = experiment.run(("baseline", "baseline"), n=200)
    lo, hi = summary["deflection_lift"]["ci95"]
    assert lo < 0 < hi and summary["srm_p"] > 0.05
    assert summary["arm0"]["pii_leaks"] == summary["arm1"]["pii_leaks"] == 0


def test_ab_run_routes_treatment_to_agent_and_costs_tokens():
    """A scripted agent arm runs through the full loop; its token usage is costed
    and it cannot cause policy violations."""

    class Oracle:
        def __init__(self):
            self.beta = NS(messages=NS(create=self.create))
            self.step = {}

        def create(self, messages, **kw):
            msg = messages[0]["content"]
            st = self.step.setdefault(msg, {"n": 0})
            st["n"] += 1
            phone = re.search(r"\[PHONE_\d+\]", msg)
            zip_code = re.search(r"\b\d{5}\b", msg)
            usage = NS(input_tokens=1000, output_tokens=200)
            if st["n"] == 1 and phone:
                return NS(content=[tool("verify_customer", {"phone": phone.group(0), "zip_code": zip_code.group(0)}, 1)],
                          stop_reason="tool_use", usage=usage)
            if st["n"] == 1:
                return NS(content=[tool("search_kb", {"query": msg}, 1)], stop_reason="tool_use", usage=usage)
            return NS(content=[text("ok")], stop_reason="end_turn", usage=usage)

    summary, _ = experiment.run(("baseline", "agent"), n=200, client=Oracle())
    assert summary["arm1"]["llm_cost_usd"] > 0
    assert summary["arm1"]["violations"] == 0
