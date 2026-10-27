"""A/B harness: route conversations to the control (scripted bot) or treatment
(Claude agent) arm, grade each one, and report deflection, resolution, policy
violations, a 95% CI on the deflection lift, an SRM check and ROI.

    python -m care.experiment --arms baseline baseline   # A/A, no API key needed
    python -m care.experiment --arms baseline agent      # A/B, needs Claude credentials
"""
import argparse
import hashlib
import json
import math

from . import agent, baseline, data, scenarios
from .backends import Backends
from .pii import PATTERNS
from .rag import BM25, load_chunks

# Claude Opus 5.5 list prices, USD per 1M tokens.
PRICE_IN, PRICE_OUT = 4.00, 20.00


def assign(key, salt="exp1"):
    """Deterministic 50/50 split on a hash of the unit id."""
    return int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest(), 16) % 2


def two_prop_ci(x1, n1, x2, n2, z=1.96):
    p1, p2 = x1 / n1, x2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p2 - p1
    return d, (d - z * se, d + z * se)


def srm_p(n0, n1):
    """Chi-square (1 df) p-value for a 50/50 sample-ratio mismatch."""
    exp = (n0 + n1) / 2
    chi = (n0 - exp) ** 2 / exp + (n1 - exp) ** 2 / exp
    return math.erfc(math.sqrt(chi / 2))


def run(arms=("baseline", "agent"), n=200, human_cost=6.0, client=None, salt="exp1", effort="medium"):
    db = data.build()
    backends = Backends(db)
    kb = BM25(load_chunks())
    convs = scenarios.build(db, backends, n=n)
    rows = []
    for i, s in enumerate(convs):
        arm = assign(s["customer_id"] or f"anon{i}", salt)
        kind = arms[arm]
        if kind == "agent":
            res = agent.run(client, s["message"], backends, kb, effort=effort)
        else:
            res = baseline.run(s["message"], backends, kb)
        g = scenarios.grade(s, res.reply, backends)
        cost = res.input_tokens * PRICE_IN / 1e6 + res.output_tokens * PRICE_OUT / 1e6
        leak = any(pat.search(res.redacted_message) for _, pat in PATTERNS)
        rows.append({"arm": arm, "intent": s["intent"], **g, "llm_cost": cost, "pii_in_prompt": leak})
    return summarize(rows, arms, human_cost), rows


def summarize(rows, arms, human_cost):
    out = {"arms": list(arms), "human_cost_assumption_usd": human_cost}
    for a in (0, 1):
        r = [x for x in rows if x["arm"] == a]
        n = len(r)
        defl = sum(x["deflected"] for x in r)
        llm = sum(x["llm_cost"] for x in r)
        out[f"arm{a}"] = {
            "kind": arms[a], "n": n,
            "deflection_rate": round(defl / n, 4),
            "resolution_rate": round(sum(x["resolved"] for x in r) / n, 4),
            "violations": sum(x["violation"] for x in r),
            "pii_leaks": sum(x["pii_in_prompt"] for x in r),
            "llm_cost_usd": round(llm, 4),
            # Net saving per conversation versus sending every contact to a human.
            "roi_per_conv_usd": round((defl * human_cost - llm) / n, 4),
            "by_intent": {
                i: round(sum(x["resolved"] for x in r if x["intent"] == i) / max(1, sum(1 for x in r if x["intent"] == i)), 3)
                for i in sorted({x["intent"] for x in r})
            },
        }
    a0, a1 = out["arm0"], out["arm1"]
    d, ci = two_prop_ci(round(a0["deflection_rate"] * a0["n"]), a0["n"], round(a1["deflection_rate"] * a1["n"]), a1["n"])
    out["deflection_lift"] = {"diff": round(d, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)]}
    out["srm_p"] = round(srm_p(a0["n"], a1["n"]), 4)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs=2, default=["baseline", "agent"], choices=["baseline", "agent"])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--human-cost", type=float, default=6.0, help="assumed cost of one human-handled contact, USD")
    p.add_argument("--effort", default="medium")
    p.add_argument("--out", default="results.json")
    args = p.parse_args()
    client = None
    if "agent" in args.arms:
        import anthropic
        client = anthropic.Anthropic()
    summary, rows = run(args.arms, args.n, args.human_cost, client, effort=args.effort)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
