# Customer-Care AI Agent

A Claude agent that resolves mobile-carrier support requests by calling tools against mock **CRM, billing and ticketing** systems, answers policy questions from a **RAG** knowledge base, and is measured against a scripted bot in an **A/B harness** that reports deflection, resolution, policy violations and ROI.

## How it works

| Piece | File | What it does |
|---|---|---|
| Systems of record | `care/data.py`, `care/backends.py` | Seeded SQLite: 200 customers, 2 invoices each with line items, credits, tickets. Business rules (one late-fee waiver, $50 credit cap) are enforced here, not in the prompt. |
| Tools | `care/tools.py` | `verify_customer`, `get_invoices`, `issue_credit`, `create_ticket`, `search_kb`, all `strict: true`. Account tools refuse to run before verification. |
| PII | `care/pii.py` | Phone, email, card and SSN are replaced with tokens like `[PHONE_1]` before the model sees the message; the dispatcher resolves tokens when a tool runs. |
| RAG | `care/rag.py`, `care/kb/` | BM25 over policy sections (roaming, billing, network, account). |
| Agent | `care/agent.py` | Manual tool-use loop on `claude-opus-5-5`, parallel tool results returned in one message, tool errors returned with `is_error`, refusal handled, server-side fallback enabled. |
| Control arm | `care/baseline.py` | FAQ bot: top KB hit for questions, a human ticket for every account request. |
| Scenarios and grader | `care/scenarios.py` | Bill increase, late-fee waiver (eligible and not), KB questions, outage, cancellation. Grading reads backend state and the tool log, not just the reply. |
| Experiment | `care/experiment.py` | Hash-based 50/50 assignment, deflection and resolution per arm and intent, 95% CI on the lift, SRM check, LLM cost from token usage, ROI per conversation. |

## Run

```bash
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python anthropic pytest
.venv/bin/python -m pytest -q                                   # 18 tests, no API key needed
.venv/bin/python -m care.experiment --arms baseline baseline    # A/A check
.venv/bin/python -m care.experiment --arms baseline agent       # A/B, needs Claude credentials
```

ROI assumes a human-handled contact costs `--human-cost` dollars (default 6.00, an assumption, not a measured figure).

## Results so far

A/A (scripted bot in both arms, 200 conversations): deflection 22.8% vs 26.3%, lift +3.5 pts with 95% CI [-8.4, +15.4] (contains 0, as it should); SRM p = 0.89; 0 PII leaks; 0 policy violations. See `results_aa.json`.

The A/B run against the live agent has not been run yet.
