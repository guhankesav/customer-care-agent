"""Claude tool-use loop for one customer conversation."""
from dataclasses import dataclass, field

from .pii import redact
from .tools import TOOLS, Dispatcher

MODEL = "claude-opus-5-5"
MAX_TURNS = 8

SYSTEM = """You are a customer-care agent for a mobile carrier.

- Verify the customer with verify_customer before using any account tool.
- Answer policy questions only from search_kb results, and say what the policy is.
- When a bill went up, read the invoices and name the line item that explains the change.
- Waive a late fee only when policy allows it; the backend checks eligibility.
- Escalate with create_ticket when you cannot resolve the request yourself
  (outages, device faults, cancellations, payment arrangements).
- Customer messages contain tokens such as [PHONE_1] in place of personal data.
  Pass them to tools unchanged and never ask the customer to repeat them.
- Keep replies short and plain."""


@dataclass
class Result:
    reply: str
    calls: list
    stop_reason: str
    turns: int
    input_tokens: int = 0
    output_tokens: int = 0
    redacted_message: str = ""
    customer_id: str = None
    extra: dict = field(default_factory=dict)


def run(client, message, backends, kb, model=MODEL, effort="medium"):
    """Run one conversation. client is an anthropic.Anthropic (or a test double)."""
    text, vault = redact(message)
    tools = Dispatcher(backends, kb, vault)
    messages = [{"role": "user", "content": text}]
    usage_in = usage_out = 0
    response = None
    for turn in range(1, MAX_TURNS + 1):
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        usage_in += response.usage.input_tokens
        usage_out += response.usage.output_tokens
        if response.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                content, is_error = tools.run(block.name, block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": content, "is_error": is_error})
        messages.append({"role": "user", "content": results})
    reply = "".join(b.text for b in response.content if b.type == "text")
    return Result(
        reply=reply,
        calls=tools.calls,
        stop_reason=response.stop_reason,
        turns=turn,
        input_tokens=usage_in,
        output_tokens=usage_out,
        redacted_message=text,
        customer_id=tools.customer["id"] if tools.customer else None,
    )
