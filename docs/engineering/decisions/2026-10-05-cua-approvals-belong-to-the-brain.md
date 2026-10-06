# CUA: approvals belong to the brain, not the hands

Status: accepted
Owner: Atlas
Date: 2026-10-05

## Context

The perception session (`rapid_mlx/computer_use/perception.py`) used to judge
actions itself:

- a money-commit gate refused clicks, AX presses and activating keys on
  controls whose label looked like a purchase ("Place order", "Pay $25"). It
  then asked for a one-shot approval bound to the window, the control and
  every amount on screen;
- typing or filling into a password, one-time-code or card field was refused
  (`needs_human`);
- text holding a card number was refused (`sensitive_data`).

A held-out run on real sites and native apps showed the hands layer cannot get
this right:

- the gates were English-only, so they missed "Comprar ahora" and "提交订单";
- they were role-gated, so they missed "Place order" rendered as a div;
- they fired falsely on "Transfer or Reset" and on TextEdit's text view.

More importantly, whether an action needs the user's confirmation depends on
what the user asked for. "Cancel my Amazon order 12345" should just run, while
"find me batteries" must not buy any. Only the brain knows the request.

## Decision

The hands and perception session only execute. They never refuse an action
because of what it does, and they keep no approval state. The brain decides
from the user's instructions whether to confirm first. To confirm, it stops
and asks the user in its reply; the chat/agent loop relays the question and
the answer. No dedicated "ask user" operation is needed.

Kept, because they are perception or mechanism, not policy:

- **Secret values never enter the model's context.** A password field's
  contents are never read. What was typed into a one-time-code or card field
  reads as `[entered by the user]` (`computer_use/privacy.py`).
- **`handoff`.** The brain calls it when it wants the user to type something
  themselves, such as a password or a code. The window is the user's until
  they finish, and agent input to it is refused (`with_human`).

Removed:

- `is_money_commit`, the approval tokens and their binding to amounts;
- `pending_approvals` / `approve` / `deny`;
- the live focused-field secret check;
- the card-number refusal;
- the secret-field typing refusal, with their tests.

## Brain-facing guidance

Tool descriptions and system prompts for a brain driving the session should
say, in substance:

> Deciding whether to confirm with the user is your job. Before an action that
> spends money, deletes or cancels something, sends something on the user's
> behalf, or enters a secret, check the user's instructions. If they did not
> clearly ask for it, stop and ask the user in your reply, and act only after
> they answer. To have the user type a password or code themselves, use
> `handoff` with a reason.

## Consequences

- The approval API never shipped. The perception session and its gates
  existed only on the unmerged computer-use PR stack, and no host or brain
  prompt in this repository drives the session. So there is no compatibility
  period. The one consumer is the mock-eval harness (`evals/cua_mock`), whose
  `/human pending|approve|deny` goes with this change. Only `human_act` and
  `human_done` remain on the user's channel during a handoff.
- Eval harnesses must score "asked before committing" from the brain's reply,
  not from an approval event.
- Any future brain entry point that drives this session must ship the
  guidance above in its prompt or tool descriptions.
- The older run-based CUA product (`rapid_mlx/cua`, `routes/cua.py`, the Mac
  app's `needs_approval` status) is a separate path and is unchanged here.
