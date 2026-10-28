# Judge validation

An unvalidated LLM judge makes every groundedness number above it decorative.
Kappa rather than raw agreement, because the label distribution is skewed: a
judge answering "entailed" every time posts ~90% agreement with no skill.

| Judge | prompt | kappa | agreement | n | labels | usable |
|---|---|---|---|---|---|---|
| `openai/gpt-oss-20b` | v1 | **0.952** | 0.970 | 100 | model-pending-review | yes |

- Floor: kappa >= 0.70
- Shipped judge: **openai/gpt-oss-20b** -- the cheapest model clearing the floor, chosen by measurement rather than by price.
- Reference labels are model-drafted (Claude) and PENDING HUMAN REVIEW, so these figures measure agreement with another model, not with a human. (If the judge and the drafter were the same model it would be self-consistency.)
