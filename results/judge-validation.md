# Judge validation

An unvalidated LLM judge makes every groundedness number above it decorative.
Kappa rather than raw agreement, because the label distribution is skewed: a
judge answering "entailed" every time posts ~90% agreement with no skill.

| Judge | prompt | kappa | agreement | n | labels | usable |
|---|---|---|---|---|---|---|
| `claude-haiku-5-5` | v1 | **0.969** | 0.980 | 100 | Claude-drafted | yes |

- Floor: kappa >= 0.70
- Shipped judge: **claude-haiku-5-5** -- the cheapest model clearing the floor, chosen by measurement rather than by price.
- Reference labels were drafted by Claude, with every quote verified verbatim against its chunk, so these figures measure agreement within the Claude model family rather than with a human annotator.
