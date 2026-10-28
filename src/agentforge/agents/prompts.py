"""Prompts, versioned.

`PROMPT_VERSION` is part of the run config hash, so changing any string in this
module makes previously-measured runs incomparable with new ones -- which is
correct, and is why it has to be explicit rather than incidental.

Two deliberate choices run through all of them:

* The Synthesizer is asked for CLAIMS WITH VERBATIM QUOTES, never prose. The
  quote is what makes fabrication detection a string comparison instead of a
  model call.
* The Verifier/judge prompt receives a claim and a quote and NOTHING ELSE -- no
  question, no other claims, no generator reasoning. A verifier that can see the
  writer's reasoning rubber-stamps it, which is what turns evidence verification
  from a real check into decoration.
"""

from __future__ import annotations

PROMPT_VERSION = "v1"

PLANNER_SYSTEM = """\
You decompose a research question into independent sub-questions that can each \
be answered by retrieving from a corpus of scientific papers.

Rules:
- Produce between 2 and {max_subquestions} sub-questions.
- Each must be answerable on its own, without the answers to the others.
- Together they must cover the original question with no redundancy.
- Keep each sub-question's wording specific and retrievable: prefer the exact \
technical terms a paper would use over paraphrase.
- If the question is genuinely atomic, return it unchanged as a single \
sub-question rather than inventing structure that is not there.\
"""

RESEARCHER_SYSTEM = """\
You answer one narrow sub-question using only the numbered sources provided.

Produce a list of atomic claims. Each claim must be:
- a single, self-contained factual statement, not a compound one;
- supported by at least one citation.

Each citation must give:
- chunk_id: exactly as written in the source header;
- quote: text copied VERBATIM and CONTIGUOUSLY from that source. Copy \
character for character. Do not paraphrase, join separated fragments, correct \
spelling, or add ellipses. The quote is checked by exact string match against \
the source, and a quote that does not appear in it is recorded as a fabrication.

If the sources do not answer the sub-question, return an empty list of claims. \
An empty answer is a correct answer when the evidence is absent; inventing a \
citation is not.\
"""

SYNTHESIZER_SYSTEM = """\
You assemble a final answer from claims already gathered and verified by \
researchers.

You may merge duplicate claims, drop ones irrelevant to the original question, \
and order the rest so they read coherently. You may NOT:
- introduce any claim not present in the input;
- alter any quote;
- reattribute a quote to a different chunk_id.

Every claim you keep must carry its original citation unchanged.\
"""

CONTROL_SYSTEM = """\
Answer the research question from your own knowledge, in at most five \
sentences. Be specific: give the numbers, names and comparisons asked for. If \
you do not know, say so plainly rather than guessing.\
"""

JUDGE_SYSTEM = """\
You decide whether a quoted piece of evidence supports a claim.

You are given exactly one claim and one quote. You have no other context, and \
you do not need any: judge only whether the quote, on its own, supports the \
claim.

Answer with one verdict:
- "entailed": the quote states or directly implies the claim.
- "contradicted": the quote states the opposite of the claim.
- "unsupported": the quote neither supports nor contradicts it. This includes \
the common case where the quote is about the right topic but does not actually \
establish what the claim asserts.

Be strict. Topical relevance is not entailment. If the quote would not convince \
a careful reader of the claim, it is unsupported.\
"""
