# S2-2: Commit the golden dataset, seeded with GLIDER

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | FR-7 (the dataset half), ISS-15 (the missing dataset), SRS §7.4; backlog: seed the golden dataset with GLIDER |
| **Depends on** | — (can run alongside S2-1) |
| **Model** | opus-fast (Claude drafts every record; the maintainer verifies every one) |
| **Plan-first** | no |
| **Branch** | `feat/s2-2-golden-dataset`. Low risk: data plus one pure module. It is on a branch only so that CI runs before the merge. |

## Goal

The committed golden set that both eval gates score against: tier 1 in S2-3 and tier 2 in
S2-5. About 25 questions over the three corpus PDFs:

- about 20 answerable, each with a verified ground truth and the printed pages it rests
  on;
- about 5 that the corpus cannot answer.

Plus `rag_qa/evaluation/dataset.py`, which loads and validates it. Design: ARCHITECTURE.md
§2.1 DEC-15, schema in §2.4.

A wrong ground truth is worse than a missing one. It silently lowers the score of a correct
answer, or raises the score of a wrong one, in every run after it. That is why every record
is checked by a human against the PDF.

## Scope

- **`eval/eval_dataset.jsonl`**, schema in ARCHITECTURE.md §2.4.
  - **First record: GLIDER.** "What is GLIDER and what does it evaluate?", with the
    verified facts from S0-6's "Fact-check":
    - SLM is a *Small* Language Model (p. 2);
    - the base model is phi-3.5-mini;
    - the scores: GLIDER 0.654, GPT-4o-mini 0.481, Qwen-2.5-72B 0.485 (p. 7).

    The v0.2 manual test found two further traps: the benchmark categories (p. 5) and
    the p. 8 description.
  - **Spread across all three PDFs.** The corpus is lopsided: the proceedings hold 88% of
    the chunks (S0-6). Include proceedings questions that name one specific paper, so
    retrieval must find it among hundreds of pages. Those are the hard retrieval cases.
  - **Question types:**
    - exact numbers and attributions (S0-6's failure mode);
    - definitions and acronym expansions (the other one);
    - facts that span pages;
    - a comparison within one paper.
  - **Unanswerable:** about 4 plausible questions near the corpus topics, asking for a
    detail the papers do not state, plus one far-off question.
    - The far-off one is "What is the capital of Australia?", with
      `must_not_contain: ["Canberra"]`, as in S0-6.
    - Each `ground_truth` is the system prompt's refusal sentence.
  - **Each answerable record's `notes`** gives the passage it rests on: file, printed
    page label, and a short quote. The ground truth stays close to the source wording,
    because RAGAs' context recall compares against it.
- **How the records are made.** Claude drafts every record from the PDF text. Page labels
  come from `PdfLoader` (`page_label`), so they match what `citation()` prints. For each
  record, Claude lists the quote it rests on. The maintainer checks each one against the
  source, and signs off record by record under "Deviation from plan". No record ships
  unchecked.
- **`src/rag_qa/evaluation/__init__.py` and `evaluation/dataset.py`.**
  - `GoldenItem` (Pydantic, frozen, `extra="forbid"`) and
    `load_golden(path) -> list[GoldenItem]`.
  - Each bad line gets its own error, with its line number.
  - ids must be unique.
  - An answerable record needs a non-empty ground truth and `expected_sources`.
  - An unanswerable record has no sources, and is the only kind that may carry
    `must_not_contain`.
  - Pure Python with no LangChain, so the S2-5 gate can import it in CI's model-free job.
- **Tests.**
  - `tests/test_eval_dataset.py`: one case for each validation rule.
  - A test over the real file: it loads cleanly, every `source` exists under `corpus/`,
    and every page label exists in that PDF. Read `pypdf`'s `page_labels` for this, not
    the text, so the 24 MB proceedings costs no extraction time.
  - `tests/test_architecture.py`: `rag_qa.evaluation.dataset` loads no LangChain.

## Out of scope

- The metrics and the CI job (S2-3), and RAGAs (S2-5).
- Generated question sets (RAGAs testset generation). The set is hand-curated on purpose:
  a generated set inherits its generator's misreadings, and a model misreading is exactly
  the failure S0-6 found.
- Changing the corpus.

## Verification

```bash
# 1. the gates
uv run ruff check && uv run mypy && uv run pytest --cov=rag_qa -q

# 2. the dataset itself
uv run pytest tests/test_eval_dataset.py -v
uv run python -c "
from rag_qa.evaluation.dataset import load_golden
items = load_golden('eval/eval_dataset.jsonl')
print(len(items), 'records;', sum(i.answerable for i in items), 'answerable')
print(sorted({s.source for i in items for s in i.expected_sources}))"
#    → ~25 records, ~20 answerable, and all three PDFs named
head -1 eval/eval_dataset.jsonl          # → the GLIDER record, with S0-6's facts

# 3. the human check: one sign-off line per record under "Deviation from plan"

# 4. CI
gh run list --limit 1                    # → green on the branch
```

## Review notes for the human

The ground truths are the product here; the code is small. For each record:

- **Read the quoted passage in the PDF**, at the printed page, not the PDF's page index.
- **Check every number, and every "X stands for" or "X is".** Those are the facts S0-6
  showed the model getting wrong, so they are also the facts a careless ground truth would
  get wrong.
- **Ask whether the expected pages are the right ones.** A tier-1 hit on the wrong page
  counts as a miss.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out, including the maintainer's record-by-record sign-off.)
