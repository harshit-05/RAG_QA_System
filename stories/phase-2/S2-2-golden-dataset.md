# S2-2: Commit the golden dataset, seeded with GLIDER

| | |
| --- | --- |
| **Status** | Implemented 2026-10-03; gates green locally. Awaiting the maintainer's record-by-record sign-off (table below) and CI on the branch |
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

- **Which pages a record lists (input for S2-3).** `expected_sources` lists the pages the
  ground truth is written from: the passages its `notes` quote, one per fact. When the
  same fact is repeated elsewhere, `notes` names that page ("also p. 4") but does not
  list it. So a tier-1 miss can be a retrieved repeat. Read the record's notes before
  calling it a retrieval failure. Listing every repeat instead would have made recall
  punish a context that held the fact on another page.
- **The YOLOv8 PDF has two page numberings.** `citation()` prints the PDF's page labels,
  1–13 (the first page is a ResearchGate cover). The journal's own numbers, 52–62, are
  printed on the pages too, 49 higher. The records use the labels, as tier 1 must; each
  YOLOv8 record's notes give both numbers.
- **The source papers contradict themselves**, so a judge will sometimes see two figures
  in the context:
  - GLIDER is "a powerful 3B evaluator" in the abstract and "3.8B parameters" on p. 2 and
    in every table;
  - GLIDER's human agreement is "91.3%" in the abstract, 91%, 90% and 91% on p. 7, and
    0.918, 0.905 and 0.917 in Table 6;
  - the YOLOv8 paper credits its 95.4% to "object detection" (p. 3) and to "crowd
    detection and anomaly identification" (p. 8–9).

  Each ground truth takes the specific statement, and its notes name the other.
- **The proceedings are an OCR'd scan.** Some pages extract with broken spacing ("s
  ubstitut ion", pp. 3, 249 and 303), and some characters come out wrong: "19S4" for 1984
  (p. iii), "19n" for 1977 (p. 248), "wur" for the unifier-set symbol (p. 4). Retrieval
  embeds this text. The proceedings questions were therefore chosen on cleanly extracted
  passages, so they measure retrieval rather than OCR. A re-OCR of the corpus is a
  corpus change, out of scope here. It is worth knowing before anyone reads a low
  proceedings score as a retrieval problem.
- **Each quote in `notes` was checked against the text `PdfLoader` extracts at the cited
  label** (a scratch script, not committed). All 65 quotes are found, and every listed
  page carries at least one. Negative controls (a swapped score, a wrong page, an
  invented expansion) are not found. The claims in notes that carry no quote were checked
  the same way. One was wrong, a page cited from memory ("Melbourne, Australia" is on
  p. 500, not p. 499), and has been fixed. The check tolerates only line breaks,
  end-of-line hyphenation and OCR spaces around a hyphen, so a passage found on the
  printed page is the passage quoted.

## Deviation from plan

- **Rules added beyond the story's list**, all from ARCHITECTURE.md §2.4 or needed to
  apply it:
  - `source` must be corpus-relative (§2.4 says so; an absolute path would silently never
    match in tier 1);
  - fields are strictly typed, so a page label must be the string `"7"`, never the number
    7, and `answerable` must be a real boolean;
  - page labels and `must_not_contain` entries must be non-empty (an empty string is a
    substring of every answer, so every decline would fail);
  - blank lines are skipped, and a file with no records is an error.
- **Model:** this session ran on Opus 5.5, as the story's opus-fast routing asks.
- **The record mix:** 25 records. 20 are answerable: 7 on GLIDER, 6 on YOLOv8 and 7 on
  the proceedings, each proceedings question naming its paper. 5 are unanswerable: 4 near
  the corpus topics, and Canberra. Two of the unanswerable questions are traps: one
  rests on a cross-document distractor (the YOLOv8 GPU question, where only GLIDER
  names H100s), and one has a known prior-knowledge leak ("Oxford" for CADE-8). The
  question types the story asks for: numbers and attributions (`glider-purpose`,
  `glider-data-filtering`, `yolo-accuracy`, `yolo-response-overhead`, `ketonen-ekl`);
  definitions and acronyms (`glider-name`, `glider-slm`, `yolo-acronym`,
  `siekmann-unification-hierarchy`, `lusk-overbeek-itp`, `ohlbach-wrightson-mkrp`);
  facts that span pages (`yolo-objectives`, `wos-linked-inference`, `cade7-venue`,
  `glider-purpose`); a comparison within one paper (`glider-flask-vs-gpt4o`).

### Record-by-record sign-off (maintainer)

Check each against the PDF at the printed page, following "Review notes for the human".
Each record's `notes` quotes the passages; YOLOv8 pages are PDF labels (journal page =
label + 49). Mark ✓, or write what is wrong.

| # | id | File, pages | Check these facts | Signed off |
| --- | --- | --- | --- | --- |
| 1 | `glider-purpose` | GLIDER pp. 1, 2, 7 | name expansion; SLM = Small Language Model; 3.8B; Phi-3.5-mini-instruct; 0.654 / GPT-4o-mini 0.481 / Qwen-2.5-72B 0.485 | |
| 2 | `glider-name` | GLIDER p. 1 | the title's expansion | |
| 3 | `glider-slm` | GLIDER p. 2 | Small Language Model; "17x" | |
| 4 | `glider-training-data` | GLIDER pp. 1, 2 | 685 domains, 183 criteria (not swapped) | |
| 5 | `glider-flask-vs-gpt4o` | GLIDER pp. 5, 6 | FLASK is Table 1's 2nd column: GLIDER 0.615, GPT-4o 0.610 | |
| 6 | `glider-human-study` | GLIDER p. 7 | 100 points, 3 annotators; 91/90/91%; alpha 0.838 | |
| 7 | `glider-data-filtering` | GLIDER p. 3 | 18,258 samples; 14.6% | |
| 8 | `yolo-accuracy` | YOLOv8 pp. 3, 9 | 95.4% and 92.7%, and what each is attributed to | |
| 9 | `yolo-response-overhead` | YOLOv8 p. 9 | 2-3 s; 30% overhead | |
| 10 | `yolo-acronym` | YOLOv8 p. 3 | You Only Look Once | |
| 11 | `yolo-objectives` | YOLOv8 pp. 3–4 | the three objectives, across the page break | |
| 12 | `yolo-anomaly-model` | YOLOv8 p. 8 | LSTM-based; context-aware filtering + multi-modal fusion | |
| 13 | `yolo-alert-pipeline` | YOLOv8 p. 7 | severity classification; alarms, door locks, emergency messages | |
| 14 | `cade7-venue` | Proceedings pp. i, iii | May 14-16, 1984, Napa; 27 papers; Siekmann keynote, Suppes banquet | |
| 15 | `siekmann-unification-hierarchy` | Proceedings p. 4 | the four definitions | |
| 16 | `lusk-overbeek-itp` | Proceedings p. 43 | Interactive Theorem Prover; LMA; Pascal; ~fifty sites | |
| 17 | `ketonen-ekl` | Proceedings p. 65 | ~10000 lines, MACLISP; SAIL (KL10); began 1981 | |
| 18 | `stickel-ring-commutativity` | Proceedings p. 248 | x^3 = x ⇒ commutative; Bledsoe 1977; Veroff, ANL-NIU | |
| 19 | `wos-linked-inference` | Proceedings pp. 316–317 | linked UR-resolution; one step for many; semantic for syntactic criteria | |
| 20 | `ohlbach-wrightson-mkrp` | Proceedings p. 496 | "converse of contraction"; MKRP expansion; Karlsruhe and Kaiserslautern | |
| 21 | `unanswerable-yolo-dataset` | — | the paper names no dataset (p. 8); `must_not_contain` COCO | |
| 22 | `unanswerable-yolo-gpus` | — | no hardware in the YOLOv8 paper; `must_not_contain` H100 | |
| 23 | `unanswerable-glider-mmlu` | — | MMLU appears nowhere | |
| 24 | `unanswerable-cade8-venue` | — | the foreword lists only earlier venues; `must_not_contain` Oxford | |
| 25 | `unanswerable-capital-australia` | — | `must_not_contain` Canberra | |
