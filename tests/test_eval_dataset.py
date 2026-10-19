"""The golden set (S2-2): one case per validation rule, then the committed file itself.

The rules live in :mod:`rag_qa.evaluation.dataset`; ARCHITECTURE.md §2.4 has the schema.
The real-file tests check what the loader cannot: that every expected source exists in
the corpus, every page label exists in that PDF, and every passage a record's ``notes``
quotes is on the page it names. The labels come from ``pypdf``'s page labels, and the
text only from the cited pages (about thirty), so the 24 MB proceedings costs a couple of
seconds, not the minute that extracting all 516 pages takes.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from pypdf import PdfReader

from rag_qa.evaluation.dataset import GoldenItem, GoldenSetError, load_golden

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN = REPO_ROOT / "eval" / "eval_dataset.jsonl"
CORPUS = REPO_ROOT / "corpus"

ANSWERABLE: dict[str, Any] = {
    "id": "a",
    "question": "What is GLIDER?",
    "ground_truth": "A small judge model.",
    "answerable": True,
    "expected_sources": [{"source": "paper.pdf", "pages": ["7"]}],
}
UNANSWERABLE: dict[str, Any] = {
    "id": "u",
    "question": "What is the capital of Australia?",
    "ground_truth": "I could not find the answer in the provided documents.",
    "answerable": False,
    "must_not_contain": ["Canberra"],
}


def write(tmp_path: Path, *lines: dict[str, Any] | str) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        "".join((line if isinstance(line, str) else json.dumps(line)) + "\n" for line in lines)
    )
    return path


def problems(path: Path) -> str:
    with pytest.raises(GoldenSetError) as exc:
        load_golden(path)
    return str(exc.value)


# ---- the schema, one rule at a time ----


def test_a_valid_file_loads_in_order_and_skips_blank_lines(tmp_path: Path) -> None:
    items = load_golden(write(tmp_path, ANSWERABLE, "", UNANSWERABLE))
    assert [item.id for item in items] == ["a", "u"]
    assert items[0].expected_sources[0].pages == ("7",)
    assert items[1].must_not_contain == ("Canberra",)


def test_records_are_frozen(tmp_path: Path) -> None:
    item = load_golden(write(tmp_path, ANSWERABLE))[0]
    with pytest.raises(ValidationError):
        item.question = "other"  # type: ignore[misc]


def test_a_line_that_is_not_json_names_its_line(tmp_path: Path) -> None:
    assert "golden.jsonl:2: Invalid JSON" in problems(write(tmp_path, ANSWERABLE, "{oops"))


def test_every_bad_line_gets_its_own_error(tmp_path: Path) -> None:
    empty_id = {**ANSWERABLE, "id": ""}
    string_bool = {**ANSWERABLE, "id": "c", "answerable": "yes"}
    message = problems(write(tmp_path, empty_id, UNANSWERABLE, string_bool))
    assert "golden.jsonl:1: id:" in message
    assert "golden.jsonl:3: answerable:" in message
    assert ":2:" not in message


def test_an_unknown_key_is_rejected(tmp_path: Path) -> None:
    # extra="forbid": a misspelt optional key would otherwise vanish silently.
    message = problems(write(tmp_path, {**UNANSWERABLE, "must_not_contians": ["x"]}))
    assert "golden.jsonl:1: must_not_contians: Extra inputs are not permitted" in message


def test_a_missing_required_field_is_named(tmp_path: Path) -> None:
    record = {key: value for key, value in ANSWERABLE.items() if key != "ground_truth"}
    assert "golden.jsonl:1: ground_truth: Field required" in problems(write(tmp_path, record))


def test_types_are_strict(tmp_path: Path) -> None:
    # A page label must be the printed label as a string, never the number 7.
    record = {**ANSWERABLE, "expected_sources": [{"source": "paper.pdf", "pages": [7]}]}
    assert "expected_sources.0.pages.0: Input should be a valid string" in problems(
        write(tmp_path, record)
    )


def test_ids_must_be_unique(tmp_path: Path) -> None:
    message = problems(write(tmp_path, ANSWERABLE, UNANSWERABLE, {**UNANSWERABLE, "id": "a"}))
    assert "golden.jsonl:3: id 'a' repeats line 1" in message


@pytest.mark.parametrize("ground_truth", ["", "   "])
def test_an_answerable_record_needs_a_ground_truth(tmp_path: Path, ground_truth: str) -> None:
    message = problems(write(tmp_path, {**ANSWERABLE, "ground_truth": ground_truth}))
    assert "an answerable record needs a non-empty ground_truth" in message


def test_an_answerable_record_needs_expected_sources(tmp_path: Path) -> None:
    message = problems(write(tmp_path, {**ANSWERABLE, "expected_sources": []}))
    assert "an answerable record needs expected_sources" in message


def test_an_unanswerable_record_has_no_expected_sources(tmp_path: Path) -> None:
    record = {**UNANSWERABLE, "expected_sources": ANSWERABLE["expected_sources"]}
    assert "an unanswerable record has no expected_sources" in problems(write(tmp_path, record))


def test_only_an_unanswerable_record_carries_must_not_contain(tmp_path: Path) -> None:
    message = problems(write(tmp_path, {**ANSWERABLE, "must_not_contain": ["x"]}))
    assert "must_not_contain is only for unanswerable records" in message


def test_must_not_contain_entries_are_non_empty(tmp_path: Path) -> None:
    # An empty string is a substring of every answer, so it would fail every decline.
    message = problems(write(tmp_path, {**UNANSWERABLE, "must_not_contain": [""]}))
    assert "must_not_contain entries must be non-empty" in message


@pytest.mark.parametrize("source", ["/home/me/corpus/paper.pdf", "../paper.pdf", r"sub\paper.pdf"])
def test_a_source_is_corpus_relative(tmp_path: Path, source: str) -> None:
    record = {**ANSWERABLE, "expected_sources": [{"source": source, "pages": ["7"]}]}
    assert "relative to the corpus root" in problems(write(tmp_path, record))


@pytest.mark.parametrize("source", ["./paper.pdf", "a//b.pdf", "paper.pdf/", "."])
def test_a_source_must_already_be_normalised(tmp_path: Path, source: str) -> None:
    # These would pass a relative-path check yet never equal a chunk's source string.
    record = {**ANSWERABLE, "expected_sources": [{"source": source, "pages": ["7"]}]}
    assert "must be a normalised path" in problems(write(tmp_path, record))


@pytest.mark.parametrize("source", [" paper.pdf", "paper.pdf "])
def test_a_source_has_no_surrounding_spaces(tmp_path: Path, source: str) -> None:
    record = {**ANSWERABLE, "expected_sources": [{"source": source, "pages": ["7"]}]}
    assert "no surrounding spaces" in problems(write(tmp_path, record))


@pytest.mark.parametrize("field", ["pages", "also_pages"])
def test_a_page_label_has_no_surrounding_spaces(tmp_path: Path, field: str) -> None:
    expected = {"source": "paper.pdf", "pages": ["7"], field: [" 8"]}
    record = {**ANSWERABLE, "expected_sources": [expected]}
    assert "page labels must not have surrounding spaces" in problems(write(tmp_path, record))


def test_also_pages_load_beside_pages(tmp_path: Path) -> None:
    expected = {"source": "paper.pdf", "pages": ["7"], "also_pages": ["2", "9"]}
    item = load_golden(write(tmp_path, {**ANSWERABLE, "expected_sources": [expected]}))[0]
    assert item.expected_sources[0].pages == ("7",)
    assert item.expected_sources[0].also_pages == ("2", "9")


def test_a_page_is_not_both_expected_and_also(tmp_path: Path) -> None:
    expected = {"source": "paper.pdf", "pages": ["7", "8"], "also_pages": ["8"]}
    message = problems(write(tmp_path, {**ANSWERABLE, "expected_sources": [expected]}))
    assert "either in pages or in also_pages, not both: ['8']" in message


def test_also_pages_need_pages(tmp_path: Path) -> None:
    expected = {"source": "notes.txt", "also_pages": ["2"]}
    message = problems(write(tmp_path, {**ANSWERABLE, "expected_sources": [expected]}))
    assert "also_pages needs pages" in message


def test_also_pages_do_not_repeat(tmp_path: Path) -> None:
    expected = {"source": "paper.pdf", "pages": ["7"], "also_pages": ["2", "2"]}
    message = problems(write(tmp_path, {**ANSWERABLE, "expected_sources": [expected]}))
    assert "page labels must not repeat" in message


def test_page_labels_do_not_repeat(tmp_path: Path) -> None:
    record = {**ANSWERABLE, "expected_sources": [{"source": "paper.pdf", "pages": ["7", "7"]}]}
    assert "page labels must not repeat" in problems(write(tmp_path, record))


def test_a_source_is_listed_once(tmp_path: Path) -> None:
    sources = [{"source": "paper.pdf", "pages": ["7"]}, {"source": "paper.pdf", "pages": ["8"]}]
    message = problems(write(tmp_path, {**ANSWERABLE, "expected_sources": sources}))
    assert "expected_sources must not repeat a source" in message


def test_a_line_separator_inside_a_string_is_not_a_line_break(tmp_path: Path) -> None:
    # str.splitlines() splits on U+2028, which is legal inside a JSON string.
    record = {**ANSWERABLE, "notes": "before\u2028after"}
    path = tmp_path / "golden.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    assert load_golden(path)[0].notes == "before\u2028after"


def test_a_page_label_is_non_empty(tmp_path: Path) -> None:
    record = {**ANSWERABLE, "expected_sources": [{"source": "paper.pdf", "pages": [" "]}]}
    assert "page labels must be non-empty" in problems(write(tmp_path, record))


def test_a_file_without_records_is_rejected(tmp_path: Path) -> None:
    assert "golden.jsonl: no records" in problems(write(tmp_path, ""))


# ---- the committed golden set ----


@pytest.fixture(scope="module")
def golden() -> list[GoldenItem]:
    return load_golden(GOLDEN)


def test_the_golden_set_loads(golden: list[GoldenItem]) -> None:
    assert golden


def test_the_first_record_is_the_s0_6_glider_case(golden: list[GoldenItem]) -> None:
    first = golden[0]
    assert first.question == "What is GLIDER and what does it evaluate?"
    for fact in ("Small Language Model", "Phi-3.5-mini", "0.654", "0.481", "0.485"):
        assert fact in first.ground_truth


def test_every_expected_source_is_a_corpus_file(golden: list[GoldenItem]) -> None:
    missing = sorted(
        {s.source for item in golden for s in item.expected_sources}
        - {p.relative_to(CORPUS).as_posix() for p in CORPUS.rglob("*") if p.is_file()}
    )
    assert missing == []


def test_every_expected_page_label_exists_in_its_file(golden: list[GoldenItem]) -> None:
    labels: dict[str, set[str]] = {}
    unknown = []
    for item in golden:
        for expected in item.expected_sources:
            path = CORPUS / expected.source
            if path.suffix.lower() != ".pdf":
                # Formats without pages carry no page labels (ARCHITECTURE.md §2.4).
                assert expected.pages == (), f"{item.id}: {expected.source} has no pages"
                continue
            if expected.source not in labels:
                labels[expected.source] = set(PdfReader(path).page_labels)
            unknown += [
                f"{item.id}: {expected.source} p. {page}"
                for page in (*expected.pages, *expected.also_pages)
                if page not in labels[expected.source]
            ]
    assert unknown == []


def test_every_also_page_is_named_in_the_notes(golden: list[GoldenItem]) -> None:
    # also_pages count as hits in tier 1, so each must be justified where the human check
    # reads: the notes say what the page repeats (S2-2 review).
    unexplained = [
        f"{item.id}: p. {page}"
        for item in golden
        for expected in item.expected_sources
        for page in expected.also_pages
        if not re.search(rf"\bp\. {re.escape(page)}\b", item.notes)
    ]
    assert unexplained == []
    assert any(expected.also_pages for item in golden for expected in item.expected_sources)


# ---- the passages the notes quote ----

QUOTE = re.compile(r'p\. (\w+): "(.*?)"(?=[;.]| \(|$)')


def normalise(text: str) -> str:
    """Tolerate only what extraction changes: line breaks, end-of-line hyphenation and
    OCR spaces around a hyphen (hyphens, and the space around them, are dropped on both
    sides). Case is folded. A changed number, name or word still fails."""
    text = re.sub(r"\s*[-\u2010\u2011\u00ad]\s*", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def test_the_quote_matcher_tolerates_extraction_but_not_a_changed_fact() -> None:
    page = normalise(
        "GLIDER outperforms with a score of 0.654, but also much larger LLMs like\nGPT-4o-\nmini"
    )
    assert normalise("larger LLMs like GPT-4o-mini") in page  # line break and hyphenation
    assert normalise("a score of 0.654") in page
    assert normalise("a score of 0.645") not in page  # a swapped digit
    assert normalise("GPT-4o-mini and Qwen") not in page  # an invented continuation


def test_every_quoted_passage_is_on_the_page_the_notes_name(golden: list[GoldenItem]) -> None:
    readers: dict[str, PdfReader] = {}
    page_text: dict[tuple[str, str], str] = {}
    unmatched, quotes = [], 0
    for item in golden:
        if not item.answerable:
            continue
        found = QUOTE.findall(item.notes)
        # A note whose quotes stopped matching the pattern would make this test vacuous.
        assert found, f"{item.id}: notes quote no passage as p. N: \"...\""
        for label, quote in found:
            quotes += 1
            matched = False
            for expected in item.expected_sources:
                path = CORPUS / expected.source
                reader = readers.setdefault(expected.source, PdfReader(path))
                labels = reader.page_labels
                if label not in labels:
                    continue
                key = (expected.source, label)
                if key not in page_text:
                    page_text[key] = normalise(reader.pages[labels.index(label)].extract_text())
                matched = matched or normalise(quote) in page_text[key]
            if not matched:
                unmatched.append(f"{item.id}: p. {label}: {quote[:70]}")
    assert quotes >= 60  # the committed set has 64
    assert unmatched == []
