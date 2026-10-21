"""Our BM25 encoder (DEC-20, S3-5): its tokens, its indices, its weights, and its version."""

import hashlib
import json
import math

import pytest

from rag_qa import sparse
from rag_qa.sparse import (
    SPARSE_VERSION,
    STOPWORDS,
    Bm25Encoder,
    SparseVector,
    token_index,
    tokens,
)

#: The settings of config.yaml's bm25 entry.
BM25 = {"k1": 1.2, "b": 0.75, "avg_len": 100.76}


def encoder(**settings: float) -> Bm25Encoder:
    return Bm25Encoder(**{**BM25, **settings})


# --- tokens -------------------------------------------------------------------------------


def test_tokens_are_lowercase_unicode_words_without_stopwords() -> None:
    assert tokens("What is GLIDER? The GLIDER-SLM, v2.0") == ["glider", "glider", "slm", "v2", "0"]
    assert tokens("Ünïcode café_au_lait Ω42") == ["ünïcode", "café_au_lait", "ω42"]
    assert tokens("") == []
    assert tokens("is it? -- ...") == []  # stopwords and punctuation only


def test_a_soft_hyphen_at_a_line_break_joins_the_word_again() -> None:
    # PDF text marks a line-break hyphenation with U+00AD; 823 of them in the CADE
    # proceedings (S3-5's plan, D3). Without the rule, each word splits in two.
    assert tokens("con­\nstruction and algo­ rithm") == ["construction", "algorithm"]
    assert tokens("re­write") == ["rewrite"]
    # An ordinary hyphen still splits: it may join two words of their own.
    assert tokens("con-\nstruction") == ["con", "struction"]


def test_the_stopwords_are_nltk_s_english_list_less_its_apostrophe_forms() -> None:
    assert len(STOPWORDS) == 153
    assert {"what", "which", "how", "the", "don", "t", "couldn"} <= STOPWORDS
    assert not any("'" in word for word in STOPWORDS)
    assert all(word == word.lower() and word.isalpha() for word in STOPWORDS)
    # Every stopword is something \w+ can yield, or the list would hold dead words.
    assert all(tokens(word) == [] for word in STOPWORDS)


# --- indices ------------------------------------------------------------------------------


def test_a_token_s_index_is_its_4_byte_blake2b_digest_big_endian() -> None:
    assert token_index("glider") == 2366667583  # pinned: a change moves every index
    digest = hashlib.blake2b(b"glider", digest_size=4).digest()
    assert token_index("glider") == int.from_bytes(digest, "big")
    assert 0 <= token_index("ünïcode") < 2**32


# --- weights ------------------------------------------------------------------------------


def test_a_document_s_weights_follow_the_bm25_formula_by_hand() -> None:
    # "glider" twice, "slm" once: a 3-token document, against avg_len 100.
    vector = encoder(avg_len=100).encode_document("GLIDER, the glider SLM")
    norm = 1 - 0.75 + 0.75 * 3 / 100  # 0.2725
    expected = {
        token_index("glider"): 2 * 2.2 / (2 + 1.2 * norm),  # 1.890846…
        token_index("slm"): 1 * 2.2 / (1 + 1.2 * norm),  # 1.657874…
    }
    assert dict(zip(vector.indices, vector.values, strict=True)) == pytest.approx(expected)
    assert vector.indices == tuple(sorted(expected))
    assert dict(zip(vector.indices, vector.values, strict=True))[
        token_index("glider")
    ] == pytest.approx(1.890846583584014)


def weight(vector: SparseVector, token: str) -> float:
    return vector.values[vector.indices.index(token_index(token))]


def test_a_longer_document_weighs_a_term_less_and_b_0_ignores_length() -> None:
    short = encoder().encode_document("glider")
    long = encoder().encode_document("glider " + "padding " * 199)
    assert weight(long, "glider") < weight(short, "glider")
    # With b = 0 the length drops out: a term seen once weighs (k1+1)/(1+k1) = 1, always.
    flat = encoder(b=0)
    assert weight(flat.encode_document("glider"), "glider") == pytest.approx(1.0)
    assert weight(flat.encode_document("glider " + "padding " * 199), "glider") == pytest.approx(
        1.0
    )


def test_a_query_weighs_1_per_distinct_token_and_qdrant_adds_the_idf() -> None:
    query = encoder().encode_query("What is GLIDER, and why is glider small?")
    assert query.indices == tuple(sorted({token_index("glider"), token_index("small")}))
    assert query.values == (1.0, 1.0)


def test_a_text_with_no_tokens_is_an_empty_vector() -> None:
    assert encoder().encode_document("the and of") == SparseVector(indices=(), values=())
    assert encoder().encode_query("what is it?") == SparseVector(indices=(), values=())


def test_two_tokens_on_one_index_merge_their_frequency(monkeypatch: pytest.MonkeyPatch) -> None:
    # A collision must never send Qdrant one index twice: the weights merge instead.
    monkeypatch.setattr(sparse, "token_index", lambda token: 7)
    vector = encoder(avg_len=2).encode_document("alpha beta")
    assert vector.indices == (7,)
    assert vector.values == pytest.approx((2 * 2.2 / (2 + 1.2 * 1.0),))
    assert encoder().encode_query("alpha beta") == SparseVector(indices=(7,), values=(1.0,))


def test_documents_are_encoded_one_by_one_in_order() -> None:
    texts = ["glider", "small model", ""]
    assert encoder().encode_documents(texts) == [encoder().encode_document(t) for t in texts]


# --- settings and the vector type ---------------------------------------------------------


@pytest.mark.parametrize(
    ("settings", "error", "message"),
    [
        ({"k1": 0}, ValueError, "k1 must be a finite number above 0"),
        ({"k1": -1.2}, ValueError, "k1 must be a finite number above 0"),
        ({"k1": math.inf}, ValueError, "k1 must be a finite number above 0"),
        ({"k1": "1.2"}, TypeError, "k1 must be a number"),
        ({"k1": True}, TypeError, "k1 must be a number"),
        ({"b": 1.5}, ValueError, "b must be a number from 0 to 1"),
        ({"b": -0.1}, ValueError, "b must be a number from 0 to 1"),
        ({"b": None}, ValueError, "b must be a number from 0 to 1"),
        ({"avg_len": 0}, ValueError, "avg_len must be a finite number above 0"),
        ({"avg_len": math.nan}, ValueError, "avg_len must be a finite number above 0"),
    ],
)
def test_settings_out_of_range_are_refused(
    settings: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        Bm25Encoder(**{**BM25, **settings})  # type: ignore[arg-type]


def test_every_setting_is_required_so_the_spec_says_everything() -> None:
    with pytest.raises(TypeError):
        Bm25Encoder(k1=1.2, b=0.75)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Bm25Encoder(1.2, 0.75, 100.0)  # type: ignore[misc]
    assert repr(encoder()) == "Bm25Encoder(k1=1.2, b=0.75, avg_len=100.76)"


def test_a_sparse_vector_has_one_value_per_index_and_ascending_indices() -> None:
    with pytest.raises(ValueError, match="one value per index"):
        SparseVector(indices=(1, 2), values=(1.0,))
    with pytest.raises(ValueError, match="distinct and ascending"):
        SparseVector(indices=(2, 1), values=(1.0, 1.0))
    with pytest.raises(ValueError, match="distinct and ascending"):
        SparseVector(indices=(1, 1), values=(1.0, 1.0))


# --- the version's tripwire ---------------------------------------------------------------

#: Texts that exercise every rule: case, Unicode, digits, stopwords, punctuation, the soft
#: hyphen, repeats and length.
PROBE_TEXTS = (
    "What is GLIDER? GLIDER is a grader LLM; glider-slm, v2.0 (2024).",
    "con­\nstruction of the algo­ rithm, and the con-\nstruction again",
    "Ünïcode café_au_lait Ω42 naïve résumé",
    "the and of is it",
    "",
    "word " * 300,
)

#: :func:`probe_digest` under each SPARSE_VERSION.
PROBE_VECTORS = {1: "81e4db3d7bea73ab61c3b7826176bef95a118e8b9039760379a5020d2caf16ca"}


def probe_digest() -> str:
    """The sha256 of the probe texts' document and query vectors, as JSON. Python's float
    repr round-trips, so the digest moves only when a vector does."""
    enc = encoder()
    vectors = [
        [list(v.indices), list(v.values)]
        for text in PROBE_TEXTS
        for v in (enc.encode_document(text), enc.encode_query(text))
    ]
    return hashlib.sha256(json.dumps(vectors).encode()).hexdigest()


def test_the_probe_texts_encode_as_the_sparse_version_says() -> None:
    # The tripwire for SPARSE_VERSION. If the vectors changed, by a change to the
    # stopwords, the tokenizer, the hash or the weights, bump SPARSE_VERSION and pin this
    # digest under the new version: every index with sparse vectors then rebuilds once.
    digest = probe_digest()
    assert PROBE_VECTORS.get(SPARSE_VERSION) == digest, (
        f"the probe texts no longer encode as SPARSE_VERSION {SPARSE_VERSION} did: bump it in "
        f"rag_qa/sparse.py, and pin {digest} under the new version"
    )
