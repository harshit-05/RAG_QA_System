"""Our own BM25 encoder: the sparse vectors of hybrid retrieval (DEC-20, ARCHITECTURE.md §3.1).

Hybrid retrieval searches two ways at once. The dense embedding finds chunks that mean what
the question means; this encoder's sparse vectors find chunks that share its words, which
catches a name, an acronym or a number the embedder blurs. :mod:`rag_qa.vectorstore` fuses
the two candidate lists, and the cross-encoder reranks what survives.

**BM25, split between us and Qdrant.** A document's weight for a term is

    tf·(k1+1) / (tf + k1·(1 − b + b·len/avg_len))

and a query's is 1 for each distinct term. The collection's sparse vector is created with
``modifier=IDF``, so Qdrant multiplies in each term's inverse document frequency itself, from
the collection it searches. That is BM25 exactly, as Qdrant documents it. The IDF is applied
there and only there, never here as well.

**Tokens** (:func:`tokens`): a soft hyphen (U+00AD) is removed with the whitespace after it,
since PDF text marks a line-break hyphenation with one, and "con\\u00ad\\nstruction" must stay
one word; then the text is lowercased and cut into Unicode word tokens (``\\w+``), less a
fixed English stopword list. No stemming in v1.0 (ARCHITECTURE.md §3.6).

**Indices** (:func:`token_index`): each token becomes a uint32 through blake2b with a 4-byte
digest, which is stable across processes and machines, unlike ``hash()``. Term frequency is
counted per index, so a collision merges two tokens' weights rather than sending Qdrant the
same index twice.

**Versioning.** :data:`SPARSE_VERSION` versions all of the above, as
:data:`rag_qa.manifest.CHUNKING_VERSION` versions the chunks. The encoder's identity in the
manifest is its spec plus this version, so a change to the tokenizer rebuilds every index
that has sparse vectors. ``tests/test_sparse.py`` pins the sample corpus's vectors to it.

The standard library only (``tests/test_architecture.py``): no LangChain, no numpy.
"""

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

#: How this code turns text into sparse vectors. **Bump it whenever the same text and
#: settings would make other vectors:** the stopwords, the tokenizer, the hash or the
#: weights. Every index with sparse vectors then rebuilds in full.
SPARSE_VERSION: Final = 1

#: NLTK's English stopword list, less its 26 forms with an apostrophe ("don't",
#: "you're"…), which ``\w+`` never yields: the pieces they split into, "don" and "t", are
#: here already. Fixed in code, so no download and no library version can change it.
_STOPWORDS_TEXT: Final = """
    a about above after again against ain all am an and any are aren as at be because been
    before being below between both but by can couldn d did didn do does doesn doing don down
    during each few for from further had hadn has hasn have haven having he her here hers
    herself him himself his how i if in into is isn it its itself just ll m ma me mightn more
    most mustn my myself needn no nor not now o of off on once only or other our ours
    ourselves out over own re s same shan she should shouldn so some such t than that the
    their theirs them themselves then there these they this those through to too under until
    up ve very was wasn we were weren what when where which while who whom why will with won
    wouldn y you your yours yourself yourselves
"""
#: Kept as words in a text, which reads as prose: as a list, the formatter would give each
#: of the 153 a line of its own.
STOPWORDS: Final = frozenset(_STOPWORDS_TEXT.split())

#: A soft hyphen and the whitespace after it: a line-break hyphenation in PDF text.
_SOFT_HYPHEN = re.compile("­\\s*")
#: A Unicode word token: letters, digits and the underscore, in any script.
_WORD = re.compile(r"\w+")


def tokens(text: str) -> list[str]:
    """``text``'s tokens, in order: soft hyphens joined, lowercased, ``\\w+``, no stopwords."""
    words = _WORD.findall(_SOFT_HYPHEN.sub("", text).lower())
    return [word for word in words if word not in STOPWORDS]


def token_index(token: str) -> int:
    """The sparse index of ``token``: its UTF-8 bytes' 4-byte blake2b digest, big-endian."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


@dataclass(frozen=True)
class SparseVector:
    """A sparse vector: parallel ``indices`` (ascending, distinct) and ``values``.

    Ours, so this module needs no client library; :mod:`rag_qa.vectorstore` converts it for
    Qdrant. An empty one is a text with no tokens.
    """

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"a sparse vector needs one value per index: {len(self.indices)} indices, "
                f"{len(self.values)} values"
            )
        if any(a >= b for a, b in zip(self.indices, self.indices[1:], strict=False)):
            raise ValueError("a sparse vector's indices must be distinct and ascending")


def _positive(name: str, value: float) -> float:
    """``value`` as a float, when it is a finite number above 0."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number, not {value!r}")
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a finite number above 0, not {value!r}")
    return float(value)


class Bm25Encoder:
    """BM25 sparse vectors for documents and queries (the module docstring).

    The ``_target_`` of ``components.sparse_encoders.bm25``. Its three settings are
    required, so the spec the manifest and the fingerprint hash is all that decides the
    vectors:

    - ``k1``: how fast a term's weight saturates as it repeats (1.2 is usual);
    - ``b``: how much a longer chunk's weights shrink, from 0 (not at all) to 1 (0.75 is
      usual);
    - ``avg_len``: the mean number of tokens in the corpus's chunks, as :func:`tokens`
      counts them. Measured once and fixed in the config: a statistic that moved with
      every ingest would rewrite every sparse vector (DEC-20).
    """

    def __init__(self, *, k1: float, b: float, avg_len: float) -> None:
        self.k1 = _positive("k1", k1)
        if isinstance(b, bool) or not isinstance(b, int | float) or not 0 <= b <= 1:
            raise ValueError(f"b must be a number from 0 to 1, not {b!r}")
        self.b = float(b)
        self.avg_len = _positive("avg_len", avg_len)

    def __repr__(self) -> str:
        return f"Bm25Encoder(k1={self.k1}, b={self.b}, avg_len={self.avg_len})"

    def encode_document(self, text: str) -> SparseVector:
        """One document's vector: each index weighs ``tf·(k1+1) / (tf + k1·norm)``, with
        ``norm = 1 − b + b·len/avg_len`` and ``len`` the document's token count."""
        found = tokens(text)
        counts = Counter(token_index(token) for token in found)
        norm = 1 - self.b + self.b * len(found) / self.avg_len
        indices = sorted(counts)
        return SparseVector(
            indices=tuple(indices),
            values=tuple(counts[i] * (self.k1 + 1) / (counts[i] + self.k1 * norm) for i in indices),
        )

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        """Each text's vector, in order (:meth:`encode_document`)."""
        return [self.encode_document(text) for text in texts]

    def encode_query(self, text: str) -> SparseVector:
        """A query's vector: 1 for each distinct index. Qdrant adds the IDF."""
        indices = sorted({token_index(token) for token in tokens(text)})
        return SparseVector(indices=tuple(indices), values=(1.0,) * len(indices))
