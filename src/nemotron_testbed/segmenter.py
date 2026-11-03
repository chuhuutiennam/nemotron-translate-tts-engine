"""Stable punctuation sentence segmentation over cumulative streaming ASR text.

Pure text logic, shared by the engine (shadow mode) and scripts/check_segmentation.py.

A unit of text is committed when, in the part after the already committed prefix:
  * it ends with . ? ! followed by whitespace or end of text; sentences shorter than
    min_tokens, or ending with an unfinished connector word, are merged with the next
    sentence(s) until the merged unit passes,
  * every sentence in it was seen unchanged in >= punct_stable_n consecutive model steps,
    or for >= punct_stable_ms (0 = off). Both default to stable_n / stable_ms when unset;
    stable_n / stable_ms always govern the comma fallback.
Fallbacks for text without a stable sentence end:
  * comma_after_ms (0 = off): commit up to the last ", " once the segment is that old,
  * max_segment_ms (0 = off): commit the whole uncommitted text (a forced cut).
Whatever is left when the utterance ends is committed by finish().

Later ASR texts may revise words that were already committed; committed_end() finds where the
committed text ends in each new text by word alignment instead of counting tokens.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

_BOUNDARY = re.compile(r"[.?!](?=\s|$)")
_COMMA = re.compile(r",(?=\s)")
_WORD = re.compile(r"\w+", re.UNICODE)

GUARD_WORDS = {
    "vi": {"vì", "nhưng", "mà", "nên", "là", "để", "nếu", "khi", "với", "do"},
    "en": {"and", "but", "because", "so", "if", "when", "with", "to", "of", "the", "a", "that", "which", "or"},
}

COMMIT_PUNCT = "punct"
COMMIT_COMMA = "comma"
COMMIT_MAX_SEGMENT = "max_segment"
COMMIT_UTTERANCE_END = "utterance_end"

ALIGN_EXACT = "exact"
ALIGN_WORDS = "words"
ALIGN_ALIGNED = "aligned"
ALIGN_SNAP = "snap"
ALIGN_FALLBACK = "fallback"

_SENTENCE_END = frozenset(".?!")
_ALIGN_MIN_COVERAGE = 0.5  # below this share of committed words matched, the text counts as rewritten
_SNAP_EXTRA_TOKENS = 2  # how far past the revised committed words a sentence end is looked for


@dataclass(frozen=True, slots=True)
class SegmenterParams:
    stable_n: int = 2
    stable_ms: float = 0.0
    min_tokens: int = 4
    max_segment_ms: float = 8000.0
    comma_after_ms: float = 0.0
    punct_stable_n: int | None = None
    punct_stable_ms: float | None = None

    @property
    def sentence_stable_n(self) -> int:
        return self.stable_n if self.punct_stable_n is None else self.punct_stable_n

    @property
    def sentence_stable_ms(self) -> float:
        return self.stable_ms if self.punct_stable_ms is None else self.punct_stable_ms


@dataclass(frozen=True, slots=True)
class SegmentCommit:
    segment_idx: int
    text: str
    t_ms: float
    step: int
    kind: str
    first_seen_ms: float | None = None

    @property
    def forced_cut(self) -> bool:
        return self.kind == COMMIT_MAX_SEGMENT


@dataclass(frozen=True, slots=True)
class SegmentJudgement:
    segment_idx: int
    kind: str
    text: str
    t_ms: float
    gain_ms: float
    premature: bool
    revised: bool


@dataclass(frozen=True, slots=True)
class Alignment:
    """Where the uncommitted part of an ASR text starts: index into text.split(), and the rule used."""

    index: int
    how: str


_NO_ALIGNMENT = Alignment(0, ALIGN_EXACT)


@dataclass(slots=True)
class SegmenterUpdate:
    commits: list[SegmentCommit] = field(default_factory=list)
    revised: list[int] = field(default_factory=list)
    alignment: Alignment = _NO_ALIGNMENT  # of the committed text in this update's text, before its commits


@dataclass(slots=True)
class FinishResult:
    commits: list[SegmentCommit] = field(default_factory=list)
    judgements: list[SegmentJudgement] = field(default_factory=list)
    revised: list[int] = field(default_factory=list)
    alignment: Alignment = _NO_ALIGNMENT  # of the committed text in the final text


def normalize_words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def committed_end(committed_text: str, text: str) -> Alignment:
    """Find where committed_text ends in text, a newer ASR text that may revise committed words.

    Rules, first match wins:
      exact     the committed tokens are an unchanged prefix of text: cut right after them.
      words     same words, only case, punctuation or spacing changed: cut after the token holding
                the last committed word.
      aligned   word alignment (difflib): cut after the last matched committed word, then skip one
                text word per committed word revised at the very end (one for one).
      snap      the committed text ended a sentence and its last words were revised: cut after the
                first sentence end from the last matched token, within the revised words + 2 tokens.
      fallback  less than half of the committed words match (text rewritten): count the committed
                tokens, as before.
    Tokens without a word (lone punctuation) right after a words / aligned / snap cut are skipped.
    """
    committed_tokens = committed_text.split()
    tokens = text.split()
    if not committed_tokens:
        return _NO_ALIGNMENT
    if tokens[: len(committed_tokens)] == committed_tokens:
        return Alignment(len(committed_tokens), ALIGN_EXACT)
    fallback = Alignment(min(len(committed_tokens), len(tokens)), ALIGN_FALLBACK)
    committed_words = normalize_words(committed_text)
    words: list[str] = []
    owner: list[int] = []  # owner[i] = index of the token holding words[i]
    for index, token in enumerate(tokens):
        for word in normalize_words(token):
            words.append(word)
            owner.append(index)
    if not committed_words:
        return fallback
    if words[: len(committed_words)] == committed_words:
        return Alignment(_after_word(tokens, owner, len(committed_words)), ALIGN_WORDS)

    matcher = difflib.SequenceMatcher(None, committed_words, words, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size]
    if sum(block.size for block in blocks) < _ALIGN_MIN_COVERAGE * len(committed_words):
        return fallback
    sentence_end = committed_tokens[-1][-1] in _SENTENCE_END
    if sentence_end and len(blocks) > 1:
        last, prev = blocks[-1], blocks[-2]
        # A lone last word matched only after words the committed text does not have is more likely
        # a later repeat of that word than the committed end; the sentence end re-anchors the cut.
        if last.size == 1 and last.b - (prev.b + prev.size) > last.a - (prev.a + prev.size):
            blocks.pop()
    last = blocks[-1]
    a_end, b_end = last.a + last.size, last.b + last.size
    tail = len(committed_words) - a_end  # committed words revised at the very end
    if tail and sentence_end:
        first = owner[b_end - 1]
        for index in range(first, min(len(tokens), first + tail + 1 + _SNAP_EXTRA_TOKENS)):
            if tokens[index][-1] in _SENTENCE_END:
                return Alignment(_skip_punctuation(tokens, index + 1), ALIGN_SNAP)
    end_word = b_end + min(tail, len(words) - b_end)
    return Alignment(_after_word(tokens, owner, end_word), ALIGN_ALIGNED)


def remainder_after(committed_text: str, text: str) -> str:
    """The part of text after committed_text (see committed_end)."""
    return " ".join(text.split()[committed_end(committed_text, text).index :])


def _after_word(tokens: list[str], owner: list[int], end_word: int) -> int:
    """Token index right after the token holding word end_word - 1 (that whole token is committed)."""
    index = owner[end_word - 1] + 1 if end_word > 0 else 0
    return _skip_punctuation(tokens, index)


def _skip_punctuation(tokens: list[str], index: int) -> int:
    while index < len(tokens) and not _WORD.search(tokens[index]):
        index += 1
    return index


def split_sentences(text: str) -> tuple[list[str], str]:
    """Split on sentence-ending punctuation. Returns (complete sentences, unfinished tail)."""
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        sentences.append(text[start : match.end()].strip())
        start = match.end()
    return sentences, text[start:].strip()


def guard_words(language: str) -> set[str]:
    lang = (language or "").lower()
    if lang.startswith("vi"):
        return GUARD_WORDS["vi"]
    if lang.startswith("en"):
        return GUARD_WORDS["en"]
    return GUARD_WORDS["vi"] | GUARD_WORDS["en"]


def find_premature(commit_texts: list[str], final_text: str) -> list[bool]:
    """Per commit: True when any of its words is missing from the final text (in order)."""
    final_words = normalize_words(final_text)
    committed_words: list[str] = []
    spans: list[tuple[int, int]] = []
    for text in commit_texts:
        words = normalize_words(text)
        spans.append((len(committed_words), len(committed_words) + len(words)))
        committed_words.extend(words)
    matched: set[int] = set()
    matcher = difflib.SequenceMatcher(None, committed_words, final_words, autojunk=False)
    for block in matcher.get_matching_blocks():
        matched.update(range(block.a, block.a + block.size))
    return [any(index not in matched for index in range(start, end)) for start, end in spans]


class StableSentenceSegmenter:
    """Incremental segmenter for one utterance; feed it every ASR partial in order."""

    def __init__(self, params: SegmenterParams, language: str = "") -> None:
        self._params = params
        self._language = language
        self._committed_text = ""
        self._tracker: dict[str, tuple[int, float]] = {}
        self._comma_tracker: tuple[str, int, float] | None = None
        self._segment_start: float | None = None
        self._step = 0
        self._commits: list[SegmentCommit] = []
        self._words: list[str] = []
        self._spans: list[tuple[int, int]] = []
        self._revised: set[int] = set()

    @property
    def commits(self) -> list[SegmentCommit]:
        return list(self._commits)

    @property
    def committed_text(self) -> str:
        """All committed segment texts joined by a space."""
        return self._committed_text

    def set_language(self, language: str) -> None:
        self._language = language

    def update(
        self, text: str, t_ms: float, *, model_step: bool = True, step: int | None = None
    ) -> SegmenterUpdate:
        self._step = step if step is not None else self._step + 1
        if self._segment_start is None:
            self._segment_start = t_ms
        alignment = committed_end(self._committed_text, text)
        result = SegmenterUpdate(revised=self._detect_revisions(text), alignment=alignment)

        tokens = text.split()
        cut = alignment.index  # tokens before cut are committed
        sentences, _ = split_sentences(" ".join(tokens[cut:]))
        self._track_sentences(sentences, t_ms, model_step)
        guards = guard_words(self._language)

        while True:
            unit = self._next_unit(sentences, guards)
            if unit is None:
                break
            count, unit_text = unit
            if not self._is_stable(sentences[:count], t_ms):
                break
            first_seen = max(self._tracker[s][1] for s in sentences[:count])
            result.commits.append(
                self._commit(unit_text, t_ms, COMMIT_PUNCT, first_seen_ms=first_seen)
            )
            cut += len(unit_text.split())
            for sentence in sentences[:count]:
                self._tracker.pop(sentence, None)
            del sentences[:count]
            self._segment_start = t_ms

        remainder = " ".join(tokens[cut:])
        comma_commit = self._comma_candidate(remainder, t_ms, model_step, guards)
        if comma_commit is not None:
            comma_text, comma_first_seen = comma_commit
            result.commits.append(
                self._commit(comma_text, t_ms, COMMIT_COMMA, first_seen_ms=comma_first_seen)
            )
            cut += len(comma_text.split())
            self._tracker.clear()
            self._segment_start = t_ms
            remainder = " ".join(tokens[cut:])
        elif not remainder:
            self._comma_tracker = None

        if (
            self._params.max_segment_ms > 0
            and remainder
            and t_ms - self._segment_start >= self._params.max_segment_ms
        ):
            result.commits.append(self._commit(remainder, t_ms, COMMIT_MAX_SEGMENT))
            self._tracker.clear()
            self._comma_tracker = None
            self._segment_start = t_ms
        return result

    def finish(self, final_text: str, t_ms: float, *, step: int | None = None) -> FinishResult:
        """Commit whatever is left of the final text and judge every earlier commit against it."""
        self._step = step if step is not None else self._step + 1
        alignment = committed_end(self._committed_text, final_text)
        result = FinishResult(revised=self._detect_revisions(final_text), alignment=alignment)
        early = list(self._commits)
        remainder = " ".join(final_text.split()[alignment.index :])
        if remainder:
            result.commits.append(
                self._commit(remainder, t_ms, COMMIT_UTTERANCE_END, record_words=False)
            )
        premature = find_premature([c.text for c in early], final_text)
        for commit, is_premature in zip(early, premature):
            result.judgements.append(
                SegmentJudgement(
                    segment_idx=commit.segment_idx,
                    kind=commit.kind,
                    text=commit.text,
                    t_ms=commit.t_ms,
                    gain_ms=t_ms - commit.t_ms,
                    premature=is_premature,
                    revised=commit.segment_idx in self._revised,
                )
            )
        return result

    def _commit(
        self,
        text: str,
        t_ms: float,
        kind: str,
        *,
        record_words: bool = True,
        first_seen_ms: float | None = None,
    ) -> SegmentCommit:
        commit = SegmentCommit(
            len(self._commits), text, t_ms, self._step, kind, first_seen_ms
        )
        self._commits.append(commit)
        self._committed_text = f"{self._committed_text} {text}" if self._committed_text else text
        if record_words:
            words = normalize_words(text)
            self._spans.append((len(self._words), len(self._words) + len(words)))
            self._words.extend(words)
        return commit

    def _detect_revisions(self, text: str) -> list[int]:
        if not self._spans:
            return []
        words = normalize_words(text)
        changed: list[int] = []
        for index, (start, end) in enumerate(self._spans):
            if index in self._revised:
                continue
            if words[start:end] != self._words[start:end]:
                self._revised.add(index)
                changed.append(index)
        return changed

    def _track_sentences(self, sentences: list[str], t_ms: float, model_step: bool) -> None:
        updated: dict[str, tuple[int, float]] = {}
        for sentence in sentences:
            previous = self._tracker.get(sentence)
            if previous is None:
                updated[sentence] = (1, t_ms)
            else:
                updated[sentence] = (previous[0] + (1 if model_step else 0), previous[1])
        self._tracker = updated

    def _passes_guard(self, text: str, guards: set[str]) -> bool:
        words = normalize_words(text)
        return len(words) >= max(1, self._params.min_tokens) and words[-1] not in guards

    def _next_unit(self, sentences: list[str], guards: set[str]) -> tuple[int, str] | None:
        for count in range(1, len(sentences) + 1):
            unit_text = " ".join(sentences[:count])
            if self._passes_guard(unit_text, guards):
                return count, unit_text
        return None

    def _is_stable(self, sentences: list[str], t_ms: float) -> bool:
        count = min(self._tracker[s][0] for s in sentences)
        first_t = max(self._tracker[s][1] for s in sentences)
        stable_ms = self._params.sentence_stable_ms
        return count >= self._params.sentence_stable_n or (
            stable_ms > 0 and t_ms - first_t >= stable_ms
        )

    def _comma_candidate(
        self, remainder: str, t_ms: float, model_step: bool, guards: set[str]
    ) -> tuple[str, float] | None:
        if self._params.comma_after_ms <= 0 or not remainder:
            return None
        segment_start = t_ms if self._segment_start is None else self._segment_start
        if t_ms - segment_start < self._params.comma_after_ms:
            return None
        prefix = ""
        for match in reversed(list(_COMMA.finditer(remainder))):
            candidate = remainder[: match.end()].strip()
            if self._passes_guard(candidate, guards):
                prefix = candidate
                break
        if not prefix:
            self._comma_tracker = None
            return None
        if self._comma_tracker is not None and self._comma_tracker[0] == prefix:
            _, count, first_t = self._comma_tracker
            count += 1 if model_step else 0
        else:
            count, first_t = 1, t_ms
        self._comma_tracker = (prefix, count, first_t)
        stable = count >= self._params.stable_n or (
            self._params.stable_ms > 0 and t_ms - first_t >= self._params.stable_ms
        )
        if not stable:
            return None
        self._comma_tracker = None
        return prefix, first_t
