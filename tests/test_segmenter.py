from __future__ import annotations

import pytest

from nemotron_testbed.segmenter import (
    ALIGN_ALIGNED,
    ALIGN_EXACT,
    ALIGN_FALLBACK,
    ALIGN_SNAP,
    ALIGN_WORDS,
    COMMIT_COMMA,
    COMMIT_MAX_SEGMENT,
    COMMIT_PUNCT,
    COMMIT_UTTERANCE_END,
    SegmenterParams,
    StableSentenceSegmenter,
    committed_end,
    find_premature,
    guard_words,
    remainder_after,
    split_sentences,
)

SENT = "This is a complete sentence here."


def make(**kw) -> StableSentenceSegmenter:
    params = SegmenterParams(**{"stable_n": 2, "min_tokens": 4, "max_segment_ms": 0.0, **kw})
    return StableSentenceSegmenter(params, "en")


def test_split_sentences_requires_whitespace_or_end() -> None:
    sentences, tail = split_sentences("Hello world. Version 3.5 is out! And then")
    assert sentences == ["Hello world.", "Version 3.5 is out!"]
    assert tail == "And then"


def test_commit_requires_stable_n_steps() -> None:
    seg = make()
    assert seg.update(SENT, 100).commits == []
    update = seg.update(SENT, 600)
    assert [c.text for c in update.commits] == [SENT]
    assert update.commits[0].kind == COMMIT_PUNCT
    assert update.commits[0].segment_idx == 0


def test_non_model_step_does_not_count_towards_stability() -> None:
    seg = make()
    seg.update(SENT, 100)
    assert seg.update(SENT, 200, model_step=False).commits == []
    assert len(seg.update(SENT, 300).commits) == 1


def test_changed_sentence_resets_stability() -> None:
    seg = make()
    seg.update("This is a complete sentence there.", 100)
    assert seg.update(SENT, 600).commits == []
    assert len(seg.update(SENT, 1100).commits) == 1


def test_stable_ms_commits_without_enough_steps() -> None:
    seg = make(stable_n=5, stable_ms=400.0)
    seg.update(SENT, 100)
    assert seg.update(SENT, 300, model_step=False).commits == []
    assert len(seg.update(SENT, 600, model_step=False).commits) == 1


def test_punct_stable_n_overrides_stable_n_for_sentence_commit() -> None:
    seg = make(stable_n=3, punct_stable_n=1)
    update = seg.update(SENT, 100)
    assert [c.kind for c in update.commits] == [COMMIT_PUNCT]


def test_punct_stable_n_unset_inherits_stable_n() -> None:
    assert make(stable_n=3).update(SENT, 100).commits == []
    params = SegmenterParams(stable_n=3, stable_ms=400.0)
    assert (params.sentence_stable_n, params.sentence_stable_ms) == (3, 400.0)


def test_punct_stable_ms_overrides_stable_ms() -> None:
    seg = make(stable_n=5, stable_ms=0.0, punct_stable_n=5, punct_stable_ms=400.0)
    seg.update(SENT, 100)
    assert len(seg.update(SENT, 600, model_step=False).commits) == 1


def test_commit_records_first_seen_of_sentence() -> None:
    seg = make()
    seg.update("This is a complete", 50)
    seg.update(SENT, 100)
    update = seg.update(SENT, 600)
    assert update.commits[0].first_seen_ms == 100
    assert update.commits[0].t_ms == 600


def test_first_seen_resets_when_sentence_text_changes() -> None:
    seg = make()
    seg.update("This is a complete sentence there.", 100)
    seg.update(SENT, 600)
    assert seg.update(SENT, 1100).commits[0].first_seen_ms == 600


def test_comma_commit_records_first_seen_and_forced_cut_has_none() -> None:
    text = "when the weather is nice outside, we go for a walk"
    seg = make(comma_after_ms=1000.0)
    seg.update(text, 0)
    seg.update(text, 1100)
    comma = seg.update(text, 1700).commits[0]
    assert comma.first_seen_ms == 1100
    seg = make(max_segment_ms=1000.0)
    seg.update("no punctuation here at all", 0)
    forced = seg.update("no punctuation here at all", 1200).commits[0]
    assert forced.kind == COMMIT_MAX_SEGMENT and forced.first_seen_ms is None


def test_stable_n_still_governs_comma_commit() -> None:
    text = "when the weather is nice outside, we go for a walk"
    seg = make(stable_n=2, punct_stable_n=1, comma_after_ms=1000.0)
    seg.update(text, 0)
    assert seg.update(text, 1100).commits == []
    update = seg.update(text, 1700)
    assert [c.kind for c in update.commits] == [COMMIT_COMMA]


def test_short_sentence_merges_with_next() -> None:
    seg = make()
    text = "Image formats. This is a longer second sentence."
    seg.update(text, 100)
    commits = seg.update(text, 600).commits
    assert [c.text for c in commits] == [text]


def test_short_sentence_alone_is_not_committed() -> None:
    seg = make()
    seg.update("Image formats.", 100)
    assert seg.update("Image formats.", 600).commits == []


def test_connector_word_guard_blocks_commit() -> None:
    seg = make()
    text = "I would like to talk about the."
    seg.update(text, 100)
    assert seg.update(text, 600).commits == []


def test_vietnamese_guard_words() -> None:
    assert "nhưng" in guard_words("vi")
    assert "and" in guard_words("en-US")
    assert {"nhưng", "and"} <= guard_words("")


def test_two_sentences_commit_separately_when_both_pass() -> None:
    seg = make()
    first = "This is the first sentence."
    second = "And here comes another long sentence."
    both = first + " " + second
    seg.update(first, 100)
    assert [c.text for c in seg.update(both, 600).commits] == [first]
    assert [c.text for c in seg.update(both, 1100).commits] == [second]
    assert [c.segment_idx for c in seg.commits] == [0, 1]


def test_committed_prefix_is_not_recommitted() -> None:
    seg = make()
    seg.update(SENT, 100)
    seg.update(SENT, 600)
    assert seg.update(SENT, 1100).commits == []
    assert len(seg.commits) == 1


def test_comma_fallback_after_age() -> None:
    seg = make(comma_after_ms=1000.0)
    text = "when the weather is nice outside, we go for a walk"
    seg.update(text, 0)
    assert seg.update(text, 500).commits == []
    seg.update(text, 1100)
    update = seg.update(text, 1700)
    assert [c.kind for c in update.commits] == [COMMIT_COMMA]
    assert update.commits[0].text == "when the weather is nice outside,"


def test_comma_fallback_off_by_default() -> None:
    seg = make()
    text = "when the weather is nice outside, we go for a walk"
    for t in (0, 1000, 2000, 3000):
        assert seg.update(text, t).commits == []


def test_max_segment_forced_cut() -> None:
    seg = make(max_segment_ms=2000.0)
    text = "no punctuation at all in this long run of words"
    seg.update(text, 0)
    assert seg.update(text, 1000).commits == []
    update = seg.update(text, 2100)
    assert [c.kind for c in update.commits] == [COMMIT_MAX_SEGMENT]
    assert update.commits[0].forced_cut
    assert update.commits[0].text == text


def test_finish_commits_remainder_and_judges() -> None:
    seg = make()
    seg.update(SENT + " and the rest", 100)
    seg.update(SENT + " and the rest", 600)
    result = seg.finish(SENT + " and the rest of it", 1500)
    assert [c.kind for c in result.commits] == [COMMIT_UTTERANCE_END]
    assert result.commits[0].text == "and the rest of it"
    assert len(result.judgements) == 1
    judgement = result.judgements[0]
    assert judgement.gain_ms == 1500 - 600
    assert not judgement.premature
    assert not judgement.revised


def test_finish_without_remainder_has_no_commit() -> None:
    seg = make()
    seg.update(SENT, 100)
    seg.update(SENT, 600)
    result = seg.finish(SENT, 900)
    assert result.commits == []
    assert len(result.judgements) == 1


def test_revision_of_committed_text_is_detected() -> None:
    seg = make()
    seg.update(SENT, 100)
    seg.update(SENT, 600)
    revised = seg.update("This was a complete sentence here. More", 1100).revised
    assert revised == [0]
    assert seg.update("This was a complete sentence here. More words", 1600).revised == []
    result = seg.finish("This was a complete sentence here. More words", 2000)
    assert result.judgements[0].revised
    assert result.judgements[0].premature


def test_revision_only_visible_in_final_is_reported_by_finish() -> None:
    seg = make()
    seg.update(SENT, 100)
    seg.update(SENT, 600)
    result = seg.finish("This was a complete sentence here.", 1000)
    assert result.revised == [0]
    assert result.judgements[0].revised
    assert result.judgements[0].premature


def test_find_premature() -> None:
    commits = ["I like red apples.", "They are good."]
    assert find_premature(commits, "I like red apples. They are good.") == [False, False]
    assert find_premature(commits, "I like green apples. They are good.") == [True, False]
    assert find_premature([], "anything") == []


def test_params_are_hashable_and_frozen() -> None:
    assert SegmenterParams() == SegmenterParams()
    assert hash(SegmenterParams()) == hash(SegmenterParams())


# ── committed_end: where already committed text ends in a newer ASR text ──────────────────────

ALIGN_CASES = [
    # (name, committed text, newer ASR text, expected rule, expected remainder)
    ("unchanged", "I'm going to the store today.", "I'm going to the store today. And then home.",
     ALIGN_EXACT, "And then home."),
    ("contraction split", "I'm going to the store today.", "I am going to the store today. And then home.",
     ALIGN_ALIGNED, "And then home."),
    ("contraction merge", "We will meet at nine thirty.", "We'll meet at nine thirty. See you there.",
     ALIGN_ALIGNED, "See you there."),
    ("number normalised", "It costs twenty five dollars, okay.", "It costs 25 dollars, okay. Thanks.",
     ALIGN_ALIGNED, "Thanks."),
    ("word inserted", "Please send me the report now.", "Please send me the final report now. Bye.",
     ALIGN_ALIGNED, "Bye."),
    ("word dropped", "Please send me the final report now.", "Please send me the report now. Bye.",
     ALIGN_ALIGNED, "Bye."),
    ("last word revised", "I will go to the store today.", "I will go to the store to day. And then home.",
     ALIGN_SNAP, "And then home."),
    ("last word dropped", "Please send me the report now.", "Please send me the report. Bye.",
     ALIGN_SNAP, "Bye."),
    ("dropped last word said again later", "Please send me the report now.",
     "Please send me the report. Bye bye now.", ALIGN_SNAP, "Bye bye now."),
    ("word inserted before last word", "Please send me the report.", "Please send me the final report. Bye.",
     ALIGN_SNAP, "Bye."),
    ("final drops a committed sentence", "One two three four five six. Seven eight nine ten eleven twelve.",
     "One two three four five six.", ALIGN_SNAP, ""),
    ("punctuation only", "Hello my friend how are you.", "Hello, my friend, how are you? Fine.",
     ALIGN_WORDS, "Fine."),
    ("case only", "hello there my good friend.", "Hello there my good friend. Next.", ALIGN_WORDS, "Next."),
    ("lone punctuation token at the cut", "One two three four five six.",
     "One two three four five six . Seven eight.", ALIGN_WORDS, "Seven eight."),
    ("repeated phrase", "Thank you very much for that.", "Thank you very much for that. Thank you very much.",
     ALIGN_EXACT, "Thank you very much."),
    ("filler inserted first", "I'm going to the store today.", "Uh I'm going to the store today. Okay.",
     ALIGN_ALIGNED, "Okay."),
    ("two commits, first revised", "I'm going to the store today. Then we will eat lunch.",
     "I am going to the store today. Then we will eat lunch. After that home.", ALIGN_ALIGNED, "After that home."),
    ("vietnamese number", "Tôi muốn đặt hai mươi vé.", "Tôi muốn đặt 20 vé. Cảm ơn bạn.",
     ALIGN_ALIGNED, "Cảm ơn bạn."),
    ("forced cut, word inserted", "please send me the report", "please send me the final report and more",
     ALIGN_ALIGNED, "and more"),
    # Known limit: a partial whose revised last word has no sentence end yet keeps a fragment.
    ("last word revised, no punctuation yet", "I will go to the store today.",
     "I will go to the store to day and then", ALIGN_ALIGNED, "day and then"),
    ("text rewritten", "The weather is nice today.", "Completely different words here now. Yes.",
     ALIGN_FALLBACK, "Yes."),
    # One of six words in common: under half, so the token count is kept instead of snapping to "later."
    ("text rewritten, one word in common", "The weather is really nice today.",
     "Something else entirely about the market now and later. Then more.", ALIGN_FALLBACK,
     "now and later. Then more."),
    ("text shorter than committed", "One two three four five six.", "One two three", ALIGN_ALIGNED, ""),
    ("nothing committed", "", "Anything at all.", ALIGN_EXACT, "Anything at all."),
]


@pytest.mark.parametrize(
    "committed,text,how,remainder", [case[1:] for case in ALIGN_CASES], ids=[case[0] for case in ALIGN_CASES]
)
def test_committed_end_cases(committed: str, text: str, how: str, remainder: str) -> None:
    assert committed_end(committed, text).how == how
    assert remainder_after(committed, text) == remainder


def test_unchanged_prefix_keeps_the_token_count_cut() -> None:
    text = "One. Two three, four five six seven! Eight nine ten. Eleven twelve"
    tokens = text.split()
    for count in range(len(tokens) + 1):
        alignment = committed_end(" ".join(tokens[:count]), text)
        assert alignment.index == count
        assert alignment.how == ALIGN_EXACT


def test_fallback_keeps_the_token_count_cut() -> None:
    alignment = committed_end("Alpha beta gamma delta.", "Totally other words entirely here. Rest")
    assert alignment.how == ALIGN_FALLBACK
    assert alignment.index == 4


def test_contraction_revision_does_not_recommit_committed_words() -> None:
    first = "I'm going to the store today."
    revised = "I am going to the store today. And then we go home now."
    seg = make()
    seg.update(first, 100)
    assert [c.text for c in seg.update(first, 600).commits] == [first]
    update = seg.update(revised, 1100)
    assert update.alignment.how == ALIGN_ALIGNED
    assert update.commits == []
    # With a token count the next commit would start with the repeated "today."
    assert [c.text for c in seg.update(revised, 1600).commits] == ["And then we go home now."]
    assert seg.committed_text == f"{first} And then we go home now."


def test_finish_remainder_follows_revised_final() -> None:
    first = "We will meet at nine thirty."
    seg = make()
    seg.update(first, 100)
    seg.update(first, 600)
    result = seg.finish("We'll meet at nine thirty. See you there.", 1500)
    assert result.alignment.how == ALIGN_ALIGNED
    assert [(c.kind, c.text) for c in result.commits] == [(COMMIT_UTTERANCE_END, "See you there.")]


def test_update_and_finish_report_exact_alignment_when_nothing_was_revised() -> None:
    seg = make()
    assert seg.update(SENT, 100).alignment.how == ALIGN_EXACT  # nothing committed yet
    seg.update(SENT, 600)
    assert seg.update(SENT + " more", 1100).alignment.how == ALIGN_EXACT
    assert seg.finish(SENT + " more words", 1500).alignment.how == ALIGN_EXACT
