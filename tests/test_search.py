import numpy as np

from codexa_api.packs import to_int8
from codexa_api.search import rrf


def test_to_int8_rounds_halves_up_and_never_uses_minus_128():
    # -63.5 rounds up to -63, as the stored vectors were rounded (NumPy's round would give -64)
    assert to_int8(np.array([1.0, -1.0, -0.5, 1.5, -1.5])).tolist() == [127, -127, -63, 127, -127]


def test_rrf_rewards_items_ranked_high_in_several_lists():
    scores = rrf([[("kjv", 1), ("kjv", 2)], [("kjv", 2), ("kjv", 3)]])
    assert max(scores, key=scores.get) == ("kjv", 2)
    assert scores[("kjv", 1)] == 1 / 60


def test_keyword_search_finds_the_word(searcher):
    assert searcher.search("wept", mode="keyword")[0].ref == "John 11:35"


def test_query_with_no_words_finds_nothing_by_keyword(searcher):
    assert searcher.search("?!", mode="keyword") == []


def test_semantic_search_ranks_by_similarity(searcher):
    top = searcher.search("children mocked his bald head", mode="semantic", kind="verse")[0]
    assert top.ref == "2 Kings 2:23"
    assert 0 < top.score <= 1  # a cosine


def test_hybrid_search_spans_packs_unless_filtered(searcher):
    query = "God created the heaven and the earth"
    assert {h.work_id for h in searcher.search(query)} == {"kjv", "mhenry"}
    verses = searcher.search(query, kind="verse")
    assert {h.work_id for h in verses} == {"kjv"}
    assert verses[0].ref == "Genesis 1:1"
