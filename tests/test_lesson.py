"""The mock must reproduce the lesson: a same-dimension swap is silent and wrecks the results."""

import pytest

from embedstamp import FakeEmbedder
from embedstamp.demo import load_lines, mean_overlap_at_k


def test_demo_data_sizes():
    assert len(load_lines("demo_corpus.txt")) == 160
    assert len(load_lines("demo_queries.txt")) == 16


def test_swap_raises_nothing_but_results_change(demo_index):
    queries = load_lines("demo_queries.txt")
    a, b = FakeEmbedder("fake:a"), FakeEmbedder("fake:b")
    assert mean_overlap_at_k(demo_index, queries, a, a) == 1.0
    # fake:b has the same 384 dimensions, so the index happily answers without any error
    overlap = mean_overlap_at_k(demo_index, queries, a, b)
    assert overlap < 0.5


def test_search_works_inside_one_model(demo_index):
    a = FakeEmbedder("fake:a")
    top = demo_index.query(a(["how long should dough rise for bread"])[0], 5)
    texts = dict(zip(demo_index._ids, demo_index._texts))
    assert any("bread" in texts[i] or "dough" in texts[i] for i in top)


def test_wrong_dimension_does_raise(demo_index):
    with pytest.raises(ValueError, match="dimension"):
        demo_index.query(FakeEmbedder("fake:a", dim=256)(["x"])[0])
