import math

from app.metrics import (
    bucket_series,
    compare_topk,
    renormalize_union,
    top1_agreement,
)


def test_identical_distributions_zero_js():
    top = {"a": -0.1, "b": -2.0, "c": -3.0}
    m = compare_topk(top, dict(top))
    assert m["top1_agree"] is True
    assert abs(m["js"]) < 1e-9
    assert abs(m["overlap_mass"] - 1.0) < 1e-9


def test_js_symmetric_and_bounded():
    a = {"a": -0.2, "b": -2.0}
    b = {"b": -0.2, "a": -2.0}
    m1 = compare_topk(a, b)
    m2 = compare_topk(b, a)
    assert 0 < m1["js"] <= math.log(2) + 1e-9
    assert abs(m1["js"] - m2["js"]) < 1e-12


def test_missing_tokens_get_floor():
    a = {"a": -0.1}
    b = {"a": -0.1, "b": -0.1}
    m = compare_topk(a, b)
    assert m["top1_agree"] is True  # both argmax 'a'
    assert 0 < m["js"] < math.log(2)


def test_disjoint_top1_disagrees():
    a = {"a": -0.1}
    b = {"b": -0.1}
    assert top1_agreement(a, b) is False
    m = compare_topk(a, b)
    assert m["js"] > 0.5 * math.log(2)  # nearly disjoint after floors


def test_renormalize_union_is_proper_distribution():
    tokens, pa, pb = renormalize_union({"x": -1.0, "y": -2.0}, {"x": -0.5})
    assert tokens == ["x", "y"]
    assert abs(sum(pa) - 1.0) < 1e-9
    assert abs(sum(pb) - 1.0) < 1e-9


def test_bucket_series():
    rows = bucket_series([10, 300, 310], [1.0, 2.0, 3.0], 256)
    assert rows == [
        {"bucket": 0, "mean": 1.0, "n": 1},
        {"bucket": 256, "mean": 2.5, "n": 2},
    ]
