from vlm_inspect.eval.gallery import select


def _row(image, category, label, flagged, score, *, hit=False, box=None):
    findings = [{"box": box, "label": "x", "score": score}] if box else []
    return {
        "image": image,
        "category": category,
        "label": label,
        "is_defective": flagged,
        "score": score,
        "hit": hit,
        "findings": findings,
    }


def test_select_ranks_each_failure_mode_and_alternates_parts() -> None:
    small = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
    large = {"x1": 0, "y1": 0, "x2": 90, "y2": 90}
    rows = [
        _row("a/miss-low", "a", 1, False, 0.01),
        _row("a/miss-high", "a", 1, False, 0.2),
        _row("b/miss", "b", 1, False, 0.1),
        _row("a/box-small", "a", 1, True, 0.9, box=small),
        _row("a/box-large", "a", 1, True, 0.9, box=large),
        _row("a/alarm", "a", 0, True, 0.7),
        _row("a/good", "a", 0, False, 0.1),
        _row("a/found", "a", 1, True, 0.9, hit=True, box=small),
        _row("a/named-wrong", "a", 1, True, 0.8, hit=True, box=small),
    ]
    chosen = select(
        rows,
        {"a": 0.5, "b": 0.5},
        per_mode=2,
        image_size=lambda _: (100, 100),
        ungrounded={"a/named-wrong"},
    )
    assert [r["image"] for r in chosen["missed"]] == ["a/miss-low", "b/miss"]
    assert [r["image"] for r in chosen["box_misses"]] == ["a/box-large", "a/box-small"]
    assert [r["image"] for r in chosen["false_alarm"]] == ["a/alarm"]
    assert [r["image"] for r in chosen["wrong_type"]] == ["a/named-wrong"]
    assert chosen["missed"][0]["threshold"] == 0.5


def test_wrong_type_needs_report_eval() -> None:
    chosen = select([], {}, per_mode=1, image_size=lambda _: (1, 1))
    assert "wrong_type" not in chosen
