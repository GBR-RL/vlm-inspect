import pytest

from vlm_inspect.inspectors.qwen_vl import (
    classification_prompt,
    fit_longest_side,
    localization_prompt,
)
from vlm_inspect.inspectors.vlm_output import extract_json, parse_findings, yes_probability
from vlm_inspect.parts import get_part


def test_parses_qwen_grounding_json_in_code_fence() -> None:
    text = (
        "Here are the defects:\n```json\n"
        '[{"bbox_2d": [100, 200, 300, 400], "label": "scratch"}]\n```'
    )
    findings = parse_findings(text, width=1000, height=500, score=0.9)
    assert len(findings) == 1
    box = findings[0].box
    # 0-1000 normalised coordinates map to pixels per axis.
    assert (box.x1, box.y1, box.x2, box.y2) == (100.0, 100.0, 300.0, 200.0)
    assert findings[0].label == "scratch"
    assert findings[0].score == 0.9


def test_accepts_dict_wrapper_swapped_corners_and_skips_garbage() -> None:
    text = (
        '{"defects": [{"bbox_2d": [500, 500, 100, 100]}, {"bbox_2d": [1, 2]}, '
        '{"bbox_2d": ["a", 0, 1, 1]}, "junk", {"bbox": [0, 0, 1000, 1000], "label": ""}]}'
    )
    findings = parse_findings(text, width=200, height=100, score=0.7)
    assert [(f.box.x1, f.box.x2) for f in findings] == [(20.0, 100.0), (0.0, 200.0)]
    assert findings[0].label == "defect"
    assert findings[1].label == "defect"


def test_empty_list_and_unparseable_output_give_no_findings() -> None:
    assert parse_findings("[]", 100, 100, 0.8) == []
    assert parse_findings("I cannot see any defect.", 100, 100, 0.8) == []
    assert extract_json("no json here") is None


def test_boxes_are_clipped_and_degenerate_ones_dropped() -> None:
    findings = parse_findings(
        '[{"bbox_2d": [-50, 0, 1200, 1000]}, {"bbox_2d": [10, 10, 10, 20]}]', 100, 100, 1.0
    )
    assert len(findings) == 1
    assert (findings[0].box.x1, findings[0].box.x2) == (0.0, 100.0)


def test_yes_probability_is_a_two_way_softmax() -> None:
    assert yes_probability(0.0, 0.0) == pytest.approx(0.5)
    assert yes_probability(5.0, 0.0) > 0.99
    assert yes_probability(1000.0, -1000.0) == pytest.approx(1.0)  # no overflow


def test_prompts_mention_part_and_defects() -> None:
    part = get_part("pcb1")
    zero = classification_prompt(part, one_shot=False)
    one = classification_prompt(part, one_shot=True)
    assert "missing component" in zero
    assert zero.endswith("Answer Yes or No.")
    assert "defect-free example" in one
    assert "bbox_2d" in localization_prompt(part)


def test_fit_longest_side_keeps_aspect_ratio() -> None:
    from PIL import Image

    image = Image.new("RGB", (1500, 1000))
    small = fit_longest_side(image, 768)
    assert small.size == (768, 512)
    assert fit_longest_side(Image.new("RGB", (300, 200)), 768).size == (300, 200)


def test_unknown_part_is_rejected() -> None:
    with pytest.raises(KeyError, match="known"):
        get_part("gearbox")


def test_recovers_boxes_from_output_truncated_at_the_token_limit() -> None:
    # Shape of real Qwen3-VL output that hit max_new_tokens mid-list (no closing bracket).
    text = (
        '```json\n[\n  {"bbox_2d": [129, 295, 160, 330], "label": "bent component"},\n'
        '  {"bbox_2d": [129, 645, 159, 680], "label": "bent component"},\n  {"bbox_2d": [160, 64'
    )
    findings = parse_findings(text, width=1000, height=1000, score=0.6)
    assert [f.box.x1 for f in findings] == [129.0, 129.0]


def test_duplicate_boxes_are_reported_once() -> None:
    text = (
        '[{"bbox_2d": [10, 10, 50, 50], "label": "a"}, {"bbox_2d": [10, 10, 50, 50], "label": "b"}]'
    )
    assert len(parse_findings(text, 1000, 1000, 0.5)) == 1


def test_extract_json_returns_the_outer_structure() -> None:
    # Regression: an object containing a list used to come back as the inner list.
    assert extract_json('{"verdict": "REJECT", "citations": ["A", "B"]}') == {
        "verdict": "REJECT",
        "citations": ["A", "B"],
    }
    assert extract_json('text [{"bbox_2d": [1, 2, 3, 4]}] more') == [{"bbox_2d": [1, 2, 3, 4]}]
