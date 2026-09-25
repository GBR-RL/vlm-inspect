"""What each inspected part is and which defects to look for.

The same descriptions drive the VLM prompts (M1) and seed the inspection specifications used for
retrieval-grounded reports (M4). Defect types follow the VisA annotations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PartSpec:
    name: str
    description: str
    defect_types: tuple[str, ...]

    def defect_list(self) -> str:
        return ", ".join(self.defect_types)


PARTS: dict[str, PartSpec] = {
    "pcb1": PartSpec(
        name="pcb1",
        description="an HC-SR04 ultrasonic sensor module: a blue printed circuit board with two "
        "ultrasonic transducers, a crystal oscillator and four pin headers",
        defect_types=("bent component", "melted component", "missing component", "scratch"),
    ),
    "candle": PartSpec(
        name="candle",
        description="a set of four round white wax candles, seen from above",
        defect_types=(
            "missing chunk of wax",
            "damaged corner",
            "abnormal wick",
            "discoloration",
            "foreign particle",
            "wax melted onto the surface",
        ),
    ),
    "capsules": PartSpec(
        name="capsules",
        description="about twenty green translucent soft-gel capsules lying on a grey surface",
        defect_types=("bubble", "discoloration", "scratch", "squeezed or deformed capsule"),
    ),
}


# Prompt v2: descriptions and defect lists aligned with VisA's own annotation taxonomy
# (<part>/image_anno.csv). v1 above is what the first benchmark used; it simplified the candle
# ("round white wax candles", no mention of the aluminium cups, although "damaged corner of
# packaging" is the most frequent candle defect) and omitted the capsule "leak" class. pcb1's v1
# list already matched the taxonomy, so it is unchanged.
PARTS_V2: dict[str, PartSpec] = {
    "pcb1": PARTS["pcb1"],
    "candle": PartSpec(
        name="candle",
        description="four tea-light candles in round aluminium cups, seen from above",
        defect_types=(
            "damaged corner of packaging (the aluminium cup)",
            "chunk of wax missing",
            "extra wax in candle",
            "wax melded out of the candle",
            "different colour spot",
            "foreign particles on candle",
            "weird candle wick",
        ),
    ),
    "capsules": PartSpec(
        name="capsules",
        description="about twenty green translucent soft-gel capsules lying on a grey surface",
        defect_types=("bubble", "discoloration", "scratch", "leak", "misshapen capsule"),
    ),
}

PROMPT_VERSIONS: dict[str, dict[str, PartSpec]] = {"v1": PARTS, "v2": PARTS_V2}


def get_part(name: str, version: str = "v1") -> PartSpec:
    catalogue = PROMPT_VERSIONS.get(version)
    if catalogue is None:
        raise KeyError(f"unknown prompt version '{version}' (known: {', '.join(PROMPT_VERSIONS)})")
    try:
        return catalogue[name]
    except KeyError:
        known = ", ".join(sorted(catalogue))
        raise KeyError(f"unknown part '{name}' (known: {known})") from None
