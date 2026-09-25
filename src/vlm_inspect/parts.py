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


def get_part(name: str) -> PartSpec:
    try:
        return PARTS[name]
    except KeyError:
        known = ", ".join(sorted(PARTS))
        raise KeyError(f"unknown part '{name}' (known: {known})") from None
