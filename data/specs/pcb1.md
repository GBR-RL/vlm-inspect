# Inspection specification: HC-SR04 ultrasonic sensor module

Part code: `pcb1`. Final visual inspection from above, before packing. Written for this repository
as an example specification; the defect classes follow the VisA annotations for this part.

Verdicts: **REJECT** (scrap or rework), **REVIEW** (hold for human re-inspection), **ACCEPT**.

## PCB1-GEN-01 Scope and verdict rules
Applies to the assembled module seen from the component side: blue printed circuit board, two
ultrasonic transducers with metal mesh, crystal oscillator, four-pin header (VCC, Trig, Echo,
Gnd). If several findings apply, the most severe verdict wins. Findings that cannot be assigned to
any clause are REVIEW.

## PCB1-CMP-01 Missing component
Every component of the reference layout must be present: both transducers, the crystal oscillator,
all surface-mount parts and all four header pins. An empty footprint, a missing pin or a missing
transducer is a critical defect. Verdict: REJECT.

## PCB1-CMP-02 Bent component or pin
Header pins must be straight and parallel; transducers must sit perpendicular to the board. A pin
bent sideways or a tilted, pushed or deformed component is a major defect. Verdict: REJECT.

## PCB1-THM-01 Melted or heat-damaged component
Plastic, the transducer housing or the header must show no melting, bubbling, warping, charring or
heat discolouration, typically caused by soldering or overheating. Any melted area is a major
defect. Verdict: REJECT.

## PCB1-SRF-01 Scratch through the solder mask
A scratch or gouge that exposes copper, cuts a trace or damages silkscreen markings is a major
defect. Verdict: REJECT.

## PCB1-SRF-02 Superficial scratch
A light surface scratch on the solder mask that does not expose copper and does not cross a trace
is a minor defect. Verdict: REVIEW.

## PCB1-SRF-03 Contamination on the board
Solder balls, flux residue, dust or foreign particles on the board surface. Verdict: REVIEW, or
REJECT if a particle bridges two pads.
