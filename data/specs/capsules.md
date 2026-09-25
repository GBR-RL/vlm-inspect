# Inspection specification: green soft-gel capsules, lot sample

Part code: `capsules`. Visual inspection from above of a lot sample of about twenty green
translucent soft-gel capsules. Written for this repository as an example specification; the
defect classes follow the VisA annotations for this part.

Verdicts: **REJECT** (reject the lot sample), **REVIEW** (hold for human re-inspection),
**ACCEPT**.

## CAP-GEN-01 Scope and verdict rules
Applies to every capsule in the image: shell surface, fill and shape. A single defective capsule
decides the verdict of the sample. If several findings apply, the most severe verdict wins.
Findings that cannot be assigned to any clause are REVIEW.

## CAP-LEK-01 Leaking capsule
Fill leaking through the shell, a wet or oily trace next to a capsule, or a ruptured shell is a
critical defect. Verdict: REJECT.

## CAP-SHP-01 Misshapen capsule
A capsule must be a smooth symmetric oval. A squeezed, flattened, dented, bent or otherwise
deformed capsule is a major defect. Verdict: REJECT.

## CAP-BUB-01 Air bubble
A visible air bubble or void inside the fill or in the shell is a minor defect. Verdict: REVIEW;
REJECT if the bubble is larger than about a quarter of the capsule width.

## CAP-COL-01 Discolouration
The shell must be uniformly green and translucent. A dark, pale, brown or cloudy discoloured area
is a major defect. Verdict: REJECT.

## CAP-SRF-01 Scratch on the shell
A scratch, scuff or abrasion mark on the shell surface is a minor defect. Verdict: REVIEW.
