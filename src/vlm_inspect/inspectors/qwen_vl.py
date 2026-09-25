"""Open-weight vision-language model inspector (Qwen3-VL): zero-shot and one-shot.

Two model calls per image:
1. **Classification** - one forward pass, no generation. The prompt ends with "Answer Yes or
   No" and the defect score is the probability the model assigns to "Yes" versus "No" as its next
   token. That gives a continuous score (needed for AUROC), costs no decoding, and is far more
   stable than parsing a free-text verdict.
2. **Localisation** - only for images classified as defective: the model generates JSON boxes
   (Qwen3-VL grounding format, coordinates normalised to 0-1000).

One-shot mode prepends a defect-free reference image of the same part, turning "is this
defective?" into "what differs from a good part?" - the cheapest adaptation a VLM allows.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from PIL import Image

from vlm_inspect.inspectors.vlm_output import parse_findings, yes_probability
from vlm_inspect.parts import PartSpec, get_part
from vlm_inspect.types import InspectionResult

SYSTEM_PROMPT = "You are a meticulous industrial visual inspection system."
# Decoding dominates CPU cost (~0.4 s per token for a 2B model on 4 cores), and small VLMs tend
# to enumerate near-duplicate boxes until the token limit, so the list is capped explicitly.
MAX_DEFECTS = 3


def classification_prompt(part: PartSpec, one_shot: bool) -> str:
    if one_shot:
        return (
            f"The first image shows a defect-free example of {part.description}. "
            f"The second image shows another one that must be inspected. "
            f"Possible defects include: {part.defect_list()}. "
            "Compared with the defect-free example, does the second image show any defect? "
            "Answer Yes or No."
        )
    return (
        f"This image shows {part.description}. Possible defects include: {part.defect_list()}. "
        "Inspect it carefully. Does it have any defect? Answer Yes or No."
    )


def localization_prompt(part: PartSpec) -> str:
    return (
        f"This image shows {part.description}. Locate the defects, at most {MAX_DEFECTS} "
        f"(possible types: {part.defect_list()}). Output a JSON list in which each item has "
        '"bbox_2d": [x1, y1, x2, y2] and "label" (the defect type). '
        "If there is no defect, output []."
    )


def fit_longest_side(image: Image.Image, max_side: int) -> Image.Image:
    """Downscales so the longest side is at most `max_side`; visual tokens scale with area."""
    w, h = image.size
    scale = max_side / max(w, h)
    if scale >= 1.0:
        return image
    return image.resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BICUBIC
    )


class QwenVLInspector:
    def __init__(
        self,
        model_id: str,
        *,
        max_side: int = 768,
        threshold: float = 0.5,
        max_new_tokens: int = 160,
        references: dict[str, Path] | None = None,
        dtype: str = "float32",
        prompt_version: str = "v1",
        localize: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=getattr(torch, dtype)
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.max_side = max_side
        self.threshold = threshold
        self.max_new_tokens = max_new_tokens
        self.localize = localize
        self.prompt_version = prompt_version
        self.references = {
            part: fit_longest_side(Image.open(path).convert("RGB"), max_side)
            for part, path in (references or {}).items()
        }
        self.name = "qwen-oneshot" if self.references else "qwen-zero"
        tokenizer = self.processor.tokenizer
        self._yes_ids = self._token_ids(tokenizer, ("Yes", "yes"))
        self._no_ids = self._token_ids(tokenizer, ("No", "no"))

    @staticmethod
    def _token_ids(tokenizer: Any, words: tuple[str, ...]) -> list[int]:
        ids = {tokenizer(word, add_special_tokens=False).input_ids[0] for word in words}
        return sorted(ids)

    def _inputs(self, images: list[Image.Image], prompt: str) -> Any:
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    def defect_probability(self, image: Image.Image, part: PartSpec) -> float:
        reference = self.references.get(part.name)
        images = [reference, image] if reference is not None else [image]
        inputs = self._inputs(images, classification_prompt(part, one_shot=reference is not None))
        with self._torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        # Sum probability mass over "Yes"/"yes" and "No"/"no" (log-sum-exp of the logits).
        logit_yes = float(self._torch.logsumexp(logits[self._yes_ids], dim=0))
        logit_no = float(self._torch.logsumexp(logits[self._no_ids], dim=0))
        return yes_probability(logit_yes, logit_no)

    def locate(
        self, image: Image.Image, part: PartSpec, width: int, height: int, score: float
    ) -> tuple[list[Any], str]:
        inputs = self._inputs([image], localization_prompt(part))
        with self._torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        text = str(self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0])
        # Coordinates are normalised to the (downscaled) input, which keeps the original aspect
        # ratio - so they map straight onto the original image.
        return parse_findings(text, width, height, score), text

    def generate_text(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Text-only generation with the same weights (used to write grounded reports)."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        return str(self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0])

    def inspect(self, image: Image.Image, part: str) -> InspectionResult:
        start = time.perf_counter()
        spec = get_part(part, self.prompt_version)
        original = image.convert("RGB")
        small = fit_longest_side(original, self.max_side)
        score = self.defect_probability(small, spec)
        if math.isnan(score):
            score = 0.0
        is_defective = score >= self.threshold
        findings: list[Any] = []
        raw = None
        if is_defective and self.localize:
            findings, raw = self.locate(small, spec, original.width, original.height, score)
        return InspectionResult(
            inspector=self.name,
            part=part,
            is_defective=is_defective,
            score=score,
            findings=findings,
            latency_ms=(time.perf_counter() - start) * 1000,
            raw_output=raw,
        )
