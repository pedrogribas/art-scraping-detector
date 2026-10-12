from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pavd.core.preprocessing import load_rgb


CATEGORIES: tuple[str, ...] = (
    "photograph",
    "digital painting",
    "vector art",
    "logo",
    "collage",
    "ambiguous or mixed",
)


@dataclass(frozen=True, slots=True)
class CategoryPrediction:
    category: str
    confidence: float
    source: str = "clip-zero-shot"


@dataclass(frozen=True, slots=True)
class ImageAttributes:
    color_mode: str
    aspect_ratio_class: str
    texture_level: str
    edge_density: float
    saturation_mean: float
    has_text: str
    has_face: str
    background_type: str
    source: str = "clip-zero-shot-plus-image-statistics"


class VisualCategoryClassifier:
    """Classificação operacional; não representa uma ontologia definitiva."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "auto") -> None:
        from transformers import CLIPModel, CLIPProcessor
        import torch

        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

    def classify(self, path: Path) -> CategoryPrediction:
        prediction, _ = self.classify_with_attributes(path)
        return prediction

    def classify_with_attributes(self, path: Path) -> tuple[CategoryPrediction, ImageAttributes]:
        import torch

        prompts = [f"a visual image that is {category}" for category in CATEGORIES]
        image = load_rgb(path)
        inputs = self.processor(text=prompts, images=image, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            probabilities = self.model(**inputs).logits_per_image.softmax(dim=-1)[0]
        winner = int(probabilities.argmax().item())
        category = CategoryPrediction(CATEGORIES[winner], float(probabilities[winner].item()))
        attributes = self._attributes(image, path)
        return category, attributes

    def _attributes(self, image: Any, path: Path) -> ImageAttributes:
        import torch

        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        gray = pixels.mean(axis=2)
        saturation = pixels.max(axis=2) - pixels.min(axis=2)
        edge_x = np.abs(np.diff(gray, axis=1))
        edge_y = np.abs(np.diff(gray, axis=0))
        edge_density = float(
            0.5 * (
                (np.mean(edge_x > 0.12) if edge_x.size else 0.0)
                + (np.mean(edge_y > 0.12) if edge_y.size else 0.0)
            )
        )
        saturation_mean = float(saturation.mean())
        if saturation_mean < 0.06 and float(gray.std()) > 0.08:
            color_mode = "black-and-white"
        elif saturation_mean < 0.12:
            color_mode = "muted-or-grayscale"
        else:
            color_mode = "color"
        width, height = image.size
        ratio = width / max(height, 1)
        aspect = "square" if 0.9 <= ratio <= 1.1 else ("portrait" if ratio < 0.9 else "landscape")
        texture = "low" if edge_density < 0.04 else ("medium" if edge_density < 0.12 else "high")

        def zero_shot(options: list[str], template: str) -> str:
            prompts = [template.format(option=option) for option in options]
            inputs = self.processor(text=prompts, images=image, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                scores = self.model(**inputs).logits_per_image[0]
            return options[int(scores.argmax().item())]

        return ImageAttributes(
            color_mode=color_mode,
            aspect_ratio_class=aspect,
            texture_level=texture,
            edge_density=edge_density,
            saturation_mean=saturation_mean,
            has_text=zero_shot(["no visible text", "visible text"], "an image with {option}"),
            has_face=zero_shot(["no visible face", "a visible face"], "an image with {option}"),
            background_type=zero_shot(["a uniform background", "a complex background"], "an image with {option}"),
        )
