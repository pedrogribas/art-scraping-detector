from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pavd.core.categories import VisualCategoryClassifier


def main() -> int:
    parser = argparse.ArgumentParser(description="Classifica a galeria oficial nas categorias do protocolo")
    parser.add_argument("--source", default="data/raw")
    parser.add_argument("--output", default="data/gallery/official_categories.parquet")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    classifier = VisualCategoryClassifier()
    rows = []
    for path in sorted(Path(args.source).iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        if args.limit is not None and len(rows) >= args.limit:
            break
        prediction, attributes = classifier.classify_with_attributes(path)
        rows.append({
            "gallery_id": path.stem,
            "local_path": str(path),
            "visual_category": prediction.category,
            "visual_category_confidence": prediction.confidence,
            "category_source": prediction.source,
            "manual_review_status": "pending",
            "color_mode": attributes.color_mode,
            "aspect_ratio_class": attributes.aspect_ratio_class,
            "texture_level": attributes.texture_level,
            "edge_density": attributes.edge_density,
            "saturation_mean": attributes.saturation_mean,
            "has_text": attributes.has_text,
            "has_face": attributes.has_face,
            "background_type": attributes.background_type,
            "attribute_source": attributes.source,
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
