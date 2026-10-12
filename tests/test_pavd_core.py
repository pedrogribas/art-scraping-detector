from pathlib import Path

import numpy as np
from PIL import Image

from pavd.config import DegradationConfig, OpenSetConfig
from pavd.core.degradation import DegradationPipeline
from pavd.core.metrics import ann_recall_at_k, mrr, open_set_metrics, recall_at_k, top1
from pavd.core.open_set import is_present
from pavd.core.remote_gallery import remote_items
from pavd.utils.hashing import sha256_file
from pavd.utils.seeds import seed_from_gallery_id


def test_seed_is_stable_and_uses_gallery_id() -> None:
    assert seed_from_gallery_id("abc") == seed_from_gallery_id("abc")
    assert seed_from_gallery_id("abc") != seed_from_gallery_id("def")


def test_degradation_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output_a = tmp_path / "a.jpg"
    output_b = tmp_path / "b.jpg"
    Image.fromarray(np.full((40, 50, 3), 128, dtype=np.uint8)).save(source)
    pipeline = DegradationPipeline(DegradationConfig())
    pipeline.generate(source, output_a, gallery_id="gallery-1")
    pipeline.generate(source, output_b, gallery_id="gallery-1")
    assert sha256_file(output_a) == sha256_file(output_b)


def test_metrics_and_open_set_rule() -> None:
    assert top1([1, 2, None]) == 0.5
    assert recall_at_k([1, 2, 7], 5) == 2 / 3
    assert mrr([1, 2]) == 0.75
    assert ann_recall_at_k(["a", "b"], [["a"], ["c", "b"]], 1) == 0.5
    config = OpenSetConfig()
    assert is_present([0.83, 0.2], config)
    assert is_present([0.76, 0.65], config)
    assert not is_present([0.76, 0.72], config)
    metrics = open_set_metrics([True, False], [True, True])
    assert metrics.tpr == 1.0
    assert metrics.far == 1.0


def test_remote_catalog_is_metadata_only() -> None:
    items = remote_items([
        {"id": "a", "url": "https://example.test/a.jpg", "caption": "a"},
        {"id": "b", "url": "https://example.test/b.jpg", "aesthetic_score": 7.5},
    ], limit=2)
    assert len(items) == 2
    assert all(item.metadata_only for item in items)
    assert all(item.download_status == "metadata_only" for item in items)
    assert all(item.local_path == "" for item in items)
