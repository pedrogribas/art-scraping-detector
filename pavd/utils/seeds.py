from __future__ import annotations

import hashlib


def seed_from_gallery_id(gallery_id: str) -> int:
    """Produz uma semente estável de 32 bits derivada do identificador."""
    digest = hashlib.sha256(gallery_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)
