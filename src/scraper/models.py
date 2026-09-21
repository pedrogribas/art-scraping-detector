"""Estruturas de dados compartilhadas entre os módulos de coleta.

Todo scraper produz ``ImageCandidate``; o downloader consome esses candidatos e
devolve ``DownloadResult``. Esse contrato único permite adicionar novas fontes
sem alterar o orquestrador.

Autor: Pedro Garcia Ribas - TCC Sistemas de Informação (IFMG Sabará)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ImageCandidate:
    """Imagem identificada por um scraper, ainda não baixada.

    Attributes:
        url: URL direta do arquivo de imagem.
        source: identificador curto da fonte (``reddit``, ``deviantart``...).
        title: título da obra, quando disponível.
        author: autor/creditado da obra, quando disponível.
        page_url: página de origem, usada como ``Referer`` no download.
        external_id: identificador na plataforma de origem (deduplicação).
        extra: metadados adicionais específicos da fonte.
    """

    url: str
    source: str
    title: str = ""
    author: str = ""
    page_url: str = ""
    external_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, Any]:
        """Serializa o candidato para registro no manifesto."""
        return {
            "url": self.url,
            "source": self.source,
            "title": self.title,
            "author": self.author,
            "page_url": self.page_url,
            "external_id": self.external_id,
            **self.extra,
        }


@dataclass(slots=True)
class DownloadResult:
    """Resultado da tentativa de download de um candidato."""

    candidate: ImageCandidate
    success: bool
    filename: str = ""
    sha256: str = ""
    size_bytes: int = 0
    reason: str = ""

    @classmethod
    def ok(
        cls,
        candidate: ImageCandidate,
        filename: str,
        sha256: str,
        size_bytes: int,
    ) -> "DownloadResult":
        """Constrói um resultado de sucesso."""
        return cls(
            candidate=candidate,
            success=True,
            filename=filename,
            sha256=sha256,
            size_bytes=size_bytes,
        )

    @classmethod
    def fail(cls, candidate: ImageCandidate, reason: str) -> "DownloadResult":
        """Constrói um resultado de falha com o motivo registrado."""
        return cls(candidate=candidate, success=False, reason=reason)
