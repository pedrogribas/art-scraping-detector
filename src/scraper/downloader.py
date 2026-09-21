"""Download, validação e persistência das imagens coletadas.

Responsabilidades:
    * baixar o binário com retentativas e backoff exponencial;
    * validar tipo real do arquivo (magic bytes) e faixa de tamanho;
    * deduplicar por URL e por hash SHA-256 do conteúdo;
    * nomear os arquivos de forma padronizada (``arte_<fonte>_<seq>.<ext>``);
    * manter um manifesto JSON que permite retomar a coleta de onde parou.

Autor: Pedro Garcia Ribas - TCC Sistemas de Informação (IFMG Sabará)
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import requests

from . import config
from .models import DownloadResult, ImageCandidate

logger = config.get_logger("scraper.downloader")

# Assinaturas binárias aceitas (mais confiáveis que a extensão da URL).
_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
)

_FILENAME_RE = re.compile(r"^arte_[a-z0-9]+_(\d+)\.[a-z]+$", re.IGNORECASE)


def detect_extension(conteudo: bytes) -> str | None:
    """Identifica a extensão real da imagem pelos primeiros bytes.

    Returns:
        A extensão (``.jpg``, ``.png``, ``.webp``) ou ``None`` se o conteúdo não
        for uma imagem suportada.
    """
    for assinatura, extensao in _MAGIC_SIGNATURES:
        if conteudo.startswith(assinatura):
            return extensao
    if conteudo[:4] == b"RIFF" and conteudo[8:12] == b"WEBP":
        return ".webp"
    return None


class Manifest:
    """Registro persistente da coleta (estado + metadados das imagens)."""

    def __init__(self, caminho: Path | None = None) -> None:
        self.caminho = caminho or config.MANIFEST_PATH
        self.images: dict[str, dict[str, Any]] = {}
        self.seen_urls: set[str] = set()
        self.next_index: int = 1
        self._load()

    def _load(self) -> None:
        """Carrega o manifesto do disco, tolerando arquivo ausente ou corrompido."""
        if not self.caminho.exists():
            return
        try:
            dados = json.loads(self.caminho.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as erro:
            logger.warning("Manifesto ilegível (%s). Iniciando um novo.", erro)
            return

        self.images = dados.get("images", {})
        self.seen_urls = set(dados.get("seen_urls", []))
        self.next_index = int(dados.get("next_index", len(self.images) + 1))
        logger.info(
            "Manifesto carregado: %d imagem(ns) já registrada(s).", len(self.images)
        )

    def sync_with_disk(self, diretorio: Path) -> None:
        """Alinha o contador sequencial com os arquivos presentes em disco.

        Protege contra manifesto apagado manualmente ou arquivos copiados de
        outra execução, evitando sobrescrever imagens já coletadas.
        """
        maior = 0
        for arquivo in diretorio.glob("arte_*"):
            achado = _FILENAME_RE.match(arquivo.name)
            if achado:
                maior = max(maior, int(achado.group(1)))
        if maior >= self.next_index:
            logger.info(
                "Contador ajustado pelo disco: próximo índice = %d.", maior + 1
            )
            self.next_index = maior + 1

    @property
    def total_saved(self) -> int:
        """Quantidade de imagens únicas já salvas."""
        return len(self.images)

    def counts_by_source(self) -> dict[str, int]:
        """Total de imagens salvas por fonte."""
        contagem: dict[str, int] = {}
        for registro in self.images.values():
            fonte = registro.get("source", "desconhecida")
            contagem[fonte] = contagem.get(fonte, 0) + 1
        return contagem

    def is_known_url(self, url: str) -> bool:
        """Indica se a URL já foi processada (com sucesso ou não)."""
        return url in self.seen_urls

    def is_known_hash(self, sha256: str) -> bool:
        """Indica se o conteúdo binário já foi salvo anteriormente."""
        return sha256 in self.images

    def mark_url(self, url: str) -> None:
        """Marca a URL como já visitada."""
        self.seen_urls.add(url)

    def register(self, resultado: DownloadResult) -> None:
        """Registra uma imagem salva com sucesso."""
        self.images[resultado.sha256] = {
            "filename": resultado.filename,
            "size_bytes": resultado.size_bytes,
            "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **resultado.candidate.as_metadata(),
        }

    def take_index(self) -> int:
        """Reserva e devolve o próximo número sequencial."""
        indice = self.next_index
        self.next_index += 1
        return indice

    def save(self) -> None:
        """Grava o manifesto em disco de forma atômica."""
        payload = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_images": len(self.images),
            "counts_by_source": self.counts_by_source(),
            "next_index": self.next_index,
            "seen_urls": sorted(self.seen_urls),
            "images": self.images,
        }
        try:
            self.caminho.parent.mkdir(parents=True, exist_ok=True)
            temporario = self.caminho.with_suffix(".tmp")
            temporario.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporario.replace(self.caminho)
        except OSError as erro:
            logger.error("Falha ao salvar o manifesto: %s", erro)


class ImageDownloader:
    """Baixa candidatos e persiste as imagens válidas em ``data/raw``."""

    def __init__(
        self,
        destino: Path | None = None,
        manifest: Manifest | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.destino = destino or config.RAW_DIR
        self.destino.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest or Manifest()
        self.manifest.sync_with_disk(self.destino)
        self.session = session or requests.Session()

    # -- API pública --------------------------------------------------------

    def download(self, candidato: ImageCandidate) -> DownloadResult:
        """Baixa, valida e salva um candidato.

        Nunca levanta exceção: qualquer falha vira um ``DownloadResult`` com o
        motivo preenchido, para que a coleta continue.
        """
        url = candidato.url
        if self.manifest.is_known_url(url):
            return DownloadResult.fail(candidato, "url_duplicada")

        self.manifest.mark_url(url)

        conteudo = self._fetch_bytes(url, referer=candidato.page_url)
        if conteudo is None:
            return DownloadResult.fail(candidato, "falha_de_rede")

        tamanho = len(conteudo)
        if tamanho < config.MIN_FILE_SIZE_BYTES:
            return DownloadResult.fail(candidato, "arquivo_muito_pequeno")
        if tamanho > config.MAX_FILE_SIZE_BYTES:
            return DownloadResult.fail(candidato, "arquivo_muito_grande")

        extensao = detect_extension(conteudo)
        if extensao is None:
            return DownloadResult.fail(candidato, "formato_nao_suportado")

        sha256 = hashlib.sha256(conteudo).hexdigest()
        if self.manifest.is_known_hash(sha256):
            return DownloadResult.fail(candidato, "conteudo_duplicado")

        indice = self.manifest.take_index()
        nome = f"{config.FILENAME_PREFIX}_{candidato.source}_{indice:04d}{extensao}"
        caminho = self.destino / nome

        try:
            caminho.write_bytes(conteudo)
        except OSError as erro:
            logger.error("Erro ao gravar %s: %s", nome, erro)
            return DownloadResult.fail(candidato, "erro_de_escrita")

        resultado = DownloadResult.ok(candidato, nome, sha256, tamanho)
        self.manifest.register(resultado)
        logger.debug("Salvo: %s (%.1f KB)", nome, tamanho / 1024)
        return resultado

    def download_many(
        self, candidatos: Iterable[ImageCandidate]
    ) -> Iterable[DownloadResult]:
        """Baixa uma sequência de candidatos, respeitando o delay entre requisições."""
        for candidato in candidatos:
            resultado = self.download(candidato)
            if resultado.success or resultado.reason == "falha_de_rede":
                # Só espera quando houve tráfego real de rede.
                config.polite_sleep(0.4, 1.2)
            yield resultado

    def close(self) -> None:
        """Encerra a sessão HTTP e persiste o manifesto."""
        self.manifest.save()
        try:
            self.session.close()
        except Exception:  # noqa: BLE001 - encerramento não pode quebrar o fluxo
            pass

    # -- Interno ------------------------------------------------------------

    def _fetch_bytes(self, url: str, referer: str = "") -> bytes | None:
        """Baixa o binário com retentativas e backoff exponencial."""
        for tentativa in range(1, config.MAX_RETRIES + 1):
            try:
                resposta = self.session.get(
                    url,
                    headers=config.get_headers(
                        referer=referer or None, accept="image/*,*/*;q=0.8"
                    ),
                    timeout=config.REQUEST_TIMEOUT,
                    stream=True,
                )

                if resposta.status_code == 429:
                    espera = float(resposta.headers.get("Retry-After", 10))
                    logger.warning(
                        "HTTP 429 (rate limit) em %s. Aguardando %.0fs.", url, espera
                    )
                    time.sleep(min(espera, 60))
                    continue

                if resposta.status_code in (403, 404, 410):
                    logger.debug("HTTP %s em %s. Descartando.", resposta.status_code, url)
                    return None

                resposta.raise_for_status()

                tipo = resposta.headers.get("Content-Type", "").split(";")[0].strip()
                if tipo and not tipo.startswith("image/"):
                    logger.debug("Content-Type inesperado (%s) em %s.", tipo, url)
                    return None

                conteudo = resposta.content
                if not conteudo:
                    return None
                return conteudo

            except requests.exceptions.Timeout:
                logger.debug("Timeout (%d/%d) em %s.", tentativa, config.MAX_RETRIES, url)
            except requests.exceptions.RequestException as erro:
                logger.debug(
                    "Erro de rede (%d/%d) em %s: %s",
                    tentativa,
                    config.MAX_RETRIES,
                    url,
                    erro,
                )
            except Exception as erro:  # noqa: BLE001 - blindagem do loop de coleta
                logger.warning("Erro inesperado ao baixar %s: %s", url, erro)
                return None

            if tentativa < config.MAX_RETRIES:
                time.sleep(config.BACKOFF_FACTOR**tentativa)

        logger.debug("Desistindo de %s após %d tentativas.", url, config.MAX_RETRIES)
        return None
