"""Pacote de coleta de dados (Web Scraping) da Amostra de Controle.

Módulos:
    config              parâmetros globais, credenciais e logging;
    models              estruturas de dados trocadas entre os módulos;
    downloader          download, validação, deduplicação e manifesto;
    reddit_scraper      coleta via API oficial do Reddit (praw);
    deviantart_scraper  coleta via feed RSS público do DeviantArt;
    web_scraper         scraping HTML genérico (padrão: Unsplash);
    main_scraper        orquestração do pipeline.
"""

from .models import DownloadResult, ImageCandidate

__all__ = ["DownloadResult", "ImageCandidate"]
__author__ = "Pedro Garcia Ribas"
__version__ = "1.0.0"
