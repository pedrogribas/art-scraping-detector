"""Pacote de coleta de dados (Web Scraping) da Amostra de Controle.

Módulos:
    config              parâmetros globais, cotas, credenciais e logging;
    models              estruturas de dados trocadas entre os módulos;
    downloader          download, validação, deduplicação e manifesto;
    museum_api_scraper  APIs do Met e do Art Institute of Chicago;
    booru_scraper       API pública do Safebooru;
    pexels_scraper      API oficial do Pexels (requer chave no .env);
    rss_scraper         feeds RSS do DeviantArt e do Flickr;
    web_scraper         scraping HTML genérico (Openverse/Unsplash), de reserva;
    main_scraper        orquestração balanceada do pipeline.
"""

from .models import DownloadResult, ImageCandidate

__all__ = ["DownloadResult", "ImageCandidate"]
__author__ = "Pedro Garcia Ribas"
__version__ = "2.0.0"
