"""Academic crawlers."""

from .arxiv import ArxivCrawlerIntegrated
from .standalone import StandaloneMissingAspectCrawler

__all__ = ["ArxivCrawlerIntegrated", "StandaloneMissingAspectCrawler"]
