"""
Tools module for the Intelligent Web Scraper.

This module provides the integrated AtomicScraperTool and supporting
factory classes for proper configuration management and dependency injection.
"""

from .atomic_scraper_tool import (
    AtomicScraperTool,
    AtomicScraperToolConfig,
    AtomicScraperInputSchema,
    AtomicScraperOutputSchema,
    ScrapingError,
    NetworkError,
    QualityError
)

from .tool_factory import (
    AtomicScraperToolFactory,
    ConfigurationManager,
    ToolConfigurationError
)

__all__ = [
    'AtomicScraperTool',
    'AtomicScraperToolConfig',
    'AtomicScraperInputSchema',
    'AtomicScraperOutputSchema',
    'ScrapingError',
    'NetworkError',
    'QualityError',
    'AtomicScraperToolFactory',
    'ConfigurationManager',
    'ToolConfigurationError'
]