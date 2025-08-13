"""
Intelligent Web Scraper - Nuclear-Powered AI Web Scraping Platform

A reactor-ready, nuclear-grade web scraping solution built on the Atomic Agents 
framework. This platform provides nuclear-powered AI web scraping orchestration 
with fusion-level capabilities including:

- Intelligent AI orchestration with natural language processing
- Fusion-level scraping strategy planning and optimization
- Context-aware processing with dynamic context injection
- Reactor-grade thread management and concurrency control
- Real-time monitoring, analytics, and performance tracking
- Reactor-ready error handling, recovery, and resilience
- Batch processing with multi-URL orchestration capabilities
- Comprehensive export options and data validation

The system integrates the atomic_scraper_tool into a comprehensive nuclear platform,
providing scalable, reliable, and intelligent web scraping capabilities for production use.
"""

__version__ = "0.1.0"
__author__ = "Atomic Agents Team"

from .agents.orchestrator import IntelligentScrapingOrchestrator
from .config import IntelligentScrapingConfig
from .context_providers import (
    WebsiteAnalysisContextProvider,
    ScrapingResultsContextProvider,
    ConfigurationContextProvider,
)
from .ecosystem import (
    TOOL_METADATA,
    get_tool_info,
    get_agent_factory,
    validate_ecosystem_compatibility
)

__all__ = [
    "IntelligentScrapingOrchestrator",
    "IntelligentScrapingConfig", 
    "WebsiteAnalysisContextProvider",
    "ScrapingResultsContextProvider",
    "ConfigurationContextProvider",
    "TOOL_METADATA",
    "get_tool_info",
    "get_agent_factory", 
    "validate_ecosystem_compatibility"
]