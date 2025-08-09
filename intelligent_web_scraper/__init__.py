"""
Intelligent Web Scraper - An Advanced Atomic Agents Example

This package demonstrates sophisticated AI-powered web scraping orchestration
using the Atomic Agents framework. It showcases advanced patterns including:

- Intelligent agent orchestration
- AI-powered scraping strategy planning  
- Context providers and dynamic context injection
- Thread management and concurrency control
- Real-time monitoring and instance management
- Production-ready error handling and recovery

The system integrates the atomic_scraper_tool into the atomic-agents ecosystem,
creating a comprehensive example of how to build complex, multi-agent workflows.
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