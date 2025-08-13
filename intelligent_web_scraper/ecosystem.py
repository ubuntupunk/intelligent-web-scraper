"""
Atomic Agents ecosystem integration metadata.

This module provides metadata and integration points for the atomic-agents
ecosystem, enabling tool discovery and proper integration.
"""

from typing import Dict, Any, List
from dataclasses import dataclass

from .agents.orchestrator import IntelligentScrapingOrchestrator
from .config import IntelligentScrapingConfig


@dataclass
class AtomicAgentsToolMetadata:
    """Metadata for atomic-agents tool discovery."""
    
    name: str
    description: str
    version: str
    author: str
    category: str
    tags: List[str]
    agent_class: type
    config_class: type
    example_usage: Dict[str, Any]
    requirements: List[str]


# Tool metadata for ecosystem discovery
TOOL_METADATA = AtomicAgentsToolMetadata(
    name="intelligent-web-scraper",
    description="An advanced example application demonstrating intelligent web scraping orchestration with AI-powered strategy planning",
    version="0.1.0",
    author="Atomic Agents Team",
    category="web-scraping",
    tags=[
        "web-scraping",
        "ai-orchestration", 
        "data-extraction",
        "atomic-agents-example",
        "reactor-ready",
        "monitoring",
        "concurrent-processing"
    ],
    agent_class=IntelligentScrapingOrchestrator,
    config_class=IntelligentScrapingConfig,
    example_usage={
        "basic_scraping": {
            "scraping_request": "Extract all product names and prices from this e-commerce page",
            "target_url": "https://example-store.com/products",
            "max_results": 20,
            "quality_threshold": 70.0,
            "export_format": "json"
        },
        "news_extraction": {
            "scraping_request": "Get all article titles, authors, and publication dates",
            "target_url": "https://example-news.com",
            "max_results": 50,
            "quality_threshold": 80.0,
            "export_format": "csv"
        },
        "directory_scraping": {
            "scraping_request": "Scrape contact information from company directory",
            "target_url": "https://example-directory.com/companies",
            "max_results": 100,
            "quality_threshold": 60.0,
            "export_format": "excel"
        }
    },
    requirements=[
        "atomic-agents>=0.1.0",
        "atomic-scraper-tool>=0.1.0",
        "requests>=2.32.0",
        "beautifulsoup4>=4.12.0",
        "rich>=13.7.0",
        "aiohttp>=3.9.0",
        "pydantic>=2.0.0"
    ]
)


def get_tool_info() -> Dict[str, Any]:
    """Get tool information for atomic-agents ecosystem discovery."""
    return {
        "name": TOOL_METADATA.name,
        "description": TOOL_METADATA.description,
        "version": TOOL_METADATA.version,
        "author": TOOL_METADATA.author,
        "category": TOOL_METADATA.category,
        "tags": TOOL_METADATA.tags,
        "agent_class": f"{TOOL_METADATA.agent_class.__module__}.{TOOL_METADATA.agent_class.__name__}",
        "config_class": f"{TOOL_METADATA.config_class.__module__}.{TOOL_METADATA.config_class.__name__}",
        "example_usage": TOOL_METADATA.example_usage,
        "requirements": TOOL_METADATA.requirements,
        "cli_commands": [
            "intelligent-web-scraper",
            "iws"
        ],
        "entry_points": {
            "console_scripts": [
                "intelligent-web-scraper = intelligent_web_scraper.cli:main",
                "iws = intelligent_web_scraper.cli:main"
            ]
        }
    }


def get_agent_factory():
    """Get agent factory function for dynamic instantiation."""
    def create_agent(config: Dict[str, Any] = None) -> IntelligentScrapingOrchestrator:
        """Create an IntelligentScrapingOrchestrator instance."""
        if config:
            agent_config = IntelligentScrapingConfig(**config)
        else:
            agent_config = IntelligentScrapingConfig.from_env()
        
        return IntelligentScrapingOrchestrator(config=agent_config)
    
    return create_agent


def validate_ecosystem_compatibility() -> Dict[str, bool]:
    """Validate compatibility with atomic-agents ecosystem."""
    compatibility_checks = {}
    
    try:
        # Check atomic-agents import
        import atomic_agents
        compatibility_checks["atomic_agents_import"] = True
    except ImportError:
        compatibility_checks["atomic_agents_import"] = False
    
    try:
        # Check BaseAgent inheritance
        from atomic_agents.agents.base_agent import BaseAgent
        compatibility_checks["base_agent_inheritance"] = issubclass(
            IntelligentScrapingOrchestrator, 
            BaseAgent
        )
    except ImportError:
        compatibility_checks["base_agent_inheritance"] = False
    
    try:
        # Check schema compatibility
        from atomic_agents.lib.base.base_io_schema import BaseIOSchema
        config = IntelligentScrapingConfig()
        orchestrator = IntelligentScrapingOrchestrator(config=config)
        
        compatibility_checks["input_schema_valid"] = issubclass(
            orchestrator.input_schema, 
            BaseIOSchema
        )
        compatibility_checks["output_schema_valid"] = issubclass(
            orchestrator.output_schema, 
            BaseIOSchema
        )
    except (ImportError, Exception):
        compatibility_checks["input_schema_valid"] = False
        compatibility_checks["output_schema_valid"] = False
    
    try:
        # Check context provider compatibility
        from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase
        from .context_providers import WebsiteAnalysisContextProvider
        
        compatibility_checks["context_provider_valid"] = issubclass(
            WebsiteAnalysisContextProvider,
            SystemPromptContextProviderBase
        )
    except (ImportError, Exception):
        compatibility_checks["context_provider_valid"] = False
    
    try:
        # Check CLI integration
        from .cli import main as cli_main
        compatibility_checks["cli_integration"] = callable(cli_main)
    except ImportError:
        compatibility_checks["cli_integration"] = False
    
    return compatibility_checks


# Export for ecosystem discovery
__all__ = [
    "TOOL_METADATA",
    "get_tool_info", 
    "get_agent_factory",
    "validate_ecosystem_compatibility"
]