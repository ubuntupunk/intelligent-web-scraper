"""
Basic setup tests to verify the project structure is working correctly.
"""

import pytest
from intelligent_web_scraper import IntelligentScrapingConfig, IntelligentScrapingOrchestrator
from intelligent_web_scraper.context_providers import (
    WebsiteAnalysisContextProvider,
    ScrapingResultsContextProvider,
    ConfigurationContextProvider,
)


def test_config_creation():
    """Test that configuration can be created successfully."""
    config = IntelligentScrapingConfig()
    assert config.orchestrator_model == "gpt-4o-mini"
    assert config.default_quality_threshold == 50.0
    assert config.enable_monitoring is True


def test_orchestrator_creation():
    """Test that orchestrator can be created successfully."""
    config = IntelligentScrapingConfig()
    orchestrator = IntelligentScrapingOrchestrator(config=config)
    assert orchestrator.config == config
    assert orchestrator.is_running is False


def test_context_providers_creation():
    """Test that context providers can be created successfully."""
    config = IntelligentScrapingConfig()
    
    # Test website analysis context provider
    website_context = WebsiteAnalysisContextProvider()
    assert website_context.analysis_results is None
    
    # Test scraping results context provider
    results_context = ScrapingResultsContextProvider()
    assert len(results_context.results) == 0
    
    # Test configuration context provider
    config_context = ConfigurationContextProvider(config)
    assert config_context.config == config


def test_config_from_env():
    """Test configuration creation from environment variables."""
    import os
    
    # Set some test environment variables
    os.environ["QUALITY_THRESHOLD"] = "75.0"
    os.environ["MAX_CONCURRENT_REQUESTS"] = "8"
    
    try:
        config = IntelligentScrapingConfig.from_env()
        assert config.default_quality_threshold == 75.0
        assert config.max_concurrent_requests == 8
    finally:
        # Clean up environment variables
        os.environ.pop("QUALITY_THRESHOLD", None)
        os.environ.pop("MAX_CONCURRENT_REQUESTS", None)


@pytest.mark.asyncio
async def test_orchestrator_basic_run():
    """Test that orchestrator can handle a basic run request."""
    config = IntelligentScrapingConfig()
    orchestrator = IntelligentScrapingOrchestrator(config=config)
    
    # Test input data
    input_data = {
        "scraping_request": "Test scraping request",
        "target_url": "https://example.com",
        "max_results": 5,
        "quality_threshold": 60.0,
        "export_format": "json"
    }
    
    # Run the orchestrator (should return mock data for now)
    result = await orchestrator.run(input_data)
    
    # Verify the result structure
    assert result.scraping_plan is not None
    assert isinstance(result.extracted_data, list)
    assert result.metadata is not None
    assert result.quality_score >= 0.0
    assert result.reasoning is not None
    assert isinstance(result.export_options, dict)
    assert result.monitoring_report is not None
    assert isinstance(result.instance_statistics, list)