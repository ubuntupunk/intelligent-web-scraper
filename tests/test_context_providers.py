"""
Unit tests for context providers and dynamic context injection.

This module tests the context provider functionality and integration
with the orchestrator agent according to atomic-agents patterns.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from intelligent_web_scraper.context_providers import (
    WebsiteAnalysisContextProvider,
    ScrapingResultsContextProvider,
    ConfigurationContextProvider
)
from intelligent_web_scraper.context_providers.website_analysis import (
    WebsiteStructureAnalysis,
    ContentPattern,
    NavigationInfo
)
from intelligent_web_scraper.context_providers.scraping_results import (
    ScrapedItem,
    ExtractionStatistics,
    QualityMetrics
)
from intelligent_web_scraper.context_providers.configuration import (
    ConfigurationValidationResult,
    EnvironmentInfo
)
from intelligent_web_scraper.config import IntelligentScrapingConfig
from intelligent_web_scraper.agents.orchestrator import IntelligentScrapingOrchestrator


class TestWebsiteAnalysisContextProvider:
    """Test the WebsiteAnalysisContextProvider class."""
    
    @pytest.fixture
    def provider(self):
        """Create a website analysis context provider for testing."""
        return WebsiteAnalysisContextProvider()
    
    @pytest.fixture
    def sample_analysis(self):
        """Create a sample website analysis for testing."""
        analysis = WebsiteStructureAnalysis("https://example.com", "Example Website")
        analysis.quality_score = 85.0
        analysis.analysis_confidence = 90.0
        
        # Add content patterns
        pattern1 = ContentPattern("article", ".article, .post", 10, 95.0)
        pattern1.add_example("Main article content")
        analysis.add_content_pattern(pattern1)
        
        pattern2 = ContentPattern("navigation", ".nav, nav", 3, 88.0)
        pattern2.add_example("Navigation menu")
        analysis.add_content_pattern(pattern2)
        
        # Set navigation info
        analysis.navigation_info.main_menu_selectors = [".nav", "nav"]
        analysis.navigation_info.pagination_selectors = [".pagination"]
        analysis.navigation_info.has_infinite_scroll = True
        
        return analysis
    
    def test_provider_initialization(self, provider):
        """Test that the provider initializes correctly."""
        assert provider.title == "Website Analysis Context"
        assert provider.analysis_results is None
        assert len(provider.content_patterns) == 0
        assert provider.navigation_info is None
        assert len(provider.analysis_cache) == 0
        assert provider.cache_ttl_seconds == 3600
    
    def test_set_analysis_results(self, provider, sample_analysis):
        """Test setting analysis results."""
        provider.set_analysis_results(sample_analysis)
        
        assert provider.analysis_results == sample_analysis
        assert len(provider.content_patterns) == 2
        assert provider.navigation_info == sample_analysis.navigation_info
        assert sample_analysis.url in provider.analysis_cache
    
    def test_get_info_with_analysis(self, provider, sample_analysis):
        """Test getting context info with analysis data."""
        provider.set_analysis_results(sample_analysis)
        
        info = provider.get_info()
        
        assert "Website Analysis: Example Website" in info
        assert "https://example.com" in info
        assert "Quality Score:** 85.0%" in info
        assert "**Analysis Confidence:** 90.0%" in info
        assert "Identified Content Patterns" in info
        assert "Pattern 1: Article" in info
        assert "Navigation Structure" in info
        assert "**Infinite Scroll:** Detected" in info
        assert "Scraping Recommendations" in info
    
    def test_get_info_without_analysis(self, provider):
        """Test getting context info without analysis data."""
        info = provider.get_info()
        
        assert "No website analysis available" in info
        assert "Fallback Approach" in info
        assert "Recommended Selectors" in info
        assert "Best Practices" in info
    
    def test_analysis_caching(self, provider, sample_analysis):
        """Test analysis caching functionality."""
        # Set analysis
        provider.set_analysis_results(sample_analysis)
        
        # Get cached analysis
        cached = provider.get_cached_analysis("https://example.com")
        assert cached == sample_analysis
        
        # Test cache miss
        cached_miss = provider.get_cached_analysis("https://other.com")
        assert cached_miss is None
    
    def test_cache_expiration(self, provider):
        """Test cache expiration functionality."""
        # Create old analysis
        old_analysis = WebsiteStructureAnalysis("https://example.com", "Old")
        old_analysis.analyzed_at = datetime.utcnow() - timedelta(hours=2)
        
        # Manually add to cache
        provider.analysis_cache["https://example.com"] = old_analysis
        
        # Should return None due to expiration
        cached = provider.get_cached_analysis("https://example.com")
        assert cached is None
        assert "https://example.com" not in provider.analysis_cache
    
    def test_clear_cache(self, provider, sample_analysis):
        """Test clearing the analysis cache."""
        provider.set_analysis_results(sample_analysis)
        assert len(provider.analysis_cache) == 1
        
        provider.clear_cache()
        assert len(provider.analysis_cache) == 0
    
    def test_pattern_summary(self, provider, sample_analysis):
        """Test getting pattern summary."""
        provider.set_analysis_results(sample_analysis)
        
        summary = provider.get_pattern_summary()
        
        assert summary["total_patterns"] == 2
        assert summary["confidence"] == 90.0
        assert summary["quality_score"] == 85.0
        assert len(summary["patterns"]) == 2
        assert summary["patterns"][0]["type"] == "article"
    
    def test_navigation_summary(self, provider, sample_analysis):
        """Test getting navigation summary."""
        provider.set_analysis_results(sample_analysis)
        
        summary = provider.get_navigation_summary()
        
        assert summary["available"] is True
        assert summary["pagination_available"] is True
        assert summary["infinite_scroll"] is True
        assert summary["load_more"] is False
    
    def test_analysis_freshness(self, provider, sample_analysis):
        """Test analysis freshness checking."""
        provider.set_analysis_results(sample_analysis)
        
        # Should be fresh
        assert provider.is_analysis_fresh()
        assert provider.is_analysis_fresh(max_age_seconds=7200)
        
        # Test with very short max age
        assert not provider.is_analysis_fresh(max_age_seconds=0)


class TestScrapingResultsContextProvider:
    """Test the ScrapingResultsContextProvider class."""
    
    @pytest.fixture
    def provider(self):
        """Create a scraping results context provider for testing."""
        return ScrapingResultsContextProvider()
    
    @pytest.fixture
    def sample_results(self):
        """Create sample scraping results for testing."""
        return [
            {"title": "Article 1", "content": "Content 1", "url": "https://example.com/1"},
            {"title": "Article 2", "content": "Content 2", "url": "https://example.com/2"},
            {"title": "Article 3", "content": "Content 3"}  # Missing URL
        ]
    
    @pytest.fixture
    def sample_quality_scores(self):
        """Create sample quality scores for testing."""
        return [95.0, 85.0, 60.0]
    
    def test_provider_initialization(self, provider):
        """Test that the provider initializes correctly."""
        assert provider.title == "Scraping Results Context"
        assert len(provider.results) == 0
        assert isinstance(provider.extraction_statistics, ExtractionStatistics)
        assert isinstance(provider.quality_metrics, QualityMetrics)
        assert len(provider.operation_metadata) == 0
    
    def test_set_results(self, provider, sample_results, sample_quality_scores):
        """Test setting scraping results."""
        provider.set_results(sample_results, sample_quality_scores)
        
        assert len(provider.results) == 3
        assert provider.results[0].quality_score == 95.0
        assert provider.results[1].quality_score == 85.0
        assert provider.results[2].quality_score == 60.0
        
        # Check that statistics were updated
        assert provider.extraction_statistics.total_items_extracted == 3
        assert provider.quality_metrics.overall_quality_score == 80.0  # Average
    
    def test_add_result(self, provider):
        """Test adding individual results."""
        provider.add_result({"title": "Test Article"}, 90.0)
        
        assert len(provider.results) == 1
        assert provider.results[0].data["title"] == "Test Article"
        assert provider.results[0].quality_score == 90.0
    
    def test_set_operation_metadata(self, provider):
        """Test setting operation metadata."""
        metadata = {
            "operation_id": "test-123",
            "target_url": "https://example.com",
            "strategy": "list"
        }
        
        provider.set_operation_metadata(metadata)
        assert provider.operation_metadata == metadata
    
    def test_get_info_with_results(self, provider, sample_results, sample_quality_scores):
        """Test getting context info with results."""
        provider.set_results(sample_results, sample_quality_scores)
        provider.set_operation_metadata({"operation_id": "test-123"})
        
        info = provider.get_info()
        
        assert "Scraping Results Analysis" in info
        assert "**Total Items:** 3" in info
        assert "**Average Quality:** 80.0%" in info
        assert "Operation Details" in info
        assert "Extraction Statistics" in info
        assert "Quality Analysis" in info
        assert "Quality Distribution" in info
        assert "Sample Results" in info
    
    def test_get_info_without_results(self, provider):
        """Test getting context info without results."""
        info = provider.get_info()
        
        assert "No scraping results available" in info
        assert "Analysis Pending" in info
        assert "Expected Analysis" in info
    
    def test_quality_distribution_calculation(self, provider, sample_results, sample_quality_scores):
        """Test quality distribution calculation."""
        provider.set_results(sample_results, sample_quality_scores)
        
        distribution = provider.quality_metrics.quality_distribution
        assert distribution["excellent"] == 1  # 95.0%
        assert distribution["good"] == 1       # 85.0%
        assert distribution["fair"] == 1       # 60.0%
        assert distribution["poor"] == 0
    
    def test_field_completion_rates(self, provider, sample_results, sample_quality_scores):
        """Test field completion rate calculation."""
        provider.set_results(sample_results, sample_quality_scores)
        
        completion_rates = provider.extraction_statistics.field_completion_rates
        assert completion_rates["title"] == 100.0  # All items have title
        assert completion_rates["content"] == 100.0  # All items have content
        assert abs(completion_rates["url"] - 66.7) < 0.1  # 2 out of 3 items have URL (approximately)
    
    def test_get_top_quality_items(self, provider, sample_results, sample_quality_scores):
        """Test getting top quality items."""
        provider.set_results(sample_results, sample_quality_scores)
        
        top_items = provider.get_top_quality_items(2)
        
        assert len(top_items) == 2
        assert top_items[0].quality_score == 95.0
        assert top_items[1].quality_score == 85.0
    
    def test_get_failed_items(self, provider, sample_results, sample_quality_scores):
        """Test getting failed items."""
        provider.set_results(sample_results, sample_quality_scores)
        
        # Add validation error to one item
        provider.results[2].add_validation_error("Missing required field")
        
        failed_items = provider.get_failed_items()
        
        assert len(failed_items) == 1
        assert not failed_items[0].is_valid()
        assert "Missing required field" in failed_items[0].validation_errors
    
    def test_clear_results(self, provider, sample_results, sample_quality_scores):
        """Test clearing results."""
        provider.set_results(sample_results, sample_quality_scores)
        provider.set_operation_metadata({"test": "data"})
        
        provider.clear_results()
        
        assert len(provider.results) == 0
        assert len(provider.operation_metadata) == 0
        assert provider.extraction_statistics.total_items_extracted == 0


class TestConfigurationContextProvider:
    """Test the ConfigurationContextProvider class."""
    
    @pytest.fixture
    def provider(self):
        """Create a configuration context provider for testing."""
        return ConfigurationContextProvider()
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        return IntelligentScrapingConfig(
            orchestrator_model="gpt-4o-mini",
            planning_agent_model="gpt-4o-mini",
            default_quality_threshold=75.0,
            max_concurrent_requests=5,
            request_delay=1.0,
            respect_robots_txt=True,
            enable_rate_limiting=True
        )
    
    def test_provider_initialization(self, provider):
        """Test that the provider initializes correctly."""
        assert provider.title == "Configuration Context"
        assert provider.config is None
        assert provider.validation_result is None
        assert isinstance(provider.environment_info, EnvironmentInfo)
        assert len(provider.config_overrides) == 0
    
    def test_set_configuration(self, provider, sample_config):
        """Test setting configuration."""
        provider.set_configuration(sample_config)
        
        assert provider.config == sample_config
        assert provider.validation_result is not None
        assert isinstance(provider.validation_result, ConfigurationValidationResult)
    
    def test_configuration_validation(self, provider, sample_config):
        """Test configuration validation."""
        provider.set_configuration(sample_config)
        
        validation = provider.validation_result
        assert validation.is_valid is True
        assert len(validation.errors) == 0
        # May have warnings or recommendations
    
    def test_invalid_configuration_validation(self, provider):
        """Test validation of invalid configuration."""
        invalid_config = IntelligentScrapingConfig(
            orchestrator_model="",  # Invalid empty model
            default_quality_threshold=150.0,  # Invalid threshold > 100
            max_concurrent_requests=0,  # Invalid zero requests
            request_delay=-1.0  # Invalid negative delay
        )
        
        provider.set_configuration(invalid_config)
        
        validation = provider.validation_result
        assert validation.is_valid is False
        assert len(validation.errors) > 0
    
    def test_add_config_override(self, provider, sample_config):
        """Test adding configuration overrides."""
        provider.set_configuration(sample_config)
        
        provider.add_config_override("max_concurrent_requests", 10)
        provider.add_config_override("request_delay", 2.0)
        
        assert len(provider.config_overrides) == 2
        assert provider.config_overrides["max_concurrent_requests"] == 10
        assert provider.config_overrides["request_delay"] == 2.0
    
    def test_clear_overrides(self, provider, sample_config):
        """Test clearing configuration overrides."""
        provider.set_configuration(sample_config)
        provider.add_config_override("test_key", "test_value")
        
        assert len(provider.config_overrides) == 1
        
        provider.clear_overrides()
        
        assert len(provider.config_overrides) == 0
    
    def test_get_info_with_config(self, provider, sample_config):
        """Test getting context info with configuration."""
        provider.set_configuration(sample_config)
        
        info = provider.get_info()
        
        assert "System Configuration" in info
        assert "**Configuration Status:** ✅ Valid" in info
        assert "Core Settings" in info
        assert "**Orchestrator Model:** gpt-4o-mini" in info
        assert "**Default Quality Threshold:** 75.0%" in info
        assert "Output Settings" in info
        assert "Compliance Settings" in info
        assert "Monitoring Settings" in info
        assert "Concurrency Settings" in info
        assert "Usage Guidelines" in info
    
    def test_get_info_without_config(self, provider):
        """Test getting context info without configuration."""
        info = provider.get_info()
        
        assert "No configuration loaded" in info
        assert "Default Behavior" in info
        assert "Required Configuration" in info
        assert "Loading Configuration" in info
    
    def test_get_effective_config(self, provider, sample_config):
        """Test getting effective configuration with overrides."""
        provider.set_configuration(sample_config)
        provider.add_config_override("max_concurrent_requests", 15)
        
        effective_config = provider.get_effective_config()
        
        assert effective_config["max_concurrent_requests"] == 15
        assert effective_config["orchestrator_model"] == "gpt-4o-mini"
    
    def test_get_validation_summary(self, provider, sample_config):
        """Test getting validation summary."""
        provider.set_configuration(sample_config)
        
        summary = provider.get_validation_summary()
        
        assert summary["status"] == "valid"
        assert "error_count" in summary
        assert "warning_count" in summary
        assert "recommendation_count" in summary
        assert "validated_at" in summary
    
    def test_validate_current_config(self, provider, sample_config):
        """Test validating current configuration."""
        provider.set_configuration(sample_config)
        
        is_valid = provider.validate_current_config()
        
        assert is_valid is True
        
        # Test without config
        provider.config = None
        is_valid = provider.validate_current_config()
        
        assert is_valid is False


class TestOrchestratorContextIntegration:
    """Test context provider integration with the orchestrator."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return IntelligentScrapingConfig(
            orchestrator_model="gpt-4o-mini",
            planning_agent_model="gpt-4o-mini",
            default_quality_threshold=70.0
        )
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock instructor client."""
        import instructor
        mock_openai_client = Mock()
        return instructor.from_openai(mock_openai_client)
    
    def test_orchestrator_context_provider_initialization(self, mock_config, mock_client):
        """Test that orchestrator initializes context providers correctly."""
        orchestrator = IntelligentScrapingOrchestrator(mock_config, client=mock_client)
        
        # Check that context providers are initialized
        assert hasattr(orchestrator, 'website_analysis_provider')
        assert hasattr(orchestrator, 'scraping_results_provider')
        assert hasattr(orchestrator, 'configuration_provider')
        
        # Check that they are the correct types
        assert isinstance(orchestrator.website_analysis_provider, WebsiteAnalysisContextProvider)
        assert isinstance(orchestrator.scraping_results_provider, ScrapingResultsContextProvider)
        assert isinstance(orchestrator.configuration_provider, ConfigurationContextProvider)
        
        # Check that configuration was set
        assert orchestrator.configuration_provider.config == mock_config
    
    def test_update_configuration_context(self, mock_config, mock_client):
        """Test updating configuration context."""
        orchestrator = IntelligentScrapingOrchestrator(mock_config, client=mock_client)
        
        config_updates = {
            "max_concurrent_requests": 10,
            "request_delay": 2.0
        }
        
        orchestrator.update_configuration_context(config_updates)
        
        # Check that overrides were applied
        assert len(orchestrator.configuration_provider.config_overrides) == 2
        assert orchestrator.configuration_provider.config_overrides["max_concurrent_requests"] == 10
    
    def test_get_context_provider_status(self, mock_config, mock_client):
        """Test getting context provider status."""
        orchestrator = IntelligentScrapingOrchestrator(mock_config, client=mock_client)
        
        status = orchestrator.get_context_provider_status()
        
        assert "website_analysis" in status
        assert "scraping_results" in status
        assert "configuration" in status
        
        # Check website analysis status
        website_status = status["website_analysis"]
        assert "has_analysis" in website_status
        assert "pattern_count" in website_status
        assert "cache_size" in website_status
        
        # Check configuration status
        config_status = status["configuration"]
        assert config_status["has_config"] is True
        assert "override_count" in config_status
    
    def test_refresh_all_contexts(self, mock_config, mock_client):
        """Test refreshing all context providers."""
        orchestrator = IntelligentScrapingOrchestrator(mock_config, client=mock_client)
        
        # Add some data to contexts
        orchestrator.website_analysis_provider.analysis_cache["test"] = Mock()
        orchestrator.scraping_results_provider.add_result({"test": "data"}, 80.0)
        
        # Refresh contexts
        orchestrator.refresh_all_contexts()
        
        # Check that cache was cleared
        assert len(orchestrator.website_analysis_provider.analysis_cache) == 0
        
        # Check that configuration was refreshed
        assert orchestrator.configuration_provider.config == mock_config
    
    def test_inject_custom_context(self, mock_config, mock_client):
        """Test injecting custom context data."""
        orchestrator = IntelligentScrapingOrchestrator(mock_config, client=mock_client)
        
        # Test website analysis context injection
        website_context = {
            "url": "https://example.com",
            "custom_data": "test_value"
        }
        orchestrator.inject_custom_context("website_analysis", website_context)
        
        # Test scraping results context injection
        results_context = {
            "operation_metadata": {
                "operation_id": "test-123",
                "custom_field": "test_value"
            }
        }
        orchestrator.inject_custom_context("scraping_results", results_context)
        
        # Test configuration context injection
        config_context = {
            "overrides": {
                "custom_setting": "custom_value"
            }
        }
        orchestrator.inject_custom_context("configuration", config_context)
        
        # Verify injections
        assert orchestrator.scraping_results_provider.operation_metadata["operation_id"] == "test-123"
        assert orchestrator.configuration_provider.config_overrides["custom_setting"] == "custom_value"
    
    @pytest.mark.asyncio
    async def test_dynamic_context_updates_during_run(self, mock_config, mock_client):
        """Test that context providers are updated during orchestrator run."""
        orchestrator = IntelligentScrapingOrchestrator(mock_config, client=mock_client)
        
        # Mock the coordination methods to avoid external dependencies
        mock_planning_result = {
            "scraping_plan": "Test plan",
            "strategy": {"scrape_type": "list"},
            "schema_recipe": {"name": "test_schema"},
            "reasoning": "Test reasoning",
            "confidence": 0.8
        }
        
        mock_scraping_result = {
            "results": {
                "items": [{"title": "Test Item"}],
                "total_found": 1,
                "total_scraped": 1,
                "errors": []
            },
            "quality_metrics": {
                "average_quality_score": 85.0,
                "success_rate": 100.0
            }
        }
        
        orchestrator._coordinate_with_planning_agent = AsyncMock(return_value=mock_planning_result)
        orchestrator._coordinate_with_scraper_tool = AsyncMock(return_value=mock_scraping_result)
        
        # Run orchestrator
        input_data = {
            "scraping_request": "Test request",
            "target_url": "https://example.com"
        }
        
        result = await orchestrator.run(input_data)
        
        # Check that scraping results context was updated
        assert len(orchestrator.scraping_results_provider.results) == 1
        assert orchestrator.scraping_results_provider.operation_metadata["target_url"] == "https://example.com"
        
        # Check that website analysis context update was attempted
        orchestrator._coordinate_with_planning_agent.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])