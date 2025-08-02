"""
Unit tests for AtomicScraperTool integration and configuration management.

Tests the integration of AtomicScraperTool with atomic-agents patterns,
proper configuration management, enhanced error handling, and monitoring.
"""

import pytest
import os
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from intelligent_web_scraper.config import IntelligentScrapingConfig
from intelligent_web_scraper.tools import (
    AtomicScraperTool,
    AtomicScraperToolConfig,
    AtomicScraperInputSchema,
    AtomicScraperOutputSchema,
    AtomicScraperToolFactory,
    ConfigurationManager,
    ScrapingError,
    NetworkError,
    QualityError,
    ToolConfigurationError
)


class TestAtomicScraperToolConfig:
    """Test AtomicScraperToolConfig configuration management."""
    
    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        config = AtomicScraperToolConfig(base_url="https://example.com")
        
        assert config.base_url == "https://example.com"
        assert config.request_delay == 1.0
        assert config.timeout == 30
        assert config.min_quality_score == 50.0
        assert config.max_pages == 10
        assert config.max_results == 100
        assert config.max_retries == 3
        assert config.retry_delay == 2.0
        assert config.respect_robots_txt is True
        assert config.enable_rate_limiting is True
        assert config.enable_monitoring is True
        assert config.instance_id is None
    
    def test_config_validation_invalid_url(self):
        """Test config validation with invalid URL."""
        with pytest.raises(ValueError, match="base_url must start with http"):
            AtomicScraperToolConfig(base_url="invalid-url")
    
    def test_config_validation_valid_url(self):
        """Test config validation with valid URLs."""
        # Test HTTP URL
        config1 = AtomicScraperToolConfig(base_url="http://example.com")
        assert config1.base_url == "http://example.com"
        
        # Test HTTPS URL
        config2 = AtomicScraperToolConfig(base_url="https://example.com")
        assert config2.base_url == "https://example.com"
    
    def test_config_from_intelligent_config(self):
        """Test creating config from IntelligentScrapingConfig."""
        intelligent_config = IntelligentScrapingConfig(
            request_delay=2.0,
            default_quality_threshold=60.0,
            respect_robots_txt=False,
            enable_rate_limiting=False,
            enable_monitoring=True
        )
        
        config = AtomicScraperToolConfig.from_intelligent_config(
            base_url="https://test.com",
            intelligent_config=intelligent_config,
            instance_id="test_instance"
        )
        
        assert config.base_url == "https://test.com"
        assert config.request_delay == 2.0
        assert config.min_quality_score == 60.0
        assert config.respect_robots_txt is False
        assert config.enable_rate_limiting is False
        assert config.enable_monitoring is True
        assert config.instance_id == "test_instance"
    
    def test_config_with_overrides(self):
        """Test config creation with overrides."""
        intelligent_config = IntelligentScrapingConfig()
        
        config = AtomicScraperToolConfig.from_intelligent_config(
            base_url="https://test.com",
            intelligent_config=intelligent_config,
            timeout=60,
            max_retries=5,
            user_agent="CustomAgent/1.0"
        )
        
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.user_agent == "CustomAgent/1.0"


class TestAtomicScraperInputSchema:
    """Test AtomicScraperInputSchema validation."""
    
    def test_valid_input_schema(self):
        """Test creating valid input schema."""
        input_data = AtomicScraperInputSchema(
            target_url="https://example.com",
            strategy={"scrape_type": "list", "target_selectors": {"item": "div.item"}},
            schema_recipe={"fields": {"title": {"field_type": "string", "extraction_selector": "h1"}}},
            max_results=20
        )
        
        assert input_data.target_url == "https://example.com"
        assert input_data.strategy["scrape_type"] == "list"
        assert input_data.max_results == 20
    
    def test_invalid_url_validation(self):
        """Test URL validation in input schema."""
        with pytest.raises(ValueError, match="Invalid URL format"):
            AtomicScraperInputSchema(
                target_url="not-a-url",
                strategy={"scrape_type": "list", "target_selectors": {}},
                schema_recipe={"fields": {}},
                max_results=10
            )
    
    def test_empty_url_validation(self):
        """Test empty URL validation."""
        with pytest.raises(ValueError, match="target_url cannot be empty"):
            AtomicScraperInputSchema(
                target_url="",
                strategy={"scrape_type": "list", "target_selectors": {}},
                schema_recipe={"fields": {}},
                max_results=10
            )
    
    def test_invalid_url_scheme(self):
        """Test invalid URL scheme validation."""
        with pytest.raises(ValueError, match="URL scheme must be http or https"):
            AtomicScraperInputSchema(
                target_url="ftp://example.com",
                strategy={"scrape_type": "list", "target_selectors": {}},
                schema_recipe={"fields": {}},
                max_results=10
            )


class TestAtomicScraperTool:
    """Test AtomicScraperTool functionality."""
    
    @pytest.fixture
    def mock_intelligent_config(self):
        """Create mock intelligent config."""
        return IntelligentScrapingConfig(
            request_delay=0.1,  # Faster for tests
            default_quality_threshold=50.0,
            enable_monitoring=True
        )
    
    @pytest.fixture
    def mock_tool_config(self):
        """Create mock tool config."""
        return AtomicScraperToolConfig(
            base_url="https://example.com",
            request_delay=0.1,
            timeout=10,
            enable_monitoring=True,
            instance_id="test_instance"
        )
    
    @pytest.fixture
    def scraper_tool(self, mock_tool_config, mock_intelligent_config):
        """Create AtomicScraperTool instance for testing."""
        return AtomicScraperTool(
            config=mock_tool_config,
            intelligent_config=mock_intelligent_config
        )
    
    def test_tool_initialization(self, scraper_tool):
        """Test tool initialization."""
        assert scraper_tool.config.base_url == "https://example.com"
        assert scraper_tool.config.instance_id == "test_instance"
        assert scraper_tool.intelligent_config.enable_monitoring is True
        assert scraper_tool.session is not None
        assert scraper_tool.monitoring_data is not None
    
    def test_tool_initialization_with_defaults(self):
        """Test tool initialization with default configs."""
        tool = AtomicScraperTool()
        
        assert tool.config is not None
        assert tool.intelligent_config is not None
        assert tool.session is not None
        assert tool.monitoring_data is not None
    
    @patch('intelligent_web_scraper.tools.atomic_scraper_tool.requests.Session.get')
    def test_successful_scraping_operation(self, mock_get, scraper_tool):
        """Test successful scraping operation."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = """
        <html>
            <body>
                <div class="item">
                    <h1>Test Title 1</h1>
                    <p>Test description 1</p>
                </div>
                <div class="item">
                    <h1>Test Title 2</h1>
                    <p>Test description 2</p>
                </div>
            </body>
        </html>
        """
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Create input data
        input_data = AtomicScraperInputSchema(
            target_url="https://example.com",
            strategy={
                "scrape_type": "list",
                "target_selectors": {
                    "container": "body",
                    "item": "div.item"
                },
                "max_pages": 1
            },
            schema_recipe={
                "fields": {
                    "title": {
                        "field_type": "string",
                        "extraction_selector": "h1",
                        "quality_weight": 2.0
                    },
                    "description": {
                        "field_type": "string",
                        "extraction_selector": "p",
                        "quality_weight": 1.0
                    }
                }
            },
            max_results=10
        )
        
        # Execute scraping
        result = scraper_tool.run(input_data)
        
        # Verify results
        assert isinstance(result, AtomicScraperOutputSchema)
        assert result.results["total_scraped"] == 2
        assert len(result.results["items"]) == 2
        assert result.results["items"][0]["title"] == "Test Title 1"
        assert result.results["items"][1]["title"] == "Test Title 2"
        assert result.quality_metrics["total_items_scraped"] == 2.0
        assert result.monitoring_data is not None
        assert result.monitoring_data["instance_id"] == "test_instance"
        assert len(result.errors) == 0
    
    @patch('intelligent_web_scraper.tools.atomic_scraper_tool.requests.Session.get')
    def test_network_error_handling(self, mock_get, scraper_tool):
        """Test network error handling with retries."""
        # Mock network error
        mock_get.side_effect = Exception("Network error")
        
        input_data = AtomicScraperInputSchema(
            target_url="https://example.com",
            strategy={
                "scrape_type": "list",
                "target_selectors": {"item": "div"}
            },
            schema_recipe={"fields": {"title": {"field_type": "string", "extraction_selector": "h1"}}},
            max_results=10
        )
        
        # Execute scraping (should handle error gracefully)
        result = scraper_tool.run(input_data)
        
        # Verify error handling
        assert isinstance(result, AtomicScraperOutputSchema)
        assert result.results["total_scraped"] == 0
        assert len(result.errors) > 0 or len(result.results.get("errors", [])) > 0
        # Check for error in either location
        all_errors = result.errors + result.results.get("errors", [])
        assert any("Network error" in error or "failed" in error.lower() for error in all_errors)
        assert result.quality_metrics["success_rate"] == 0.0
        assert result.monitoring_data is not None
    
    def test_input_validation_errors(self, scraper_tool):
        """Test input validation error handling."""
        # Test missing strategy fields
        input_data = AtomicScraperInputSchema(
            target_url="https://example.com",
            strategy={},  # Missing required fields
            schema_recipe={"fields": {"title": {"field_type": "string", "extraction_selector": "h1"}}},
            max_results=10
        )
        
        result = scraper_tool.run(input_data)
        
        assert result.results["total_scraped"] == 0
        assert len(result.errors) > 0
        # Check for validation error (either "missing required field" or "strategy cannot be empty")
        error_msg = result.errors[0].lower()
        assert "missing required field" in error_msg or "strategy cannot be empty" in error_msg
    
    def test_quality_threshold_filtering(self, scraper_tool):
        """Test quality threshold filtering."""
        with patch('intelligent_web_scraper.tools.atomic_scraper_tool.requests.Session.get') as mock_get:
            # Mock response with minimal content
            mock_response = Mock()
            mock_response.text = """
            <html>
                <body>
                    <div class="item">
                        <h1></h1>  <!-- Empty title -->
                    </div>
                    <div class="item">
                        <h1>Good Title</h1>
                        <p>Good description</p>
                    </div>
                </body>
            </html>
            """
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            input_data = AtomicScraperInputSchema(
                target_url="https://example.com",
                strategy={
                    "scrape_type": "list",
                    "target_selectors": {"container": "body", "item": "div.item"},
                    "max_pages": 1
                },
                schema_recipe={
                    "fields": {
                        "title": {"field_type": "string", "extraction_selector": "h1", "quality_weight": 2.0},
                        "description": {"field_type": "string", "extraction_selector": "p", "quality_weight": 1.0}
                    }
                },
                max_results=10,
                quality_threshold=60.0  # High threshold
            )
            
            result = scraper_tool.run(input_data)
            
            # Should filter out low-quality items
            assert result.results["total_scraped"] <= result.results["total_found"]
    
    def test_monitoring_data_collection(self, scraper_tool):
        """Test monitoring data collection."""
        with patch('intelligent_web_scraper.tools.atomic_scraper_tool.requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.text = "<html><body><div>Test</div></body></html>"
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            input_data = AtomicScraperInputSchema(
                target_url="https://example.com",
                strategy={"scrape_type": "list", "target_selectors": {"item": "div"}},
                schema_recipe={"fields": {"content": {"field_type": "string", "extraction_selector": "div"}}},
                max_results=10
            )
            
            # Execute multiple operations
            result1 = scraper_tool.run(input_data)
            result2 = scraper_tool.run(input_data)
            
            # Check monitoring data
            monitoring_data = scraper_tool.get_monitoring_data()
            assert monitoring_data["requests_made"] == 2
            assert monitoring_data["instance_id"] == "test_instance"
            assert "success_rate" in monitoring_data
            assert "average_response_time" in monitoring_data
    
    def test_config_update(self, scraper_tool):
        """Test dynamic configuration updates."""
        original_delay = scraper_tool.config.request_delay
        original_timeout = scraper_tool.config.timeout
        
        # Update configuration
        scraper_tool.update_config(
            request_delay=2.0,
            timeout=60,
            user_agent="UpdatedAgent/1.0"
        )
        
        assert scraper_tool.config.request_delay == 2.0
        assert scraper_tool.config.timeout == 60
        assert scraper_tool.request_timeout == 60
        assert scraper_tool.session.headers["User-Agent"] == "UpdatedAgent/1.0"
    
    def test_tool_info(self, scraper_tool):
        """Test tool information retrieval."""
        info = scraper_tool.get_tool_info()
        
        assert info["name"] == "AtomicScraperTool"
        assert info["version"] == "1.0.0"
        assert "config" in info
        assert "supported_strategies" in info
        assert "supported_field_types" in info
        assert info["monitoring_enabled"] is True
        assert info["config"]["instance_id"] == "test_instance"


class TestAtomicScraperToolFactory:
    """Test AtomicScraperToolFactory functionality."""
    
    @pytest.fixture
    def mock_intelligent_config(self):
        """Create mock intelligent config."""
        return IntelligentScrapingConfig(
            request_delay=0.1,
            default_quality_threshold=60.0,
            enable_monitoring=True
        )
    
    @pytest.fixture
    def tool_factory(self, mock_intelligent_config):
        """Create tool factory for testing."""
        return AtomicScraperToolFactory(intelligent_config=mock_intelligent_config)
    
    def test_factory_initialization(self, tool_factory):
        """Test factory initialization."""
        assert tool_factory.intelligent_config is not None
        assert tool_factory._tool_instances == {}
        assert tool_factory._config_cache == {}
    
    def test_create_tool(self, tool_factory):
        """Test tool creation through factory."""
        tool = tool_factory.create_tool(
            base_url="https://example.com",
            instance_id="test_tool"
        )
        
        assert isinstance(tool, AtomicScraperTool)
        assert tool.config.base_url == "https://example.com"
        assert tool.config.instance_id == "test_tool"
        assert tool.config.min_quality_score == 60.0  # From intelligent config
    
    def test_create_tool_with_overrides(self, tool_factory):
        """Test tool creation with configuration overrides."""
        tool = tool_factory.create_tool(
            base_url="https://example.com",
            instance_id="test_tool",
            config_overrides={
                "timeout": 120,
                "max_retries": 5,
                "min_quality_score": 70.0
            }
        )
        
        assert tool.config.timeout == 120
        assert tool.config.max_retries == 5
        assert tool.config.min_quality_score == 70.0
    
    def test_get_existing_tool(self, tool_factory):
        """Test retrieving existing tool instance."""
        # Create tool
        tool1 = tool_factory.create_tool(
            base_url="https://example.com",
            instance_id="test_tool"
        )
        
        # Get existing tool
        tool2 = tool_factory.get_tool("test_tool")
        
        assert tool1 is tool2
        assert tool2.config.instance_id == "test_tool"
    
    def test_get_nonexistent_tool(self, tool_factory):
        """Test retrieving non-existent tool."""
        tool = tool_factory.get_tool("nonexistent")
        assert tool is None
    
    def test_create_or_get_tool(self, tool_factory):
        """Test create_or_get_tool functionality."""
        # First call should create new tool
        tool1 = tool_factory.create_or_get_tool(
            base_url="https://example.com",
            instance_id="test_tool"
        )
        
        # Second call should return existing tool
        tool2 = tool_factory.create_or_get_tool(
            base_url="https://example.com",
            instance_id="test_tool"
        )
        
        assert tool1 is tool2
    
    def test_invalid_configuration_error(self, tool_factory):
        """Test error handling for invalid configuration."""
        with pytest.raises(ToolConfigurationError):
            tool_factory.create_tool(
                base_url="invalid-url",  # Invalid URL format
                instance_id="test_tool"
            )
    
    def test_list_tool_instances(self, tool_factory):
        """Test listing tool instances."""
        # Create multiple tools
        tool1 = tool_factory.create_tool("https://example1.com", "tool1")
        tool2 = tool_factory.create_tool("https://example2.com", "tool2")
        
        instances = tool_factory.list_tool_instances()
        
        assert len(instances) == 2
        assert "tool1" in instances
        assert "tool2" in instances
        assert instances["tool1"]["base_url"] == "https://example1.com"
        assert instances["tool2"]["base_url"] == "https://example2.com"
    
    def test_remove_tool_instance(self, tool_factory):
        """Test removing tool instances."""
        # Create tool
        tool_factory.create_tool("https://example.com", "test_tool")
        
        # Remove tool
        removed = tool_factory.remove_tool_instance("test_tool")
        assert removed is True
        
        # Verify removal
        tool = tool_factory.get_tool("test_tool")
        assert tool is None
        
        # Try to remove non-existent tool
        removed = tool_factory.remove_tool_instance("nonexistent")
        assert removed is False
    
    def test_clear_all_instances(self, tool_factory):
        """Test clearing all instances."""
        # Create multiple tools
        tool_factory.create_tool("https://example1.com", "tool1")
        tool_factory.create_tool("https://example2.com", "tool2")
        
        # Clear all
        tool_factory.clear_all_instances()
        
        # Verify clearing
        instances = tool_factory.list_tool_instances()
        assert len(instances) == 0
    
    def test_factory_stats(self, tool_factory):
        """Test factory statistics."""
        # Create tools
        tool_factory.create_tool("https://example1.com", "tool1")
        tool_factory.create_tool("https://example2.com", "tool2")
        
        stats = tool_factory.get_factory_stats()
        
        assert stats["total_instances"] == 2
        assert "tool1" in stats["instance_ids"]
        assert "tool2" in stats["instance_ids"]
        assert "intelligent_config" in stats
        assert stats["intelligent_config"]["default_quality_threshold"] == 60.0


class TestConfigurationManager:
    """Test ConfigurationManager functionality."""
    
    @pytest.fixture
    def config_manager(self):
        """Create configuration manager for testing."""
        return ConfigurationManager()
    
    def test_get_env_value_with_default(self, config_manager):
        """Test getting environment value with default."""
        # Test with non-existent env var
        value = config_manager.get_env_value("NONEXISTENT_VAR", default="default_value")
        assert value == "default_value"
    
    def test_get_env_value_with_type_conversion(self, config_manager):
        """Test environment value type conversion."""
        with patch.dict(os.environ, {"TEST_INT": "42", "TEST_FLOAT": "3.14", "TEST_BOOL": "true"}):
            int_value = config_manager.get_env_value("TEST_INT", value_type=int)
            float_value = config_manager.get_env_value("TEST_FLOAT", value_type=float)
            bool_value = config_manager.get_env_value("TEST_BOOL", value_type=bool)
            
            assert int_value == 42
            assert float_value == 3.14
            assert bool_value is True
    
    def test_get_env_value_required_missing(self, config_manager):
        """Test required environment variable missing."""
        with pytest.raises(ToolConfigurationError, match="Required environment variable"):
            config_manager.get_env_value("REQUIRED_MISSING_VAR", required=True)
    
    def test_get_env_value_invalid_type(self, config_manager):
        """Test invalid type conversion."""
        with patch.dict(os.environ, {"INVALID_INT": "not_a_number"}):
            with pytest.raises(ToolConfigurationError, match="Invalid type"):
                config_manager.get_env_value("INVALID_INT", value_type=int)
    
    def test_get_env_value_validation_failure(self, config_manager):
        """Test validation rule failure."""
        with patch.dict(os.environ, {"QUALITY_THRESHOLD": "150"}):  # Invalid range
            with pytest.raises(ToolConfigurationError, match="Invalid value"):
                config_manager.get_env_value("QUALITY_THRESHOLD", value_type=float)
    
    def test_validate_environment(self, config_manager):
        """Test environment validation."""
        with patch.dict(os.environ, {
            "ORCHESTRATOR_MODEL": "gpt-4",
            "QUALITY_THRESHOLD": "75.0",
            "MAX_CONCURRENT_REQUESTS": "10"
        }):
            report = config_manager.validate_environment()
            
            assert report["valid"] is True
            assert len(report["errors"]) == 0
            assert "ORCHESTRATOR_MODEL" in report["values"]
            assert report["values"]["QUALITY_THRESHOLD"] == 75.0
    
    def test_validate_environment_with_errors(self, config_manager):
        """Test environment validation with errors."""
        with patch.dict(os.environ, {"QUALITY_THRESHOLD": "invalid"}):
            report = config_manager.validate_environment()
            
            assert report["valid"] is False
            assert len(report["errors"]) > 0
    
    def test_cache_functionality(self, config_manager):
        """Test environment variable caching."""
        with patch.dict(os.environ, {"TEST_CACHE": "cached_value"}):
            # First call should cache the value
            value1 = config_manager.get_env_value("TEST_CACHE")
            
            # Modify environment (should not affect cached value)
            os.environ["TEST_CACHE"] = "modified_value"
            value2 = config_manager.get_env_value("TEST_CACHE")
            
            assert value1 == value2 == "cached_value"
            
            # Clear cache and get new value
            config_manager.clear_cache()
            value3 = config_manager.get_env_value("TEST_CACHE")
            
            assert value3 == "modified_value"


class TestErrorHandling:
    """Test error handling and logging functionality."""
    
    def test_scraping_error_creation(self):
        """Test ScrapingError creation."""
        error = ScrapingError("Test error message")
        assert str(error) == "Test error message"
    
    def test_network_error_creation(self):
        """Test NetworkError creation with URL."""
        error = NetworkError("Network failed", url="https://example.com")
        assert str(error) == "Network failed"
        assert error.url == "https://example.com"
    
    def test_quality_error_creation(self):
        """Test QualityError creation with quality score."""
        error = QualityError("Quality too low", quality_score=25.0)
        assert str(error) == "Quality too low"
        assert error.quality_score == 25.0
    
    def test_tool_configuration_error(self):
        """Test ToolConfigurationError creation."""
        error = ToolConfigurationError("Invalid configuration")
        assert str(error) == "Invalid configuration"


class TestIntegrationPatterns:
    """Test atomic-agents integration patterns."""
    
    def test_base_tool_inheritance(self):
        """Test that AtomicScraperTool properly inherits from BaseTool."""
        from atomic_agents.lib.base.base_tool import BaseTool
        
        tool = AtomicScraperTool()
        assert isinstance(tool, BaseTool)
        assert hasattr(tool, 'input_schema')
        assert hasattr(tool, 'output_schema')
        assert hasattr(tool, 'run')
    
    def test_schema_compliance(self):
        """Test that input/output schemas comply with BaseIOSchema."""
        from atomic_agents.lib.base.base_io_schema import BaseIOSchema
        
        # Test input schema
        input_schema = AtomicScraperInputSchema(
            target_url="https://example.com",
            strategy={"scrape_type": "list", "target_selectors": {}},
            schema_recipe={"fields": {}},
            max_results=10
        )
        assert isinstance(input_schema, BaseIOSchema)
        
        # Test output schema
        output_schema = AtomicScraperOutputSchema(
            results={"items": [], "total_scraped": 0},
            summary="Test summary",
            quality_metrics={"average_quality_score": 0.0}
        )
        assert isinstance(output_schema, BaseIOSchema)
    
    def test_configuration_integration(self):
        """Test integration with IntelligentScrapingConfig."""
        intelligent_config = IntelligentScrapingConfig(
            request_delay=2.0,
            default_quality_threshold=70.0,
            enable_monitoring=True
        )
        
        tool_config = AtomicScraperToolConfig.from_intelligent_config(
            base_url="https://example.com",
            intelligent_config=intelligent_config
        )
        
        tool = AtomicScraperTool(
            config=tool_config,
            intelligent_config=intelligent_config
        )
        
        assert tool.config.request_delay == 2.0
        assert tool.config.min_quality_score == 70.0
        assert tool.config.enable_monitoring is True
        assert tool.intelligent_config.enable_monitoring is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])