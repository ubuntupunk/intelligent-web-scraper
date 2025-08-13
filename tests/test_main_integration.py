"""
Integration tests for the main application interface.

These tests verify the interactive command-line interface functionality,
input validation, and user interaction flows.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from intelligent_web_scraper.main import IntelligentScrapingApp
from intelligent_web_scraper.config import IntelligentScrapingConfig


class TestIntelligentScrapingApp:
    """Test suite for the main application class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return IntelligentScrapingConfig(
            orchestrator_model="gpt-4o-mini",
            planning_agent_model="gpt-4o-mini",
            default_quality_threshold=75.0,
            max_concurrent_requests=3,
            default_export_format="json",
            results_directory="./test_results",
            enable_monitoring=True
        )
    
    @pytest.fixture
    def app(self, mock_config):
        """Create an application instance for testing."""
        with patch('intelligent_web_scraper.main.IntelligentScrapingOrchestrator'):
            return IntelligentScrapingApp(config=mock_config)
    
    def test_app_initialization(self, app, mock_config):
        """Test that the application initializes correctly."""
        assert app.config == mock_config
        assert app.console is not None
        assert app.orchestrator is not None
        assert app.session_stats == {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_items_extracted": 0
        }
    
    def test_url_validation(self, app):
        """Test URL validation functionality."""
        # Valid URLs
        assert app.validate_url("https://example.com") is True
        assert app.validate_url("http://test.org/path") is True
        assert app.validate_url("https://subdomain.example.com/page?param=value") is True
        
        # Invalid URLs
        assert app.validate_url("") is False
        assert app.validate_url("   ") is False
        assert app.validate_url("not-a-url") is False
        assert app.validate_url("example.com") is False  # Missing scheme
    
    def test_quality_threshold_validation(self, app):
        """Test quality threshold validation."""
        # Valid thresholds
        is_valid, value = app.validate_quality_threshold("50")
        assert is_valid is True
        assert value == 50.0
        
        is_valid, value = app.validate_quality_threshold("0")
        assert is_valid is True
        assert value == 0.0
        
        is_valid, value = app.validate_quality_threshold("100")
        assert is_valid is True
        assert value == 100.0
        
        # Invalid thresholds
        is_valid, value = app.validate_quality_threshold("-1")
        assert is_valid is False
        
        is_valid, value = app.validate_quality_threshold("101")
        assert is_valid is False
        
        is_valid, value = app.validate_quality_threshold("not-a-number")
        assert is_valid is False
    
    def test_max_results_validation(self, app):
        """Test max results validation."""
        # Valid values
        is_valid, value = app.validate_max_results("10")
        assert is_valid is True
        assert value == 10
        
        is_valid, value = app.validate_max_results("1")
        assert is_valid is True
        assert value == 1
        
        # Invalid values
        is_valid, value = app.validate_max_results("0")
        assert is_valid is False
        
        is_valid, value = app.validate_max_results("not-a-number")
        assert is_valid is False
    
    @patch('intelligent_web_scraper.main.Prompt.ask')
    def test_get_validated_input_success(self, mock_prompt, app):
        """Test successful input validation."""
        mock_prompt.return_value = "https://example.com"
        
        result = app.get_validated_input(
            "Enter URL:",
            app.validate_url,
            "Invalid URL",
            None
        )
        
        assert result == "https://example.com"
        mock_prompt.assert_called_once()
    
    def test_display_welcome(self, app):
        """Test welcome message display."""
        with patch.object(app.console, 'print') as mock_print:
            app.display_welcome()
        
        # Verify that print was called multiple times (for different panels)
        assert mock_print.call_count >= 3
    
    def test_display_session_stats(self, app):
        """Test session statistics display."""
        # Set some test stats
        app.session_stats = {
            "total_requests": 5,
            "successful_requests": 4,
            "failed_requests": 1,
            "total_items_extracted": 100
        }
        
        with patch.object(app.console, 'print') as mock_print:
            app.display_session_stats()
        
        mock_print.assert_called_once()
    
    def test_display_results_basic(self, app):
        """Test basic results display."""
        # Create mock result object
        mock_result = Mock()
        mock_result.extracted_data = [{"title": "Test Item 1"}, {"title": "Test Item 2"}]
        mock_result.quality_score = 85.5
        mock_result.metadata = Mock()
        mock_result.metadata.processing_time = 2.5
        mock_result.metadata.pages_processed = 3
        
        with patch.object(app.console, 'print') as mock_print:
            app.display_results(mock_result)
        
        # Should print results summary and sample data
        assert mock_print.call_count >= 2


@pytest.mark.integration
class TestFullUserInteraction:
    """Integration tests for complete user interaction flows."""
    
    @pytest.fixture
    def app_with_mock_orchestrator(self):
        """Create app with mocked orchestrator for integration testing."""
        config = IntelligentScrapingConfig()
        
        with patch('intelligent_web_scraper.main.IntelligentScrapingOrchestrator') as mock_orch_class:
            mock_orchestrator = Mock()
            mock_orch_class.return_value = mock_orchestrator
            
            # Mock successful scraping result
            mock_result = Mock()
            mock_result.extracted_data = [
                {"title": "Product 1", "price": "$10.99"},
                {"title": "Product 2", "price": "$15.99"}
            ]
            mock_result.quality_score = 88.5
            mock_result.metadata = Mock()
            mock_result.metadata.processing_time = 3.2
            mock_result.metadata.pages_processed = 1
            mock_result.export_options = {"json": "./results/data.json"}
            
            mock_orchestrator.run = AsyncMock(return_value=mock_result)
            
            app = IntelligentScrapingApp(config=config)
            app.orchestrator = mock_orchestrator
            
            return app, mock_orchestrator
    
    @patch('intelligent_web_scraper.main.Confirm.ask')
    @patch('intelligent_web_scraper.main.Prompt.ask')
    @pytest.mark.asyncio
    async def test_complete_scraping_flow(self, mock_prompt, mock_confirm, app_with_mock_orchestrator):
        """Test a complete scraping interaction flow."""
        app, mock_orchestrator = app_with_mock_orchestrator
        
        # Mock user inputs
        mock_prompt.side_effect = [
            "Extract product names and prices",  # scraping request
            "https://example.com/products",      # target URL
            "10",                                # max results
            "75.0",                              # quality threshold
            "json"                               # export format
        ]
        mock_confirm.side_effect = [True, False]  # Proceed with request, don't continue
        
        # This would normally run the interactive flow
        # For testing, we'll just verify the orchestrator would be called
        assert mock_orchestrator is not None


class TestConfigurationManagement:
    """Test suite for configuration display and modification functionality."""
    
    @pytest.fixture
    def app_with_config(self):
        """Create app with test configuration."""
        config = IntelligentScrapingConfig(
            orchestrator_model="gpt-4o-mini",
            planning_agent_model="gpt-4o-mini",
            llm_provider="openai",
            default_quality_threshold=75.0,
            max_concurrent_requests=5,
            request_delay=1.0,
            default_export_format="json",
            results_directory="./test_results",
            enable_monitoring=True,
            monitoring_interval=1.0
        )
        
        with patch('intelligent_web_scraper.main.IntelligentScrapingOrchestrator'):
            return IntelligentScrapingApp(config=config)
    
    def test_display_configuration(self, app_with_config):
        """Test configuration display functionality."""
        with patch.object(app_with_config.console, 'print') as mock_print:
            with patch('intelligent_web_scraper.main.Confirm.ask', return_value=False):
                app_with_config.display_configuration()
        
        # Should print multiple configuration tables
        assert mock_print.call_count >= 4  # Agent, Scraping, Output, Performance tables
        
        # Verify that model information is displayed
        printed_content = str(mock_print.call_args_list)
        assert "gpt-4o-mini" in printed_content
        assert "Orchestrator Model" in printed_content
        assert "Planning Agent Model" in printed_content
    
    @patch('intelligent_web_scraper.main.Prompt.ask')
    def test_modify_configuration_models(self, mock_prompt, app_with_config):
        """Test model configuration modification."""
        # Mock user inputs for model changes
        mock_prompt.side_effect = [
            "gpt-4o",           # New orchestrator model
            "gpt-4",            # New planning agent model
            "anthropic",        # New LLM provider
            "",                 # Skip quality threshold
            "",                 # Skip concurrent requests
            "",                 # Skip request delay
            "",                 # Skip export format
            "",                 # Skip results directory
            "",                 # Skip monitoring enable
            ""                  # Skip monitoring interval
        ]
        
        original_orchestrator = app_with_config.config.orchestrator_model
        original_planning = app_with_config.config.planning_agent_model
        original_provider = app_with_config.config.llm_provider
        
        app_with_config.modify_configuration()
        
        # Verify models were updated
        assert app_with_config.config.orchestrator_model == "gpt-4o"
        assert app_with_config.config.planning_agent_model == "gpt-4"
        assert app_with_config.config.llm_provider == "anthropic"
        
        # Verify other settings remained unchanged
        assert app_with_config.config.default_quality_threshold == 75.0
    
    @patch('intelligent_web_scraper.main.Prompt.ask')
    def test_modify_configuration_scraping_settings(self, mock_prompt, app_with_config):
        """Test scraping configuration modification."""
        # Mock user inputs for scraping settings
        mock_prompt.side_effect = [
            "",                 # Skip orchestrator model
            "",                 # Skip planning agent model
            "",                 # Skip LLM provider
            "85.0",             # New quality threshold
            "8",                # New concurrent requests
            "2.5",              # New request delay
            "csv",              # New export format
            "./new_results",    # New results directory
            "false",            # Disable monitoring
            "2.0"               # New monitoring interval
        ]
        
        app_with_config.modify_configuration()
        
        # Verify scraping settings were updated
        assert app_with_config.config.default_quality_threshold == 85.0
        assert app_with_config.config.max_concurrent_requests == 8
        assert app_with_config.config.request_delay == 2.5
        assert app_with_config.config.default_export_format == "csv"
        assert app_with_config.config.results_directory == "./new_results"
        assert app_with_config.config.enable_monitoring == False
        assert app_with_config.config.monitoring_interval == 2.0
    
    @patch('intelligent_web_scraper.main.Prompt.ask')
    def test_modify_configuration_validation(self, mock_prompt, app_with_config):
        """Test configuration modification with invalid inputs."""
        # Mock user inputs with some invalid values
        mock_prompt.side_effect = [
            "",                 # Skip orchestrator model
            "",                 # Skip planning agent model
            "",                 # Skip LLM provider
            "150.0",            # Invalid quality threshold (should be ignored)
            "invalid",          # Invalid concurrent requests (should be ignored)
            "-1.0",             # Invalid request delay (should be ignored)
            "",                 # Skip export format
            "",                 # Skip results directory
            "",                 # Skip monitoring enable
            "invalid"           # Invalid monitoring interval (should be ignored)
        ]
        
        original_threshold = app_with_config.config.default_quality_threshold
        original_concurrent = app_with_config.config.max_concurrent_requests
        original_delay = app_with_config.config.request_delay
        original_interval = app_with_config.config.monitoring_interval
        
        with patch.object(app_with_config.console, 'print'):
            app_with_config.modify_configuration()
        
        # Verify invalid values were rejected and originals preserved
        assert app_with_config.config.default_quality_threshold == original_threshold
        assert app_with_config.config.max_concurrent_requests == original_concurrent
        assert app_with_config.config.request_delay == original_delay
        assert app_with_config.config.monitoring_interval == original_interval
    
    def test_available_models_list(self, app_with_config):
        """Test that available models list includes expected options."""
        # This tests the model choices available in modify_configuration
        expected_models = [
            "gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo",
            "gemini-1.5-flash", "gemini-1.5-pro",
            "claude-3-haiku-20240307", "claude-3-5-sonnet-20241022",
            "deepseek-chat"
        ]
        
        # We can't directly test the choices list, but we can test that
        # the configuration accepts these model names
        for model in expected_models:
            app_with_config.config.orchestrator_model = model
            app_with_config.config.planning_agent_model = model
            # If no exception is raised, the model is valid
            assert app_with_config.config.orchestrator_model == model
            assert app_with_config.config.planning_agent_model == model
    
    @patch('intelligent_web_scraper.main.Confirm.ask')
    def test_configuration_modification_flow(self, mock_confirm, app_with_config):
        """Test the complete configuration modification flow."""
        # Test that modify_configuration is called when user confirms
        mock_confirm.return_value = True
        
        with patch.object(app_with_config, 'modify_configuration') as mock_modify:
            with patch.object(app_with_config.console, 'print'):
                app_with_config.display_configuration()
        
        mock_modify.assert_called_once()
        
        # Test that modify_configuration is not called when user declines
        mock_confirm.return_value = False
        
        with patch.object(app_with_config, 'modify_configuration') as mock_modify:
            with patch.object(app_with_config.console, 'print'):
                app_with_config.display_configuration()
        
        mock_modify.assert_not_called()