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
        mock_confirm.side_effect = [
            False,  # Don't view config
            True,   # Proceed with request
            False   # Don't perform another operation
        ]
        
        mock_prompt.side_effect = [
            "Extract product information",  # Scraping request
            "https://example-shop.com",     # Target URL
            "5",                           # Max results
            "80",                          # Quality threshold
            "json"                         # Export format
        ]
        
        with patch.object(app.console, 'print'):
            with patch.object(app.console, 'clear'):
                await app.run_interactive()
        
        # Verify orchestrator was called with correct parameters
        mock_orchestrator.run.assert_called_once()
        call_args = mock_orchestrator.run.call_args[0][0]
        
        assert call_args["scraping_request"] == "Extract product information"
        assert call_args["target_url"] == "https://example-shop.com"
        assert call_args["max_results"] == 5
        assert call_args["quality_threshold"] == 80.0
        assert call_args["export_format"] == "json"
        
        # Verify session stats were updated
        assert app.session_stats["total_requests"] == 1
        assert app.session_stats["successful_requests"] == 1
        assert app.session_stats["failed_requests"] == 0
        assert app.session_stats["total_items_extracted"] == 2