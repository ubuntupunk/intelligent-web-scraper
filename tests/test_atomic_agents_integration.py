"""
Integration tests for atomic-agents ecosystem compatibility.

These tests verify that the Intelligent Web Scraper properly integrates
with atomic-agents patterns and conventions.
"""

import pytest
import asyncio
import os
import sys
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# Add the project root to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from intelligent_web_scraper.config import IntelligentScrapingConfig
from intelligent_web_scraper.agents.orchestrator import IntelligentScrapingOrchestrator
from intelligent_web_scraper.context_providers import (
    WebsiteAnalysisContextProvider,
    ScrapingResultsContextProvider,
    ConfigurationContextProvider
)
from intelligent_web_scraper.cli import main as cli_main, create_parser
from intelligent_web_scraper.main import IntelligentScrapingApp


class TestAtomicAgentsPatterns:
    """Test atomic-agents pattern compliance."""
    
    def test_orchestrator_extends_base_agent(self):
        """Test that orchestrator properly extends BaseAgent."""
        from atomic_agents.agents.base_agent import BaseAgent
        
        config = IntelligentScrapingConfig()
        orchestrator = IntelligentScrapingOrchestrator(config=config)
        
        # Verify inheritance
        assert isinstance(orchestrator, BaseAgent)
        
        # Verify required attributes
        assert hasattr(orchestrator, 'input_schema')
        assert hasattr(orchestrator, 'output_schema')
        assert hasattr(orchestrator, 'system_prompt_generator')
        
        # Verify schema types
        assert orchestrator.input_schema is not None
        assert orchestrator.output_schema is not None
    
    def test_context_providers_extend_base(self):
        """Test that context providers extend SystemPromptContextProviderBase."""
        from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase
        
        # Test WebsiteAnalysisContextProvider
        website_provider = WebsiteAnalysisContextProvider()
        assert isinstance(website_provider, SystemPromptContextProviderBase)
        assert hasattr(website_provider, 'get_info')
        assert callable(website_provider.get_info)
        
        # Test ScrapingResultsContextProvider
        results_provider = ScrapingResultsContextProvider()
        assert isinstance(results_provider, SystemPromptContextProviderBase)
        assert hasattr(results_provider, 'get_info')
        assert callable(results_provider.get_info)
        
        # Test ConfigurationContextProvider
        config_provider = ConfigurationContextProvider()
        assert isinstance(config_provider, SystemPromptContextProviderBase)
        assert hasattr(config_provider, 'get_info')
        assert callable(config_provider.get_info)
    
    def test_input_output_schemas_are_valid(self):
        """Test that input/output schemas follow atomic-agents patterns."""
        from atomic_agents.lib.base.base_io_schema import BaseIOSchema
        
        config = IntelligentScrapingConfig()
        orchestrator = IntelligentScrapingOrchestrator(config=config)
        
        # Verify schema inheritance
        assert issubclass(orchestrator.input_schema, BaseIOSchema)
        assert issubclass(orchestrator.output_schema, BaseIOSchema)
        
        # Test schema instantiation
        input_data = {
            "scraping_request": "Test request",
            "target_url": "https://example.com"
        }
        
        input_instance = orchestrator.input_schema(**input_data)
        assert input_instance.scraping_request == "Test request"
        assert input_instance.target_url == "https://example.com"
    
    def test_system_prompt_generator_integration(self):
        """Test system prompt generator integration."""
        config = IntelligentScrapingConfig()
        orchestrator = IntelligentScrapingOrchestrator(config=config)
        
        # Verify system prompt generator exists
        assert orchestrator.system_prompt_generator is not None
        
        # Test context provider registration
        website_provider = WebsiteAnalysisContextProvider()
        orchestrator.register_context_provider("website_analysis", website_provider)
        
        # Verify context provider is registered
        assert "website_analysis" in orchestrator.system_prompt_generator.context_providers
    
    @pytest.mark.asyncio
    async def test_agent_run_method_compatibility(self):
        """Test that the agent run method is compatible with atomic-agents patterns."""
        config = IntelligentScrapingConfig()
        orchestrator = IntelligentScrapingOrchestrator(config=config)
        
        # Mock the atomic scraper tool to avoid external dependencies
        with patch('intelligent_web_scraper.tools.atomic_scraper_tool.AtomicScraperTool') as mock_tool:
            mock_instance = Mock()
            mock_instance.run = AsyncMock(return_value={
                "extracted_data": [{"title": "Test", "url": "https://example.com"}],
                "metadata": {"pages_processed": 1, "processing_time": 1.0},
                "quality_score": 85.0
            })
            mock_tool.return_value = mock_instance
            
            # Test input data
            input_data = {
                "scraping_request": "Extract titles from this page",
                "target_url": "https://example.com",
                "max_results": 5,
                "quality_threshold": 50.0
            }
            
            # Run the agent
            result = await orchestrator.run(input_data)
            
            # Verify result structure
            assert hasattr(result, 'extracted_data')
            assert hasattr(result, 'metadata')
            assert hasattr(result, 'quality_score')
            assert hasattr(result, 'scraping_plan')
            assert hasattr(result, 'reasoning')


class TestCLIIntegration:
    """Test CLI integration with atomic-agents ecosystem."""
    
    def test_cli_parser_creation(self):
        """Test CLI argument parser creation."""
        parser = create_parser()
        
        # Verify parser exists and has expected arguments
        assert parser is not None
        assert parser.prog == "intelligent-web-scraper"
        
        # Test parsing basic arguments (version exits, so we catch it)
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0  # Successful exit
    
    def test_cli_direct_mode_validation(self):
        """Test CLI direct mode argument validation."""
        from intelligent_web_scraper.cli import validate_direct_mode_args
        
        # Create mock args for testing
        class MockArgs:
            def __init__(self, **kwargs):
                self.direct = kwargs.get('direct', False)
                self.url = kwargs.get('url', None)
                self.request = kwargs.get('request', None)
                self.quality_threshold = kwargs.get('quality_threshold', 50.0)
                self.max_results = kwargs.get('max_results', 10)
        
        # Test valid direct mode args
        valid_args = MockArgs(
            direct=True,
            url="https://example.com",
            request="Test request",
            quality_threshold=75.0,
            max_results=5
        )
        
        # Should not raise exception
        validate_direct_mode_args(valid_args)
        
        # Test invalid URL
        invalid_url_args = MockArgs(
            direct=True,
            url="invalid-url",
            request="Test request"
        )
        
        with pytest.raises(ValueError, match="URL must start with"):
            validate_direct_mode_args(invalid_url_args)
        
        # Test missing URL
        missing_url_args = MockArgs(
            direct=True,
            request="Test request"
        )
        
        with pytest.raises(ValueError, match="--url is required"):
            validate_direct_mode_args(missing_url_args)
    
    def test_config_loading_from_file(self):
        """Test configuration loading from JSON file."""
        from intelligent_web_scraper.cli import load_config_from_file
        import tempfile
        import json
        
        # Create temporary config file
        config_data = {
            "default_quality_threshold": 75.0,
            "max_concurrent_requests": 3,
            "default_export_format": "csv"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Test loading valid config
            loaded_config = load_config_from_file(temp_path)
            assert loaded_config == config_data
            
            # Test loading non-existent file
            with pytest.raises(FileNotFoundError):
                load_config_from_file("non_existent_file.json")
                
        finally:
            os.unlink(temp_path)


class TestPackagingAndDistribution:
    """Test packaging and distribution configuration."""
    
    def test_package_structure(self):
        """Test that package structure follows atomic-agents conventions."""
        # Verify main package exists
        import intelligent_web_scraper
        assert intelligent_web_scraper.__version__ == "0.1.0"
        assert intelligent_web_scraper.__author__ == "Atomic Agents Team"
        
        # Verify main exports
        assert hasattr(intelligent_web_scraper, 'IntelligentScrapingOrchestrator')
        assert hasattr(intelligent_web_scraper, 'IntelligentScrapingConfig')
        assert hasattr(intelligent_web_scraper, 'WebsiteAnalysisContextProvider')
        assert hasattr(intelligent_web_scraper, 'ScrapingResultsContextProvider')
        assert hasattr(intelligent_web_scraper, 'ConfigurationContextProvider')
    
    def test_entry_points_configuration(self):
        """Test that entry points are properly configured."""
        # This would typically be tested by installing the package
        # and verifying the CLI commands are available
        # For now, we test that the CLI module can be imported
        from intelligent_web_scraper import cli
        assert hasattr(cli, 'main')
        assert callable(cli.main)
    
    def test_dependencies_compatibility(self):
        """Test that dependencies are compatible with atomic-agents."""
        # Verify atomic-agents can be imported
        try:
            import atomic_agents
            from atomic_agents.agents.base_agent import BaseAgent
            from atomic_agents.lib.base.base_io_schema import BaseIOSchema
            from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase
            
            # Basic compatibility check
            assert BaseAgent is not None
            assert BaseIOSchema is not None
            assert SystemPromptContextProviderBase is not None
            
        except ImportError as e:
            pytest.fail(f"Failed to import atomic-agents dependencies: {e}")
    
    def test_tool_discovery_compatibility(self):
        """Test compatibility with atomic-agents tool discovery patterns."""
        # Verify that the orchestrator can be discovered and instantiated
        config = IntelligentScrapingConfig()
        orchestrator = IntelligentScrapingOrchestrator(config=config)
        
        # Verify it has the expected interface for tool discovery
        assert hasattr(orchestrator, 'run')
        assert callable(orchestrator.run)
        assert hasattr(orchestrator, 'input_schema')
        assert hasattr(orchestrator, 'output_schema')


class TestConfigurationManagement:
    """Test configuration management patterns."""
    
    def test_config_from_environment(self):
        """Test configuration loading from environment variables."""
        # Set test environment variables
        test_env = {
            "ORCHESTRATOR_MODEL": "gpt-4",
            "PLANNING_AGENT_MODEL": "gpt-3.5-turbo",
            "DEFAULT_QUALITY_THRESHOLD": "75.0",
            "MAX_CONCURRENT_REQUESTS": "3"
        }
        
        with patch.dict(os.environ, test_env):
            config = IntelligentScrapingConfig.from_env()
            
            assert config.orchestrator_model == "gpt-4"
            assert config.planning_agent_model == "gpt-3.5-turbo"
            assert config.default_quality_threshold == 75.0
            assert config.max_concurrent_requests == 3
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        valid_config = IntelligentScrapingConfig(
            default_quality_threshold=75.0,
            max_concurrent_requests=5,
            request_delay=1.0
        )
        assert valid_config.default_quality_threshold == 75.0
        
        # Test invalid quality threshold
        with pytest.raises(ValueError):
            IntelligentScrapingConfig(default_quality_threshold=150.0)
        
        # Test invalid concurrent requests
        with pytest.raises(ValueError):
            IntelligentScrapingConfig(max_concurrent_requests=0)


class TestErrorHandlingPatterns:
    """Test error handling patterns compatibility."""
    
    @pytest.mark.asyncio
    async def test_graceful_error_handling(self):
        """Test that errors are handled gracefully following atomic-agents patterns."""
        config = IntelligentScrapingConfig()
        orchestrator = IntelligentScrapingOrchestrator(config=config)
        
        # Mock tool to raise an exception
        with patch('intelligent_web_scraper.tools.atomic_scraper_tool.AtomicScraperTool') as mock_tool:
            mock_instance = Mock()
            mock_instance.run = AsyncMock(side_effect=Exception("Test error"))
            mock_tool.return_value = mock_instance
            
            input_data = {
                "scraping_request": "Test request",
                "target_url": "https://example.com"
            }
            
            # Should handle error gracefully
            with pytest.raises(Exception) as exc_info:
                await orchestrator.run(input_data)
            
            assert "Test error" in str(exc_info.value)
    
    def test_input_validation_errors(self):
        """Test input validation error handling."""
        config = IntelligentScrapingConfig()
        orchestrator = IntelligentScrapingOrchestrator(config=config)
        
        # Test invalid input data
        with pytest.raises(Exception):
            # Missing required fields
            invalid_input = {}
            orchestrator.input_schema(**invalid_input)


if __name__ == "__main__":
    pytest.main([__file__])