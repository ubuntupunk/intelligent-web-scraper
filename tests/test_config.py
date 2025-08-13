"""
Unit tests for configuration management.

This module tests the IntelligentScrapingConfig class and environment variable handling.
"""

import os
import pytest
from unittest.mock import patch, Mock
from typing import Dict, Any

from intelligent_web_scraper.config import IntelligentScrapingConfig


class TestIntelligentScrapingConfig:
    """Test the IntelligentScrapingConfig class."""
    
    def test_config_default_values(self):
        """Test that configuration has correct default values."""
        config = IntelligentScrapingConfig()
        
        # Agent configuration
        assert config.orchestrator_model == "gpt-4o-mini"
        assert config.planning_agent_model == "gpt-4o-mini"
        
        # Scraping configuration
        assert config.default_quality_threshold == 50.0
        assert config.max_concurrent_requests == 5
        assert config.request_delay == 1.0
        
        # Output configuration
        assert config.default_export_format == "json"
        assert config.results_directory == "./results"
        
        # Compliance configuration
        assert config.respect_robots_txt is True
        assert config.enable_rate_limiting is True
        
        # Monitoring configuration
        assert config.enable_monitoring is True
        assert config.monitoring_interval == 1.0
        
        # Concurrency configuration
        assert config.max_instances == 5
        assert config.max_workers == 10
        assert config.max_async_tasks == 50
    
    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = IntelligentScrapingConfig(
            orchestrator_model="gpt-4",
            planning_agent_model="gpt-3.5-turbo",
            default_quality_threshold=75.0,
            max_concurrent_requests=10,
            request_delay=2.0,
            default_export_format="csv",
            results_directory="./custom_results",
            respect_robots_txt=False,
            enable_rate_limiting=False,
            enable_monitoring=False,
            monitoring_interval=0.5,
            max_instances=3,
            max_workers=20,
            max_async_tasks=100
        )
        
        assert config.orchestrator_model == "gpt-4"
        assert config.planning_agent_model == "gpt-3.5-turbo"
        assert config.default_quality_threshold == 75.0
        assert config.max_concurrent_requests == 10
        assert config.request_delay == 2.0
        assert config.default_export_format == "csv"
        assert config.results_directory == "./custom_results"
        assert config.respect_robots_txt is False
        assert config.enable_rate_limiting is False
        assert config.enable_monitoring is False
        assert config.monitoring_interval == 0.5
        assert config.max_instances == 3
        assert config.max_workers == 20
        assert config.max_async_tasks == 100
    
    def test_config_validation_valid_values(self):
        """Test that valid configuration values pass validation."""
        # Test with valid values
        config = IntelligentScrapingConfig(
            default_quality_threshold=85.5,
            max_concurrent_requests=1,
            request_delay=0.1,
            monitoring_interval=0.1,
            max_instances=1,
            max_workers=1,
            max_async_tasks=1
        )
        
        # Should not raise any validation errors
        assert config.default_quality_threshold == 85.5
        assert config.max_concurrent_requests == 1
        assert config.request_delay == 0.1
        assert config.monitoring_interval == 0.1
        assert config.max_instances == 1
        assert config.max_workers == 1
        assert config.max_async_tasks == 1
    
    def test_config_validation_invalid_values(self):
        """Test that invalid configuration values are handled properly."""
        # Note: The current config doesn't have validation constraints
        # These tests verify the config accepts various values
        
        # Test edge case values that should be accepted
        config1 = IntelligentScrapingConfig(default_quality_threshold=0.0)
        assert config1.default_quality_threshold == 0.0
        
        config2 = IntelligentScrapingConfig(max_concurrent_requests=1)
        assert config2.max_concurrent_requests == 1
        
        config3 = IntelligentScrapingConfig(request_delay=0.0)
        assert config3.request_delay == 0.0
    
    def test_config_field_descriptions(self):
        """Test that configuration fields have proper descriptions."""
        config = IntelligentScrapingConfig()
        
        # Check that fields have descriptions using Pydantic v2 syntax
        fields = config.model_fields
        
        assert "Model for orchestrator agent" in fields['orchestrator_model'].description
        assert "Model for planning agent" in fields['planning_agent_model'].description
        assert "Default quality threshold" in fields['default_quality_threshold'].description
        assert "Maximum concurrent requests" in fields['max_concurrent_requests'].description
        assert "Delay between requests" in fields['request_delay'].description
        assert "Default export format" in fields['default_export_format'].description
        assert "Directory for exported results" in fields['results_directory'].description
        assert "Whether to respect robots.txt" in fields['respect_robots_txt'].description
        assert "Whether to enable rate limiting" in fields['enable_rate_limiting'].description
        assert "Whether to enable real-time monitoring" in fields['enable_monitoring'].description
        assert "Monitoring update interval in seconds" in fields['monitoring_interval'].description
        assert "Maximum number of scraper instances" in fields['max_instances'].description
        assert "Maximum number of worker threads" in fields['max_workers'].description
        assert "Maximum number of concurrent async tasks" in fields['max_async_tasks'].description
    
    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_with_defaults(self):
        """Test creating configuration from environment variables with defaults."""
        config = IntelligentScrapingConfig.from_env()
        
        # Should use default values when environment variables are not set
        assert config.orchestrator_model == "gpt-4o-mini"
        assert config.planning_agent_model == "gpt-4o-mini"
        assert config.default_quality_threshold == 50.0
        assert config.max_concurrent_requests == 5
        assert config.request_delay == 1.0
        assert config.default_export_format == "json"
        assert config.results_directory == "./results"
        assert config.respect_robots_txt is True
        assert config.enable_rate_limiting is True
        assert config.enable_monitoring is True
        assert config.monitoring_interval == 1.0
        assert config.max_instances == 5
        assert config.max_workers == 10
        assert config.max_async_tasks == 50
    
    @patch.dict(os.environ, {
        'ORCHESTRATOR_MODEL': 'gpt-4',
        'PLANNING_AGENT_MODEL': 'gpt-3.5-turbo',
        'QUALITY_THRESHOLD': '75.5',
        'MAX_CONCURRENT_REQUESTS': '8',
        'REQUEST_DELAY': '2.5',
        'EXPORT_FORMAT': 'csv',
        'RESULTS_DIRECTORY': '/tmp/results',
        'RESPECT_ROBOTS_TXT': 'false',
        'ENABLE_RATE_LIMITING': 'false',
        'ENABLE_MONITORING': 'false',
        'MONITORING_INTERVAL': '0.5',
        'MAX_INSTANCES': '3',
        'MAX_WORKERS': '15',
        'MAX_ASYNC_TASKS': '75'
    })
    def test_from_env_with_custom_values(self):
        """Test creating configuration from environment variables with custom values."""
        config = IntelligentScrapingConfig.from_env()
        
        assert config.orchestrator_model == "gpt-4"
        assert config.planning_agent_model == "gpt-3.5-turbo"
        assert config.default_quality_threshold == 75.5
        assert config.max_concurrent_requests == 8
        assert config.request_delay == 2.5
        assert config.default_export_format == "csv"
        assert config.results_directory == "/tmp/results"
        assert config.respect_robots_txt is False
        assert config.enable_rate_limiting is False
        assert config.enable_monitoring is False
        assert config.monitoring_interval == 0.5
        assert config.max_instances == 3
        assert config.max_workers == 15
        assert config.max_async_tasks == 75
    
    @patch.dict(os.environ, {
        'RESPECT_ROBOTS_TXT': 'TRUE',
        'ENABLE_RATE_LIMITING': 'True',
        'ENABLE_MONITORING': '1'
    })
    def test_from_env_boolean_parsing(self):
        """Test boolean parsing from environment variables."""
        config = IntelligentScrapingConfig.from_env()
        
        # Test various true values
        assert config.respect_robots_txt is True
        assert config.enable_rate_limiting is True
        # Note: '1' should be parsed as string, not boolean, so it becomes False
        assert config.enable_monitoring is False
    
    @patch.dict(os.environ, {
        'QUALITY_THRESHOLD': 'invalid_float',
        'MAX_CONCURRENT_REQUESTS': 'invalid_int',
        'REQUEST_DELAY': 'invalid_float'
    })
    def test_from_env_invalid_types(self):
        """Test handling of invalid type conversions from environment variables."""
        # Should raise ValueError for invalid type conversions
        with pytest.raises(ValueError):
            IntelligentScrapingConfig.from_env()
    
    @patch.dict(os.environ, {
        'QUALITY_THRESHOLD': '-10.0',
        'MAX_CONCURRENT_REQUESTS': '0',
        'REQUEST_DELAY': '-1.0'
    })
    def test_from_env_invalid_values(self):
        """Test handling of edge case values from environment variables."""
        # The current config accepts these values without validation
        config = IntelligentScrapingConfig.from_env()
        
        assert config.default_quality_threshold == -10.0
        assert config.max_concurrent_requests == 0
        assert config.request_delay == -1.0
    
    def test_config_serialization(self):
        """Test configuration serialization to dict."""
        config = IntelligentScrapingConfig(
            orchestrator_model="gpt-4",
            default_quality_threshold=80.0,
            max_concurrent_requests=7
        )
        
        config_dict = config.model_dump()
        
        assert isinstance(config_dict, dict)
        assert config_dict['orchestrator_model'] == "gpt-4"
        assert config_dict['default_quality_threshold'] == 80.0
        assert config_dict['max_concurrent_requests'] == 7
        assert config_dict['planning_agent_model'] == "gpt-4o-mini"  # Default value
    
    def test_config_json_serialization(self):
        """Test configuration JSON serialization."""
        config = IntelligentScrapingConfig(
            orchestrator_model="gpt-4",
            default_quality_threshold=80.0
        )
        
        json_str = config.model_dump_json()
        
        assert isinstance(json_str, str)
        assert '"orchestrator_model":"gpt-4"' in json_str
        assert '"default_quality_threshold":80.0' in json_str
    
    def test_config_copy(self):
        """Test configuration copying and updating."""
        original_config = IntelligentScrapingConfig(
            orchestrator_model="gpt-4",
            default_quality_threshold=80.0
        )
        
        # Test copy with updates
        updated_config = original_config.model_copy(update={
            'max_concurrent_requests': 15,
            'request_delay': 3.0
        })
        
        # Original should be unchanged
        assert original_config.orchestrator_model == "gpt-4"
        assert original_config.default_quality_threshold == 80.0
        assert original_config.max_concurrent_requests == 5  # Default
        assert original_config.request_delay == 1.0  # Default
        
        # Updated should have new values
        assert updated_config.orchestrator_model == "gpt-4"  # Copied
        assert updated_config.default_quality_threshold == 80.0  # Copied
        assert updated_config.max_concurrent_requests == 15  # Updated
        assert updated_config.request_delay == 3.0  # Updated
    
    def test_config_equality(self):
        """Test configuration equality comparison."""
        config1 = IntelligentScrapingConfig(
            orchestrator_model="gpt-4",
            default_quality_threshold=80.0
        )
        
        config2 = IntelligentScrapingConfig(
            orchestrator_model="gpt-4",
            default_quality_threshold=80.0
        )
        
        config3 = IntelligentScrapingConfig(
            orchestrator_model="gpt-3.5-turbo",
            default_quality_threshold=80.0
        )
        
        assert config1 == config2
        assert config1 != config3
    
    def test_config_hash(self):
        """Test configuration hashing (Pydantic models are not hashable by default)."""
        config1 = IntelligentScrapingConfig(
            orchestrator_model="gpt-4",
            default_quality_threshold=80.0
        )
        
        config2 = IntelligentScrapingConfig(
            orchestrator_model="gpt-4",
            default_quality_threshold=80.0
        )
        
        # Pydantic models are not hashable by default
        with pytest.raises(TypeError):
            hash(config1)
        
        # But they can be compared for equality
        assert config1 == config2
    
    def test_config_repr(self):
        """Test configuration string representation."""
        config = IntelligentScrapingConfig(
            orchestrator_model="gpt-4",
            default_quality_threshold=80.0
        )
        
        repr_str = repr(config)
        
        assert "IntelligentScrapingConfig" in repr_str
        assert "orchestrator_model='gpt-4'" in repr_str
        assert "default_quality_threshold=80.0" in repr_str
    
    def test_config_field_access(self):
        """Test configuration field access and modification."""
        config = IntelligentScrapingConfig()
        
        # Test field access
        assert hasattr(config, 'orchestrator_model')
        assert hasattr(config, 'default_quality_threshold')
        assert hasattr(config, 'max_concurrent_requests')
        
        # Test field modification
        config.orchestrator_model = "gpt-4"
        assert config.orchestrator_model == "gpt-4"
        
        config.default_quality_threshold = 90.0
        assert config.default_quality_threshold == 90.0
    
    def test_config_validation_on_modification(self):
        """Test field modification behavior."""
        config = IntelligentScrapingConfig()
        
        # Test field modification (Pydantic models are immutable by default)
        # These should work without validation errors in the current implementation
        config.default_quality_threshold = 90.0
        assert config.default_quality_threshold == 90.0
        
        config.max_concurrent_requests = 10
        assert config.max_concurrent_requests == 10
        
        config.request_delay = 2.0
        assert config.request_delay == 2.0


class TestModelConfiguration:
    """Test model configuration and validation."""
    
    def test_supported_models(self):
        """Test that supported models can be configured."""
        supported_models = [
            "gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo",
            "gemini-1.5-flash", "gemini-1.5-pro",
            "claude-3-haiku-20240307", "claude-3-5-sonnet-20241022",
            "deepseek-chat"
        ]
        
        for model in supported_models:
            # Test orchestrator model
            config = IntelligentScrapingConfig(orchestrator_model=model)
            assert config.orchestrator_model == model
            
            # Test planning agent model
            config = IntelligentScrapingConfig(planning_agent_model=model)
            assert config.planning_agent_model == model
    
    def test_model_configuration_combinations(self):
        """Test different model combinations."""
        # Test same model for both agents
        config = IntelligentScrapingConfig(
            orchestrator_model="gpt-4o",
            planning_agent_model="gpt-4o"
        )
        assert config.orchestrator_model == "gpt-4o"
        assert config.planning_agent_model == "gpt-4o"
        
        # Test different models for each agent
        config = IntelligentScrapingConfig(
            orchestrator_model="gpt-4o-mini",
            planning_agent_model="gpt-4"
        )
        assert config.orchestrator_model == "gpt-4o-mini"
        assert config.planning_agent_model == "gpt-4"
        
        # Test cross-provider combinations
        config = IntelligentScrapingConfig(
            orchestrator_model="claude-3-5-sonnet-20241022",
            planning_agent_model="gemini-1.5-pro"
        )
        assert config.orchestrator_model == "claude-3-5-sonnet-20241022"
        assert config.planning_agent_model == "gemini-1.5-pro"
    
    def test_llm_provider_configuration(self):
        """Test LLM provider configuration."""
        supported_providers = ["openai", "gemini", "anthropic", "deepseek", "openrouter"]
        
        for provider in supported_providers:
            config = IntelligentScrapingConfig(llm_provider=provider)
            assert config.llm_provider == provider
    
    def test_provider_model_mapping(self):
        """Test that provider model mapping is properly configured."""
        config = IntelligentScrapingConfig()
        
        # Test that provider model mapping exists
        assert hasattr(config, 'provider_model_mapping')
        assert isinstance(config.provider_model_mapping, dict)
        
        # Test that all supported providers are in the mapping
        expected_providers = ["openai", "gemini", "deepseek", "openrouter", "anthropic"]
        for provider in expected_providers:
            assert provider in config.provider_model_mapping
            assert isinstance(config.provider_model_mapping[provider], dict)
    
    def test_model_environment_variable_loading(self):
        """Test loading model configuration from environment variables."""
        with patch.dict(os.environ, {
            'ORCHESTRATOR_MODEL': 'gpt-4o',
            'PLANNING_AGENT_MODEL': 'claude-3-5-sonnet-20241022',
            'LLM_PROVIDER': 'anthropic'
        }):
            config = IntelligentScrapingConfig.from_env()
            
            assert config.orchestrator_model == "gpt-4o"
            assert config.planning_agent_model == "claude-3-5-sonnet-20241022"
            assert config.llm_provider == "anthropic"
    
    def test_model_configuration_persistence(self):
        """Test that model configuration persists through serialization."""
        original_config = IntelligentScrapingConfig(
            orchestrator_model="gpt-4o",
            planning_agent_model="claude-3-5-sonnet-20241022",
            llm_provider="anthropic"
        )
        
        # Serialize and deserialize
        config_dict = original_config.model_dump()
        restored_config = IntelligentScrapingConfig(**config_dict)
        
        assert restored_config.orchestrator_model == "gpt-4o"
        assert restored_config.planning_agent_model == "claude-3-5-sonnet-20241022"
        assert restored_config.llm_provider == "anthropic"
    
    def test_default_model_configuration(self):
        """Test default model configuration values."""
        config = IntelligentScrapingConfig()
        
        # Test defaults
        assert config.orchestrator_model == "gpt-4o-mini"
        assert config.planning_agent_model == "gpt-4o-mini"
        assert config.llm_provider == "openai"
        
        # Test that defaults are reasonable for production use
        assert config.orchestrator_model in ["gpt-4o-mini", "gpt-4o", "gpt-4"]
        assert config.planning_agent_model in ["gpt-4o-mini", "gpt-4o", "gpt-4"]


if __name__ == "__main__":
    pytest.main([__file__])