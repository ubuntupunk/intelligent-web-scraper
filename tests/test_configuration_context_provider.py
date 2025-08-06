"""
Unit tests for ConfigurationContextProvider.

Tests the configuration context provider functionality including
configuration validation, environment variable handling, and profile management.
"""

import pytest
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, List

from intelligent_web_scraper.config import IntelligentScrapingConfig
from intelligent_web_scraper.context_providers.configuration import (
    ConfigurationContextProvider,
    ConfigurationValidationResult,
    EnvironmentInfo,
    ConfigurationProfile
)


class TestConfigurationValidationResult:
    """Test ConfigurationValidationResult functionality."""
    
    def test_validation_result_creation(self):
        """Test creating validation result."""
        result = ConfigurationValidationResult()
        
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.recommendations == []
        assert isinstance(result.validated_at, datetime)
    
    def test_add_error(self):
        """Test adding validation errors."""
        result = ConfigurationValidationResult()
        
        result.add_error("Test error 1")
        result.add_error("Test error 2")
        
        assert result.is_valid is False
        assert len(result.errors) == 2
        assert "Test error 1" in result.errors
        assert "Test error 2" in result.errors
    
    def test_add_warning(self):
        """Test adding validation warnings."""
        result = ConfigurationValidationResult()
        
        result.add_warning("Test warning")
        
        assert result.is_valid is True  # Warnings don't affect validity
        assert len(result.warnings) == 1
        assert "Test warning" in result.warnings
    
    def test_add_recommendation(self):
        """Test adding recommendations."""
        result = ConfigurationValidationResult()
        
        result.add_recommendation("Test recommendation")
        
        assert len(result.recommendations) == 1
        assert "Test recommendation" in result.recommendations
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = ConfigurationValidationResult()
        result.add_error("Error message")
        result.add_warning("Warning message")
        result.add_recommendation("Recommendation message")
        
        result_dict = result.to_dict()
        
        assert result_dict["is_valid"] is False
        assert result_dict["error_count"] == 1
        assert result_dict["warning_count"] == 1
        assert "Error message" in result_dict["errors"]
        assert "Warning message" in result_dict["warnings"]
        assert "Recommendation message" in result_dict["recommendations"]
        assert "validated_at" in result_dict


class TestEnvironmentInfo:
    """Test EnvironmentInfo functionality."""
    
    def test_environment_info_creation(self):
        """Test creating environment info."""
        env_info = EnvironmentInfo()
        
        assert env_info.environment_variables == {}
        assert env_info.system_info == {}
        assert env_info.paths_info == {}
        assert isinstance(env_info.detected_at, datetime)
    
    def test_add_env_var(self):
        """Test adding environment variables."""
        env_info = EnvironmentInfo()
        
        env_info.add_env_var("TEST_VAR", "test_value")
        env_info.add_env_var("SECRET_VAR", "secret_value", is_sensitive=True)
        
        assert env_info.environment_variables["TEST_VAR"] == "test_value"
        assert env_info.environment_variables["SECRET_VAR"] == "***MASKED***"
    
    def test_add_system_info(self):
        """Test adding system information."""
        env_info = EnvironmentInfo()
        
        env_info.add_system_info("python_version", "3.11.0")
        env_info.add_system_info("platform", "linux")
        
        assert env_info.system_info["python_version"] == "3.11.0"
        assert env_info.system_info["platform"] == "linux"
    
    def test_add_path_info(self, tmp_path):
        """Test adding path information."""
        env_info = EnvironmentInfo()
        
        # Test with existing directory
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        env_info.add_path_info("test_path", str(test_dir))
        
        path_info = env_info.paths_info["test_path"]
        assert path_info["path"] == str(test_dir)
        assert path_info["exists"] is True
        assert path_info["is_directory"] is True
        assert path_info["is_writable"] is True
        assert "absolute_path" in path_info
    
    def test_add_path_info_nonexistent(self):
        """Test adding path info for non-existent path."""
        env_info = EnvironmentInfo()
        
        env_info.add_path_info("nonexistent", "/nonexistent/path")
        
        path_info = env_info.paths_info["nonexistent"]
        assert path_info["exists"] is False
        assert path_info["is_directory"] is None
    
    def test_to_dict(self):
        """Test converting environment info to dictionary."""
        env_info = EnvironmentInfo()
        env_info.add_env_var("TEST", "value")
        env_info.add_system_info("os", "linux")
        
        result = env_info.to_dict()
        
        assert "environment_variables" in result
        assert "system_info" in result
        assert "paths_info" in result
        assert "detected_at" in result
        assert result["environment_variables"]["TEST"] == "value"
        assert result["system_info"]["os"] == "linux"


class TestConfigurationProfile:
    """Test ConfigurationProfile functionality."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return IntelligentScrapingConfig(
            orchestrator_model="gpt-4",
            planning_agent_model="gpt-4",
            default_quality_threshold=75.0,
            max_concurrent_requests=10,
            request_delay=1.5,
            results_directory="./test_results"
        )
    
    def test_profile_creation(self, sample_config):
        """Test creating configuration profile."""
        profile = ConfigurationProfile("test_profile", sample_config, "Test profile description")
        
        assert profile.name == "test_profile"
        assert profile.config == sample_config
        assert profile.description == "Test profile description"
        assert isinstance(profile.created_at, datetime)
        assert profile.last_validated is None
        assert profile.validation_result is None
        assert profile.usage_count == 0
        assert profile.performance_metrics == {}
    
    def test_profile_validation_success(self, sample_config, tmp_path):
        """Test successful profile validation."""
        # Create results directory
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        
        config = sample_config.model_copy()
        config.results_directory = str(results_dir)
        
        profile = ConfigurationProfile("test", config)
        result = profile.validate()
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert profile.validation_result == result
        assert profile.last_validated is not None
    
    def test_profile_validation_errors(self, sample_config):
        """Test profile validation with errors."""
        # Create config with invalid values
        config = sample_config.model_copy()
        config.orchestrator_model = ""  # Invalid
        config.default_quality_threshold = 150.0  # Invalid range
        config.max_concurrent_requests = -5  # Invalid
        config.results_directory = "/nonexistent/invalid/path"  # Invalid path
        
        profile = ConfigurationProfile("test", config)
        result = profile.validate()
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("orchestrator model cannot be empty" in error.lower() for error in result.errors)
        assert any("quality threshold must be between" in error.lower() for error in result.errors)
        assert any("max concurrent requests must be positive" in error.lower() for error in result.errors)
    
    def test_profile_validation_warnings(self, sample_config, tmp_path):
        """Test profile validation with warnings."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        
        # Create config with values that trigger warnings
        config = sample_config.model_copy()
        config.default_quality_threshold = 25.0  # Very low
        config.max_concurrent_requests = 25  # High
        config.request_delay = 0.1  # Very low
        config.max_instances = 15  # High
        config.results_directory = str(results_dir)
        
        profile = ConfigurationProfile("test", config)
        result = profile.validate()
        
        assert result.is_valid is True  # Warnings don't make it invalid
        assert len(result.warnings) > 0
        assert any("very low quality threshold" in warning.lower() for warning in result.warnings)
        assert any("high concurrent request count" in warning.lower() for warning in result.warnings)
    
    def test_profile_validation_recommendations(self, sample_config, tmp_path):
        """Test profile validation recommendations."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        
        config = sample_config.model_copy()
        config.default_quality_threshold = 85.0  # High
        config.enable_monitoring = False
        config.enable_rate_limiting = False
        config.results_directory = str(results_dir)
        
        profile = ConfigurationProfile("test", config)
        result = profile.validate()
        
        assert len(result.recommendations) > 0
        assert any("consider lowering quality threshold" in rec.lower() for rec in result.recommendations)
        assert any("enable monitoring" in rec.lower() for rec in result.recommendations)
        assert any("enable rate limiting" in rec.lower() for rec in result.recommendations)
    
    def test_increment_usage(self, sample_config):
        """Test incrementing usage counter."""
        profile = ConfigurationProfile("test", sample_config)
        
        assert profile.usage_count == 0
        
        profile.increment_usage()
        assert profile.usage_count == 1
        
        profile.increment_usage()
        assert profile.usage_count == 2
    
    def test_update_performance_metrics(self, sample_config):
        """Test updating performance metrics."""
        profile = ConfigurationProfile("test", sample_config)
        
        metrics = {
            "avg_response_time": 1.5,
            "success_rate": 95.0,
            "throughput": 10.5
        }
        
        profile.update_performance_metrics(metrics)
        
        assert profile.performance_metrics["avg_response_time"] == 1.5
        assert profile.performance_metrics["success_rate"] == 95.0
        assert profile.performance_metrics["throughput"] == 10.5
        
        # Update with additional metrics
        new_metrics = {"error_rate": 2.0}
        profile.update_performance_metrics(new_metrics)
        
        assert profile.performance_metrics["error_rate"] == 2.0
        assert profile.performance_metrics["success_rate"] == 95.0  # Should still be there
    
    def test_to_dict(self, sample_config):
        """Test converting profile to dictionary."""
        profile = ConfigurationProfile("test", sample_config, "Test description")
        profile.increment_usage()
        profile.update_performance_metrics({"metric": 1.0})
        profile.validate()
        
        result = profile.to_dict()
        
        assert result["name"] == "test"
        assert result["description"] == "Test description"
        assert "config" in result
        assert "created_at" in result
        assert "last_validated" in result
        assert "validation_result" in result
        assert result["usage_count"] == 1
        assert result["performance_metrics"]["metric"] == 1.0


class TestConfigurationContextProvider:
    """Test ConfigurationContextProvider functionality."""
    
    @pytest.fixture
    def context_provider(self):
        """Create a context provider for testing."""
        with patch.dict(os.environ, {}, clear=True):
            return ConfigurationContextProvider()
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return IntelligentScrapingConfig(
            orchestrator_model="gpt-4",
            planning_agent_model="gpt-4",
            default_quality_threshold=80.0,
            max_concurrent_requests=8,
            request_delay=1.0,
            results_directory="./results"
        )
    
    def test_context_provider_initialization(self, context_provider):
        """Test context provider initialization."""
        assert context_provider.current_config is not None  # Should load default
        assert len(context_provider.config_profiles) >= 1  # Should have default profile
        assert isinstance(context_provider.environment_info, EnvironmentInfo)
        # Validation history may have entries from default profile creation
        assert isinstance(context_provider.validation_history, list)
        # Config changes log may have entries from default profile creation
        assert isinstance(context_provider.config_changes_log, list)
        assert context_provider.default_profile_name == "default"
        assert context_provider.active_profile_name is not None
    
    def test_set_configuration(self, context_provider, sample_config):
        """Test setting configuration."""
        context_provider.set_configuration(sample_config, "test_profile")
        
        assert context_provider.current_config == sample_config
        assert context_provider.active_profile_name == "test_profile"
        assert "test_profile" in context_provider.config_profiles
        assert len(context_provider.config_changes_log) > 0
    
    def test_load_configuration_from_env(self, context_provider):
        """Test loading configuration from environment."""
        with patch.dict(os.environ, {
            "ORCHESTRATOR_MODEL": "gpt-3.5-turbo",
            "QUALITY_THRESHOLD": "60.0",
            "MAX_CONCURRENT_REQUESTS": "15"
        }):
            config = context_provider.load_configuration_from_env("env_profile")
            
            assert config.orchestrator_model == "gpt-3.5-turbo"
            assert config.default_quality_threshold == 60.0
            assert config.max_concurrent_requests == 15
            assert context_provider.active_profile_name == "env_profile"
    
    def test_create_profile(self, context_provider, sample_config):
        """Test creating configuration profile."""
        profile = context_provider.create_profile("new_profile", sample_config, "New test profile")
        
        assert profile.name == "new_profile"
        assert profile.description == "New test profile"
        assert "new_profile" in context_provider.config_profiles
        assert len(context_provider.validation_history) > 0
        assert len(context_provider.config_changes_log) > 0
    
    def test_activate_profile(self, context_provider, sample_config):
        """Test activating configuration profile."""
        # Create profile first
        context_provider.create_profile("test_profile", sample_config)
        
        # Activate it
        result = context_provider.activate_profile("test_profile")
        
        assert result is True
        assert context_provider.active_profile_name == "test_profile"
        assert context_provider.current_config == sample_config
        assert context_provider.config_profiles["test_profile"].usage_count == 1
    
    def test_activate_nonexistent_profile(self, context_provider):
        """Test activating non-existent profile."""
        result = context_provider.activate_profile("nonexistent")
        
        assert result is False
        assert context_provider.active_profile_name != "nonexistent"
    
    def test_validate_current_configuration(self, context_provider, sample_config, tmp_path):
        """Test validating current configuration."""
        # Set up valid configuration
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        
        config = sample_config.model_copy()
        config.results_directory = str(results_dir)
        
        context_provider.set_configuration(config, "valid_profile")
        
        result = context_provider.validate_current_configuration()
        
        assert isinstance(result, ConfigurationValidationResult)
        assert result.is_valid is True
        assert len(context_provider.validation_history) > 0
    
    def test_validate_no_configuration(self):
        """Test validating when no configuration is set."""
        provider = ConfigurationContextProvider()
        provider.current_config = None
        
        result = provider.validate_current_configuration()
        
        assert result.is_valid is False
        assert any("no configuration is currently set" in error.lower() for error in result.errors)
    
    def test_get_configuration_recommendations(self, context_provider, sample_config):
        """Test getting configuration recommendations."""
        context_provider.set_configuration(sample_config, "test")
        
        recommendations = context_provider.get_configuration_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should have recommendations about environment variables, monitoring, etc.
    
    def test_get_configuration_recommendations_no_config(self):
        """Test getting recommendations with no configuration."""
        provider = ConfigurationContextProvider()
        provider.current_config = None
        
        recommendations = provider.get_configuration_recommendations()
        
        assert len(recommendations) == 1
        assert "set up a configuration" in recommendations[0].lower()
    
    def test_get_info_no_configuration(self):
        """Test getting context info with no configuration."""
        provider = ConfigurationContextProvider()
        provider.current_config = None
        
        info = provider.get_info()
        
        assert "No active configuration" in info
        assert "Load a configuration profile" in info
    
    def test_get_info_with_configuration(self, context_provider, sample_config, tmp_path):
        """Test getting context info with configuration."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        
        config = sample_config.model_copy()
        config.results_directory = str(results_dir)
        
        context_provider.set_configuration(config, "test_profile")
        
        info = context_provider.get_info()
        
        # Check header
        assert "Configuration Context: test_profile" in info
        assert "Active configuration loaded" in info
        
        # Check configuration sections
        assert "Current Configuration Summary" in info
        assert "Agent Models:" in info
        assert "gpt-4" in info
        assert "Quality Threshold: 80.0%" in info
        assert "Max Concurrent Requests: 8" in info
        
        # Check validation section
        assert "Configuration Validation" in info
        assert "Configuration is valid" in info or "Configuration has issues" in info
        
        # Check environment section
        assert "Environment Information" in info
        assert "Results Directory:" in info
    
    def test_get_configuration_summary(self, context_provider, sample_config):
        """Test getting configuration summary."""
        context_provider.set_configuration(sample_config, "test")
        
        summary = context_provider.get_configuration_summary()
        
        assert summary["status"] == "active"
        assert summary["active_profile"] == "test"
        assert "configuration" in summary
        assert "validation" in summary
        assert "profiles_count" in summary
        assert "environment_info" in summary
        assert "recommendations" in summary
        assert "last_updated" in summary
    
    def test_get_configuration_summary_no_config(self):
        """Test getting summary with no configuration."""
        provider = ConfigurationContextProvider()
        provider.current_config = None
        
        summary = provider.get_configuration_summary()
        
        assert summary["status"] == "no_configuration"
        assert summary["active_profile"] is None
    
    def test_export_configuration(self, context_provider, sample_config):
        """Test exporting configuration in different formats."""
        context_provider.set_configuration(sample_config, "test")
        
        # Test dict format
        dict_export = context_provider.export_configuration("dict")
        assert isinstance(dict_export, dict)
        assert dict_export["orchestrator_model"] == "gpt-4"
        
        # Test JSON format
        json_export = context_provider.export_configuration("json")
        assert isinstance(json_export, str)
        assert "gpt-4" in json_export
        
        # Test env format
        env_export = context_provider.export_configuration("env")
        assert isinstance(env_export, str)
        assert "ORCHESTRATOR_MODEL=gpt-4" in env_export
    
    def test_export_configuration_invalid_format(self, context_provider, sample_config):
        """Test exporting with invalid format."""
        context_provider.set_configuration(sample_config, "test")
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            context_provider.export_configuration("invalid")
    
    def test_export_configuration_no_config(self):
        """Test exporting with no configuration."""
        provider = ConfigurationContextProvider()
        provider.current_config = None
        
        dict_export = provider.export_configuration("dict")
        assert dict_export == {}
        
        json_export = provider.export_configuration("json")
        assert json_export == ""
    
    def test_get_validation_history(self, context_provider, sample_config):
        """Test getting validation history."""
        context_provider.set_configuration(sample_config, "test")
        context_provider.validate_current_configuration()
        context_provider.validate_current_configuration()
        
        history = context_provider.get_validation_history()
        
        assert isinstance(history, list)
        assert len(history) >= 2
        assert all("is_valid" in item for item in history)
    
    def test_get_config_changes_log(self, context_provider, sample_config):
        """Test getting configuration changes log."""
        context_provider.set_configuration(sample_config, "test1")
        context_provider.create_profile("test2", sample_config)
        
        changes_log = context_provider.get_config_changes_log()
        
        assert isinstance(changes_log, list)
        assert len(changes_log) >= 2
        assert all("timestamp" in item for item in changes_log)
        assert all("action" in item for item in changes_log)
    
    def test_clear_validation_history(self, context_provider, sample_config):
        """Test clearing validation history."""
        context_provider.set_configuration(sample_config, "test")
        context_provider.validate_current_configuration()
        
        assert len(context_provider.validation_history) > 0
        
        context_provider.clear_validation_history()
        
        assert len(context_provider.validation_history) == 0
    
    def test_clear_config_changes_log(self, context_provider, sample_config):
        """Test clearing configuration changes log."""
        context_provider.set_configuration(sample_config, "test")
        
        assert len(context_provider.config_changes_log) > 0
        
        context_provider.clear_config_changes_log()
        
        assert len(context_provider.config_changes_log) == 0
    
    def test_update_profile_performance(self, context_provider, sample_config):
        """Test updating profile performance metrics."""
        context_provider.create_profile("test", sample_config)
        
        metrics = {"response_time": 1.5, "success_rate": 95.0}
        result = context_provider.update_profile_performance("test", metrics)
        
        assert result is True
        assert context_provider.config_profiles["test"].performance_metrics["response_time"] == 1.5
        
        # Test with non-existent profile
        result = context_provider.update_profile_performance("nonexistent", metrics)
        assert result is False
    
    def test_get_profile_comparison(self, context_provider, sample_config):
        """Test comparing configuration profiles."""
        # Create two different profiles
        config1 = sample_config.model_copy()
        config1.default_quality_threshold = 70.0
        
        config2 = sample_config.model_copy()
        config2.default_quality_threshold = 80.0
        config2.max_concurrent_requests = 10
        
        context_provider.create_profile("profile1", config1)
        context_provider.create_profile("profile2", config2)
        
        comparison = context_provider.get_profile_comparison("profile1", "profile2")
        
        assert comparison["profile1"] == "profile1"
        assert comparison["profile2"] == "profile2"
        assert "differences" in comparison
        assert comparison["differences_count"] > 0
        assert "performance_comparison" in comparison
        
        # Check that differences are detected
        differences = comparison["differences"]
        quality_diff = next((d for d in differences if d["field"] == "default_quality_threshold"), None)
        assert quality_diff is not None
        assert quality_diff["profile1_value"] == 70.0
        assert quality_diff["profile2_value"] == 80.0
    
    def test_get_profile_comparison_nonexistent(self, context_provider):
        """Test comparing non-existent profiles."""
        comparison = context_provider.get_profile_comparison("nonexistent1", "nonexistent2")
        
        assert "error" in comparison
        assert "not found" in comparison["error"].lower()
    
    @patch.dict(os.environ, {
        "ORCHESTRATOR_MODEL": "gpt-4",
        "QUALITY_THRESHOLD": "75.0",
        "RESULTS_DIRECTORY": "/tmp/test"
    })
    def test_environment_detection(self):
        """Test environment variable detection."""
        provider = ConfigurationContextProvider()
        
        env_info = provider.environment_info
        
        assert "ORCHESTRATOR_MODEL" in env_info.environment_variables
        assert "QUALITY_THRESHOLD" in env_info.environment_variables
        assert "RESULTS_DIRECTORY" in env_info.environment_variables
        assert env_info.environment_variables["ORCHESTRATOR_MODEL"] == "gpt-4"
    
    def test_config_changes_log_limit(self, context_provider, sample_config):
        """Test that config changes log is limited in size."""
        # Create many profile changes to test limit
        for i in range(60):  # More than the 50 limit
            config = sample_config.model_copy()
            config.default_quality_threshold = 50.0 + i
            context_provider.create_profile(f"profile_{i}", config)
        
        # Should be limited to 50 entries
        assert len(context_provider.config_changes_log) == 50
    
    def test_validation_history_limit(self, context_provider, sample_config):
        """Test that validation history is limited when retrieved."""
        context_provider.set_configuration(sample_config, "test")
        
        # Create many validations
        for _ in range(15):
            context_provider.validate_current_configuration()
        
        # Should return only last 10
        history = context_provider.get_validation_history()
        assert len(history) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])