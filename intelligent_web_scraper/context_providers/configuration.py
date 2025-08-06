"""
Configuration Context Provider.

This module provides system configuration context to agents,
demonstrating configuration validation, environment variable handling,
and dynamic configuration management patterns in atomic-agents.
"""

import os
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase

from ..config import IntelligentScrapingConfig


class ConfigurationValidationResult:
    """Results of configuration validation."""
    
    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.recommendations: List[str] = []
        self.validated_at = datetime.utcnow()
    
    def add_error(self, error: str) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)
    
    def add_recommendation(self, recommendation: str) -> None:
        """Add a configuration recommendation."""
        self.recommendations.append(recommendation)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "validated_at": self.validated_at.isoformat(),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }


class EnvironmentInfo:
    """Information about the environment and system configuration."""
    
    def __init__(self):
        self.environment_variables: Dict[str, str] = {}
        self.system_info: Dict[str, Any] = {}
        self.paths_info: Dict[str, Dict[str, Any]] = {}
        self.detected_at = datetime.utcnow()
    
    def add_env_var(self, key: str, value: str, is_sensitive: bool = False) -> None:
        """Add environment variable information."""
        if is_sensitive:
            # Mask sensitive values
            self.environment_variables[key] = "***MASKED***"
        else:
            self.environment_variables[key] = value
    
    def add_system_info(self, key: str, value: Any) -> None:
        """Add system information."""
        self.system_info[key] = value
    
    def add_path_info(self, path_name: str, path_value: str) -> None:
        """Add path information with validation."""
        path_obj = Path(path_value)
        self.paths_info[path_name] = {
            "path": path_value,
            "exists": path_obj.exists(),
            "is_directory": path_obj.is_dir() if path_obj.exists() else None,
            "is_writable": os.access(path_obj.parent if not path_obj.exists() else path_obj, os.W_OK),
            "absolute_path": str(path_obj.absolute())
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "environment_variables": self.environment_variables,
            "system_info": self.system_info,
            "paths_info": self.paths_info,
            "detected_at": self.detected_at.isoformat()
        }


class ConfigurationProfile:
    """Represents a configuration profile with validation and metadata."""
    
    def __init__(self, name: str, config: IntelligentScrapingConfig, description: str = ""):
        self.name = name
        self.config = config
        self.description = description
        self.created_at = datetime.utcnow()
        self.last_validated = None
        self.validation_result: Optional[ConfigurationValidationResult] = None
        self.usage_count = 0
        self.performance_metrics: Dict[str, float] = {}
    
    def validate(self) -> ConfigurationValidationResult:
        """Validate the configuration profile."""
        result = ConfigurationValidationResult()
        
        # Validate agent configuration
        if not self.config.orchestrator_model:
            result.add_error("Orchestrator model cannot be empty")
        
        if not self.config.planning_agent_model:
            result.add_error("Planning agent model cannot be empty")
        
        # Validate scraping configuration
        if self.config.default_quality_threshold < 0 or self.config.default_quality_threshold > 100:
            result.add_error(f"Quality threshold must be between 0 and 100, got {self.config.default_quality_threshold}")
        
        if self.config.max_concurrent_requests <= 0:
            result.add_error(f"Max concurrent requests must be positive, got {self.config.max_concurrent_requests}")
        
        if self.config.request_delay < 0:
            result.add_error(f"Request delay cannot be negative, got {self.config.request_delay}")
        
        # Validate concurrency configuration
        if self.config.max_instances <= 0:
            result.add_error(f"Max instances must be positive, got {self.config.max_instances}")
        
        if self.config.max_workers <= 0:
            result.add_error(f"Max workers must be positive, got {self.config.max_workers}")
        
        if self.config.max_async_tasks <= 0:
            result.add_error(f"Max async tasks must be positive, got {self.config.max_async_tasks}")
        
        # Validate paths
        results_path = Path(self.config.results_directory)
        if not results_path.parent.exists():
            result.add_error(f"Results directory parent does not exist: {results_path.parent}")
        elif not os.access(results_path.parent, os.W_OK):
            result.add_error(f"Results directory is not writable: {results_path.parent}")
        
        # Add warnings for potentially problematic configurations
        if self.config.default_quality_threshold < 30:
            result.add_warning("Very low quality threshold may result in poor data quality")
        
        if self.config.max_concurrent_requests > 20:
            result.add_warning("High concurrent request count may overwhelm target servers")
        
        if self.config.request_delay < 0.5:
            result.add_warning("Very low request delay may trigger rate limiting")
        
        if self.config.max_instances > 10:
            result.add_warning("High instance count may consume excessive resources")
        
        # Add recommendations
        if self.config.default_quality_threshold > 80:
            result.add_recommendation("Consider lowering quality threshold if extraction yields are low")
        
        if not self.config.enable_monitoring:
            result.add_recommendation("Enable monitoring for better observability")
        
        if not self.config.enable_rate_limiting:
            result.add_recommendation("Enable rate limiting to be respectful to target servers")
        
        self.validation_result = result
        self.last_validated = datetime.utcnow()
        return result
    
    def increment_usage(self) -> None:
        """Increment usage counter."""
        self.usage_count += 1
    
    def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Update performance metrics for this profile."""
        self.performance_metrics.update(metrics)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "config": self.config.model_dump(),
            "created_at": self.created_at.isoformat(),
            "last_validated": self.last_validated.isoformat() if self.last_validated else None,
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "usage_count": self.usage_count,
            "performance_metrics": self.performance_metrics
        }


class ConfigurationContextProvider(SystemPromptContextProviderBase):
    """
    Provides system configuration context to agents.
    
    This context provider demonstrates advanced configuration management patterns
    including validation, environment variable handling, profile management,
    and dynamic configuration recommendations for atomic-agents applications.
    """
    
    def __init__(self, title: str = "Configuration Context"):
        super().__init__(title=title)
        self.current_config: Optional[IntelligentScrapingConfig] = None
        self.config_profiles: Dict[str, ConfigurationProfile] = {}
        self.environment_info = EnvironmentInfo()
        self.validation_history: List[ConfigurationValidationResult] = []
        self.config_changes_log: List[Dict[str, Any]] = []
        self.default_profile_name = "default"
        self.active_profile_name: Optional[str] = None
        
        # Initialize with current environment
        self._detect_environment_info()
        self._load_default_configuration()
    
    def set_configuration(self, config: IntelligentScrapingConfig, profile_name: str = "current") -> None:
        """
        Set the current configuration.
        
        Args:
            config: Configuration to set as current
            profile_name: Name for this configuration profile
        """
        self.current_config = config
        
        # Create or update profile
        profile = ConfigurationProfile(profile_name, config, f"Configuration profile: {profile_name}")
        profile.validate()
        self.config_profiles[profile_name] = profile
        self.active_profile_name = profile_name
        
        # Log configuration change
        self._log_config_change("set_configuration", profile_name, config)
    
    def load_configuration_from_env(self, profile_name: str = "environment") -> IntelligentScrapingConfig:
        """
        Load configuration from environment variables.
        
        Args:
            profile_name: Name for the environment-based profile
            
        Returns:
            Configuration loaded from environment
        """
        config = IntelligentScrapingConfig.from_env()
        self.set_configuration(config, profile_name)
        return config
    
    def create_profile(self, name: str, config: IntelligentScrapingConfig, description: str = "") -> ConfigurationProfile:
        """
        Create a new configuration profile.
        
        Args:
            name: Profile name
            config: Configuration for the profile
            description: Profile description
            
        Returns:
            Created configuration profile
        """
        profile = ConfigurationProfile(name, config, description)
        validation_result = profile.validate()
        
        self.config_profiles[name] = profile
        self.validation_history.append(validation_result)
        
        self._log_config_change("create_profile", name, config)
        
        return profile
    
    def activate_profile(self, profile_name: str) -> bool:
        """
        Activate a configuration profile.
        
        Args:
            profile_name: Name of profile to activate
            
        Returns:
            True if profile was activated, False if not found
        """
        if profile_name not in self.config_profiles:
            return False
        
        profile = self.config_profiles[profile_name]
        self.current_config = profile.config
        self.active_profile_name = profile_name
        profile.increment_usage()
        
        self._log_config_change("activate_profile", profile_name, profile.config)
        
        return True
    
    def validate_current_configuration(self) -> ConfigurationValidationResult:
        """
        Validate the current configuration.
        
        Returns:
            Validation result
        """
        if not self.current_config:
            result = ConfigurationValidationResult()
            result.add_error("No configuration is currently set")
            return result
        
        if self.active_profile_name and self.active_profile_name in self.config_profiles:
            profile = self.config_profiles[self.active_profile_name]
            result = profile.validate()
        else:
            # Create temporary profile for validation
            temp_profile = ConfigurationProfile("temp", self.current_config)
            result = temp_profile.validate()
        
        self.validation_history.append(result)
        return result
    
    def get_configuration_recommendations(self) -> List[str]:
        """
        Get configuration recommendations based on current setup.
        
        Returns:
            List of configuration recommendations
        """
        recommendations = []
        
        if not self.current_config:
            recommendations.append("Set up a configuration to get started")
            return recommendations
        
        # Environment-based recommendations
        if "ORCHESTRATOR_MODEL" not in os.environ:
            recommendations.append("Set ORCHESTRATOR_MODEL environment variable for consistent model selection")
        
        if "QUALITY_THRESHOLD" not in os.environ:
            recommendations.append("Set QUALITY_THRESHOLD environment variable to standardize quality requirements")
        
        # Performance recommendations
        if self.current_config.max_concurrent_requests > self.current_config.max_instances * 2:
            recommendations.append("Consider increasing max_instances to handle concurrent requests efficiently")
        
        if self.current_config.max_workers < self.current_config.max_instances:
            recommendations.append("Consider increasing max_workers to match or exceed max_instances")
        
        # Resource recommendations
        results_path = Path(self.current_config.results_directory)
        if not results_path.exists():
            recommendations.append(f"Create results directory: {results_path}")
        
        # Monitoring recommendations
        if not self.current_config.enable_monitoring:
            recommendations.append("Enable monitoring for better system observability")
        
        # Compliance recommendations
        if not self.current_config.respect_robots_txt:
            recommendations.append("Enable robots.txt compliance for ethical scraping")
        
        if not self.current_config.enable_rate_limiting:
            recommendations.append("Enable rate limiting to avoid overwhelming target servers")
        
        return recommendations
    
    def get_info(self) -> str:
        """
        Return formatted configuration information for agent context.
        
        This method demonstrates how to format complex configuration data
        into clear, actionable context for AI agents.
        """
        context_parts = []
        
        # Header with current configuration status
        if self.current_config and self.active_profile_name:
            context_parts.append(f"## Configuration Context: {self.active_profile_name}")
            context_parts.append(f"**Status:** Active configuration loaded")
        else:
            context_parts.append("## Configuration Context")
            context_parts.append("**Status:** No active configuration")
            return "\n".join(context_parts + ["", "### Recommendation", "- Load a configuration profile to get started"])
        
        context_parts.append("")
        
        # Current configuration summary
        config = self.current_config
        context_parts.append("### Current Configuration Summary")
        context_parts.append(f"**Agent Models:**")
        context_parts.append(f"  - Orchestrator: {config.orchestrator_model}")
        context_parts.append(f"  - Planning Agent: {config.planning_agent_model}")
        context_parts.append("")
        
        context_parts.append(f"**Scraping Settings:**")
        context_parts.append(f"  - Quality Threshold: {config.default_quality_threshold}%")
        context_parts.append(f"  - Max Concurrent Requests: {config.max_concurrent_requests}")
        context_parts.append(f"  - Request Delay: {config.request_delay}s")
        context_parts.append(f"  - Export Format: {config.default_export_format}")
        context_parts.append("")
        
        context_parts.append(f"**Concurrency Settings:**")
        context_parts.append(f"  - Max Instances: {config.max_instances}")
        context_parts.append(f"  - Max Workers: {config.max_workers}")
        context_parts.append(f"  - Max Async Tasks: {config.max_async_tasks}")
        context_parts.append("")
        
        context_parts.append(f"**Compliance & Monitoring:**")
        context_parts.append(f"  - Respect Robots.txt: {'Yes' if config.respect_robots_txt else 'No'}")
        context_parts.append(f"  - Rate Limiting: {'Enabled' if config.enable_rate_limiting else 'Disabled'}")
        context_parts.append(f"  - Monitoring: {'Enabled' if config.enable_monitoring else 'Disabled'}")
        context_parts.append("")
        
        # Validation status
        validation_result = self.validate_current_configuration()
        context_parts.append("### Configuration Validation")
        if validation_result.is_valid:
            context_parts.append("**Status:** ✅ Configuration is valid")
        else:
            context_parts.append("**Status:** ❌ Configuration has issues")
            context_parts.append(f"**Errors:** {len(validation_result.errors)}")
            for error in validation_result.errors[:3]:  # Show first 3 errors
                context_parts.append(f"  - {error}")
        
        if validation_result.warnings:
            context_parts.append(f"**Warnings:** {len(validation_result.warnings)}")
            for warning in validation_result.warnings[:2]:  # Show first 2 warnings
                context_parts.append(f"  - {warning}")
        
        context_parts.append("")
        
        # Environment information
        context_parts.append("### Environment Information")
        context_parts.append(f"**Results Directory:** {config.results_directory}")
        
        # Check if results directory exists and is writable
        results_path = Path(config.results_directory)
        if results_path.exists():
            context_parts.append(f"  - Status: ✅ Exists and accessible")
        else:
            context_parts.append(f"  - Status: ⚠️ Directory does not exist")
        
        # Show relevant environment variables
        relevant_env_vars = [
            "ORCHESTRATOR_MODEL", "PLANNING_AGENT_MODEL", "QUALITY_THRESHOLD",
            "MAX_CONCURRENT_REQUESTS", "RESULTS_DIRECTORY"
        ]
        
        set_env_vars = []
        for var in relevant_env_vars:
            if var in os.environ:
                set_env_vars.append(f"{var}={os.environ[var]}")
        
        if set_env_vars:
            context_parts.append("**Environment Variables Set:**")
            for env_var in set_env_vars[:3]:  # Show first 3
                context_parts.append(f"  - {env_var}")
        
        context_parts.append("")
        
        # Available profiles
        if len(self.config_profiles) > 1:
            context_parts.append("### Available Configuration Profiles")
            for name, profile in list(self.config_profiles.items())[:5]:  # Show first 5
                status = "🔴" if profile.validation_result and not profile.validation_result.is_valid else "🟢"
                usage = f"(used {profile.usage_count} times)" if profile.usage_count > 0 else ""
                context_parts.append(f"- **{name}** {status} {usage}")
                if profile.description:
                    context_parts.append(f"  {profile.description}")
            context_parts.append("")
        
        # Recommendations
        recommendations = self.get_configuration_recommendations()
        if recommendations:
            context_parts.append("### Configuration Recommendations")
            for rec in recommendations[:5]:  # Show first 5 recommendations
                context_parts.append(f"- {rec}")
            context_parts.append("")
        
        # Performance insights
        if self.active_profile_name and self.active_profile_name in self.config_profiles:
            profile = self.config_profiles[self.active_profile_name]
            if profile.performance_metrics:
                context_parts.append("### Performance Insights")
                for metric, value in list(profile.performance_metrics.items())[:3]:
                    context_parts.append(f"- **{metric.replace('_', ' ').title()}:** {value}")
                context_parts.append("")
        
        # Configuration best practices
        context_parts.append("### Configuration Best Practices")
        context_parts.append("- Use environment variables for deployment-specific settings")
        context_parts.append("- Enable monitoring and rate limiting for production use")
        context_parts.append("- Set appropriate quality thresholds based on your data requirements")
        context_parts.append("- Configure concurrency settings based on your system resources")
        
        return "\n".join(context_parts)
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get a programmatic summary of current configuration.
        
        Returns:
            Dictionary containing configuration summary
        """
        if not self.current_config:
            return {"status": "no_configuration", "active_profile": None}
        
        validation_result = self.validate_current_configuration()
        
        return {
            "status": "active",
            "active_profile": self.active_profile_name,
            "configuration": self.current_config.model_dump(),
            "validation": validation_result.to_dict(),
            "profiles_count": len(self.config_profiles),
            "environment_info": self.environment_info.to_dict(),
            "recommendations": self.get_configuration_recommendations(),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def export_configuration(self, format_type: str = "dict") -> Union[Dict[str, Any], str]:
        """
        Export current configuration in specified format.
        
        Args:
            format_type: Export format ("dict", "json", "env")
            
        Returns:
            Configuration in requested format
        """
        if not self.current_config:
            return {} if format_type == "dict" else ""
        
        if format_type == "dict":
            return self.current_config.model_dump()
        elif format_type == "json":
            return self.current_config.model_dump_json(indent=2)
        elif format_type == "env":
            # Generate environment variable format
            config_dict = self.current_config.model_dump()
            env_lines = []
            for key, value in config_dict.items():
                env_key = key.upper()
                env_lines.append(f"{env_key}={value}")
            return "\n".join(env_lines)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _detect_environment_info(self) -> None:
        """Detect and collect environment information."""
        # Collect relevant environment variables
        relevant_vars = [
            "ORCHESTRATOR_MODEL", "PLANNING_AGENT_MODEL", "QUALITY_THRESHOLD",
            "MAX_CONCURRENT_REQUESTS", "REQUEST_DELAY", "EXPORT_FORMAT",
            "RESULTS_DIRECTORY", "RESPECT_ROBOTS_TXT", "ENABLE_RATE_LIMITING",
            "ENABLE_MONITORING", "MAX_INSTANCES", "MAX_WORKERS", "MAX_ASYNC_TASKS"
        ]
        
        for var in relevant_vars:
            if var in os.environ:
                self.environment_info.add_env_var(var, os.environ[var])
        
        # Collect system information
        self.environment_info.add_system_info("python_version", os.sys.version)
        self.environment_info.add_system_info("platform", os.name)
        self.environment_info.add_system_info("working_directory", os.getcwd())
        
        # Collect path information
        self.environment_info.add_path_info("current_directory", os.getcwd())
        if "RESULTS_DIRECTORY" in os.environ:
            self.environment_info.add_path_info("results_directory", os.environ["RESULTS_DIRECTORY"])
    
    def _load_default_configuration(self) -> None:
        """Load default configuration."""
        try:
            default_config = IntelligentScrapingConfig.from_env()
            self.create_profile(self.default_profile_name, default_config, "Default configuration from environment")
            self.activate_profile(self.default_profile_name)
        except Exception as e:
            # If loading from environment fails, create minimal config
            minimal_config = IntelligentScrapingConfig()
            self.create_profile(self.default_profile_name, minimal_config, "Minimal default configuration")
            self.activate_profile(self.default_profile_name)
    
    def _log_config_change(self, action: str, profile_name: str, config: IntelligentScrapingConfig) -> None:
        """Log configuration changes."""
        change_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "profile_name": profile_name,
            "config_summary": {
                "orchestrator_model": config.orchestrator_model,
                "quality_threshold": config.default_quality_threshold,
                "max_concurrent_requests": config.max_concurrent_requests,
                "enable_monitoring": config.enable_monitoring
            }
        }
        
        self.config_changes_log.append(change_entry)
        
        # Keep only last 50 changes
        if len(self.config_changes_log) > 50:
            self.config_changes_log = self.config_changes_log[-50:]
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get validation history."""
        return [result.to_dict() for result in self.validation_history[-10:]]  # Last 10 validations
    
    def get_config_changes_log(self) -> List[Dict[str, Any]]:
        """Get configuration changes log."""
        return self.config_changes_log[-20:]  # Last 20 changes
    
    def clear_validation_history(self) -> None:
        """Clear validation history."""
        self.validation_history.clear()
    
    def clear_config_changes_log(self) -> None:
        """Clear configuration changes log."""
        self.config_changes_log.clear()
    
    def update_profile_performance(self, profile_name: str, metrics: Dict[str, float]) -> bool:
        """
        Update performance metrics for a profile.
        
        Args:
            profile_name: Name of the profile to update
            metrics: Performance metrics to update
            
        Returns:
            True if profile was updated, False if not found
        """
        if profile_name not in self.config_profiles:
            return False
        
        profile = self.config_profiles[profile_name]
        profile.update_performance_metrics(metrics)
        return True
    
    def get_profile_comparison(self, profile1: str, profile2: str) -> Dict[str, Any]:
        """
        Compare two configuration profiles.
        
        Args:
            profile1: First profile name
            profile2: Second profile name
            
        Returns:
            Comparison results
        """
        if profile1 not in self.config_profiles or profile2 not in self.config_profiles:
            return {"error": "One or both profiles not found"}
        
        p1 = self.config_profiles[profile1]
        p2 = self.config_profiles[profile2]
        
        differences = []
        config1_dict = p1.config.model_dump()
        config2_dict = p2.config.model_dump()
        
        for key in config1_dict:
            if config1_dict[key] != config2_dict[key]:
                differences.append({
                    "field": key,
                    f"{profile1}_value": config1_dict[key],
                    f"{profile2}_value": config2_dict[key]
                })
        
        return {
            "profile1": profile1,
            "profile2": profile2,
            "differences": differences,
            "differences_count": len(differences),
            "performance_comparison": {
                f"{profile1}_usage": p1.usage_count,
                f"{profile2}_usage": p2.usage_count,
                f"{profile1}_metrics": p1.performance_metrics,
                f"{profile2}_metrics": p2.performance_metrics
            }
        }