"""
Configuration Context Provider.

This module provides system configuration context to agents,
demonstrating configuration management patterns in atomic-agents.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase
from ..config import IntelligentScrapingConfig


class ConfigurationValidationResult:
    """Result of configuration validation."""
    
    def __init__(self):
        self.is_valid: bool = True
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.recommendations: List[str] = []
        self.validated_at = datetime.utcnow()
    
    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)
    
    def add_error(self, message: str) -> None:
        """Add a validation error."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_recommendation(self, message: str) -> None:
        """Add a configuration recommendation."""
        self.recommendations.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "warnings": self.warnings,
            "errors": self.errors,
            "recommendations": self.recommendations,
            "validated_at": self.validated_at.isoformat()
        }


class EnvironmentInfo:
    """Information about the runtime environment."""
    
    def __init__(self):
        self.python_version: str = ""
        self.platform: str = ""
        self.available_memory_mb: float = 0.0
        self.cpu_count: int = 0
        self.disk_space_gb: float = 0.0
        self.network_available: bool = True
        self.dependencies_status: Dict[str, str] = {}
        self.environment_variables: Dict[str, str] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "available_memory_mb": self.available_memory_mb,
            "cpu_count": self.cpu_count,
            "disk_space_gb": self.disk_space_gb,
            "network_available": self.network_available,
            "dependencies_status": self.dependencies_status,
            "environment_variables": self.environment_variables
        }


class ConfigurationContextProvider(SystemPromptContextProviderBase):
    """
    Provides system configuration context to agents.
    
    This context provider demonstrates advanced patterns for exposing
    system configuration, validation results, and environment information
    to agents for informed decision-making.
    """
    
    def __init__(self, title: str = "Configuration Context"):
        super().__init__(title=title)
        self.config: Optional[IntelligentScrapingConfig] = None
        self.validation_result: Optional[ConfigurationValidationResult] = None
        self.environment_info = EnvironmentInfo()
        self.config_overrides: Dict[str, Any] = {}
        self.last_updated = datetime.utcnow()
    
    def set_configuration(self, config: IntelligentScrapingConfig) -> None:
        """
        Set the current configuration.
        
        Args:
            config: Configuration to use for context
        """
        self.config = config
        self.last_updated = datetime.utcnow()
        
        # Automatically validate the configuration
        self.validation_result = self._validate_configuration(config)
        
        # Update environment info
        self._update_environment_info()
    
    def add_config_override(self, key: str, value: Any) -> None:
        """
        Add a configuration override.
        
        Args:
            key: Configuration key to override
            value: New value for the configuration
        """
        self.config_overrides[key] = value
        self.last_updated = datetime.utcnow()
    
    def clear_overrides(self) -> None:
        """Clear all configuration overrides."""
        self.config_overrides.clear()
        self.last_updated = datetime.utcnow()
    
    def get_info(self) -> str:
        """
        Return formatted configuration information for agent context.
        
        This method demonstrates how to present configuration data
        in a clear, actionable format for AI agents.
        """
        if not self.config:
            return self._get_no_config_context()
        
        context_parts = []
        
        # Header
        context_parts.append("## System Configuration")
        context_parts.append(f"**Last Updated:** {self.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        context_parts.append("")
        
        # Configuration validation status
        if self.validation_result:
            validation = self.validation_result
            status = "✅ Valid" if validation.is_valid else "❌ Invalid"
            context_parts.append(f"**Configuration Status:** {status}")
            
            if validation.errors:
                context_parts.append("**Configuration Errors:**")
                for error in validation.errors:
                    context_parts.append(f"- ❌ {error}")
                context_parts.append("")
            
            if validation.warnings:
                context_parts.append("**Configuration Warnings:**")
                for warning in validation.warnings:
                    context_parts.append(f"- ⚠️ {warning}")
                context_parts.append("")
        
        # Core configuration
        config = self.config
        context_parts.append("### Core Settings")
        context_parts.append(f"- **Orchestrator Model:** {config.orchestrator_model}")
        context_parts.append(f"- **Planning Agent Model:** {config.planning_agent_model}")
        context_parts.append(f"- **Default Quality Threshold:** {config.default_quality_threshold}%")
        context_parts.append(f"- **Max Concurrent Requests:** {config.max_concurrent_requests}")
        context_parts.append(f"- **Request Delay:** {config.request_delay}s")
        context_parts.append("")
        
        # Output configuration
        context_parts.append("### Output Settings")
        context_parts.append(f"- **Default Export Format:** {config.default_export_format}")
        context_parts.append(f"- **Results Directory:** {config.results_directory}")
        context_parts.append("")
        
        # Compliance settings
        context_parts.append("### Compliance Settings")
        context_parts.append(f"- **Respect robots.txt:** {'Yes' if config.respect_robots_txt else 'No'}")
        context_parts.append(f"- **Rate Limiting Enabled:** {'Yes' if config.enable_rate_limiting else 'No'}")
        context_parts.append("")
        
        # Monitoring configuration
        context_parts.append("### Monitoring Settings")
        context_parts.append(f"- **Monitoring Enabled:** {'Yes' if config.enable_monitoring else 'No'}")
        context_parts.append(f"- **Monitoring Interval:** {config.monitoring_interval}s")
        context_parts.append("")
        
        # Concurrency settings
        context_parts.append("### Concurrency Settings")
        context_parts.append(f"- **Max Instances:** {config.max_instances}")
        context_parts.append(f"- **Max Workers:** {config.max_workers}")
        context_parts.append(f"- **Max Async Tasks:** {config.max_async_tasks}")
        context_parts.append("")
        
        # Configuration overrides
        if self.config_overrides:
            context_parts.append("### Active Overrides")
            for key, value in self.config_overrides.items():
                context_parts.append(f"- **{key}:** {value}")
            context_parts.append("")
        
        # Environment information
        env = self.environment_info
        if env.python_version or env.platform:
            context_parts.append("### Environment Information")
            if env.python_version:
                context_parts.append(f"- **Python Version:** {env.python_version}")
            if env.platform:
                context_parts.append(f"- **Platform:** {env.platform}")
            if env.cpu_count > 0:
                context_parts.append(f"- **CPU Cores:** {env.cpu_count}")
            if env.available_memory_mb > 0:
                context_parts.append(f"- **Available Memory:** {env.available_memory_mb:.1f} MB")
            context_parts.append(f"- **Network Available:** {'Yes' if env.network_available else 'No'}")
            context_parts.append("")
        
        # Dependencies status
        if env.dependencies_status:
            context_parts.append("### Dependencies Status")
            for dep, status in env.dependencies_status.items():
                status_icon = "✅" if status == "available" else "❌"
                context_parts.append(f"- {status_icon} **{dep}:** {status}")
            context_parts.append("")
        
        # Recommendations
        if self.validation_result and self.validation_result.recommendations:
            context_parts.append("### Configuration Recommendations")
            for rec in self.validation_result.recommendations:
                context_parts.append(f"- 💡 {rec}")
            context_parts.append("")
        
        # Usage guidelines
        context_parts.append("### Usage Guidelines")
        guidelines = self._generate_usage_guidelines()
        for guideline in guidelines:
            context_parts.append(f"- {guideline}")
        
        return "\n".join(context_parts)
    
    def _get_no_config_context(self) -> str:
        """Return context when no configuration is available."""
        return """## System Configuration

**Status:** No configuration loaded

### Default Behavior
- System will use built-in defaults for all settings
- Configuration should be loaded before starting operations
- Environment variables will be checked for overrides

### Required Configuration
- Orchestrator and planning agent model settings
- Quality thresholds and processing limits
- Output format and directory preferences
- Compliance and monitoring settings

### Loading Configuration
- Use IntelligentScrapingConfig.from_env() for environment-based config
- Set configuration via ConfigurationContextProvider.set_configuration()
- Validate configuration before use to ensure proper operation"""
    
    def _validate_configuration(self, config: IntelligentScrapingConfig) -> ConfigurationValidationResult:
        """
        Validate the configuration and return validation results.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation result with errors, warnings, and recommendations
        """
        result = ConfigurationValidationResult()
        
        # Validate model settings
        if not config.orchestrator_model:
            result.add_error("Orchestrator model is not specified")
        elif not config.orchestrator_model.startswith(('gpt-', 'claude-', 'llama-')):
            result.add_warning(f"Unusual orchestrator model: {config.orchestrator_model}")
        
        if not config.planning_agent_model:
            result.add_error("Planning agent model is not specified")
        
        # Validate thresholds
        if config.default_quality_threshold < 0 or config.default_quality_threshold > 100:
            result.add_error("Quality threshold must be between 0 and 100")
        elif config.default_quality_threshold < 30:
            result.add_warning("Very low quality threshold may result in poor data quality")
        elif config.default_quality_threshold > 95:
            result.add_warning("Very high quality threshold may result in few extracted items")
        
        # Validate concurrency settings
        if config.max_concurrent_requests < 1:
            result.add_error("Max concurrent requests must be at least 1")
        elif config.max_concurrent_requests > 20:
            result.add_warning("High concurrent request count may overwhelm target servers")
        
        if config.request_delay < 0:
            result.add_error("Request delay cannot be negative")
        elif config.request_delay < 0.5:
            result.add_warning("Very short request delay may violate rate limiting policies")
        
        # Validate instance limits
        if config.max_instances < 1:
            result.add_error("Max instances must be at least 1")
        elif config.max_instances > 10:
            result.add_warning("High instance count may consume excessive resources")
        
        if config.max_workers < 1:
            result.add_error("Max workers must be at least 1")
        
        if config.max_async_tasks < 1:
            result.add_error("Max async tasks must be at least 1")
        
        # Validate monitoring settings
        if config.monitoring_interval < 0.1:
            result.add_warning("Very short monitoring interval may impact performance")
        elif config.monitoring_interval > 10:
            result.add_warning("Long monitoring interval may delay issue detection")
        
        # Generate recommendations
        if config.default_quality_threshold < 70:
            result.add_recommendation("Consider increasing quality threshold to 70% or higher for better data quality")
        
        if config.max_concurrent_requests == 1 and config.request_delay > 2:
            result.add_recommendation("With single concurrent request, consider reducing request delay for better performance")
        
        if not config.enable_monitoring:
            result.add_recommendation("Enable monitoring for better visibility into scraping operations")
        
        if not config.respect_robots_txt:
            result.add_recommendation("Enable robots.txt respect for ethical scraping practices")
        
        return result
    
    def _update_environment_info(self) -> None:
        """Update environment information."""
        import sys
        import platform
        import os
        
        env = self.environment_info
        
        # Basic system info
        env.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        env.platform = platform.system()
        env.cpu_count = os.cpu_count() or 0
        
        # Check network availability (simplified)
        env.network_available = True  # Would implement actual network check in production
        
        # Check dependencies
        dependencies = ['requests', 'beautifulsoup4', 'openai', 'instructor', 'pydantic']
        for dep in dependencies:
            try:
                __import__(dep)
                env.dependencies_status[dep] = "available"
            except ImportError:
                env.dependencies_status[dep] = "missing"
        
        # Environment variables (only safe ones)
        safe_env_vars = [
            'ORCHESTRATOR_MODEL', 'PLANNING_AGENT_MODEL', 'QUALITY_THRESHOLD',
            'MAX_CONCURRENT_REQUESTS', 'EXPORT_FORMAT', 'RESULTS_DIRECTORY'
        ]
        for var in safe_env_vars:
            if var in os.environ:
                env.environment_variables[var] = os.environ[var]
    
    def _generate_usage_guidelines(self) -> List[str]:
        """Generate usage guidelines based on current configuration."""
        guidelines = []
        
        if not self.config:
            return ["Load configuration before generating guidelines"]
        
        config = self.config
        
        # Quality guidelines
        if config.default_quality_threshold >= 80:
            guidelines.append("High quality threshold set - expect fewer but higher quality results")
        elif config.default_quality_threshold <= 50:
            guidelines.append("Low quality threshold set - review results carefully for data quality")
        
        # Concurrency guidelines
        if config.max_concurrent_requests > 5:
            guidelines.append("High concurrency configured - monitor target server response and respect rate limits")
        
        if config.request_delay < 1.0:
            guidelines.append("Short request delay configured - ensure compliance with target site policies")
        
        # Monitoring guidelines
        if config.enable_monitoring:
            guidelines.append("Monitoring enabled - use monitoring reports to optimize performance")
        else:
            guidelines.append("Monitoring disabled - consider enabling for better operational visibility")
        
        # Compliance guidelines
        if config.respect_robots_txt and config.enable_rate_limiting:
            guidelines.append("Compliance features enabled - scraping will follow ethical practices")
        else:
            guidelines.append("Some compliance features disabled - ensure manual compliance with site policies")
        
        # Resource guidelines
        if config.max_instances > 3:
            guidelines.append("Multiple instances configured - monitor system resources during operation")
        
        return guidelines or ["Configuration appears standard - follow general scraping best practices"]
    
    def get_effective_config(self) -> Dict[str, Any]:
        """
        Get the effective configuration including overrides.
        
        Returns:
            Dictionary containing the effective configuration values
        """
        if not self.config:
            return {}
        
        # Start with base configuration
        effective_config = self.config.model_dump()
        
        # Apply overrides
        effective_config.update(self.config_overrides)
        
        return effective_config
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of configuration validation results.
        
        Returns:
            Dictionary containing validation summary
        """
        if not self.validation_result:
            return {"status": "not_validated"}
        
        return {
            "status": "valid" if self.validation_result.is_valid else "invalid",
            "error_count": len(self.validation_result.errors),
            "warning_count": len(self.validation_result.warnings),
            "recommendation_count": len(self.validation_result.recommendations),
            "validated_at": self.validation_result.validated_at.isoformat()
        }
    
    def get_environment_summary(self) -> Dict[str, Any]:
        """
        Get a summary of environment information.
        
        Returns:
            Dictionary containing environment summary
        """
        env = self.environment_info
        
        missing_deps = [dep for dep, status in env.dependencies_status.items() if status != "available"]
        
        return {
            "python_version": env.python_version,
            "platform": env.platform,
            "cpu_count": env.cpu_count,
            "network_available": env.network_available,
            "dependencies_ok": len(missing_deps) == 0,
            "missing_dependencies": missing_deps
        }
    
    def validate_current_config(self) -> bool:
        """
        Validate the current configuration and update validation results.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        if not self.config:
            return False
        
        self.validation_result = self._validate_configuration(self.config)
        return self.validation_result.is_valid