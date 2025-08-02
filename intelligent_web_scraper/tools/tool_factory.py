"""
Tool Factory for Atomic Scraper Tool

Provides dependency injection and configuration management patterns
for clean separation of concerns and proper tool initialization.
"""

import logging
import os
from typing import Dict, Any, Optional, Type
from uuid import uuid4

from ..config import IntelligentScrapingConfig
from .atomic_scraper_tool import AtomicScraperTool, AtomicScraperToolConfig


logger = logging.getLogger(__name__)


class ToolConfigurationError(Exception):
    """Exception raised for tool configuration errors."""
    pass


class AtomicScraperToolFactory:
    """
    Factory class for creating and configuring AtomicScraperTool instances.
    
    This factory demonstrates proper dependency injection patterns and
    configuration management for atomic-agents tools.
    """
    
    def __init__(self, intelligent_config: Optional[IntelligentScrapingConfig] = None):
        """
        Initialize the tool factory.
        
        Args:
            intelligent_config: Intelligent scraping system configuration
        """
        self.intelligent_config = intelligent_config or IntelligentScrapingConfig.from_env()
        self._tool_instances: Dict[str, AtomicScraperTool] = {}
        self._config_cache: Dict[str, AtomicScraperToolConfig] = {}
        
        logger.info("AtomicScraperToolFactory initialized")
    
    def create_tool(
        self,
        base_url: str,
        instance_id: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> AtomicScraperTool:
        """
        Create a new AtomicScraperTool instance with proper configuration.
        
        Args:
            base_url: Base URL for scraping operations
            instance_id: Optional instance identifier for monitoring
            config_overrides: Optional configuration overrides
            
        Returns:
            Configured AtomicScraperTool instance
            
        Raises:
            ToolConfigurationError: If configuration is invalid
        """
        try:
            # Generate instance ID if not provided
            if instance_id is None:
                instance_id = f"scraper_{uuid4().hex[:8]}"
            
            # Create tool configuration
            tool_config = self._create_tool_config(
                base_url=base_url,
                instance_id=instance_id,
                config_overrides=config_overrides or {}
            )
            
            # Create tool instance
            tool = AtomicScraperTool(
                config=tool_config,
                intelligent_config=self.intelligent_config
            )
            
            # Cache the instance
            self._tool_instances[instance_id] = tool
            
            logger.info(f"Created AtomicScraperTool instance: {instance_id}")
            return tool
            
        except Exception as e:
            raise ToolConfigurationError(f"Failed to create tool instance: {e}") from e
    
    def get_tool(self, instance_id: str) -> Optional[AtomicScraperTool]:
        """
        Get an existing tool instance by ID.
        
        Args:
            instance_id: Instance identifier
            
        Returns:
            Tool instance or None if not found
        """
        return self._tool_instances.get(instance_id)
    
    def create_or_get_tool(
        self,
        base_url: str,
        instance_id: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> AtomicScraperTool:
        """
        Create a new tool or return existing one if instance_id is provided and exists.
        
        Args:
            base_url: Base URL for scraping operations
            instance_id: Optional instance identifier
            config_overrides: Optional configuration overrides
            
        Returns:
            AtomicScraperTool instance
        """
        if instance_id and instance_id in self._tool_instances:
            logger.info(f"Returning existing tool instance: {instance_id}")
            return self._tool_instances[instance_id]
        
        return self.create_tool(
            base_url=base_url,
            instance_id=instance_id,
            config_overrides=config_overrides
        )
    
    def _create_tool_config(
        self,
        base_url: str,
        instance_id: str,
        config_overrides: Dict[str, Any]
    ) -> AtomicScraperToolConfig:
        """
        Create tool configuration with validation and caching.
        
        Args:
            base_url: Base URL for scraping
            instance_id: Instance identifier
            config_overrides: Configuration overrides
            
        Returns:
            Validated AtomicScraperToolConfig instance
        """
        # Create cache key
        cache_key = f"{base_url}_{hash(frozenset(config_overrides.items()))}"
        
        # Check cache first
        if cache_key in self._config_cache:
            cached_config = self._config_cache[cache_key]
            # Create new config with updated instance_id
            config_dict = cached_config.model_dump()
            config_dict['instance_id'] = instance_id
            return AtomicScraperToolConfig(**config_dict)
        
        # Create new configuration
        try:
            config = AtomicScraperToolConfig.from_intelligent_config(
                base_url=base_url,
                intelligent_config=self.intelligent_config,
                instance_id=instance_id,
                **config_overrides
            )
            
            # Validate configuration
            self._validate_tool_config(config)
            
            # Cache the configuration (without instance_id for reuse)
            cache_config_dict = config.model_dump()
            cache_config_dict.pop('instance_id', None)
            self._config_cache[cache_key] = AtomicScraperToolConfig(**cache_config_dict)
            
            return config
            
        except Exception as e:
            raise ToolConfigurationError(f"Invalid tool configuration: {e}") from e
    
    def _validate_tool_config(self, config: AtomicScraperToolConfig) -> None:
        """
        Validate tool configuration.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ToolConfigurationError: If configuration is invalid
        """
        # Validate URL format
        if not config.base_url.startswith(('http://', 'https://')):
            raise ToolConfigurationError(f"Invalid base_url format: {config.base_url}")
        
        # Validate numeric ranges
        if config.min_quality_score < 0 or config.min_quality_score > 100:
            raise ToolConfigurationError(f"min_quality_score must be between 0 and 100, got {config.min_quality_score}")
        
        if config.request_delay < 0.1:
            raise ToolConfigurationError(f"request_delay must be at least 0.1 seconds, got {config.request_delay}")
        
        if config.timeout < 5:
            raise ToolConfigurationError(f"timeout must be at least 5 seconds, got {config.timeout}")
        
        if config.max_retries < 0:
            raise ToolConfigurationError(f"max_retries must be non-negative, got {config.max_retries}")
    
    def update_intelligent_config(self, new_config: IntelligentScrapingConfig) -> None:
        """
        Update the intelligent scraping configuration.
        
        Args:
            new_config: New intelligent scraping configuration
        """
        self.intelligent_config = new_config
        
        # Clear config cache to force recreation with new settings
        self._config_cache.clear()
        
        logger.info("Updated intelligent scraping configuration")
    
    def list_tool_instances(self) -> Dict[str, Dict[str, Any]]:
        """
        List all tool instances with their basic information.
        
        Returns:
            Dictionary mapping instance IDs to tool information
        """
        instances_info = {}
        
        for instance_id, tool in self._tool_instances.items():
            instances_info[instance_id] = {
                'instance_id': instance_id,
                'base_url': tool.config.base_url,
                'monitoring_enabled': tool.config.enable_monitoring,
                'created_at': getattr(tool, '_created_at', 'unknown'),
                'tool_info': tool.get_tool_info()
            }
        
        return instances_info
    
    def remove_tool_instance(self, instance_id: str) -> bool:
        """
        Remove a tool instance from the factory.
        
        Args:
            instance_id: Instance identifier to remove
            
        Returns:
            True if instance was removed, False if not found
        """
        if instance_id in self._tool_instances:
            del self._tool_instances[instance_id]
            logger.info(f"Removed tool instance: {instance_id}")
            return True
        
        return False
    
    def clear_all_instances(self) -> None:
        """Clear all tool instances and caches."""
        self._tool_instances.clear()
        self._config_cache.clear()
        logger.info("Cleared all tool instances and caches")
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """
        Get factory statistics and health information.
        
        Returns:
            Dictionary containing factory statistics
        """
        return {
            'total_instances': len(self._tool_instances),
            'cached_configs': len(self._config_cache),
            'instance_ids': list(self._tool_instances.keys()),
            'intelligent_config': {
                'orchestrator_model': self.intelligent_config.orchestrator_model,
                'planning_agent_model': self.intelligent_config.planning_agent_model,
                'default_quality_threshold': self.intelligent_config.default_quality_threshold,
                'max_concurrent_requests': self.intelligent_config.max_concurrent_requests,
                'enable_monitoring': self.intelligent_config.enable_monitoring
            }
        }


class ConfigurationManager:
    """
    Configuration manager for environment variable handling and validation.
    
    Demonstrates proper configuration management patterns for production applications.
    """
    
    def __init__(self):
        """Initialize configuration manager."""
        self._env_cache: Dict[str, Any] = {}
        self._validation_rules: Dict[str, callable] = {
            'QUALITY_THRESHOLD': lambda x: 0 <= float(x) <= 100,
            'MAX_CONCURRENT_REQUESTS': lambda x: int(x) > 0,
            'REQUEST_DELAY': lambda x: float(x) >= 0.1,
            'MAX_INSTANCES': lambda x: int(x) > 0,
            'MAX_WORKERS': lambda x: int(x) > 0,
            'MONITORING_INTERVAL': lambda x: float(x) > 0
        }
    
    def get_env_value(
        self,
        key: str,
        default: Any = None,
        value_type: Type = str,
        required: bool = False
    ) -> Any:
        """
        Get environment variable value with type conversion and validation.
        
        Args:
            key: Environment variable key
            default: Default value if not found
            value_type: Type to convert value to
            required: Whether the value is required
            
        Returns:
            Environment variable value
            
        Raises:
            ToolConfigurationError: If required value is missing or invalid
        """
        # Check cache first
        if key in self._env_cache:
            return self._env_cache[key]
        
        # Get value from environment
        raw_value = os.getenv(key)
        
        if raw_value is None:
            if required:
                raise ToolConfigurationError(f"Required environment variable {key} is not set")
            value = default
        else:
            try:
                # Convert to requested type
                if value_type == bool:
                    value = raw_value.lower() in ('true', '1', 'yes', 'on')
                else:
                    value = value_type(raw_value)
                
                # Validate if rule exists
                if key in self._validation_rules:
                    if not self._validation_rules[key](value):
                        raise ToolConfigurationError(f"Invalid value for {key}: {value}")
                
            except (ValueError, TypeError) as e:
                raise ToolConfigurationError(f"Invalid type for {key}: {raw_value} (expected {value_type.__name__})") from e
        
        # Cache the value
        self._env_cache[key] = value
        return value
    
    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate all environment variables and return validation report.
        
        Returns:
            Dictionary containing validation results
        """
        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'values': {}
        }
        
        # Define expected environment variables
        expected_vars = {
            'ORCHESTRATOR_MODEL': {'type': str, 'default': 'gpt-4o-mini', 'required': False},
            'PLANNING_AGENT_MODEL': {'type': str, 'default': 'gpt-4o-mini', 'required': False},
            'QUALITY_THRESHOLD': {'type': float, 'default': 50.0, 'required': False},
            'MAX_CONCURRENT_REQUESTS': {'type': int, 'default': 5, 'required': False},
            'REQUEST_DELAY': {'type': float, 'default': 1.0, 'required': False},
            'RESULTS_DIRECTORY': {'type': str, 'default': './results', 'required': False},
            'ENABLE_MONITORING': {'type': bool, 'default': True, 'required': False},
            'MAX_INSTANCES': {'type': int, 'default': 5, 'required': False},
            'MAX_WORKERS': {'type': int, 'default': 10, 'required': False}
        }
        
        # Validate each variable
        for var_name, var_config in expected_vars.items():
            try:
                value = self.get_env_value(
                    var_name,
                    default=var_config['default'],
                    value_type=var_config['type'],
                    required=var_config['required']
                )
                validation_report['values'][var_name] = value
                
            except ToolConfigurationError as e:
                validation_report['valid'] = False
                validation_report['errors'].append(str(e))
        
        # Check for deprecated or unknown variables
        all_env_vars = {k: v for k, v in os.environ.items() if k.startswith(('SCRAPER_', 'INTELLIGENT_'))}
        for env_var in all_env_vars:
            if env_var not in expected_vars:
                validation_report['warnings'].append(f"Unknown environment variable: {env_var}")
        
        return validation_report
    
    def clear_cache(self) -> None:
        """Clear environment variable cache."""
        self._env_cache.clear()