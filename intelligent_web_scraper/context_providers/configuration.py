"""
Configuration Context Provider.

This module provides system configuration context to agents,
demonstrating context provider patterns in atomic-agents.
"""

from typing import Dict, Any, Optional
from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase
from ..config import IntelligentScrapingConfig


class ConfigurationContextProvider(SystemPromptContextProviderBase):
    """Provides system configuration context to agents."""
    
    def __init__(self, config: IntelligentScrapingConfig, title: str = "System Configuration"):
        super().__init__(title=title)
        self.config = config
        self.runtime_overrides: Dict[str, Any] = {}
        self.environment_info: Optional[Dict[str, Any]] = None
    
    def set_runtime_overrides(self, overrides: Dict[str, Any]) -> None:
        """Set runtime configuration overrides."""
        self.runtime_overrides = overrides
    
    def set_environment_info(self, env_info: Dict[str, Any]) -> None:
        """Set environment information."""
        self.environment_info = env_info
    
    def get_effective_config_value(self, key: str) -> Any:
        """Get effective configuration value considering overrides."""
        if key in self.runtime_overrides:
            return self.runtime_overrides[key]
        return getattr(self.config, key, None)
    
    def get_info(self) -> str:
        """Return formatted configuration information."""
        info_parts = []
        
        # Core configuration
        info_parts.append("Core Configuration:")
        info_parts.append(f"- Orchestrator Model: {self.get_effective_config_value('orchestrator_model')}")
        info_parts.append(f"- Planning Agent Model: {self.get_effective_config_value('planning_agent_model')}")
        info_parts.append(f"- Quality Threshold: {self.get_effective_config_value('default_quality_threshold')}%")
        
        # Performance settings
        info_parts.append("\nPerformance Settings:")
        info_parts.append(f"- Max Concurrent Requests: {self.get_effective_config_value('max_concurrent_requests')}")
        info_parts.append(f"- Request Delay: {self.get_effective_config_value('request_delay')}s")
        info_parts.append(f"- Max Instances: {self.get_effective_config_value('max_instances')}")
        info_parts.append(f"- Max Workers: {self.get_effective_config_value('max_workers')}")
        
        # Compliance settings
        info_parts.append("\nCompliance Settings:")
        info_parts.append(f"- Respect Robots.txt: {self.get_effective_config_value('respect_robots_txt')}")
        info_parts.append(f"- Rate Limiting Enabled: {self.get_effective_config_value('enable_rate_limiting')}")
        
        # Monitoring settings
        info_parts.append("\nMonitoring Settings:")
        info_parts.append(f"- Monitoring Enabled: {self.get_effective_config_value('enable_monitoring')}")
        info_parts.append(f"- Monitoring Interval: {self.get_effective_config_value('monitoring_interval')}s")
        
        # Output settings
        info_parts.append("\nOutput Settings:")
        info_parts.append(f"- Default Export Format: {self.get_effective_config_value('default_export_format')}")
        info_parts.append(f"- Results Directory: {self.get_effective_config_value('results_directory')}")
        
        # Runtime overrides
        if self.runtime_overrides:
            info_parts.append("\nRuntime Overrides:")
            for key, value in self.runtime_overrides.items():
                info_parts.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        # Environment information
        if self.environment_info:
            info_parts.append("\nEnvironment Information:")
            for key, value in self.environment_info.items():
                info_parts.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(info_parts)