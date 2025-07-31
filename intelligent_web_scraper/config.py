"""
Configuration management for the Intelligent Web Scraper.

This module demonstrates proper configuration patterns for atomic-agents
applications, including environment variable handling and validation.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field


class IntelligentScrapingConfig(BaseModel):
    """Configuration for the intelligent scraping system."""
    
    # Agent configuration
    orchestrator_model: str = Field(
        default="gpt-4o-mini", 
        description="Model for orchestrator agent"
    )
    planning_agent_model: str = Field(
        default="gpt-4o-mini", 
        description="Model for planning agent"
    )
    
    # Scraping configuration
    default_quality_threshold: float = Field(
        default=50.0, 
        description="Default quality threshold"
    )
    max_concurrent_requests: int = Field(
        default=5, 
        description="Maximum concurrent requests"
    )
    request_delay: float = Field(
        default=1.0, 
        description="Delay between requests"
    )
    
    # Output configuration
    default_export_format: str = Field(
        default="json", 
        description="Default export format"
    )
    results_directory: str = Field(
        default="./results", 
        description="Directory for exported results"
    )
    
    # Compliance configuration
    respect_robots_txt: bool = Field(
        default=True, 
        description="Whether to respect robots.txt"
    )
    enable_rate_limiting: bool = Field(
        default=True, 
        description="Whether to enable rate limiting"
    )
    
    # Monitoring configuration
    enable_monitoring: bool = Field(
        default=True, 
        description="Whether to enable real-time monitoring"
    )
    monitoring_interval: float = Field(
        default=1.0, 
        description="Monitoring update interval in seconds"
    )
    
    # Concurrency configuration
    max_instances: int = Field(
        default=5, 
        description="Maximum number of scraper instances"
    )
    max_workers: int = Field(
        default=10, 
        description="Maximum number of worker threads"
    )
    max_async_tasks: int = Field(
        default=50, 
        description="Maximum number of concurrent async tasks"
    )
    
    @classmethod
    def from_env(cls) -> "IntelligentScrapingConfig":
        """Create configuration from environment variables."""
        return cls(
            orchestrator_model=os.getenv("ORCHESTRATOR_MODEL", "gpt-4o-mini"),
            planning_agent_model=os.getenv("PLANNING_AGENT_MODEL", "gpt-4o-mini"),
            default_quality_threshold=float(os.getenv("QUALITY_THRESHOLD", "50.0")),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "5")),
            request_delay=float(os.getenv("REQUEST_DELAY", "1.0")),
            default_export_format=os.getenv("EXPORT_FORMAT", "json"),
            results_directory=os.getenv("RESULTS_DIRECTORY", "./results"),
            respect_robots_txt=os.getenv("RESPECT_ROBOTS_TXT", "true").lower() == "true",
            enable_rate_limiting=os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true",
            enable_monitoring=os.getenv("ENABLE_MONITORING", "true").lower() == "true",
            monitoring_interval=float(os.getenv("MONITORING_INTERVAL", "1.0")),
            max_instances=int(os.getenv("MAX_INSTANCES", "5")),
            max_workers=int(os.getenv("MAX_WORKERS", "10")),
            max_async_tasks=int(os.getenv("MAX_ASYNC_TASKS", "50")),
        )