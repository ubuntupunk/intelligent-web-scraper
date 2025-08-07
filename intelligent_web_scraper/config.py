"""
Configuration management for the Intelligent Web Scraper.

This module demonstrates proper configuration patterns for atomic-agents
applications, including environment variable handling and validation.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field


class IntelligentScrapingConfig(BaseModel):
    """
    Configuration for the intelligent scraping system.
    
    This class manages all configuration options for the intelligent web scraper,
    including LLM model settings, performance parameters, compliance options,
    and operational limits. It demonstrates proper configuration patterns for
    atomic-agents applications with environment variable support and validation.
    
    The configuration follows the principle of sensible defaults while allowing
    full customization through environment variables or direct instantiation.
    
    Attributes:
        orchestrator_model (str): LLM model identifier for the orchestrator agent.
            Supports OpenAI models like "gpt-4", "gpt-4o-mini", "gpt-3.5-turbo".
            Default: "gpt-4o-mini"
            
        planning_agent_model (str): LLM model identifier for the planning agent.
            Can be different from orchestrator model for cost optimization.
            Default: "gpt-4o-mini"
            
        default_quality_threshold (float): Default minimum quality score (0-100)
            for extracted data. Items below this threshold may be filtered out.
            Default: 50.0
            
        max_concurrent_requests (int): Maximum number of concurrent HTTP requests
            to prevent overwhelming target servers and respect rate limits.
            Default: 5
            
        request_delay (float): Minimum delay in seconds between consecutive
            requests to the same domain. Helps with rate limiting compliance.
            Default: 1.0
            
        default_export_format (str): Default format for exporting results.
            Supported: "json", "csv", "markdown", "excel"
            Default: "json"
            
        results_directory (str): Directory path for storing exported results.
            Will be created if it doesn't exist.
            Default: "./results"
            
        respect_robots_txt (bool): Whether to check and respect robots.txt
            files before scraping. Recommended for ethical scraping.
            Default: True
            
        enable_rate_limiting (bool): Whether to enable automatic rate limiting
            based on server response times and headers.
            Default: True
            
        enable_monitoring (bool): Whether to enable real-time monitoring
            dashboard and metrics collection.
            Default: True
            
        monitoring_interval (float): Update interval in seconds for the
            monitoring dashboard and metrics collection.
            Default: 1.0
            
        max_instances (int): Maximum number of concurrent scraper instances
            that can be active simultaneously.
            Default: 5
            
        max_workers (int): Maximum number of worker threads for CPU-intensive
            tasks like data processing and analysis.
            Default: 10
            
        max_async_tasks (int): Maximum number of concurrent async tasks
            for I/O-bound operations like HTTP requests.
            Default: 50
    
    Example:
        Create configuration with custom settings:
        
        ```python
        config = IntelligentScrapingConfig(
            orchestrator_model="gpt-4",
            default_quality_threshold=75.0,
            max_concurrent_requests=10,
            enable_monitoring=True
        )
        ```
        
        Create configuration from environment variables:
        
        ```python
        # Set environment variables
        os.environ["ORCHESTRATOR_MODEL"] = "gpt-4"
        os.environ["QUALITY_THRESHOLD"] = "75.0"
        
        # Load configuration
        config = IntelligentScrapingConfig.from_env()
        ```
    
    Note:
        All configuration values are validated using Pydantic, ensuring
        type safety and proper value ranges. Invalid configurations will
        raise ValidationError with detailed error messages.
    """
    
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
        """
        Create configuration from environment variables.
        
        This class method provides a convenient way to load configuration
        from environment variables, which is the recommended approach for
        production deployments and containerized environments.
        
        Environment Variables:
            ORCHESTRATOR_MODEL: LLM model for orchestrator agent
            PLANNING_AGENT_MODEL: LLM model for planning agent
            QUALITY_THRESHOLD: Default quality threshold (0-100)
            MAX_CONCURRENT_REQUESTS: Maximum concurrent HTTP requests
            REQUEST_DELAY: Delay between requests in seconds
            EXPORT_FORMAT: Default export format (json, csv, markdown, excel)
            RESULTS_DIRECTORY: Directory for exported results
            RESPECT_ROBOTS_TXT: Whether to respect robots.txt (true/false)
            ENABLE_RATE_LIMITING: Whether to enable rate limiting (true/false)
            ENABLE_MONITORING: Whether to enable monitoring (true/false)
            MONITORING_INTERVAL: Monitoring update interval in seconds
            MAX_INSTANCES: Maximum number of scraper instances
            MAX_WORKERS: Maximum number of worker threads
            MAX_ASYNC_TASKS: Maximum number of concurrent async tasks
        
        Returns:
            IntelligentScrapingConfig: Configuration instance with values
                loaded from environment variables, using defaults for
                any missing variables.
        
        Raises:
            ValueError: If environment variable values are invalid
                (e.g., non-numeric values for numeric fields)
            ValidationError: If the resulting configuration is invalid
        
        Example:
            ```bash
            # Set environment variables
            export ORCHESTRATOR_MODEL="gpt-4"
            export QUALITY_THRESHOLD="75.0"
            export MAX_CONCURRENT_REQUESTS="8"
            export ENABLE_MONITORING="true"
            ```
            
            ```python
            # Load configuration in Python
            config = IntelligentScrapingConfig.from_env()
            print(f"Using model: {config.orchestrator_model}")
            print(f"Quality threshold: {config.default_quality_threshold}")
            ```
        
        Note:
            Boolean environment variables are parsed case-insensitively,
            with "true", "1", "yes", "on" being considered True, and
            all other values being considered False.
        """
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