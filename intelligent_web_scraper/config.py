"""
Configuration management for the Intelligent Web Scraper.

This module demonstrates proper configuration patterns for atomic-agents
applications, including environment variable handling and validation.
"""

import os
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator


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
    
    # LLM Provider configuration
    llm_provider: Literal["openai", "gemini", "deepseek", "openrouter", "anthropic"] = Field(
        default="openai",
        description="LLM provider to use for AI models"
    )
    
    # Provider-specific API configurations
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    openai_base_url: Optional[str] = Field(
        default=None,
        description="OpenAI API base URL (for custom endpoints)"
    )
    
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key"
    )
    
    deepseek_api_key: Optional[str] = Field(
        default=None,
        description="DeepSeek API key"
    )
    deepseek_base_url: Optional[str] = Field(
        default="https://api.deepseek.com/v1",
        description="DeepSeek API base URL"
    )
    
    openrouter_api_key: Optional[str] = Field(
        default=None,
        description="OpenRouter API key"
    )
    openrouter_base_url: Optional[str] = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL"
    )
    
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    
    # Model mapping for different providers
    provider_model_mapping: Dict[str, Dict[str, str]] = Field(
        default_factory=lambda: {
            "openai": {
                "gpt-4o-mini": "gpt-4o-mini",
                "gpt-4o": "gpt-4o", 
                "gpt-4": "gpt-4",
                "gpt-3.5-turbo": "gpt-3.5-turbo"
            },
            "gemini": {
                "gpt-4o-mini": "gemini-1.5-flash",
                "gpt-4o": "gemini-1.5-pro",
                "gpt-4": "gemini-1.5-pro",
                "gpt-3.5-turbo": "gemini-1.5-flash"
            },
            "deepseek": {
                "gpt-4o-mini": "deepseek-chat",
                "gpt-4o": "deepseek-chat",
                "gpt-4": "deepseek-chat",
                "gpt-3.5-turbo": "deepseek-chat"
            },
            "openrouter": {
                "gpt-4o-mini": "openai/gpt-4o-mini",
                "gpt-4o": "openai/gpt-4o",
                "gpt-4": "openai/gpt-4",
                "gpt-3.5-turbo": "openai/gpt-3.5-turbo"
            },
            "anthropic": {
                "gpt-4o-mini": "claude-3-haiku-20240307",
                "gpt-4o": "claude-3-5-sonnet-20241022",
                "gpt-4": "claude-3-5-sonnet-20241022",
                "gpt-3.5-turbo": "claude-3-haiku-20240307"
            }
        },
        description="Mapping of generic model names to provider-specific model names"
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
    
    @validator('default_quality_threshold')
    def validate_quality_threshold(cls, v):
        """Validate quality threshold is between 0 and 100."""
        if not 0 <= v <= 100:
            raise ValueError('Quality threshold must be between 0 and 100')
        return v
    
    @validator('max_concurrent_requests')
    def validate_max_concurrent_requests(cls, v):
        """Validate max concurrent requests is positive."""
        if v <= 0:
            raise ValueError('Max concurrent requests must be positive')
        return v
    
    @validator('request_delay')
    def validate_request_delay(cls, v):
        """Validate request delay is non-negative."""
        if v < 0:
            raise ValueError('Request delay must be non-negative')
        return v
    
    @validator('default_export_format')
    def validate_export_format(cls, v):
        """Validate export format is supported."""
        valid_formats = {'json', 'csv', 'markdown', 'excel'}
        if v not in valid_formats:
            raise ValueError(f'Export format must be one of: {valid_formats}')
        return v
    
    @validator('monitoring_interval')
    def validate_monitoring_interval(cls, v):
        """Validate monitoring interval is positive."""
        if v <= 0:
            raise ValueError('Monitoring interval must be positive')
        return v
    
    @validator('max_instances')
    def validate_max_instances(cls, v):
        """Validate max instances is positive."""
        if v <= 0:
            raise ValueError('Max instances must be positive')
        return v
    
    @validator('max_workers')
    def validate_max_workers(cls, v):
        """Validate max workers is positive."""
        if v <= 0:
            raise ValueError('Max workers must be positive')
        return v
    
    @validator('max_async_tasks')
    def validate_max_async_tasks(cls, v):
        """Validate max async tasks is positive."""
        if v <= 0:
            raise ValueError('Max async tasks must be positive')
        return v
    
    @validator('orchestrator_model', 'planning_agent_model')
    def validate_model_names(cls, v, values):
        """Validate that model names are supported by the selected provider."""
        provider = values.get('llm_provider', 'openai')
        model_mapping = values.get('provider_model_mapping', {})
        
        if provider in model_mapping and v not in model_mapping[provider]:
            available_models = list(model_mapping[provider].keys())
            raise ValueError(f'Model "{v}" not supported by provider "{provider}". Available models: {available_models}')
        
        return v
    
    def get_provider_model_name(self, generic_model: str) -> str:
        """Get the provider-specific model name for a generic model name."""
        if self.llm_provider in self.provider_model_mapping:
            return self.provider_model_mapping[self.llm_provider].get(generic_model, generic_model)
        return generic_model
    
    def get_provider_config(self) -> Dict[str, Any]:
        """Get the configuration for the selected LLM provider."""
        config = {
            "provider": self.llm_provider,
            "orchestrator_model": self.get_provider_model_name(self.orchestrator_model),
            "planning_agent_model": self.get_provider_model_name(self.planning_agent_model)
        }
        
        if self.llm_provider == "openai":
            config.update({
                "api_key": self.openai_api_key or os.getenv("OPENAI_API_KEY"),
                "base_url": self.openai_base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            })
        elif self.llm_provider == "gemini":
            config.update({
                "api_key": self.gemini_api_key or os.getenv("GEMINI_API_KEY"),
                "base_url": "https://generativelanguage.googleapis.com/v1beta"
            })
        elif self.llm_provider == "deepseek":
            config.update({
                "api_key": self.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY"),
                "base_url": self.deepseek_base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            })
        elif self.llm_provider == "openrouter":
            config.update({
                "api_key": self.openrouter_api_key or os.getenv("OPENROUTER_API_KEY"),
                "base_url": self.openrouter_base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            })
        elif self.llm_provider == "anthropic":
            config.update({
                "api_key": self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
                "base_url": "https://api.anthropic.com"
            })
        
        return config
    
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
            DEFAULT_QUALITY_THRESHOLD: Default quality threshold (0-100)
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
            export DEFAULT_QUALITY_THRESHOLD="75.0"
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
            default_quality_threshold=float(os.getenv("DEFAULT_QUALITY_THRESHOLD", "50.0")),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "5")),
            request_delay=float(os.getenv("REQUEST_DELAY", "1.0")),
            default_export_format=os.getenv("EXPORT_FORMAT", "json"),
            results_directory=os.getenv("RESULTS_DIRECTORY", "./results"),
            respect_robots_txt=os.getenv("RESPECT_ROBOTS_TXT", "true").lower() in ("true", "1", "yes", "on"),
            enable_rate_limiting=os.getenv("ENABLE_RATE_LIMITING", "true").lower() in ("true", "1", "yes", "on"),
            enable_monitoring=os.getenv("ENABLE_MONITORING", "true").lower() in ("true", "1", "yes", "on"),
            monitoring_interval=float(os.getenv("MONITORING_INTERVAL", "1.0")),
            max_instances=int(os.getenv("MAX_INSTANCES", "5")),
            max_workers=int(os.getenv("MAX_WORKERS", "10")),
            max_async_tasks=int(os.getenv("MAX_ASYNC_TASKS", "50")),
            
            # LLM Provider configuration
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )