# Intelligent Web Scraper API Documentation

This document provides comprehensive API documentation for the Intelligent Web Scraper system, including all classes, methods, and usage examples.

## Table of Contents

1. [Core Components](#core-components)
2. [Orchestrator Agent](#orchestrator-agent)
3. [Configuration Management](#configuration-management)
4. [Context Providers](#context-providers)
5. [Tools and Factories](#tools-and-factories)
6. [Export System](#export-system)
7. [Monitoring and Metrics](#monitoring-and-metrics)
8. [Error Handling](#error-handling)

## Core Components

### IntelligentScrapingOrchestrator

The main orchestrator agent that coordinates the entire scraping workflow.

```python
class IntelligentScrapingOrchestrator(BaseAgent):
    """
    Main orchestrator agent for intelligent web scraping operations.
    
    This agent demonstrates advanced atomic-agents patterns by coordinating
    between planning agents, scraper tools, and context providers to execute
    sophisticated scraping workflows with natural language understanding.
    
    Attributes:
        config (IntelligentScrapingConfig): Configuration for the orchestrator
        planning_agent (AtomicScraperPlanningAgent): AI planning agent
        scraper_tool (AtomicScraperTool): Scraping execution tool
        context_providers (List[SystemPromptContextProviderBase]): Dynamic context providers
        instance_manager (ScraperInstanceManager): Manages scraper instances
        export_manager (ExportManager): Handles result export
    
    Example:
        ```python
        config = IntelligentScrapingConfig(
            orchestrator_model="gpt-4o-mini",
            enable_monitoring=True
        )
        orchestrator = IntelligentScrapingOrchestrator(config=config)
        
        result = await orchestrator.run({
            "scraping_request": "Extract product information",
            "target_url": "https://example.com",
            "max_results": 20
        })
        ```
    """
```

#### Methods

##### `__init__(config: IntelligentScrapingConfig)`

Initialize the orchestrator with configuration.

**Parameters:**
- `config` (IntelligentScrapingConfig): Configuration object containing all settings

**Raises:**
- `ValueError`: If configuration is invalid
- `ImportError`: If required dependencies are missing

##### `async run(params: IntelligentScrapingOrchestratorInputSchema) -> IntelligentScrapingOrchestratorOutputSchema`

Execute the intelligent scraping workflow.

**Parameters:**
- `params` (IntelligentScrapingOrchestratorInputSchema): Input parameters for scraping

**Returns:**
- `IntelligentScrapingOrchestratorOutputSchema`: Complete scraping results with metadata

**Raises:**
- `ScrapingError`: If scraping operation fails
- `ValidationError`: If input parameters are invalid
- `TimeoutError`: If operation exceeds timeout limits

**Example:**
```python
result = await orchestrator.run({
    "scraping_request": "Extract all product names and prices",
    "target_url": "https://example-store.com/products",
    "max_results": 50,
    "quality_threshold": 70.0,
    "export_format": "csv"
})

print(f"Extracted {len(result.extracted_data)} items")
print(f"Quality score: {result.quality_score}")
```

##### `add_context_provider(provider: SystemPromptContextProviderBase)`

Add a context provider for dynamic prompt enhancement.

**Parameters:**
- `provider` (SystemPromptContextProviderBase): Context provider instance

**Example:**
```python
website_context = WebsiteAnalysisContextProvider()
orchestrator.add_context_provider(website_context)
```

##### `async get_monitoring_report() -> ScrapingMonitoringReport`

Get comprehensive monitoring report for current operations.

**Returns:**
- `ScrapingMonitoringReport`: Detailed monitoring data and metrics

## Configuration Management

### IntelligentScrapingConfig

Configuration class for the intelligent scraping system.

```python
class IntelligentScrapingConfig(BaseModel):
    """
    Configuration for the intelligent scraping system.
    
    This class manages all configuration options for the scraping system,
    including LLM settings, performance parameters, and operational limits.
    
    Attributes:
        orchestrator_model (str): LLM model for orchestrator agent
        planning_agent_model (str): LLM model for planning agent
        default_quality_threshold (float): Default minimum quality score
        max_concurrent_requests (int): Maximum concurrent HTTP requests
        request_delay (float): Delay between requests in seconds
        enable_monitoring (bool): Enable real-time monitoring
        results_directory (str): Directory for exported results
        respect_robots_txt (bool): Whether to respect robots.txt
        enable_rate_limiting (bool): Enable request rate limiting
    """
```

#### Class Methods

##### `from_env() -> IntelligentScrapingConfig`

Create configuration from environment variables.

**Returns:**
- `IntelligentScrapingConfig`: Configuration loaded from environment

**Environment Variables:**
- `ORCHESTRATOR_MODEL`: Model for orchestrator (default: "gpt-4o-mini")
- `PLANNING_AGENT_MODEL`: Model for planning agent (default: "gpt-4o-mini")
- `QUALITY_THRESHOLD`: Default quality threshold (default: 50.0)
- `MAX_CONCURRENT_REQUESTS`: Max concurrent requests (default: 5)
- `REQUEST_DELAY`: Delay between requests (default: 1.0)
- `ENABLE_MONITORING`: Enable monitoring (default: true)
- `RESULTS_DIRECTORY`: Results directory (default: "./results")

**Example:**
```python
# Set environment variables
os.environ["ORCHESTRATOR_MODEL"] = "gpt-4"
os.environ["QUALITY_THRESHOLD"] = "75.0"

# Load configuration
config = IntelligentScrapingConfig.from_env()
```

## Context Providers

Context providers inject dynamic information into agent prompts to enhance decision-making.

### WebsiteAnalysisContextProvider

Provides website structure analysis context.

```python
class WebsiteAnalysisContextProvider(SystemPromptContextProviderBase):
    """
    Provides dynamic website analysis context to agents.
    
    This context provider analyzes website structure, content patterns,
    and navigation information to help agents make better scraping decisions.
    
    Attributes:
        analysis_results (Optional[WebsiteStructureAnalysis]): Current analysis
        content_patterns (List[ContentPattern]): Detected content patterns
        navigation_info (Optional[NavigationInfo]): Navigation structure
    """
```

#### Methods

##### `set_analysis_results(analysis: Dict[str, Any])`

Set website analysis results for context.

**Parameters:**
- `analysis` (Dict[str, Any]): Website analysis data

**Example:**
```python
provider = WebsiteAnalysisContextProvider()
provider.set_analysis_results({
    "site_type": "e-commerce",
    "complexity_score": 7.5,
    "content_patterns": [
        {"pattern": "product_grid", "confidence": 0.9}
    ]
})
```

##### `get_info() -> str`

Get formatted context information.

**Returns:**
- `str`: Formatted context string for agent prompts

### ScrapingResultsContextProvider

Provides historical scraping results context.

```python
class ScrapingResultsContextProvider(SystemPromptContextProviderBase):
    """
    Provides scraping results context for result processing.
    
    This provider maintains historical scraping data and performance
    metrics to help agents optimize future scraping operations.
    """
```

## Tools and Factories

### ToolFactory

Factory class for creating and managing scraper tool instances.

```python
class ToolFactory:
    """
    Factory for creating and managing scraper tool instances.
    
    This factory provides a centralized way to create, configure, and
    manage multiple scraper tool instances with different configurations.
    
    Attributes:
        intelligent_config (IntelligentScrapingConfig): Base configuration
        tool_instances (Dict[str, AtomicScraperTool]): Active tool instances
        cached_configs (Dict[str, AtomicScraperToolConfig]): Cached configurations
    """
```

#### Methods

##### `create_tool(base_url: str, instance_id: str, config_overrides: Dict[str, Any] = None) -> AtomicScraperTool`

Create a new scraper tool instance.

**Parameters:**
- `base_url` (str): Base URL for the scraper
- `instance_id` (str): Unique identifier for the instance
- `config_overrides` (Dict[str, Any], optional): Configuration overrides

**Returns:**
- `AtomicScraperTool`: Configured scraper tool instance

**Example:**
```python
factory = ToolFactory(intelligent_config)
scraper = factory.create_tool(
    base_url="https://example.com",
    instance_id="main_scraper",
    config_overrides={"timeout": 30, "min_quality_score": 70.0}
)
```

##### `list_tool_instances() -> Dict[str, Dict[str, Any]]`

List all active tool instances.

**Returns:**
- `Dict[str, Dict[str, Any]]`: Dictionary of instance information

##### `get_factory_stats() -> Dict[str, Any]`

Get factory statistics and metrics.

**Returns:**
- `Dict[str, Any]`: Factory statistics including instance counts and cache info

## Export System

### ExportManager

Manages data export in multiple formats.

```python
class ExportManager:
    """
    Manages export of scraped data to various formats.
    
    Supports JSON, CSV, Markdown, and Excel export formats with
    configurable options and validation.
    
    Attributes:
        config (ExportConfiguration): Export configuration
        validators (Dict[ExportFormat, ExportValidator]): Format validators
    """
```

#### Methods

##### `export_data(data: ExportData) -> ExportResult`

Export data to configured format.

**Parameters:**
- `data` (ExportData): Data to export including results and metadata

**Returns:**
- `ExportResult`: Export result with file path and statistics

**Raises:**
- `ValidationError`: If data validation fails
- `ExportError`: If export operation fails

**Example:**
```python
export_config = ExportConfiguration(
    format=ExportFormat.JSON,
    output_directory="./exports",
    include_metadata=True
)

manager = ExportManager(export_config)
result = manager.export_data(export_data)

print(f"Exported to: {result.file_path}")
print(f"Records: {result.records_exported}")
```

### ExportConfiguration

Configuration for export operations.

```python
class ExportConfiguration(BaseModel):
    """
    Configuration for data export operations.
    
    Attributes:
        format (ExportFormat): Target export format
        output_directory (str): Output directory path
        filename_prefix (str): Prefix for generated filenames
        include_timestamp (bool): Include timestamp in filename
        include_metadata (bool): Include metadata in export
        json_indent (int): JSON indentation spaces
        csv_delimiter (str): CSV field delimiter
        excel_sheet_name (str): Excel worksheet name
    """
```

## Monitoring and Metrics

### MonitoringDashboard

Real-time monitoring dashboard with Rich interface.

```python
class MonitoringDashboard:
    """
    Real-time monitoring dashboard for scraper operations.
    
    Provides live updating Rich console interface with metrics display,
    visual alerts, and performance tracking.
    
    Attributes:
        console (Console): Rich console instance
        refresh_rate (float): Dashboard refresh rate in Hz
        enable_sound_alerts (bool): Enable audio alerts
        max_history (int): Maximum history entries to keep
    """
```

#### Methods

##### `start()`

Start the monitoring dashboard.

**Example:**
```python
dashboard = MonitoringDashboard(refresh_rate=2.0)
dashboard.start()
```

##### `update_instance_data(instances: List[Dict[str, Any]])`

Update dashboard with instance data.

**Parameters:**
- `instances` (List[Dict[str, Any]]): List of instance statistics

##### `stop()`

Stop the monitoring dashboard and cleanup resources.

### AlertManager

Manages alerts and notifications.

```python
class AlertManager:
    """
    Manages alerts and notifications for the monitoring system.
    
    Provides alert creation, management, and notification capabilities
    with different severity levels and filtering options.
    """
```

#### Methods

##### `create_alert(title: str, message: str, level: AlertLevel, source: str = "system")`

Create a new alert.

**Parameters:**
- `title` (str): Alert title
- `message` (str): Alert message
- `level` (AlertLevel): Alert severity level
- `source` (str): Alert source identifier

## Error Handling

### Custom Exceptions

The system defines several custom exceptions for specific error conditions:

```python
class ScrapingError(Exception):
    """Base exception for scraping operations."""
    pass

class ValidationError(ScrapingError):
    """Raised when input validation fails."""
    pass

class ConfigurationError(ScrapingError):
    """Raised when configuration is invalid."""
    pass

class TimeoutError(ScrapingError):
    """Raised when operations exceed timeout limits."""
    pass

class QualityError(ScrapingError):
    """Raised when quality thresholds are not met."""
    pass
```

### Error Recovery

The system implements automatic error recovery mechanisms:

```python
async def run_with_retry(
    operation: Callable,
    max_retries: int = 3,
    backoff_factor: float = 2.0
) -> Any:
    """
    Execute operation with automatic retry and exponential backoff.
    
    Parameters:
        operation (Callable): Operation to execute
        max_retries (int): Maximum number of retry attempts
        backoff_factor (float): Exponential backoff multiplier
    
    Returns:
        Any: Operation result
    
    Raises:
        Exception: Last exception if all retries fail
    """
```

## Usage Examples

### Basic Usage

```python
from intelligent_web_scraper import (
    IntelligentScrapingOrchestrator,
    IntelligentScrapingConfig
)

# Create configuration
config = IntelligentScrapingConfig(
    orchestrator_model="gpt-4o-mini",
    default_quality_threshold=70.0,
    enable_monitoring=True
)

# Initialize orchestrator
orchestrator = IntelligentScrapingOrchestrator(config=config)

# Execute scraping
result = await orchestrator.run({
    "scraping_request": "Extract product information",
    "target_url": "https://example.com/products",
    "max_results": 25,
    "export_format": "json"
})

print(f"Extracted {len(result.extracted_data)} items")
```

### Advanced Usage with Context Providers

```python
from intelligent_web_scraper.context_providers import (
    WebsiteAnalysisContextProvider,
    ScrapingResultsContextProvider
)

# Create context providers
website_context = WebsiteAnalysisContextProvider()
website_context.set_analysis_results({
    "site_type": "e-commerce",
    "complexity_score": 8.0
})

results_context = ScrapingResultsContextProvider()
results_context.set_historical_data(historical_results)

# Add to orchestrator
orchestrator.add_context_provider(website_context)
orchestrator.add_context_provider(results_context)

# Execute with enhanced context
result = await orchestrator.run(scraping_request)
```

### Monitoring and Export

```python
from intelligent_web_scraper.monitoring import MonitoringDashboard
from intelligent_web_scraper.export import ExportManager, ExportConfiguration

# Start monitoring
dashboard = MonitoringDashboard(refresh_rate=1.0)
dashboard.start()

# Configure export
export_config = ExportConfiguration(
    format=ExportFormat.CSV,
    include_metadata=True,
    output_directory="./results"
)

export_manager = ExportManager(export_config)

# Execute scraping with monitoring
result = await orchestrator.run(request)

# Export results
export_result = export_manager.export_data(result)
print(f"Results exported to: {export_result.file_path}")

# Stop monitoring
dashboard.stop()
```

## Type Definitions

### Input/Output Schemas

All API methods use strongly-typed Pydantic schemas for input and output:

```python
# Input schema for orchestrator
class IntelligentScrapingOrchestratorInputSchema(BaseIOSchema):
    scraping_request: str
    target_url: str
    max_results: Optional[int] = 10
    quality_threshold: Optional[float] = 50.0
    export_format: Optional[str] = "json"
    enable_monitoring: Optional[bool] = True
    concurrent_instances: Optional[int] = 1

# Output schema for orchestrator
class IntelligentScrapingOrchestratorOutputSchema(BaseIOSchema):
    scraping_plan: str
    extracted_data: List[Dict[str, Any]]
    metadata: ScrapingMetadata
    quality_score: float
    reasoning: str
    export_options: Dict[str, str]
    monitoring_report: ScrapingMonitoringReport
    instance_statistics: List[ScraperInstanceStats]
```

### Enums

```python
class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    EXCEL = "excel"

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class InstanceStatus(Enum):
    """Scraper instance status values."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"
```

This API documentation provides comprehensive coverage of all public interfaces, methods, and usage patterns in the Intelligent Web Scraper system.