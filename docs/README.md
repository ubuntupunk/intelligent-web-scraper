# Intelligent Web Scraper Documentation

Welcome to the Intelligent Web Scraper documentation. This system provides a comprehensive, production-ready web scraping framework with advanced error handling, structured logging, and monitoring capabilities.

## Table of Contents

1. [Core Systems Overview](core_systems_overview.md) - Architecture and integration of core systems
2. [Error Handling System](error_handling.md) - Comprehensive error handling with retry logic and circuit breakers
3. [Logging System](logging_system.md) - Structured logging with performance monitoring and audit trails
4. [Export Functionality](export_functionality.md) - Data export capabilities

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd intelligent-web-scraper

# Install dependencies
poetry install
```

### Basic Usage

```python
from intelligent_web_scraper.core import EnhancedErrorHandler, StructuredLogger, LogContext
from intelligent_web_scraper.config import IntelligentScrapingConfig

# Initialize configuration
config = IntelligentScrapingConfig()

# Initialize core systems
error_handler = EnhancedErrorHandler(config)
logger = StructuredLogger(config)

# Set logging context
context = LogContext(
    operation_id="scrape_001",
    operation_type="web_scraping",
    url="https://example.com"
)
logger.set_context(context)

# Use error handling with retry
@error_handler.retry_with_backoff()
def scrape_website():
    logger.info("Starting scraping operation")
    # Your scraping logic here
    return {"items": [], "status": "success"}

# Execute with full protection
try:
    result = scrape_website()
    logger.info(f"Scraping completed: {len(result['items'])} items")
except Exception as e:
    logger.error(f"Scraping failed: {str(e)}")
    raise
```

## System Architecture

The Intelligent Web Scraper is built around two core systems:

### 1. Error Handling System
- **Automatic Error Classification**: Categorizes errors into 12+ types
- **Circuit Breaker Pattern**: Prevents cascading failures
- **Intelligent Retry Logic**: Exponential backoff with jitter
- **Graceful Degradation**: Partial result extraction on failures

### 2. Logging System
- **Structured JSON Logging**: Machine-readable logs with rich context
- **Performance Monitoring**: Real-time metrics and threshold monitoring
- **Audit Trail**: Compliance-focused event logging
- **Multiple Output Formats**: Console, file, and specialized logs

## Key Features

### Resilient Operations
- Automatic retry with configurable backoff strategies
- Circuit breakers to prevent service overload
- Graceful degradation to maintain partial functionality
- Comprehensive error statistics and monitoring

### Comprehensive Observability
- Structured JSON logs with contextual information
- Real-time performance monitoring and alerting
- Complete audit trail for compliance and debugging
- Integration between error handling and logging systems

### Production Ready
- Thread-safe implementations
- Configurable retention policies
- Performance threshold monitoring
- Compliance and governance features

## Core Components

### Error Handling Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| `ErrorClassifier` | Automatic error categorization | 12+ error categories, recovery strategies |
| `CircuitBreaker` | Prevent cascading failures | Configurable thresholds, automatic recovery |
| `RetryConfig` | Intelligent retry logic | Exponential/linear/fixed backoff, jitter |
| `GracefulDegradationManager` | Partial result extraction | Category-specific degradation strategies |
| `EnhancedErrorHandler` | Error management orchestration | Statistics, monitoring, context management |

### Logging Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| `StructuredLogger` | JSON logging with context | Multiple log levels, contextual information |
| `PerformanceMonitor` | Real-time metrics collection | Configurable thresholds, trend analysis |
| `AuditTrail` | Compliance event logging | 15+ event types, risk assessment |
| `LogContext` | Rich contextual information | Operation tracking, metadata support |
| `StructuredFormatter` | JSON log formatting | Machine-readable, exception handling |

## Configuration

### Basic Configuration

```python
from intelligent_web_scraper.config import IntelligentScrapingConfig

config = IntelligentScrapingConfig(
    max_workers=4,
    max_async_tasks=10,
    enable_monitoring=True,
    results_directory="./results"
)
```

### Error Handling Configuration

```python
from intelligent_web_scraper.core import RetryConfig, CircuitBreakerConfig

# Retry configuration
retry_config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
    backoff_strategy="exponential"
)

# Circuit breaker configuration
breaker_config = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    name="web_scraper"
)
```

### Logging Configuration

```python
# Performance monitoring thresholds
performance_thresholds = {
    'max_duration_ms': 30000,  # 30 seconds
    'max_memory_mb': 512,      # 512MB
    'max_cpu_percent': 80.0,   # 80%
    'min_success_rate': 0.9    # 90%
}

# Audit trail configuration
audit_config = {
    'enabled': True,
    'retention_days': 90,
    'compliance_mode': True,
    'encrypt_sensitive_data': True
}
```

## Usage Patterns

### Pattern 1: Simple Scraping with Protection

```python
@error_handler.retry_with_backoff(circuit_breaker_name="scraper")
def simple_scrape(url):
    with logger.context_manager(LogContext("scrape", "simple", url=url)):
        logger.info(f"Scraping {url}")
        # Your scraping logic
        return scrape_data(url)
```

### Pattern 2: Batch Processing with Monitoring

```python
def batch_scrape(urls):
    batch_id = f"batch_{int(time.time())}"
    context = LogContext(batch_id, "batch_processing")
    
    with logger.context_manager(context):
        logger.audit(EventType.SCRAPING_STARTED, {"batch_size": len(urls)})
        
        results = []
        for url in urls:
            try:
                result = simple_scrape(url)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to scrape {url}: {str(e)}")
                continue
        
        logger.audit(EventType.SCRAPING_COMPLETED, {
            "total_urls": len(urls),
            "successful": len(results),
            "success_rate": len(results) / len(urls)
        })
        
        return results
```

### Pattern 3: Health Monitoring

```python
def monitor_health():
    error_stats = error_handler.get_error_statistics()
    log_stats = logger.get_log_statistics()
    
    # Check circuit breaker states
    for name, state in error_stats['circuit_breakers'].items():
        if state['state'] == 'open':
            logger.audit(
                EventType.CIRCUIT_BREAKER_OPENED,
                {"breaker": name, "failures": state['failure_count']},
                risk_level="high"
            )
    
    # Check error rates
    if error_stats['total_errors'] > 0:
        error_rate = error_stats['total_errors'] / (error_stats['total_errors'] + log_stats['performance_summary']['completed_operations'])
        if error_rate > 0.1:  # 10% threshold
            logger.audit(
                EventType.PERFORMANCE_THRESHOLD_EXCEEDED,
                {"metric": "error_rate", "value": error_rate},
                risk_level="medium"
            )
```

## Testing

The system includes comprehensive test suites:

```bash
# Run all tests
poetry run pytest

# Run specific test suites
poetry run pytest tests/test_enhanced_error_handling.py
poetry run pytest tests/test_logging_system.py

# Run with coverage
poetry run pytest --cov=intelligent_web_scraper
```

## Monitoring and Observability

### Log Files

The system creates several log files:

- `logs/intelligent_scraper.log` - Main application logs
- `logs/audit.log` - Audit trail and compliance events
- `logs/performance.log` - Performance metrics and monitoring

### Metrics and Statistics

```python
# Get comprehensive statistics
error_stats = error_handler.get_error_statistics()
log_stats = logger.get_log_statistics()

# Monitor performance
performance_summary = logger.performance_monitor.get_performance_summary()

# Generate compliance reports
compliance_report = logger.audit_trail.get_compliance_report(days=7)
```

## Best Practices

### 1. Initialize Systems Early
Set up error handling and logging at application startup.

### 2. Use Consistent Context
Maintain consistent context across operations for better traceability.

### 3. Configure Appropriately
Tune retry logic, circuit breakers, and performance thresholds for your use case.

### 4. Monitor Proactively
Set up regular health checks and alerting for system issues.

### 5. Handle Sensitive Data Carefully
Configure audit trail settings to protect sensitive information.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[License information here]

## Support

For questions, issues, or contributions, please refer to the project repository or contact the development team.

---

This documentation provides a comprehensive guide to using the Intelligent Web Scraper's core systems. For detailed information about specific components, refer to the individual documentation files linked above.