    # Structured Logging and Audit Trail System

The Intelligent Web Scraper includes a comprehensive structured logging system that provides JSON-formatted logs with contextual information, performance monitoring, metrics collection, and audit trail functionality for tracking scraping activities and compliance.

## Overview

The logging system is designed to provide:

- **Structured JSON Logging**: Machine-readable logs with rich contextual information
- **Performance Monitoring**: Real-time metrics collection and performance tracking
- **Audit Trail**: Compliance-focused event logging with risk assessment
- **Contextual Information**: Rich context for all log entries including operation IDs, user IDs, and metadata
- **Multiple Output Formats**: Console, file, and specialized audit logs

## Core Components

### Log Levels

The system supports standard log levels plus specialized levels:

```python
from intelligent_web_scraper.core import LogLevel

LogLevel.TRACE        # Detailed trace information
LogLevel.DEBUG        # Debug information
LogLevel.INFO         # General information
LogLevel.WARNING      # Warning messages
LogLevel.ERROR        # Error messages
LogLevel.CRITICAL     # Critical errors
LogLevel.AUDIT        # Audit events
LogLevel.PERFORMANCE  # Performance metrics
LogLevel.SECURITY     # Security-related events
```

### Event Types

Audit events are categorized by type for compliance tracking:

```python
from intelligent_web_scraper.core import EventType

EventType.SCRAPING_STARTED           # Scraping operation began
EventType.SCRAPING_COMPLETED         # Scraping operation completed
EventType.SCRAPING_FAILED            # Scraping operation failed
EventType.DATA_EXTRACTED             # Data successfully extracted
EventType.ERROR_OCCURRED             # Error during operation
EventType.RATE_LIMIT_HIT             # Rate limit encountered
EventType.CIRCUIT_BREAKER_OPENED     # Circuit breaker opened
EventType.CIRCUIT_BREAKER_CLOSED     # Circuit breaker closed
EventType.DEGRADATION_APPLIED        # Graceful degradation applied
EventType.CONFIGURATION_CHANGED      # Configuration modified
EventType.INSTANCE_CREATED           # Instance created
EventType.INSTANCE_DESTROYED         # Instance destroyed
EventType.PERFORMANCE_THRESHOLD_EXCEEDED  # Performance threshold exceeded
EventType.SECURITY_VIOLATION         # Security violation detected
EventType.COMPLIANCE_CHECK           # Compliance check performed
```

## Usage Examples

### Basic Logging

```python
from intelligent_web_scraper.core import StructuredLogger, LogContext
from intelligent_web_scraper.config import IntelligentScrapingConfig

# Initialize logger
config = IntelligentScrapingConfig()
logger = StructuredLogger(config)

# Set logging context
context = LogContext(
    operation_id="scrape_001",
    operation_type="web_scraping",
    url="https://example.com",
    user_id="user_123",
    session_id="session_456"
)

logger.set_context(context)

# Log messages with different levels
logger.info("Starting scraping operation")
logger.debug("Processing page content")
logger.warning("Rate limit approaching")
logger.error("Failed to parse content")
logger.critical("System resource exhausted")
```

### Context Management

```python
# Temporary context using context manager
base_context = LogContext("base_op", "system")
logger.set_context(base_context)

with logger.context_manager(LogContext("sub_op", "scraping", url="https://example.com")) as ctx:
    logger.info("This log has the sub-operation context")
    # Context is automatically restored after the block

logger.info("Back to base context")
```

### Performance Monitoring

```python
from intelligent_web_scraper.core import PerformanceMetrics
from datetime import datetime

# Start performance monitoring
performance_monitor = logger.performance_monitor
metrics = performance_monitor.start_operation("scrape_001", "web_scraping")

# Update metrics during operation
performance_monitor.update_operation_metrics(
    "scrape_001",
    memory_usage_mb=128.5,
    cpu_usage_percent=45.2,
    items_processed=50,
    custom_metrics={"pages_scraped": 5}
)

# Complete monitoring
completed_metrics = performance_monitor.complete_operation(
    "scrape_001",
    success=True,
    error_count=2
)

# Log performance metrics
logger.performance(completed_metrics)
```

### Audit Trail

```python
from intelligent_web_scraper.core import EventType

# Log audit events
event_id = logger.audit(
    event_type=EventType.SCRAPING_STARTED,
    event_data={
        "url": "https://example.com",
        "user_agent": "IntelligentScraper/1.0",
        "expected_items": 100
    },
    compliance_tags=["gdpr", "data_protection"],
    risk_level="low"
)

# Log high-risk security event
logger.audit(
    event_type=EventType.SECURITY_VIOLATION,
    event_data={
        "violation_type": "rate_limit_exceeded",
        "attempts": 10,
        "blocked_duration": 300
    },
    risk_level="high"
)
```

### Named Loggers

```python
# Get specialized loggers for different modules
scraper_logger = logger.get_logger("scraper")
parser_logger = logger.get_logger("parser")
export_logger = logger.get_logger("export")

# Each logger inherits the main configuration but can be identified separately
scraper_logger.info("Scraper module initialized")
parser_logger.debug("Parsing HTML content")
export_logger.info("Exporting data to CSV")
```

## Log Output Format

### Structured JSON Format

All logs are output in structured JSON format:

```json
{
  "timestamp": "2024-01-15T10:30:45.123456Z",
  "level": "INFO",
  "logger": "intelligent_web_scraper.scraper",
  "message": "Scraping operation completed successfully",
  "module": "scraper",
  "function": "scrape_website",
  "line": 145,
  "thread_id": 12345,
  "process_id": 67890,
  "context": {
    "operation_id": "scrape_001",
    "operation_type": "web_scraping",
    "url": "https://example.com",
    "user_id": "user_123",
    "session_id": "session_456",
    "timestamp": "2024-01-15T10:30:45.123456Z",
    "metadata": {
      "pages_processed": 5,
      "items_extracted": 50
    }
  },
  "performance_metrics": {
    "operation_id": "scrape_001",
    "operation_type": "web_scraping",
    "start_time": "2024-01-15T10:30:40.000000Z",
    "end_time": "2024-01-15T10:30:45.123456Z",
    "duration_ms": 5123.456,
    "memory_usage_mb": 128.5,
    "cpu_usage_percent": 45.2,
    "items_processed": 50,
    "success_rate": 0.96,
    "error_count": 2
  }
}
```

### Exception Logging

When exceptions occur, they are automatically captured with full context:

```json
{
  "timestamp": "2024-01-15T10:30:45.123456Z",
  "level": "ERROR",
  "logger": "intelligent_web_scraper.scraper",
  "message": "Failed to scrape website",
  "exception": {
    "type": "ConnectionError",
    "message": "Connection timeout after 30 seconds",
    "traceback": [
      "Traceback (most recent call last):",
      "  File \"scraper.py\", line 145, in scrape_website",
      "    response = requests.get(url, timeout=30)",
      "requests.exceptions.ConnectionError: Connection timeout"
    ]
  },
  "context": {
    "operation_id": "scrape_001",
    "url": "https://example.com"
  }
}
```

## Performance Monitoring

### Real-time Metrics

```python
# Get performance summary
summary = logger.performance_monitor.get_performance_summary()

print(f"Active operations: {summary['active_operations']}")
print(f"Completed operations: {summary['completed_operations']}")
print(f"Average duration: {summary['aggregated_metrics']['average_duration_ms']:.2f}ms")
print(f"Success rate: {summary['aggregated_metrics']['successful_operations'] / summary['aggregated_metrics']['total_operations']:.2%}")
```

### Performance Thresholds

The system automatically monitors performance thresholds:

```python
# Check if metrics exceed thresholds
violations = logger.performance_monitor.check_performance_thresholds(metrics)

if violations:
    for violation in violations:
        logger.warning(f"Performance threshold exceeded: {violation}")
        
        # Log performance threshold event
        logger.audit(
            event_type=EventType.PERFORMANCE_THRESHOLD_EXCEEDED,
            event_data={"violation": violation, "metrics": metrics.to_dict()},
            risk_level="medium"
        )
```

### Custom Metrics

```python
# Add custom metrics to operations
performance_monitor.update_operation_metrics(
    "scrape_001",
    custom_metrics={
        "pages_scraped": 10,
        "images_downloaded": 25,
        "api_calls_made": 5,
        "cache_hits": 15,
        "cache_misses": 3
    }
)
```

## Audit Trail and Compliance

### Event Filtering

```python
from datetime import datetime, timedelta

audit_trail = logger.audit_trail

# Get events by type
scraping_events = audit_trail.get_events(
    event_type=EventType.SCRAPING_STARTED,
    limit=50
)

# Get events by time range
yesterday = datetime.utcnow() - timedelta(days=1)
recent_events = audit_trail.get_events(
    start_time=yesterday,
    limit=100
)

# Get security events
security_events = audit_trail.get_events(
    event_type=EventType.SECURITY_VIOLATION
)
```

### Compliance Reporting

```python
# Generate compliance report
report = audit_trail.get_compliance_report(days=7)

print(f"Report period: {report['report_period']['days']} days")
print(f"Total events: {report['total_events']}")
print(f"Event breakdown:")
for event_type, count in report['event_counts'].items():
    print(f"  {event_type}: {count}")

print(f"Risk level breakdown:")
for risk_level, count in report['risk_level_counts'].items():
    print(f"  {risk_level}: {count}")

print(f"Violations: {report['violations_count']}")
print(f"Security events: {report['security_events_count']}")
```

### Compliance Tags

```python
# Tag events for compliance tracking
logger.audit(
    event_type=EventType.DATA_EXTRACTED,
    event_data={"items_count": 100, "data_types": ["personal", "public"]},
    compliance_tags=["gdpr", "ccpa", "data_protection", "privacy"],
    risk_level="medium"
)

# Query events by compliance tags
gdpr_events = [
    event for event in audit_trail.get_events(limit=1000)
    if "gdpr" in event.compliance_tags
]
```

## Configuration

### Log File Configuration

The system automatically creates rotating log files:

- **Main Log**: `logs/intelligent_scraper.log` (10MB, 5 backups)
- **Audit Log**: `logs/audit.log` (50MB, 10 backups)  
- **Performance Log**: `logs/performance.log` (20MB, 5 backups)

### Custom Configuration

```python
# Configure audit trail
audit_trail.audit_config.update({
    'retention_days': 180,  # Keep audit logs for 6 months
    'compliance_mode': True,
    'encrypt_sensitive_data': True,
    'include_request_data': False,  # For privacy
    'include_response_data': False
})

# Configure performance thresholds
performance_monitor.thresholds.update({
    'max_duration_ms': 60000,  # 60 seconds
    'max_memory_mb': 1024,     # 1GB
    'max_cpu_percent': 90.0,   # 90%
    'min_success_rate': 0.95   # 95%
})
```

## Integration Patterns

### Pattern 1: Operation Lifecycle Logging

```python
def scrape_with_full_logging(url):
    operation_id = f"scrape_{int(time.time())}"
    context = LogContext(operation_id, "web_scraping", url=url)
    
    with logger.context_manager(context):
        # Start audit trail
        logger.audit(EventType.SCRAPING_STARTED, {"url": url})
        
        # Start performance monitoring
        metrics = logger.performance_monitor.start_operation(operation_id, "scraping")
        
        try:
            logger.info(f"Starting scraping operation for {url}")
            
            # Perform scraping
            result = perform_scraping(url)
            
            # Update metrics
            logger.performance_monitor.update_operation_metrics(
                operation_id,
                items_processed=len(result.get('items', [])),
                custom_metrics={"pages_scraped": result.get('pages_count', 1)}
            )
            
            logger.info(f"Scraping completed successfully: {len(result.get('items', []))} items")
            
            # Complete monitoring
            completed_metrics = logger.performance_monitor.complete_operation(
                operation_id, success=True
            )
            
            # Log performance
            logger.performance(completed_metrics)
            
            # Audit completion
            logger.audit(
                EventType.SCRAPING_COMPLETED,
                {"items_extracted": len(result.get('items', []))}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Scraping failed: {str(e)}")
            
            # Complete monitoring with failure
            logger.performance_monitor.complete_operation(operation_id, success=False)
            
            # Audit failure
            logger.audit(
                EventType.SCRAPING_FAILED,
                {"error": str(e), "error_type": type(e).__name__},
                risk_level="medium"
            )
            
            raise
```

### Pattern 2: Batch Processing with Monitoring

```python
def batch_process_with_monitoring(urls):
    batch_id = f"batch_{int(time.time())}"
    context = LogContext(batch_id, "batch_processing")
    
    with logger.context_manager(context):
        logger.info(f"Starting batch processing of {len(urls)} URLs")
        
        results = []
        for i, url in enumerate(urls):
            item_context = LogContext(f"{batch_id}_item_{i}", "item_processing", url=url)
            
            with logger.context_manager(item_context):
                try:
                    result = scrape_with_full_logging(url)
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Failed to process {url}: {str(e)}")
                    continue
        
        logger.info(f"Batch processing completed: {len(results)}/{len(urls)} successful")
        
        # Log batch completion
        logger.audit(
            EventType.DATA_EXTRACTED,
            {
                "batch_size": len(urls),
                "successful_items": len(results),
                "success_rate": len(results) / len(urls)
            }
        )
        
        return results
```

### Pattern 3: Health Monitoring

```python
def monitor_system_health():
    """Monitor system health and log alerts."""
    
    # Get logging statistics
    stats = logger.get_log_statistics()
    
    # Check error rates
    total_logs = stats['log_metrics']['total_logs']
    error_count = stats['log_metrics']['errors_count']
    
    if total_logs > 0:
        error_rate = error_count / total_logs
        if error_rate > 0.1:  # 10% error rate threshold
            logger.audit(
                EventType.PERFORMANCE_THRESHOLD_EXCEEDED,
                {
                    "metric": "error_rate",
                    "value": error_rate,
                    "threshold": 0.1,
                    "total_logs": total_logs,
                    "error_count": error_count
                },
                risk_level="high"
            )
    
    # Check performance metrics
    perf_summary = stats['performance_summary']
    if perf_summary['active_operations'] > 10:  # Too many concurrent operations
        logger.warning(f"High number of active operations: {perf_summary['active_operations']}")
    
    # Check audit trail
    audit_summary = stats['audit_summary']
    if audit_summary['violations_count'] > 0:
        logger.critical(f"Compliance violations detected: {audit_summary['violations_count']}")
    
    return stats
```

## Best Practices

### 1. Use Appropriate Log Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General operational information
- **WARNING**: Something unexpected but not critical
- **ERROR**: Error conditions that don't stop the application
- **CRITICAL**: Serious errors that might stop the application

### 2. Provide Rich Context

Always set appropriate context for operations:

```python
context = LogContext(
    operation_id="unique_operation_id",
    operation_type="descriptive_type",
    url="target_url",
    user_id="user_identifier",
    session_id="session_identifier",
    metadata={"additional": "context"}
)
```

### 3. Monitor Performance Proactively

- Set appropriate performance thresholds
- Monitor trends over time
- Set up alerts for threshold violations
- Use custom metrics for domain-specific monitoring

### 4. Maintain Audit Trail Integrity

- Use appropriate risk levels for events
- Include relevant compliance tags
- Provide detailed event data for investigations
- Regularly review compliance reports

### 5. Handle Sensitive Data Carefully

- Avoid logging sensitive information
- Use audit configuration to control data inclusion
- Consider encryption for sensitive audit data
- Follow data retention policies

This structured logging system provides comprehensive visibility into your web scraping operations while maintaining compliance and performance monitoring capabilities.