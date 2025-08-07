# Core Systems Overview

The Intelligent Web Scraper includes two foundational core systems that provide robust error handling and comprehensive logging capabilities. These systems work together to ensure reliable, observable, and maintainable web scraping operations.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Intelligent Web Scraper                     │
├─────────────────────────────────────────────────────────────┤
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│                      Core Systems                           │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │  Error Handling     │    │    Logging System           │ │
│  │  System             │◄──►│                             │ │
│  │                     │    │                             │ │
│  │ • ErrorClassifier   │    │ • StructuredLogger          │ │
│  │ • CircuitBreaker    │    │ • PerformanceMonitor        │ │
│  │ • RetryConfig       │    │ • AuditTrail                │ │
│  │ • GracefulDegradation│   │ • LogContext                │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Configuration Layer                      │
└─────────────────────────────────────────────────────────────┘
```

## Integration Between Systems

The error handling and logging systems are designed to work seamlessly together:

### 1. Automatic Error Logging

When errors occur, they are automatically logged with full context:

```python
from intelligent_web_scraper.core import EnhancedErrorHandler, StructuredLogger, LogContext

# Initialize both systems
config = IntelligentScrapingConfig()
error_handler = EnhancedErrorHandler(config)
logger = StructuredLogger(config)

# Set logging context
context = LogContext("op_123", "scraping", url="https://example.com")
logger.set_context(context)

# Errors are automatically logged with context
with error_handler.error_handling_context("op_123", "scraping") as ctx:
    try:
        result = risky_operation()
    except Exception as e:
        # Error is automatically classified and logged
        logger.error(f"Operation failed: {str(e)}")
        raise
```

### 2. Performance Monitoring with Error Correlation

Performance metrics include error information for comprehensive monitoring:

```python
# Start performance monitoring
metrics = logger.performance_monitor.start_operation("op_123", "scraping")

# Use error handling with retry
@error_handler.retry_with_backoff()
def monitored_operation():
    # Operation that might fail and retry
    return perform_scraping()

try:
    result = monitored_operation()
    # Complete with success
    completed_metrics = logger.performance_monitor.complete_operation(
        "op_123", success=True
    )
except Exception as e:
    # Complete with failure and error count
    completed_metrics = logger.performance_monitor.complete_operation(
        "op_123", success=False, error_count=1
    )
    raise

# Log performance metrics
logger.performance(completed_metrics)
```

### 3. Audit Trail for Error Events

Error events are automatically recorded in the audit trail:

```python
from intelligent_web_scraper.core import EventType

# Circuit breaker events are automatically audited
breaker = error_handler.get_circuit_breaker("service_name")

@breaker
def protected_operation():
    return service_call()

try:
    result = protected_operation()
except CircuitBreakerOpenError:
    # Circuit breaker open event is automatically logged
    logger.audit(
        EventType.CIRCUIT_BREAKER_OPENED,
        {"service": "service_name", "reason": "failure_threshold_exceeded"},
        risk_level="high"
    )
```

## Core System Features

### Error Handling System

**Key Components:**
- **ErrorClassifier**: Automatic error categorization and strategy determination
- **CircuitBreaker**: Prevents cascading failures with configurable thresholds
- **RetryConfig**: Intelligent retry logic with exponential backoff
- **GracefulDegradationManager**: Partial result extraction on failures
- **EnhancedErrorHandler**: Comprehensive error management orchestration

**Capabilities:**
- Automatic error classification into 12+ categories
- 6 different recovery strategies (retry, fallback, skip, abort, degrade, escalate)
- Circuit breaker pattern with automatic recovery
- Exponential, linear, and fixed backoff strategies
- Graceful degradation with partial result extraction
- Comprehensive error statistics and monitoring

### Logging System

**Key Components:**
- **StructuredLogger**: JSON-formatted logging with rich context
- **PerformanceMonitor**: Real-time metrics collection and analysis
- **AuditTrail**: Compliance-focused event logging
- **LogContext**: Rich contextual information for all log entries
- **StructuredFormatter**: Custom JSON formatter for machine-readable logs

**Capabilities:**
- Structured JSON logging with contextual information
- Performance monitoring with configurable thresholds
- Audit trail with 15+ event types and risk assessment
- Multiple log levels including specialized levels (AUDIT, PERFORMANCE, SECURITY)
- Rotating log files with configurable retention
- Real-time performance monitoring and alerting

## Usage Patterns

### Pattern 1: Complete Operation Lifecycle

```python
def complete_scraping_operation(url):
    """Demonstrates full integration of both core systems."""
    
    operation_id = f"scrape_{int(time.time())}"
    context = LogContext(operation_id, "web_scraping", url=url)
    
    # Initialize systems
    error_handler = EnhancedErrorHandler(config)
    logger = StructuredLogger(config)
    logger.set_context(context)
    
    # Start audit trail
    logger.audit(EventType.SCRAPING_STARTED, {"url": url})
    
    # Start performance monitoring
    metrics = logger.performance_monitor.start_operation(operation_id, "scraping")
    
    # Configure retry with circuit breaker
    retry_config = RetryConfig(max_attempts=3, base_delay=2.0)
    
    @error_handler.retry_with_backoff(retry_config, circuit_breaker_name="scraper")
    def scrape_with_protection():
        with error_handler.error_handling_context(operation_id, "scraping", url=url):
            logger.info(f"Attempting to scrape {url}")
            return perform_actual_scraping(url)
    
    try:
        # Execute with full protection
        result = scrape_with_protection()
        
        # Update performance metrics
        logger.performance_monitor.update_operation_metrics(
            operation_id,
            items_processed=len(result.get('items', [])),
            custom_metrics={"pages_scraped": result.get('page_count', 1)}
        )
        
        # Complete monitoring
        completed_metrics = logger.performance_monitor.complete_operation(
            operation_id, success=True
        )
        
        # Log performance
        logger.performance(completed_metrics)
        
        # Audit success
        logger.audit(
            EventType.SCRAPING_COMPLETED,
            {"items_extracted": len(result.get('items', []))}
        )
        
        logger.info(f"Scraping completed successfully: {len(result.get('items', []))} items")
        return result
        
    except Exception as e:
        # Handle with degradation if possible
        partial_data = {"items": [], "total_expected": 100}
        
        try:
            degraded_result = error_handler.handle_with_degradation(
                lambda: {"items": [], "status": "failed"},
                partial_data,
                context
            )
            
            # Log degradation
            logger.audit(
                EventType.DEGRADATION_APPLIED,
                {
                    "reason": str(e),
                    "partial_items": len(degraded_result.get('partial_results', []))
                },
                risk_level="medium"
            )
            
            return degraded_result
            
        except Exception as final_error:
            # Complete monitoring with failure
            logger.performance_monitor.complete_operation(operation_id, success=False)
            
            # Audit failure
            logger.audit(
                EventType.SCRAPING_FAILED,
                {"error": str(final_error), "error_type": type(final_error).__name__},
                risk_level="high"
            )
            
            logger.error(f"Scraping failed completely: {str(final_error)}")
            raise
```

### Pattern 2: System Health Monitoring

```python
def monitor_system_health():
    """Monitor both error handling and logging system health."""
    
    # Get error handling statistics
    error_stats = error_handler.get_error_statistics()
    
    # Get logging statistics
    log_stats = logger.get_log_statistics()
    
    # Combined health report
    health_report = {
        "timestamp": datetime.utcnow().isoformat(),
        "error_handling": {
            "total_errors": error_stats['total_errors'],
            "recovery_success_rate": error_stats.get('recovery_success_rate', 0),
            "degradation_rate": error_stats.get('degradation_rate', 0),
            "circuit_breakers": {
                name: state['state'] 
                for name, state in error_stats['circuit_breakers'].items()
            }
        },
        "logging": {
            "total_logs": log_stats['log_metrics']['total_logs'],
            "error_rate": log_stats['log_metrics']['errors_count'] / max(log_stats['log_metrics']['total_logs'], 1),
            "active_operations": log_stats['performance_summary']['active_operations'],
            "audit_violations": log_stats['audit_summary']['violations_count']
        }
    }
    
    # Log health report
    logger.audit(
        EventType.COMPLIANCE_CHECK,
        health_report,
        compliance_tags=["system_health", "monitoring"],
        risk_level="low" if health_report['logging']['audit_violations'] == 0 else "medium"
    )
    
    return health_report
```

### Pattern 3: Configuration Management

```python
def configure_core_systems(custom_config):
    """Configure both core systems with custom settings."""
    
    # Configure error handling
    error_handler = EnhancedErrorHandler(custom_config)
    
    # Configure circuit breakers
    critical_breaker = CircuitBreakerConfig(
        failure_threshold=2,
        recovery_timeout=30.0,
        name="critical_service"
    )
    error_handler.get_circuit_breaker("critical_service", critical_breaker)
    
    # Configure logging
    logger = StructuredLogger(custom_config)
    
    # Configure performance thresholds
    logger.performance_monitor.thresholds.update({
        'max_duration_ms': 30000,  # 30 seconds
        'max_memory_mb': 512,      # 512MB
        'max_cpu_percent': 75.0,   # 75%
        'min_success_rate': 0.9    # 90%
    })
    
    # Configure audit trail
    logger.audit_trail.audit_config.update({
        'retention_days': 365,     # 1 year retention
        'compliance_mode': True,
        'encrypt_sensitive_data': True
    })
    
    # Log configuration change
    logger.audit(
        EventType.CONFIGURATION_CHANGED,
        {
            "error_handling_config": {
                "circuit_breakers": list(error_handler.circuit_breakers.keys())
            },
            "logging_config": {
                "performance_thresholds": logger.performance_monitor.thresholds,
                "audit_retention_days": logger.audit_trail.audit_config['retention_days']
            }
        },
        compliance_tags=["configuration", "system_setup"],
        risk_level="low"
    )
    
    return error_handler, logger
```

## Benefits of Integration

### 1. **Comprehensive Observability**
- All errors are automatically logged with full context
- Performance metrics include error correlation
- Audit trail captures all significant events

### 2. **Operational Resilience**
- Circuit breakers prevent cascading failures
- Retry logic handles transient issues
- Graceful degradation maintains partial functionality

### 3. **Compliance and Governance**
- Complete audit trail for all operations
- Risk-based event classification
- Configurable data retention and privacy controls

### 4. **Performance Optimization**
- Real-time performance monitoring
- Threshold-based alerting
- Historical trend analysis

### 5. **Debugging and Troubleshooting**
- Structured logs with rich context
- Error classification and suggested actions
- Performance correlation with errors

## Best Practices for Core Systems

### 1. **Initialize Early**
Set up both systems at application startup:

```python
def initialize_application():
    config = IntelligentScrapingConfig()
    error_handler = EnhancedErrorHandler(config)
    logger = StructuredLogger(config)
    
    # Make available globally or through dependency injection
    return error_handler, logger
```

### 2. **Use Consistent Context**
Maintain consistent context across both systems:

```python
def operation_with_context(operation_id, operation_type, **context_data):
    context = LogContext(operation_id, operation_type, **context_data)
    logger.set_context(context)
    
    with error_handler.error_handling_context(operation_id, operation_type, **context_data):
        # Your operation here
        pass
```

### 3. **Monitor System Health**
Regularly check the health of both systems:

```python
# Schedule regular health checks
def scheduled_health_check():
    health_report = monitor_system_health()
    
    # Take action based on health report
    if health_report['logging']['audit_violations'] > 0:
        # Alert administrators
        send_alert("Compliance violations detected")
    
    if any(state == 'open' for state in health_report['error_handling']['circuit_breakers'].values()):
        # Alert about service issues
        send_alert("Circuit breakers open - service degradation")
```

### 4. **Configure Appropriately**
Tune both systems for your specific use case:

```python
# Production configuration
production_config = {
    "error_handling": {
        "circuit_breaker_threshold": 5,
        "retry_max_attempts": 3,
        "retry_base_delay": 2.0
    },
    "logging": {
        "performance_threshold_ms": 30000,
        "audit_retention_days": 365,
        "log_level": "INFO"
    }
}
```

The core systems provide a solid foundation for building reliable, observable, and maintainable web scraping applications. By leveraging both systems together, you get comprehensive error handling, detailed logging, performance monitoring, and compliance tracking out of the box.