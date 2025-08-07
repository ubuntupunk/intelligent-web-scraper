# Enhanced Error Handling System

The Intelligent Web Scraper includes a comprehensive error handling system that provides categorized error handling strategies, retry logic with exponential backoff, circuit breaker patterns, and graceful degradation with partial result extraction capabilities.

## Overview

The error handling system is designed to make web scraping operations more resilient and reliable by:

- Automatically categorizing errors and applying appropriate recovery strategies
- Implementing retry logic with intelligent backoff to handle transient failures
- Using circuit breaker patterns to prevent cascading failures
- Providing graceful degradation to extract partial results when full operations fail
- Offering comprehensive error statistics and monitoring

## Core Components

### ErrorCategory

Errors are automatically categorized into different types for targeted handling:

```python
from intelligent_web_scraper.core import ErrorCategory

# Available categories:
ErrorCategory.NETWORK      # Network connectivity issues
ErrorCategory.PARSING      # HTML/JSON parsing errors
ErrorCategory.VALIDATION   # Data validation failures
ErrorCategory.RATE_LIMIT   # Rate limiting responses
ErrorCategory.AUTHENTICATION  # Auth failures
ErrorCategory.PERMISSION   # Access denied errors
ErrorCategory.RESOURCE     # Memory/CPU resource issues
ErrorCategory.TIMEOUT      # Operation timeouts
ErrorCategory.QUALITY      # Data quality issues
ErrorCategory.CONFIGURATION  # Config errors
ErrorCategory.SYSTEM       # System-level errors
ErrorCategory.UNKNOWN      # Unclassified errors
```

### Recovery Strategies

Each error category has an associated recovery strategy:

```python
from intelligent_web_scraper.core import RecoveryStrategy

RecoveryStrategy.RETRY     # Retry the operation
RecoveryStrategy.FALLBACK  # Use fallback mechanism
RecoveryStrategy.SKIP      # Skip and continue
RecoveryStrategy.ABORT     # Stop operation
RecoveryStrategy.DEGRADE   # Apply graceful degradation
RecoveryStrategy.ESCALATE  # Escalate to higher level
```

## Usage Examples

### Basic Error Handling

```python
from intelligent_web_scraper.core import EnhancedErrorHandler, ErrorContext
from intelligent_web_scraper.config import IntelligentScrapingConfig

# Initialize error handler
config = IntelligentScrapingConfig()
error_handler = EnhancedErrorHandler(config)

# Use error handling context
with error_handler.error_handling_context("op_123", "scraping", url="https://example.com") as context:
    # Your scraping operation here
    result = perform_scraping_operation()
```

### Retry with Exponential Backoff

```python
from intelligent_web_scraper.core import RetryConfig

# Configure retry behavior
retry_config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
    backoff_strategy="exponential"
)

# Apply retry decorator
@error_handler.retry_with_backoff(retry_config)
def scrape_with_retry():
    # This function will be retried on failure
    return scrape_website()

result = scrape_with_retry()
```

### Circuit Breaker Pattern

```python
from intelligent_web_scraper.core import CircuitBreakerConfig

# Configure circuit breaker
breaker_config = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    name="website_scraper"
)

# Get circuit breaker
breaker = error_handler.get_circuit_breaker("website_scraper", breaker_config)

# Use circuit breaker
@breaker
def protected_operation():
    return scrape_website()

try:
    result = protected_operation()
except CircuitBreakerOpenError:
    print("Circuit breaker is open - service unavailable")
```

### Graceful Degradation

```python
from intelligent_web_scraper.core import ErrorContext

def scraping_operation_with_degradation():
    context = ErrorContext("op_123", "scraping")
    partial_data = {"items": [], "total_expected": 100}
    
    def scraping_function():
        # Your scraping logic here
        return full_scraping_result()
    
    # Handle with degradation on failure
    result = error_handler.handle_with_degradation(
        scraping_function,
        partial_data,
        context
    )
    
    if result.get('status') == 'degraded':
        print(f"Operation degraded: {result['degradation_note']}")
        print(f"Extracted {result['total_extracted']} of {result['total_expected']} items")
    
    return result
```

### Async Operations

```python
import asyncio

# Async retry decorator
@error_handler.async_retry_with_backoff(retry_config)
async def async_scrape_with_retry():
    # Async scraping operation
    return await async_scrape_website()

# Usage
result = await async_scrape_with_retry()
```

## Error Classification

The system automatically classifies errors based on their type and context:

```python
from intelligent_web_scraper.core import ErrorClassifier, ErrorContext

classifier = ErrorClassifier()
context = ErrorContext("op_123", "scraping")

try:
    # Some operation that might fail
    result = risky_operation()
except Exception as e:
    error_info = classifier.classify_error(e, context)
    
    print(f"Error category: {error_info.category}")
    print(f"Severity: {error_info.severity}")
    print(f"Recovery strategy: {error_info.recovery_strategy}")
    print(f"Is retryable: {error_info.is_retryable}")
    print(f"Suggested actions: {error_info.suggested_actions}")
```

## Configuration

### Retry Configuration

```python
# Exponential backoff (default)
exponential_config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
    backoff_strategy="exponential"
)

# Linear backoff
linear_config = RetryConfig(
    max_attempts=5,
    base_delay=2.0,
    backoff_strategy="linear"
)

# Fixed delay
fixed_config = RetryConfig(
    max_attempts=3,
    base_delay=5.0,
    backoff_strategy="fixed"
)
```

### Circuit Breaker Configuration

```python
# Basic circuit breaker
basic_breaker = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    name="basic_service"
)

# Sensitive service (lower threshold)
sensitive_breaker = CircuitBreakerConfig(
    failure_threshold=2,
    recovery_timeout=30.0,
    expected_exception=requests.exceptions.RequestException,
    name="sensitive_service"
)
```

## Monitoring and Statistics

### Error Statistics

```python
# Get comprehensive error statistics
stats = error_handler.get_error_statistics()

print(f"Total errors: {stats['total_errors']}")
print(f"Errors by category: {stats['errors_by_category']}")
print(f"Errors by severity: {stats['errors_by_severity']}")
print(f"Recovery success rate: {stats['recovery_success_rate']:.2%}")
print(f"Degradation rate: {stats['degradation_rate']:.2%}")

# Circuit breaker states
for name, state in stats['circuit_breakers'].items():
    print(f"Circuit breaker '{name}': {state['state']}")
    print(f"  Failures: {state['failure_count']}/{state['failure_threshold']}")
```

### Reset Statistics

```python
# Reset all error statistics and circuit breakers
error_handler.reset_statistics()
```

## Best Practices

### 1. Use Appropriate Error Categories

Ensure your custom exceptions are properly classified by extending standard exception types or configuring custom classification rules.

### 2. Configure Retry Logic Appropriately

- Use exponential backoff for network operations
- Use linear backoff for resource contention
- Use fixed delays for rate-limited APIs
- Always include jitter to prevent thundering herd problems

### 3. Set Reasonable Circuit Breaker Thresholds

- Lower thresholds (2-3 failures) for critical services
- Higher thresholds (5-10 failures) for less critical operations
- Adjust recovery timeouts based on expected service recovery time

### 4. Implement Graceful Degradation

- Always provide partial data when possible
- Include clear degradation messages for users
- Consider the business impact of degraded operations

### 5. Monitor Error Patterns

- Regularly review error statistics
- Set up alerts for high error rates
- Monitor circuit breaker state changes
- Track degradation rates to identify systemic issues

## Integration with Logging

The error handling system integrates seamlessly with the structured logging system:

```python
from intelligent_web_scraper.core import StructuredLogger, LogContext

# Initialize logger
logger = StructuredLogger(config)

# Set context for error correlation
context = LogContext("op_123", "scraping", url="https://example.com")
logger.set_context(context)

# Errors are automatically logged with context
with error_handler.error_handling_context("op_123", "scraping") as ctx:
    try:
        result = scraping_operation()
    except Exception as e:
        # Error is automatically logged with full context
        logger.error(f"Scraping failed: {str(e)}")
        raise
```

## Error Handling Patterns

### Pattern 1: Resilient Web Scraping

```python
@error_handler.retry_with_backoff(
    RetryConfig(max_attempts=3, base_delay=2.0),
    circuit_breaker_name="website_scraper"
)
def resilient_scrape(url):
    with error_handler.error_handling_context("scrape", "web_scraping", url=url):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return parse_response(response)
```

### Pattern 2: Batch Processing with Degradation

```python
def batch_scrape_with_degradation(urls):
    results = []
    partial_data = {"items": results, "total_expected": len(urls)}
    
    for url in urls:
        context = ErrorContext(f"batch_{len(results)}", "batch_scraping", url=url)
        
        def scrape_single():
            return resilient_scrape(url)
        
        try:
            result = error_handler.handle_with_degradation(
                scrape_single,
                partial_data,
                context
            )
            
            if result.get('status') == 'degraded':
                # Log degradation but continue
                logger.warning(f"Degraded scraping for {url}: {result['degradation_note']}")
            
            results.extend(result.get('partial_results', []))
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {str(e)}")
            continue
    
    return results
```

### Pattern 3: Service Health Monitoring

```python
def monitor_service_health():
    stats = error_handler.get_error_statistics()
    
    # Check circuit breaker states
    for name, state in stats['circuit_breakers'].items():
        if state['state'] == 'open':
            logger.warning(f"Circuit breaker '{name}' is open")
            # Send alert or take corrective action
    
    # Check error rates
    if stats['total_errors'] > 0:
        error_rate = stats['total_errors'] / (stats['total_errors'] + stats.get('successful_operations', 1))
        if error_rate > 0.1:  # 10% error rate threshold
            logger.warning(f"High error rate detected: {error_rate:.2%}")
    
    return stats
```

This error handling system provides a robust foundation for building resilient web scraping applications that can handle various failure scenarios gracefully while maintaining operational visibility and control.