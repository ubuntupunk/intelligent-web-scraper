"""
Enhanced Error Handling System for Intelligent Web Scraper.

This module implements comprehensive error handling with categorized error handling strategies,
retry logic with exponential backoff, circuit breaker patterns, and graceful degradation
with partial result extraction capabilities.
"""

import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Type, Tuple
from functools import wraps
import random
import threading
from contextlib import contextmanager

from ..config import IntelligentScrapingConfig


logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors for different handling strategies."""
    NETWORK = "network"
    PARSING = "parsing"
    VALIDATION = "validation"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    QUALITY = "quality"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity levels for error classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    DEGRADE = "degrade"
    ESCALATE = "escalate"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation_id: str
    operation_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    url: Optional[str] = None
    attempt_number: int = 1
    max_attempts: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'operation_id': self.operation_id,
            'operation_type': self.operation_type,
            'timestamp': self.timestamp.isoformat(),
            'url': self.url,
            'attempt_number': self.attempt_number,
            'max_attempts': self.max_attempts,
            'metadata': self.metadata
        }


@dataclass
class ErrorInfo:
    """Comprehensive error information."""
    error: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    context: ErrorContext
    is_retryable: bool = True
    retry_delay: float = 1.0
    max_retries: int = 3
    partial_results: Optional[Dict[str, Any]] = None
    suggested_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and monitoring."""
        return {
            'error_type': type(self.error).__name__,
            'error_message': str(self.error),
            'category': self.category.value,
            'severity': self.severity.value,
            'recovery_strategy': self.recovery_strategy.value,
            'is_retryable': self.is_retryable,
            'retry_delay': self.retry_delay,
            'max_retries': self.max_retries,
            'context': self.context.to_dict(),
            'partial_results_available': self.partial_results is not None,
            'suggested_actions': self.suggested_actions,
            'traceback': traceback.format_exception(type(self.error), self.error, self.error.__traceback__)
        }


class CircuitBreakerState(Enum):
    """States for circuit breaker pattern."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception
    name: str = "default"


class CircuitBreaker:
    """
    Circuit breaker implementation for preventing cascading failures.
    
    The circuit breaker monitors failures and prevents calls to failing services
    when failure rate exceeds threshold, allowing time for recovery.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.success_count = 0
        self.lock = threading.RLock()
        
        logger.info(f"Initialized circuit breaker '{config.name}' with threshold {config.failure_threshold}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for applying circuit breaker to functions."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker '{self.config.name}' moved to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.config.name}' is OPEN. "
                        f"Next attempt in {self._time_until_reset():.1f} seconds"
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.config.expected_exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.config.recovery_timeout
    
    def _time_until_reset(self) -> float:
        """Calculate time until circuit breaker can attempt reset."""
        if self.last_failure_time is None:
            return 0.0
        
        time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return max(0.0, self.config.recovery_timeout - time_since_failure)
    
    def _on_success(self) -> None:
        """Handle successful operation."""
        self.failure_count = 0
        self.success_count += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info(f"Circuit breaker '{self.config.name}' reset to CLOSED")
    
    def _on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(
                f"Circuit breaker '{self.config.name}' opened after {self.failure_count} failures"
            )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self.lock:
            return {
                'name': self.config.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'failure_threshold': self.config.failure_threshold,
                'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
                'time_until_reset': self._time_until_reset() if self.state == CircuitBreakerState.OPEN else 0.0
            }
    
    def reset(self) -> None:
        """Manually reset circuit breaker."""
        with self.lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
            logger.info(f"Circuit breaker '{self.config.name}' manually reset")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        backoff_strategy: str = "exponential"
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.backoff_strategy = backoff_strategy
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.backoff_strategy == "exponential":
            delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        elif self.backoff_strategy == "linear":
            delay = self.base_delay * attempt
        elif self.backoff_strategy == "fixed":
            delay = self.base_delay
        else:
            delay = self.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            jitter_amount = delay * 0.1 * random.random()
            delay += jitter_amount
        
        return delay


class ErrorClassifier:
    """Classifies errors into categories and determines handling strategies."""
    
    def __init__(self):
        self.classification_rules = self._build_classification_rules()
    
    def _build_classification_rules(self) -> Dict[str, Dict[str, Any]]:
        """Build error classification rules."""
        return {
            # Network errors
            'requests.exceptions.ConnectionError': {
                'category': ErrorCategory.NETWORK,
                'severity': ErrorSeverity.HIGH,
                'recovery_strategy': RecoveryStrategy.RETRY,
                'is_retryable': True,
                'max_retries': 3,
                'retry_delay': 2.0
            },
            'requests.exceptions.Timeout': {
                'category': ErrorCategory.TIMEOUT,
                'severity': ErrorSeverity.MEDIUM,
                'recovery_strategy': RecoveryStrategy.RETRY,
                'is_retryable': True,
                'max_retries': 2,
                'retry_delay': 5.0
            },
            'requests.exceptions.HTTPError': {
                'category': ErrorCategory.NETWORK,
                'severity': ErrorSeverity.MEDIUM,
                'recovery_strategy': RecoveryStrategy.RETRY,
                'is_retryable': True,
                'max_retries': 2,
                'retry_delay': 1.0
            },
            
            # Parsing errors
            'bs4.FeatureNotFound': {
                'category': ErrorCategory.PARSING,
                'severity': ErrorSeverity.HIGH,
                'recovery_strategy': RecoveryStrategy.FALLBACK,
                'is_retryable': False,
                'max_retries': 0
            },
            'json.JSONDecodeError': {
                'category': ErrorCategory.PARSING,
                'severity': ErrorSeverity.MEDIUM,
                'recovery_strategy': RecoveryStrategy.FALLBACK,
                'is_retryable': False,
                'max_retries': 0
            },
            
            # Validation errors
            'pydantic.ValidationError': {
                'category': ErrorCategory.VALIDATION,
                'severity': ErrorSeverity.MEDIUM,
                'recovery_strategy': RecoveryStrategy.DEGRADE,
                'is_retryable': False,
                'max_retries': 0
            },
            
            # Rate limiting
            'requests.exceptions.TooManyRedirects': {
                'category': ErrorCategory.RATE_LIMIT,
                'severity': ErrorSeverity.MEDIUM,
                'recovery_strategy': RecoveryStrategy.RETRY,
                'is_retryable': True,
                'max_retries': 1,
                'retry_delay': 10.0
            },
            
            # Resource errors
            'MemoryError': {
                'category': ErrorCategory.RESOURCE,
                'severity': ErrorSeverity.CRITICAL,
                'recovery_strategy': RecoveryStrategy.DEGRADE,
                'is_retryable': False,
                'max_retries': 0
            },
            
            # System errors
            'OSError': {
                'category': ErrorCategory.SYSTEM,
                'severity': ErrorSeverity.HIGH,
                'recovery_strategy': RecoveryStrategy.RETRY,
                'is_retryable': True,
                'max_retries': 1,
                'retry_delay': 5.0
            }
        }
    
    def classify_error(self, error: Exception, context: ErrorContext) -> ErrorInfo:
        """Classify an error and determine handling strategy."""
        error_type = f"{error.__class__.__module__}.{error.__class__.__name__}"
        
        # Check for specific error type
        if error_type in self.classification_rules:
            rule = self.classification_rules[error_type]
        else:
            # Check for parent classes
            rule = None
            for rule_type, rule_config in self.classification_rules.items():
                try:
                    rule_class = self._get_class_from_string(rule_type)
                    if isinstance(error, rule_class):
                        rule = rule_config
                        break
                except (ImportError, AttributeError):
                    continue
            
            # Default classification for unknown errors
            if rule is None:
                rule = {
                    'category': ErrorCategory.UNKNOWN,
                    'severity': ErrorSeverity.MEDIUM,
                    'recovery_strategy': RecoveryStrategy.RETRY,
                    'is_retryable': True,
                    'max_retries': 1,
                    'retry_delay': 1.0
                }
        
        # Create error info
        error_info = ErrorInfo(
            error=error,
            category=rule['category'],
            severity=rule['severity'],
            recovery_strategy=rule['recovery_strategy'],
            context=context,
            is_retryable=rule['is_retryable'],
            retry_delay=rule.get('retry_delay', 1.0),
            max_retries=rule.get('max_retries', 1)
        )
        
        # Add context-specific suggestions
        error_info.suggested_actions = self._generate_suggestions(error_info)
        
        return error_info
    
    def _get_class_from_string(self, class_string: str) -> Type:
        """Get class object from string representation."""
        module_name, class_name = class_string.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    
    def _generate_suggestions(self, error_info: ErrorInfo) -> List[str]:
        """Generate suggested actions based on error type."""
        suggestions = []
        
        if error_info.category == ErrorCategory.NETWORK:
            suggestions.extend([
                "Check network connectivity",
                "Verify target URL is accessible",
                "Consider increasing timeout values",
                "Check for rate limiting or blocking"
            ])
        
        elif error_info.category == ErrorCategory.PARSING:
            suggestions.extend([
                "Verify website structure hasn't changed",
                "Check CSS selectors are still valid",
                "Consider updating parsing logic",
                "Enable fallback parsing strategies"
            ])
        
        elif error_info.category == ErrorCategory.VALIDATION:
            suggestions.extend([
                "Review data validation rules",
                "Check for unexpected data formats",
                "Consider relaxing validation constraints",
                "Enable partial result extraction"
            ])
        
        elif error_info.category == ErrorCategory.RATE_LIMIT:
            suggestions.extend([
                "Increase delay between requests",
                "Implement exponential backoff",
                "Check robots.txt compliance",
                "Consider using different user agents"
            ])
        
        elif error_info.category == ErrorCategory.RESOURCE:
            suggestions.extend([
                "Reduce concurrent operations",
                "Implement memory management",
                "Consider processing in smaller batches",
                "Monitor system resources"
            ])
        
        return suggestions


class GracefulDegradationManager:
    """Manages graceful degradation and partial result extraction."""
    
    def __init__(self):
        self.degradation_strategies = self._build_degradation_strategies()
    
    def _build_degradation_strategies(self) -> Dict[ErrorCategory, Callable]:
        """Build degradation strategies for different error categories."""
        return {
            ErrorCategory.NETWORK: self._degrade_network_error,
            ErrorCategory.PARSING: self._degrade_parsing_error,
            ErrorCategory.VALIDATION: self._degrade_validation_error,
            ErrorCategory.QUALITY: self._degrade_quality_error,
            ErrorCategory.RESOURCE: self._degrade_resource_error
        }
    
    def apply_degradation(self, error_info: ErrorInfo, partial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply graceful degradation strategy."""
        strategy = self.degradation_strategies.get(error_info.category)
        
        if strategy:
            return strategy(error_info, partial_data)
        else:
            return self._default_degradation(error_info, partial_data)
    
    def _degrade_network_error(self, error_info: ErrorInfo, partial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle network error degradation."""
        return {
            'status': 'degraded',
            'reason': 'network_error',
            'partial_results': partial_data.get('items', []),
            'total_expected': partial_data.get('total_expected', 0),
            'total_extracted': len(partial_data.get('items', [])),
            'degradation_note': 'Network issues prevented complete extraction',
            'retry_recommended': True
        }
    
    def _degrade_parsing_error(self, error_info: ErrorInfo, partial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle parsing error degradation."""
        return {
            'status': 'degraded',
            'reason': 'parsing_error',
            'partial_results': partial_data.get('items', []),
            'total_expected': partial_data.get('total_expected', 0),
            'total_extracted': len(partial_data.get('items', [])),
            'degradation_note': 'Parsing issues prevented complete extraction',
            'retry_recommended': False,
            'suggested_action': 'Update selectors or parsing logic'
        }
    
    def _degrade_validation_error(self, error_info: ErrorInfo, partial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle validation error degradation."""
        # Filter out invalid items and keep valid ones
        valid_items = []
        invalid_items = []
        
        for item in partial_data.get('items', []):
            if self._is_item_valid(item):
                valid_items.append(item)
            else:
                invalid_items.append(item)
        
        return {
            'status': 'degraded',
            'reason': 'validation_error',
            'partial_results': valid_items,
            'invalid_items': invalid_items,
            'total_expected': partial_data.get('total_expected', 0),
            'total_extracted': len(valid_items),
            'total_invalid': len(invalid_items),
            'degradation_note': 'Validation errors filtered out some results',
            'retry_recommended': False
        }
    
    def _degrade_quality_error(self, error_info: ErrorInfo, partial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quality error degradation."""
        # Keep items above minimum quality threshold
        quality_threshold = 30.0  # Lower threshold for degraded mode
        quality_items = []
        low_quality_items = []
        
        for item in partial_data.get('items', []):
            quality_score = item.get('quality_score', 0)
            if quality_score >= quality_threshold:
                quality_items.append(item)
            else:
                low_quality_items.append(item)
        
        return {
            'status': 'degraded',
            'reason': 'quality_error',
            'partial_results': quality_items,
            'low_quality_items': low_quality_items,
            'quality_threshold_used': quality_threshold,
            'total_expected': partial_data.get('total_expected', 0),
            'total_extracted': len(quality_items),
            'degradation_note': 'Quality issues reduced result set',
            'retry_recommended': True
        }
    
    def _degrade_resource_error(self, error_info: ErrorInfo, partial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource error degradation."""
        # Return whatever we managed to extract before resource exhaustion
        return {
            'status': 'degraded',
            'reason': 'resource_error',
            'partial_results': partial_data.get('items', []),
            'total_expected': partial_data.get('total_expected', 0),
            'total_extracted': len(partial_data.get('items', [])),
            'degradation_note': 'Resource constraints limited extraction',
            'retry_recommended': True,
            'suggested_action': 'Reduce batch size or concurrent operations'
        }
    
    def _default_degradation(self, error_info: ErrorInfo, partial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Default degradation strategy."""
        return {
            'status': 'degraded',
            'reason': 'unknown_error',
            'partial_results': partial_data.get('items', []),
            'total_expected': partial_data.get('total_expected', 0),
            'total_extracted': len(partial_data.get('items', [])),
            'degradation_note': f'Unknown error caused degradation: {error_info.error}',
            'retry_recommended': error_info.is_retryable
        }
    
    def _is_item_valid(self, item: Dict[str, Any]) -> bool:
        """Check if an item meets minimum validation requirements."""
        # Basic validation - item should have some content
        if not isinstance(item, dict):
            return False
        
        # Check for required fields
        required_fields = ['title', 'url', 'description']
        valid_field_count = sum(1 for field in required_fields if item.get(field))
        
        # Item is valid if it has at least one required field with content
        return valid_field_count > 0


class EnhancedErrorHandler:
    """
    Enhanced error handling system with comprehensive error management.
    
    This class provides categorized error handling, retry logic with exponential backoff,
    circuit breaker patterns, and graceful degradation capabilities.
    """
    
    def __init__(self, config: IntelligentScrapingConfig):
        self.config = config
        self.error_classifier = ErrorClassifier()
        self.degradation_manager = GracefulDegradationManager()
        
        # Circuit breakers for different services
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Error statistics
        self.error_stats = {
            'total_errors': 0,
            'errors_by_category': {},
            'errors_by_severity': {},
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            'degraded_operations': 0
        }
        self.stats_lock = threading.Lock()
        
        logger.info("Enhanced error handler initialized")
    
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker for a service."""
        if name not in self.circuit_breakers:
            if config is None:
                config = CircuitBreakerConfig(name=name)
            self.circuit_breakers[name] = CircuitBreaker(config)
        
        return self.circuit_breakers[name]
    
    @contextmanager
    def error_handling_context(self, operation_id: str, operation_type: str, **context_kwargs):
        """Context manager for comprehensive error handling."""
        context = ErrorContext(
            operation_id=operation_id,
            operation_type=operation_type,
            **context_kwargs
        )
        
        try:
            yield context
        except Exception as e:
            error_info = self.error_classifier.classify_error(e, context)
            self._update_error_stats(error_info)
            
            # Log the error
            logger.error(
                f"Error in operation {operation_id}: {error_info.to_dict()}",
                exc_info=True
            )
            
            # Re-raise for handling by retry decorator or caller
            raise
    
    def retry_with_backoff(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_name: Optional[str] = None
    ):
        """Decorator for retry logic with exponential backoff."""
        if retry_config is None:
            retry_config = RetryConfig()
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_with_retry(
                    func, args, kwargs, retry_config, circuit_breaker_name
                )
            return wrapper
        return decorator
    
    def _execute_with_retry(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        retry_config: RetryConfig,
        circuit_breaker_name: Optional[str]
    ) -> Any:
        """Execute function with retry logic and circuit breaker protection."""
        last_error = None
        
        for attempt in range(1, retry_config.max_attempts + 1):
            try:
                # Apply circuit breaker if specified
                if circuit_breaker_name:
                    circuit_breaker = self.get_circuit_breaker(circuit_breaker_name)
                    return circuit_breaker.call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_error = e
                
                # Create context for error classification
                context = ErrorContext(
                    operation_id=f"retry_{int(time.time())}",
                    operation_type=func.__name__,
                    attempt_number=attempt,
                    max_attempts=retry_config.max_attempts
                )
                
                error_info = self.error_classifier.classify_error(e, context)
                self._update_error_stats(error_info)
                
                # Check if error is retryable
                if not error_info.is_retryable or attempt >= retry_config.max_attempts:
                    logger.error(f"Non-retryable error or max attempts reached: {error_info.to_dict()}")
                    raise
                
                # Calculate delay and wait
                delay = retry_config.calculate_delay(attempt)
                logger.warning(
                    f"Attempt {attempt}/{retry_config.max_attempts} failed: {str(e)}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                
                time.sleep(delay)
        
        # This should never be reached, but just in case
        if last_error:
            raise last_error
    
    async def async_retry_with_backoff(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_name: Optional[str] = None
    ):
        """Async decorator for retry logic with exponential backoff."""
        if retry_config is None:
            retry_config = RetryConfig()
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await self._execute_async_with_retry(
                    func, args, kwargs, retry_config, circuit_breaker_name
                )
            return wrapper
        return decorator
    
    async def _execute_async_with_retry(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        retry_config: RetryConfig,
        circuit_breaker_name: Optional[str]
    ) -> Any:
        """Execute async function with retry logic and circuit breaker protection."""
        last_error = None
        
        for attempt in range(1, retry_config.max_attempts + 1):
            try:
                # Apply circuit breaker if specified
                if circuit_breaker_name:
                    circuit_breaker = self.get_circuit_breaker(circuit_breaker_name)
                    if asyncio.iscoroutinefunction(func):
                        return await circuit_breaker.call(func, *args, **kwargs)
                    else:
                        return circuit_breaker.call(func, *args, **kwargs)
                else:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
            except Exception as e:
                last_error = e
                
                # Create context for error classification
                context = ErrorContext(
                    operation_id=f"async_retry_{int(time.time())}",
                    operation_type=func.__name__,
                    attempt_number=attempt,
                    max_attempts=retry_config.max_attempts
                )
                
                error_info = self.error_classifier.classify_error(e, context)
                self._update_error_stats(error_info)
                
                # Check if error is retryable
                if not error_info.is_retryable or attempt >= retry_config.max_attempts:
                    logger.error(f"Non-retryable error or max attempts reached: {error_info.to_dict()}")
                    raise
                
                # Calculate delay and wait
                delay = retry_config.calculate_delay(attempt)
                logger.warning(
                    f"Async attempt {attempt}/{retry_config.max_attempts} failed: {str(e)}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                
                await asyncio.sleep(delay)
        
        # This should never be reached, but just in case
        if last_error:
            raise last_error
    
    def handle_with_degradation(
        self,
        operation_func: Callable,
        partial_data: Dict[str, Any],
        context: ErrorContext
    ) -> Dict[str, Any]:
        """Handle operation with graceful degradation on failure."""
        try:
            return operation_func()
        except Exception as e:
            error_info = self.error_classifier.classify_error(e, context)
            self._update_error_stats(error_info)
            
            # Apply degradation if strategy allows it
            if error_info.recovery_strategy == RecoveryStrategy.DEGRADE:
                with self.stats_lock:
                    self.error_stats['degraded_operations'] += 1
                
                degraded_result = self.degradation_manager.apply_degradation(error_info, partial_data)
                
                logger.warning(
                    f"Operation degraded due to {error_info.category.value} error: "
                    f"{degraded_result['degradation_note']}"
                )
                
                return degraded_result
            else:
                # Re-raise if degradation is not appropriate
                raise
    
    def _update_error_stats(self, error_info: ErrorInfo) -> None:
        """Update error statistics."""
        with self.stats_lock:
            self.error_stats['total_errors'] += 1
            
            # Update category stats
            category = error_info.category.value
            if category not in self.error_stats['errors_by_category']:
                self.error_stats['errors_by_category'][category] = 0
            self.error_stats['errors_by_category'][category] += 1
            
            # Update severity stats
            severity = error_info.severity.value
            if severity not in self.error_stats['errors_by_severity']:
                self.error_stats['errors_by_severity'][severity] = 0
            self.error_stats['errors_by_severity'][severity] += 1
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self.stats_lock:
            stats = self.error_stats.copy()
        
        # Add circuit breaker states
        stats['circuit_breakers'] = {
            name: breaker.get_state()
            for name, breaker in self.circuit_breakers.items()
        }
        
        # Calculate derived metrics
        total_errors = stats['total_errors']
        if total_errors > 0:
            stats['recovery_success_rate'] = (
                stats['successful_recoveries'] / stats['recovery_attempts'] * 100
                if stats['recovery_attempts'] > 0 else 0
            )
            stats['degradation_rate'] = (
                stats['degraded_operations'] / total_errors * 100
            )
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset error statistics."""
        with self.stats_lock:
            self.error_stats = {
                'total_errors': 0,
                'errors_by_category': {},
                'errors_by_severity': {},
                'recovery_attempts': 0,
                'successful_recoveries': 0,
                'degraded_operations': 0
            }
        
        # Reset circuit breakers
        for breaker in self.circuit_breakers.values():
            breaker.reset()
        
        logger.info("Error statistics and circuit breakers reset")