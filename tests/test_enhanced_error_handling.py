"""
Unit tests for enhanced error handling system.

Tests comprehensive error handling with categorized error handling strategies,
retry logic with exponential backoff, circuit breaker patterns, and graceful
degradation with partial result extraction capabilities.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import threading

from intelligent_web_scraper.core.error_handling import (
    EnhancedErrorHandler,
    ErrorCategory,
    ErrorSeverity,
    RecoveryStrategy,
    ErrorContext,
    ErrorInfo,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    RetryConfig,
    GracefulDegradationManager,
    ErrorClassifier
)
from intelligent_web_scraper.config import IntelligentScrapingConfig


class TestErrorContext:
    """Test ErrorContext functionality."""
    
    def test_error_context_creation(self):
        """Test creating error context."""
        context = ErrorContext(
            operation_id="test_op_123",
            operation_type="scraping",
            url="https://example.com",
            attempt_number=2,
            max_attempts=3
        )
        
        assert context.operation_id == "test_op_123"
        assert context.operation_type == "scraping"
        assert context.url == "https://example.com"
        assert context.attempt_number == 2
        assert context.max_attempts == 3
        assert isinstance(context.timestamp, datetime)
    
    def test_error_context_to_dict(self):
        """Test converting error context to dictionary."""
        context = ErrorContext(
            operation_id="test_op_123",
            operation_type="scraping",
            metadata={"key": "value"}
        )
        
        context_dict = context.to_dict()
        
        assert context_dict['operation_id'] == "test_op_123"
        assert context_dict['operation_type'] == "scraping"
        assert 'timestamp' in context_dict
        assert context_dict['metadata'] == {"key": "value"}


class TestErrorInfo:
    """Test ErrorInfo functionality."""
    
    def test_error_info_creation(self):
        """Test creating error info."""
        error = ValueError("Test error")
        context = ErrorContext("op_123", "test")
        
        error_info = ErrorInfo(
            error=error,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.RETRY,
            context=context,
            is_retryable=True,
            retry_delay=2.0,
            max_retries=3
        )
        
        assert error_info.error == error
        assert error_info.category == ErrorCategory.VALIDATION
        assert error_info.severity == ErrorSeverity.MEDIUM
        assert error_info.recovery_strategy == RecoveryStrategy.RETRY
        assert error_info.is_retryable is True
        assert error_info.retry_delay == 2.0
        assert error_info.max_retries == 3
    
    def test_error_info_to_dict(self):
        """Test converting error info to dictionary."""
        error = ValueError("Test error")
        context = ErrorContext("op_123", "test")
        
        error_info = ErrorInfo(
            error=error,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.RETRY,
            context=context,
            suggested_actions=["action1", "action2"]
        )
        
        error_dict = error_info.to_dict()
        
        assert error_dict['error_type'] == "ValueError"
        assert error_dict['error_message'] == "Test error"
        assert error_dict['category'] == "validation"
        assert error_dict['severity'] == "medium"
        assert error_dict['recovery_strategy'] == "retry"
        assert error_dict['suggested_actions'] == ["action1", "action2"]
        assert 'traceback' in error_dict


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""
    
    def test_circuit_breaker_creation(self):
        """Test creating circuit breaker."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            name="test_breaker"
        )
        
        breaker = CircuitBreaker(config)
        
        assert breaker.config == config
        assert breaker.failure_count == 0
        assert breaker.success_count == 0
    
    def test_circuit_breaker_success(self):
        """Test circuit breaker with successful operations."""
        config = CircuitBreakerConfig(failure_threshold=3, name="test")
        breaker = CircuitBreaker(config)
        
        def successful_operation():
            return "success"
        
        result = breaker.call(successful_operation)
        
        assert result == "success"
        assert breaker.success_count == 1
        assert breaker.failure_count == 0
    
    def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker opening after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=2, name="test")
        breaker = CircuitBreaker(config)
        
        def failing_operation():
            raise ValueError("Test failure")
        
        # First failure
        with pytest.raises(ValueError):
            breaker.call(failing_operation)
        assert breaker.failure_count == 1
        
        # Second failure - should open circuit
        with pytest.raises(ValueError):
            breaker.call(failing_operation)
        assert breaker.failure_count == 2
        
        # Third call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            breaker.call(failing_operation)
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,  # Short timeout for testing
            name="test"
        )
        breaker = CircuitBreaker(config)
        
        def failing_operation():
            raise ValueError("Test failure")
        
        def successful_operation():
            return "success"
        
        # Trigger failure to open circuit
        with pytest.raises(ValueError):
            breaker.call(failing_operation)
        
        # Should be open
        with pytest.raises(CircuitBreakerOpenError):
            breaker.call(successful_operation)
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should allow one attempt (half-open)
        result = breaker.call(successful_operation)
        assert result == "success"
        
        # Should be closed now
        result = breaker.call(successful_operation)
        assert result == "success"


class TestRetryConfig:
    """Test RetryConfig functionality."""
    
    def test_retry_config_creation(self):
        """Test creating retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=30.0,
            exponential_base=2.5,
            jitter=False,
            backoff_strategy="exponential"
        )
        
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.5
        assert config.jitter is False
        assert config.backoff_strategy == "exponential"
    
    def test_exponential_backoff_delay(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            jitter=False,
            backoff_strategy="exponential"
        )
        
        assert config.calculate_delay(1) == 1.0  # 1.0 * 2^0
        assert config.calculate_delay(2) == 2.0  # 1.0 * 2^1
        assert config.calculate_delay(3) == 4.0  # 1.0 * 2^2
        assert config.calculate_delay(4) == 8.0  # 1.0 * 2^3
    
    def test_linear_backoff_delay(self):
        """Test linear backoff delay calculation."""
        config = RetryConfig(
            base_delay=2.0,
            jitter=False,
            backoff_strategy="linear"
        )
        
        assert config.calculate_delay(1) == 2.0  # 2.0 * 1
        assert config.calculate_delay(2) == 4.0  # 2.0 * 2
        assert config.calculate_delay(3) == 6.0  # 2.0 * 3
    
    def test_fixed_backoff_delay(self):
        """Test fixed backoff delay calculation."""
        config = RetryConfig(
            base_delay=3.0,
            jitter=False,
            backoff_strategy="fixed"
        )
        
        assert config.calculate_delay(1) == 3.0
        assert config.calculate_delay(2) == 3.0
        assert config.calculate_delay(3) == 3.0
    
    def test_max_delay_limit(self):
        """Test maximum delay limit."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=5.0,
            exponential_base=2.0,
            jitter=False,
            backoff_strategy="exponential"
        )
        
        assert config.calculate_delay(1) == 1.0
        assert config.calculate_delay(2) == 2.0
        assert config.calculate_delay(3) == 4.0
        assert config.calculate_delay(4) == 5.0  # Limited by max_delay
        assert config.calculate_delay(5) == 5.0  # Limited by max_delay


class TestEnhancedErrorHandler:
    """Test EnhancedErrorHandler functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return IntelligentScrapingConfig(
            max_workers=2,
            max_async_tasks=5,
            enable_monitoring=True
        )
    
    @pytest.fixture
    def error_handler(self, config):
        """Create error handler for testing."""
        return EnhancedErrorHandler(config)
    
    def test_error_handler_creation(self, error_handler):
        """Test creating enhanced error handler."""
        assert error_handler.error_classifier is not None
        assert error_handler.degradation_manager is not None
        assert error_handler.circuit_breakers == {}
        assert error_handler.error_stats['total_errors'] == 0
    
    def test_get_circuit_breaker(self, error_handler):
        """Test getting circuit breaker."""
        breaker = error_handler.get_circuit_breaker("test_service")
        
        assert isinstance(breaker, CircuitBreaker)
        assert breaker.config.name == "test_service"
        assert "test_service" in error_handler.circuit_breakers
        
        # Getting same breaker should return existing instance
        same_breaker = error_handler.get_circuit_breaker("test_service")
        assert same_breaker is breaker
    
    def test_error_handling_context(self, error_handler):
        """Test error handling context manager."""
        with error_handler.error_handling_context("op_123", "test") as context:
            assert context.operation_id == "op_123"
            assert context.operation_type == "test"
        
        # Test with exception
        with pytest.raises(ValueError):
            with error_handler.error_handling_context("op_456", "test"):
                raise ValueError("Test error")
        
        # Should have recorded the error
        assert error_handler.error_stats['total_errors'] == 1
    
    def test_retry_with_backoff_decorator(self, error_handler):
        """Test retry with backoff decorator."""
        call_count = 0
        
        @error_handler.retry_with_backoff(RetryConfig(max_attempts=3, base_delay=0.01))
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = flaky_function()
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_retry_with_backoff(self, error_handler):
        """Test async retry with backoff decorator."""
        call_count = 0
        
        @error_handler.async_retry_with_backoff(RetryConfig(max_attempts=3, base_delay=0.01))
        async def async_flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "async_success"
        
        result = await async_flaky_function()
        
        assert result == "async_success"
        assert call_count == 3
    
    def test_handle_with_degradation_success(self, error_handler):
        """Test handling with degradation - successful operation."""
        def successful_operation():
            return {"status": "success", "data": "result"}
        
        context = ErrorContext("op_123", "test")
        result = error_handler.handle_with_degradation(
            successful_operation,
            {"items": []},
            context
        )
        
        assert result["status"] == "success"
        assert result["data"] == "result"
    
    def test_error_statistics(self, error_handler):
        """Test error statistics tracking."""
        # Simulate some errors
        context = ErrorContext("op_123", "test")
        
        # Network error
        network_error = ConnectionError("Network failed")
        network_error.__class__.__module__ = 'requests.exceptions'
        network_error.__class__.__name__ = 'ConnectionError'
        
        error_info = error_handler.error_classifier.classify_error(network_error, context)
        error_handler._update_error_stats(error_info)
        
        # Validation error
        validation_error = ValueError("Validation failed")
        validation_error.__class__.__module__ = 'pydantic'
        validation_error.__class__.__name__ = 'ValidationError'
        
        error_info = error_handler.error_classifier.classify_error(validation_error, context)
        error_handler._update_error_stats(error_info)
        
        stats = error_handler.get_error_statistics()
        
        assert stats['total_errors'] == 2
        assert stats['errors_by_category']['network'] == 1
        assert stats['errors_by_category']['validation'] == 1
        assert stats['errors_by_severity']['high'] == 1
        assert stats['errors_by_severity']['medium'] == 1
    
    def test_reset_statistics(self, error_handler):
        """Test resetting error statistics."""
        # Add some errors first
        context = ErrorContext("op_123", "test")
        error = ValueError("Test error")
        error_info = error_handler.error_classifier.classify_error(error, context)
        error_handler._update_error_stats(error_info)
        
        # Create a circuit breaker and trigger failure
        breaker = error_handler.get_circuit_breaker("test")
        try:
            breaker.call(lambda: exec('raise ValueError("test")'))
        except ValueError:
            pass
        
        # Verify stats exist
        assert error_handler.error_stats['total_errors'] > 0
        assert breaker.failure_count > 0
        
        # Reset statistics
        error_handler.reset_statistics()
        
        # Verify reset
        assert error_handler.error_stats['total_errors'] == 0
        assert error_handler.error_stats['errors_by_category'] == {}
        assert breaker.failure_count == 0


if __name__ == "__main__":
    pytest.main([__file__])