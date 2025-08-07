"""
Comprehensive unit tests for the enhanced error handling system.

This module tests error classification, retry logic, circuit breaker patterns,
and graceful degradation mechanisms.
"""

import pytest
import asyncio
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from intelligent_web_scraper.core.error_handling import (
    ErrorCategory,
    ErrorSeverity,
    RecoveryStrategy,
    ErrorContext,
    ErrorInfo,
    CircuitBreakerState,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitBreakerOpenError,
    RetryConfig,
    ErrorClassifier,
    GracefulDegradationManager,
    EnhancedErrorHandler
)
from intelligent_web_scraper.config import IntelligentScrapingConfig


class TestErrorContext:
    """Test the ErrorContext dataclass."""
    
    def test_error_context_creation(self):
        """Test error context creation with required fields."""
        context = ErrorContext(
            operation_id="test-op-123",
            operation_type="scraping"
        )
        
        assert context.operation_id == "test-op-123"
        assert context.operation_type == "scraping"
        assert isinstance(context.timestamp, datetime)
        assert context.url is None
        assert context.attempt_number == 1
        assert context.max_attempts == 3
        assert isinstance(context.metadata, dict)
        assert len(context.metadata) == 0
    
    def test_error_context_with_optional_fields(self):
        """Test error context creation with optional fields."""
        metadata = {"key": "value", "retry_count": 2}
        context = ErrorContext(
            operation_id="test-op-456",
            operation_type="parsing",
            url="https://example.com",
            attempt_number=2,
            max_attempts=5,
            metadata=metadata
        )
        
        assert context.operation_id == "test-op-456"
        assert context.operation_type == "parsing"
        assert context.url == "https://example.com"
        assert context.attempt_number == 2
        assert context.max_attempts == 5
        assert context.metadata == metadata
    
    def test_error_context_to_dict(self):
        """Test error context serialization to dictionary."""
        context = ErrorContext(
            operation_id="test-op-789",
            operation_type="validation",
            url="https://test.com",
            metadata={"test": "data"}
        )
        
        context_dict = context.to_dict()
        
        assert isinstance(context_dict, dict)
        assert context_dict['operation_id'] == "test-op-789"
        assert context_dict['operation_type'] == "validation"
        assert context_dict['url'] == "https://test.com"
        assert context_dict['attempt_number'] == 1
        assert context_dict['max_attempts'] == 3
        assert context_dict['metadata'] == {"test": "data"}
        assert 'timestamp' in context_dict
        assert isinstance(context_dict['timestamp'], str)  # ISO format


if __name__ == "__main__":
    pytest.main([__file__])