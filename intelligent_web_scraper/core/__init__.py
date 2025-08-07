"""
Core error handling, logging, and recovery mechanisms for Intelligent Web Scraper.
"""

from .error_handling import (
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
    GracefulDegradationManager
)

from .logging_system import (
    StructuredLogger,
    LogLevel,
    EventType,
    LogContext,
    PerformanceMetrics,
    AuditEvent,
    AuditTrail,
    PerformanceMonitor,
    StructuredFormatter
)

__all__ = [
    # Error handling
    'EnhancedErrorHandler',
    'ErrorCategory',
    'ErrorSeverity', 
    'RecoveryStrategy',
    'ErrorContext',
    'ErrorInfo',
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'CircuitBreakerOpenError',
    'RetryConfig',
    'GracefulDegradationManager',
    
    # Logging system
    'StructuredLogger',
    'LogLevel',
    'EventType',
    'LogContext',
    'PerformanceMetrics',
    'AuditEvent',
    'AuditTrail',
    'PerformanceMonitor',
    'StructuredFormatter'
]