"""
Unit tests for structured logging and audit trail system.

Tests comprehensive structured logging with appropriate levels, contextual information,
performance monitoring, metrics collection, and audit trail functionality.
"""

import json
import logging
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import threading

from intelligent_web_scraper.core.logging_system import (
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
from intelligent_web_scraper.config import IntelligentScrapingConfig


class TestLogContext:
    """Test LogContext functionality."""
    
    def test_log_context_creation(self):
        """Test creating log context."""
        context = LogContext(
            operation_id="test_op_123",
            operation_type="scraping",
            user_id="user_456",
            session_id="session_789",
            url="https://example.com",
            metadata={"key": "value"}
        )
        
        assert context.operation_id == "test_op_123"
        assert context.operation_type == "scraping"
        assert context.user_id == "user_456"
        assert context.session_id == "session_789"
        assert context.url == "https://example.com"
        assert context.metadata == {"key": "value"}
        assert isinstance(context.timestamp, datetime)
    
    def test_log_context_to_dict(self):
        """Test converting log context to dictionary."""
        context = LogContext(
            operation_id="test_op_123",
            operation_type="scraping",
            metadata={"test": "data"}
        )
        
        context_dict = context.to_dict()
        
        assert context_dict['operation_id'] == "test_op_123"
        assert context_dict['operation_type'] == "scraping"
        assert 'timestamp' in context_dict
        assert context_dict['metadata'] == {"test": "data"}
        
        # Verify timestamp is ISO format string
        datetime.fromisoformat(context_dict['timestamp'])


class TestPerformanceMetrics:
    """Test PerformanceMetrics functionality."""
    
    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        start_time = datetime.utcnow()
        metrics = PerformanceMetrics(
            operation_id="op_123",
            operation_type="scraping",
            start_time=start_time,
            memory_usage_mb=128.5,
            cpu_usage_percent=45.2,
            items_processed=100
        )
        
        assert metrics.operation_id == "op_123"
        assert metrics.operation_type == "scraping"
        assert metrics.start_time == start_time
        assert metrics.memory_usage_mb == 128.5
        assert metrics.cpu_usage_percent == 45.2
        assert metrics.items_processed == 100
        assert metrics.end_time is None
        assert metrics.duration_ms is None
    
    def test_performance_metrics_finalize(self):
        """Test finalizing performance metrics."""
        start_time = datetime.utcnow()
        metrics = PerformanceMetrics(
            operation_id="op_123",
            operation_type="scraping",
            start_time=start_time
        )
        
        # Wait a small amount to ensure duration > 0
        time.sleep(0.01)
        
        end_time = datetime.utcnow()
        metrics.finalize(end_time)
        
        assert metrics.end_time == end_time
        assert metrics.duration_ms is not None
        assert metrics.duration_ms > 0
    
    def test_performance_metrics_to_dict(self):
        """Test converting performance metrics to dictionary."""
        start_time = datetime.utcnow()
        metrics = PerformanceMetrics(
            operation_id="op_123",
            operation_type="scraping",
            start_time=start_time,
            custom_metrics={"test": "value"}
        )
        
        metrics.finalize()
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict['operation_id'] == "op_123"
        assert metrics_dict['operation_type'] == "scraping"
        assert 'start_time' in metrics_dict
        assert 'end_time' in metrics_dict
        assert 'duration_ms' in metrics_dict
        assert metrics_dict['custom_metrics'] == {"test": "value"}
        
        # Verify timestamps are ISO format strings
        datetime.fromisoformat(metrics_dict['start_time'])
        datetime.fromisoformat(metrics_dict['end_time'])


class TestAuditEvent:
    """Test AuditEvent functionality."""
    
    def test_audit_event_creation(self):
        """Test creating audit event."""
        context = LogContext("op_123", "scraping")
        event = AuditEvent(
            event_id="event_456",
            event_type=EventType.SCRAPING_STARTED,
            timestamp=datetime.utcnow(),
            context=context,
            event_data={"url": "https://example.com"},
            compliance_tags=["gdpr", "privacy"],
            risk_level="medium"
        )
        
        assert event.event_id == "event_456"
        assert event.event_type == EventType.SCRAPING_STARTED
        assert event.context == context
        assert event.event_data == {"url": "https://example.com"}
        assert event.compliance_tags == ["gdpr", "privacy"]
        assert event.risk_level == "medium"
    
    def test_audit_event_to_dict(self):
        """Test converting audit event to dictionary."""
        context = LogContext("op_123", "scraping")
        event = AuditEvent(
            event_id="event_456",
            event_type=EventType.DATA_EXTRACTED,
            timestamp=datetime.utcnow(),
            context=context,
            event_data={"items": 50}
        )
        
        event_dict = event.to_dict()
        
        assert event_dict['event_id'] == "event_456"
        assert event_dict['event_type'] == "data_extracted"
        assert 'timestamp' in event_dict
        assert 'context' in event_dict
        assert event_dict['event_data'] == {"items": 50}
        
        # Verify timestamp is ISO format string
        datetime.fromisoformat(event_dict['timestamp'])


class TestStructuredFormatter:
    """Test StructuredFormatter functionality."""
    
    def test_structured_formatter_basic(self):
        """Test basic structured formatting."""
        formatter = StructuredFormatter(include_context=False)
        
        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data['level'] == 'INFO'
        assert log_data['logger'] == 'test_logger'
        assert log_data['message'] == 'Test message'
        assert log_data['line'] == 10
        assert 'timestamp' in log_data
    
    def test_structured_formatter_with_context(self):
        """Test structured formatting with context."""
        formatter = StructuredFormatter(include_context=True)
        context = LogContext("op_123", "test")
        
        # Create a log record with context
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message with context",
            args=(),
            exc_info=None
        )
        record.context = context
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data['message'] == 'Test message with context'
        assert 'context' in log_data
        assert log_data['context']['operation_id'] == "op_123"
        assert log_data['context']['operation_type'] == "test"


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return IntelligentScrapingConfig(
            max_workers=2,
            max_async_tasks=5,
            enable_monitoring=True
        )
    
    @pytest.fixture
    def performance_monitor(self, config):
        """Create performance monitor for testing."""
        return PerformanceMonitor(config)
    
    def test_performance_monitor_creation(self, performance_monitor):
        """Test creating performance monitor."""
        assert performance_monitor.active_operations == {}
        assert len(performance_monitor.completed_operations) == 0
        assert 'max_duration_ms' in performance_monitor.thresholds
        assert 'total_operations' in performance_monitor.aggregated_metrics
    
    def test_start_operation(self, performance_monitor):
        """Test starting operation monitoring."""
        metrics = performance_monitor.start_operation("op_123", "scraping")
        
        assert metrics.operation_id == "op_123"
        assert metrics.operation_type == "scraping"
        assert isinstance(metrics.start_time, datetime)
        assert "op_123" in performance_monitor.active_operations
    
    def test_complete_operation(self, performance_monitor):
        """Test completing operation monitoring."""
        # Start operation
        metrics = performance_monitor.start_operation("op_123", "scraping")
        
        # Wait a bit
        time.sleep(0.01)
        
        # Complete operation
        completed_metrics = performance_monitor.complete_operation("op_123", success=True)
        
        assert completed_metrics is not None
        assert completed_metrics.operation_id == "op_123"
        assert completed_metrics.duration_ms is not None
        assert completed_metrics.duration_ms > 0
        assert completed_metrics.success_rate == 1.0
        assert "op_123" not in performance_monitor.active_operations
        assert len(performance_monitor.completed_operations) == 1


class TestAuditTrail:
    """Test AuditTrail functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return IntelligentScrapingConfig(
            max_workers=2,
            max_async_tasks=5,
            enable_monitoring=True
        )
    
    @pytest.fixture
    def audit_trail(self, config):
        """Create audit trail for testing."""
        return AuditTrail(config)
    
    def test_audit_trail_creation(self, audit_trail):
        """Test creating audit trail."""
        assert len(audit_trail.events) == 0
        assert audit_trail.audit_config['enabled'] is True
        assert audit_trail.audit_config['retention_days'] == 90
        assert len(audit_trail.event_counters) == 0
    
    def test_log_event(self, audit_trail):
        """Test logging audit event."""
        context = LogContext("op_123", "scraping")
        
        event_id = audit_trail.log_event(
            event_type=EventType.SCRAPING_STARTED,
            context=context,
            event_data={"url": "https://example.com"},
            compliance_tags=["gdpr"],
            risk_level="low"
        )
        
        assert event_id != ""
        assert len(audit_trail.events) == 1
        assert audit_trail.event_counters[EventType.SCRAPING_STARTED.value] == 1
        
        # Check the logged event
        event = audit_trail.events[0]
        assert event.event_id == event_id
        assert event.event_type == EventType.SCRAPING_STARTED
        assert event.context == context
        assert event.event_data == {"url": "https://example.com"}
        assert event.compliance_tags == ["gdpr"]
        assert event.risk_level == "low"


class TestStructuredLogger:
    """Test StructuredLogger functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = IntelligentScrapingConfig(
                max_workers=2,
                max_async_tasks=5,
                enable_monitoring=True,
                results_directory=temp_dir
            )
            yield config
    
    @pytest.fixture
    def structured_logger(self, config):
        """Create structured logger for testing."""
        return StructuredLogger(config)
    
    def test_structured_logger_creation(self, structured_logger):
        """Test creating structured logger."""
        assert structured_logger.performance_monitor is not None
        assert structured_logger.audit_trail is not None
        assert structured_logger.logger is not None
        assert len(structured_logger._loggers) == 0
        assert structured_logger.log_metrics['total_logs'] == 0
    
    def test_basic_logging_methods(self, structured_logger):
        """Test basic logging methods."""
        context = LogContext("op_123", "test")
        structured_logger.set_context(context)
        
        # Test different log levels
        structured_logger.info("Info message")
        structured_logger.debug("Debug message")
        structured_logger.warning("Warning message")
        structured_logger.error("Error message")
        structured_logger.critical("Critical message")
        
        # Check metrics
        assert structured_logger.log_metrics['total_logs'] == 5
        assert structured_logger.log_metrics['logs_by_level']['INFO'] == 1
        assert structured_logger.log_metrics['logs_by_level']['DEBUG'] == 1
        assert structured_logger.log_metrics['logs_by_level']['WARNING'] == 1
        assert structured_logger.log_metrics['logs_by_level']['ERROR'] == 1
        assert structured_logger.log_metrics['logs_by_level']['CRITICAL'] == 1
        assert structured_logger.log_metrics['errors_count'] == 1
        assert structured_logger.log_metrics['warnings_count'] == 1


if __name__ == "__main__":
    pytest.main([__file__])