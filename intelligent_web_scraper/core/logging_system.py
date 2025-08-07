"""
Structured Logging and Audit Trail System for Intelligent Web Scraper.

This module implements comprehensive structured logging with appropriate levels,
contextual information, performance monitoring, metrics collection, and audit
trail functionality for tracking scraping activities and compliance.
"""

import json
import logging
import logging.handlers
import os
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import contextmanager
import uuid
import traceback
from collections import defaultdict, deque

from ..config import IntelligentScrapingConfig


class LogLevel(Enum):
    """Enhanced log levels for structured logging."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"
    PERFORMANCE = "PERFORMANCE"
    SECURITY = "SECURITY"


class EventType(Enum):
    """Types of events for audit trail."""
    SCRAPING_STARTED = "scraping_started"
    SCRAPING_COMPLETED = "scraping_completed"
    SCRAPING_FAILED = "scraping_failed"
    DATA_EXTRACTED = "data_extracted"
    ERROR_OCCURRED = "error_occurred"
    RATE_LIMIT_HIT = "rate_limit_hit"
    CIRCUIT_BREAKER_OPENED = "circuit_breaker_opened"
    CIRCUIT_BREAKER_CLOSED = "circuit_breaker_closed"
    DEGRADATION_APPLIED = "degradation_applied"
    CONFIGURATION_CHANGED = "configuration_changed"
    INSTANCE_CREATED = "instance_created"
    INSTANCE_DESTROYED = "instance_destroyed"
    PERFORMANCE_THRESHOLD_EXCEEDED = "performance_threshold_exceeded"
    SECURITY_VIOLATION = "security_violation"
    COMPLIANCE_CHECK = "compliance_check"


@dataclass
class LogContext:
    """Context information for structured logging."""
    operation_id: str
    operation_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    instance_id: Optional[str] = None
    url: Optional[str] = None
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    correlation_id: Optional[str] = None
    parent_operation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    operation_id: str
    operation_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    network_bytes_sent: Optional[int] = None
    network_bytes_received: Optional[int] = None
    items_processed: Optional[int] = None
    success_rate: Optional[float] = None
    error_count: Optional[int] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def finalize(self, end_time: Optional[datetime] = None) -> None:
        """Finalize metrics calculation."""
        if end_time is None:
            end_time = datetime.utcnow()
        
        self.end_time = end_time
        self.duration_ms = (end_time - self.start_time).total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


@dataclass
class AuditEvent:
    """Audit event for compliance tracking."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    context: LogContext
    event_data: Dict[str, Any] = field(default_factory=dict)
    compliance_tags: List[str] = field(default_factory=list)
    risk_level: str = "low"
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['context'] = self.context.to_dict()
        return data


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread_id': record.thread,
            'process_id': record.process
        }
        
        # Add context if available
        if self.include_context and hasattr(record, 'context'):
            log_data['context'] = record.context.to_dict()
        
        # Add performance metrics if available
        if hasattr(record, 'performance_metrics'):
            log_data['performance_metrics'] = record.performance_metrics.to_dict()
        
        # Add audit event if available
        if hasattr(record, 'audit_event'):
            log_data['audit_event'] = record.audit_event.to_dict()
        
        # Add exception information
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info', 'context', 'performance_metrics',
                          'audit_event']:
                log_data[key] = value
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self, config: IntelligentScrapingConfig):
        self.config = config
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self.completed_operations: deque = deque(maxlen=1000)
        self.metrics_lock = threading.Lock()
        
        # Performance thresholds
        self.thresholds = {
            'max_duration_ms': 30000,  # 30 seconds
            'max_memory_mb': 512,
            'max_cpu_percent': 80.0,
            'min_success_rate': 0.8
        }
        
        # Aggregated metrics
        self.aggregated_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_duration_ms': 0.0,
            'average_memory_mb': 0.0,
            'average_cpu_percent': 0.0,
            'total_items_processed': 0,
            'operations_per_minute': 0.0
        }
        
        # Time-based metrics
        self.time_windows = {
            '1min': deque(maxlen=60),
            '5min': deque(maxlen=300),
            '15min': deque(maxlen=900),
            '1hour': deque(maxlen=3600)
        }
    
    def start_operation(self, operation_id: str, operation_type: str) -> PerformanceMetrics:
        """Start monitoring an operation."""
        metrics = PerformanceMetrics(
            operation_id=operation_id,
            operation_type=operation_type,
            start_time=datetime.utcnow()
        )
        
        with self.metrics_lock:
            self.active_operations[operation_id] = metrics
        
        return metrics
    
    def complete_operation(
        self,
        operation_id: str,
        success: bool = True,
        error_count: Optional[int] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[PerformanceMetrics]:
        """Complete monitoring an operation."""
        with self.metrics_lock:
            if operation_id not in self.active_operations:
                return None
            
            metrics = self.active_operations.pop(operation_id)
            metrics.finalize(end_time)
            
            # Calculate success rate
            if error_count is not None:
                metrics.error_count = error_count
                total_items = metrics.items_processed or 1
                metrics.success_rate = max(0.0, (total_items - error_count) / total_items)
            else:
                metrics.success_rate = 1.0 if success else 0.0
            
            # Add to completed operations
            self.completed_operations.append(metrics)
            
            return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self.metrics_lock:
            return {
                'active_operations': len(self.active_operations),
                'completed_operations': len(self.completed_operations),
                'aggregated_metrics': self.aggregated_metrics.copy()
            }


class AuditTrail:
    """Audit trail for compliance tracking and activity logging."""
    
    def __init__(self, config: IntelligentScrapingConfig):
        self.config = config
        self.events: deque = deque(maxlen=10000)
        self.events_lock = threading.Lock()
        
        # Audit configuration
        self.audit_config = {
            'enabled': True,
            'retention_days': 90,
            'compliance_mode': True,
            'encrypt_sensitive_data': True,
            'include_request_data': False,  # For privacy
            'include_response_data': False  # For privacy
        }
        
        # Event counters
        self.event_counters = defaultdict(int)
        
        # Compliance tracking
        self.compliance_violations = []
        self.security_events = []
    
    def log_event(
        self,
        event_type: EventType,
        context: LogContext,
        event_data: Optional[Dict[str, Any]] = None,
        compliance_tags: Optional[List[str]] = None,
        risk_level: str = "low"
    ) -> str:
        """Log an audit event."""
        if not self.audit_config['enabled']:
            return ""
        
        event_id = str(uuid.uuid4())
        
        audit_event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            context=context,
            event_data=event_data or {},
            compliance_tags=compliance_tags or [],
            risk_level=risk_level
        )
        
        with self.events_lock:
            self.events.append(audit_event)
            self.event_counters[event_type.value] += 1
            
            # Track compliance violations
            if risk_level in ['high', 'critical']:
                self.compliance_violations.append(audit_event)
            
            # Track security events
            if event_type == EventType.SECURITY_VIOLATION:
                self.security_events.append(audit_event)
        
        return event_id
    
    def get_events(
        self,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Get audit events with optional filtering."""
        with self.events_lock:
            filtered_events = []
            
            for event in reversed(self.events):
                # Filter by event type
                if event_type and event.event_type != event_type:
                    continue
                
                # Filter by time range
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                
                filtered_events.append(event)
                
                if len(filtered_events) >= limit:
                    break
            
            return filtered_events
    
    def get_compliance_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate compliance report for specified period."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        events = self.get_events(start_time=start_time, end_time=end_time, limit=10000)
        
        # Count events by type
        event_counts = defaultdict(int)
        for event in events:
            event_counts[event.event_type.value] += 1
        
        # Count violations by risk level
        risk_counts = defaultdict(int)
        for event in events:
            risk_counts[event.risk_level] += 1
        
        return {
            'report_period': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'days': days
            },
            'total_events': len(events),
            'event_counts': dict(event_counts),
            'risk_level_counts': dict(risk_counts),
            'violations_count': len([e for e in events if e.risk_level in ['high', 'critical']]),
            'security_events_count': len([e for e in events if e.event_type == EventType.SECURITY_VIOLATION])
        }


class StructuredLogger:
    """
    Structured logging system with comprehensive features.
    
    This class provides structured logging with appropriate levels, contextual information,
    performance monitoring, metrics collection, and audit trail functionality.
    """
    
    def __init__(self, config: IntelligentScrapingConfig):
        self.config = config
        self.performance_monitor = PerformanceMonitor(config)
        self.audit_trail = AuditTrail(config)
        
        # Setup logging configuration
        self.logger = logging.getLogger('intelligent_web_scraper')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_handlers()
        
        # Context storage
        self._context_storage = threading.local()
        
        # Logger registry
        self._loggers: Dict[str, logging.Logger] = {}
        
        # Metrics collection
        self.log_metrics = {
            'total_logs': 0,
            'logs_by_level': defaultdict(int),
            'errors_count': 0,
            'warnings_count': 0
        }
        self.metrics_lock = threading.Lock()
    
    def _setup_handlers(self) -> None:
        """Setup logging handlers."""
        # Console handler with structured formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(StructuredFormatter(include_context=True))
        self.logger.addHandler(console_handler)
        
        # File handler for general logs
        log_dir = Path(self.config.results_directory) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "intelligent_scraper.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter(include_context=True))
        self.logger.addHandler(file_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the specified name."""
        if name not in self._loggers:
            logger = logging.getLogger(f'intelligent_web_scraper.{name}')
            logger.setLevel(self.logger.level)
            
            # Add handlers from main logger
            for handler in self.logger.handlers:
                logger.addHandler(handler)
            
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def set_context(self, context: LogContext) -> None:
        """Set logging context for current thread."""
        self._context_storage.context = context
    
    def get_context(self) -> Optional[LogContext]:
        """Get current logging context."""
        return getattr(self._context_storage, 'context', None)
    
    @contextmanager
    def context_manager(self, context: LogContext):
        """Context manager for temporary logging context."""
        old_context = self.get_context()
        self.set_context(context)
        try:
            yield context
        finally:
            if old_context:
                self.set_context(old_context)
            else:
                self._context_storage.context = None
    
    def log_with_context(
        self,
        level: str,
        message: str,
        context: Optional[LogContext] = None,
        performance_metrics: Optional[PerformanceMetrics] = None,
        audit_event: Optional[AuditEvent] = None,
        **kwargs
    ) -> None:
        """Log message with structured context."""
        # Get logger level
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # Use provided context or current context
        if context is None:
            context = self.get_context()
        
        # Create log record
        record = self.logger.makeRecord(
            self.logger.name,
            log_level,
            "",
            0,
            message,
            (),
            None
        )
        
        # Add structured data
        if context:
            record.context = context
        if performance_metrics:
            record.performance_metrics = performance_metrics
        if audit_event:
            record.audit_event = audit_event
        
        # Add extra fields
        for key, value in kwargs.items():
            setattr(record, key, value)
        
        # Update metrics
        with self.metrics_lock:
            self.log_metrics['total_logs'] += 1
            self.log_metrics['logs_by_level'][level.upper()] += 1
            
            if level.upper() == 'ERROR':
                self.log_metrics['errors_count'] += 1
            elif level.upper() == 'WARNING':
                self.log_metrics['warnings_count'] += 1
        
        # Log the record
        self.logger.handle(record)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.log_with_context('INFO', message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.log_with_context('DEBUG', message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.log_with_context('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.log_with_context('ERROR', message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.log_with_context('CRITICAL', message, **kwargs)
    
    def audit(self, event_type: EventType, event_data: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """Log audit event."""
        context = self.get_context() or LogContext("unknown", "audit")
        
        event_id = self.audit_trail.log_event(
            event_type=event_type,
            context=context,
            event_data=event_data,
            **kwargs
        )
        
        # Also log to audit logger
        audit_event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            context=context,
            event_data=event_data or {}
        )
        
        self.log_with_context('INFO', f"Audit event: {event_type.value}", audit_event=audit_event)
        
        return event_id
    
    def performance(self, metrics: PerformanceMetrics, **kwargs) -> None:
        """Log performance metrics."""
        self.log_with_context('INFO', f"Performance metrics for {metrics.operation_type}", 
                            performance_metrics=metrics, **kwargs)
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        with self.metrics_lock:
            return {
                'log_metrics': dict(self.log_metrics),
                'performance_summary': self.performance_monitor.get_performance_summary(),
                'audit_summary': {
                    'total_events': len(self.audit_trail.events),
                    'event_counters': dict(self.audit_trail.event_counters),
                    'violations_count': len(self.audit_trail.compliance_violations),
                    'security_events_count': len(self.audit_trail.security_events)
                }
            }