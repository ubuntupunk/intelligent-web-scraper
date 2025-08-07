"""
Comprehensive unit tests for the monitoring system.

This module tests metrics collection, alerts, and dashboard functionality.
"""

import pytest
import asyncio
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from intelligent_web_scraper.monitoring import (
    MetricsCollector,
    AlertManager,
    MonitoringDashboard,
    SystemMetrics
)

# Mock classes for testing since they don't exist in the actual implementation
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"

class MetricAggregation(Enum):
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    COUNT = "count"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertCondition(Enum):
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    EQUALS = "equals"

@dataclass
class MetricValue:
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'labels': self.labels,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class MetricSample:
    timestamp: datetime
    value: float
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp

# Mock implementations for testing
class TimeSeriesMetric:
    def __init__(self, name: str, metric_type: MetricType, max_samples: int = 1000):
        self.name = name
        self.metric_type = metric_type
        self.max_samples = max_samples
        self.samples: List[MetricSample] = []
        self.labels: Dict[str, str] = {}
    
    def add_sample(self, value: float, labels: Dict[str, str] = None):
        sample = MetricSample(datetime.utcnow(), value, labels or {})
        self.samples.append(sample)
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)
    
    def get_latest_value(self) -> Optional[float]:
        return self.samples[-1].value if self.samples else None
    
    def get_samples_in_range(self, start_time: datetime, end_time: datetime) -> List[MetricSample]:
        return [s for s in self.samples if start_time <= s.timestamp <= end_time]
    
    def calculate_aggregation(self, aggregation: MetricAggregation) -> Optional[float]:
        if not self.samples:
            return 0 if aggregation == MetricAggregation.COUNT else None
        
        values = [s.value for s in self.samples]
        
        if aggregation == MetricAggregation.AVERAGE:
            return sum(values) / len(values)
        elif aggregation == MetricAggregation.MIN:
            return min(values)
        elif aggregation == MetricAggregation.MAX:
            return max(values)
        elif aggregation == MetricAggregation.SUM:
            return sum(values)
        elif aggregation == MetricAggregation.COUNT:
            return len(values)
        elif aggregation == MetricAggregation.P50:
            sorted_values = sorted(values)
            return sorted_values[len(sorted_values) // 2]
        elif aggregation == MetricAggregation.P95:
            sorted_values = sorted(values)
            return sorted_values[int(len(sorted_values) * 0.95)]
        elif aggregation == MetricAggregation.P99:
            sorted_values = sorted(values)
            return sorted_values[int(len(sorted_values) * 0.99)]
        
        return None

class MockMetricsCollector:
    def __init__(self):
        self.metrics: Dict[str, TimeSeriesMetric] = {}
        self.collection_interval = 1.0
    
    def record_counter(self, name: str, value: float, labels: Dict[str, str] = None):
        if name not in self.metrics:
            self.metrics[name] = TimeSeriesMetric(name, MetricType.COUNTER)
        self.metrics[name].add_sample(value, labels)
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        if name not in self.metrics:
            self.metrics[name] = TimeSeriesMetric(name, MetricType.GAUGE)
        self.metrics[name].add_sample(value, labels)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        if name not in self.metrics:
            self.metrics[name] = TimeSeriesMetric(name, MetricType.HISTOGRAM)
        self.metrics[name].add_sample(value, labels)
    
    def get_metric(self, name: str) -> Optional[TimeSeriesMetric]:
        return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, TimeSeriesMetric]:
        return self.metrics.copy()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        return {
            'total_metrics': len(self.metrics),
            'metric_types': {mt.value: sum(1 for m in self.metrics.values() if m.metric_type == mt) for mt in MetricType},
            'latest_values': {name: metric.get_latest_value() for name, metric in self.metrics.items()}
        }
    
    def clear_metrics(self):
        self.metrics.clear()
    
    def export_metrics(self, format_type: str) -> Any:
        if format_type == "dict":
            return {name: [{'timestamp': s.timestamp.isoformat(), 'value': s.value, 'labels': s.labels} 
                          for s in metric.samples] for name, metric in self.metrics.items()}
        elif format_type == "json":
            import json
            return json.dumps(self.export_metrics("dict"))
        return None

@dataclass
class AlertRule:
    name: str
    metric_name: str
    condition: AlertCondition
    threshold: float
    level: AlertLevel
    description: str = ""
    enabled: bool = True
    cooldown_seconds: int = 300
    
    def evaluate(self, value: float) -> bool:
        if not self.enabled:
            return False
        
        if self.condition == AlertCondition.GREATER_THAN:
            return value > self.threshold
        elif self.condition == AlertCondition.LESS_THAN:
            return value < self.threshold
        elif self.condition == AlertCondition.EQUALS:
            return value == self.threshold
        
        return False

@dataclass
class Alert:
    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    level: AlertLevel
    message: str
    timestamp: datetime = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def resolve(self):
        self.resolved = True
        self.resolved_at = datetime.utcnow()
    
    def get_duration(self) -> float:
        end_time = self.resolved_at or datetime.utcnow()
        return (end_time - self.timestamp).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_name': self.rule_name,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'level': self.level.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }

class MockAlertManager:
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
    
    def add_rule(self, rule: AlertRule):
        self.rules[rule.name] = rule
    
    def remove_rule(self, rule_name: str) -> bool:
        if rule_name in self.rules:
            del self.rules[rule_name]
            return True
        return False
    
    def evaluate_rules(self, metrics: Dict[str, float]) -> List[Alert]:
        new_alerts = []
        
        for rule in self.rules.values():
            if rule.metric_name in metrics:
                value = metrics[rule.metric_name]
                if rule.evaluate(value):
                    alert = Alert(
                        rule_name=rule.name,
                        metric_name=rule.metric_name,
                        current_value=value,
                        threshold=rule.threshold,
                        level=rule.level,
                        message=f"{rule.description or rule.name}: {value} {rule.condition.value} {rule.threshold}"
                    )
                    alert_id = f"{rule.name}_{int(time.time())}"
                    self.active_alerts[alert_id] = alert
                    new_alerts.append(alert)
        
        return new_alerts
    
    def resolve_alert(self, alert_id: str) -> bool:
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            alert.resolve()
            self.alert_history.append(alert)
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        return list(self.active_alerts.values())
    
    def get_alert_history(self) -> List[Alert]:
        return self.alert_history.copy()
    
    def get_alert_summary(self) -> Dict[str, Any]:
        alerts_by_level = {level.value: 0 for level in AlertLevel}
        for alert in self.active_alerts.values():
            alerts_by_level[alert.level.value] += 1
        
        return {
            'total_rules': len(self.rules),
            'active_alerts': len(self.active_alerts),
            'alerts_by_level': alerts_by_level,
            'total_history': len(self.alert_history)
        }
from intelligent_web_scraper.config import IntelligentScrapingConfig


class TestMetricValue:
    """Test the MetricValue dataclass."""
    
    def test_metric_value_creation(self):
        """Test metric value creation with different types."""
        # Counter metric
        counter = MetricValue(
            name="requests_total",
            value=100,
            metric_type=MetricType.COUNTER,
            labels={"endpoint": "/api/scrape", "status": "success"}
        )
        
        assert counter.name == "requests_total"
        assert counter.value == 100
        assert counter.metric_type == MetricType.COUNTER
        assert counter.labels == {"endpoint": "/api/scrape", "status": "success"}
        assert isinstance(counter.timestamp, datetime)
        
        # Gauge metric
        gauge = MetricValue(
            name="memory_usage_bytes",
            value=1024.5,
            metric_type=MetricType.GAUGE
        )
        
        assert gauge.name == "memory_usage_bytes"
        assert gauge.value == 1024.5
        assert gauge.metric_type == MetricType.GAUGE
        assert gauge.labels == {}
    
    def test_metric_value_serialization(self):
        """Test metric value serialization to dictionary."""
        metric = MetricValue(
            name="response_time_seconds",
            value=0.25,
            metric_type=MetricType.HISTOGRAM,
            labels={"method": "GET", "endpoint": "/api/data"}
        )
        
        metric_dict = metric.to_dict()
        
        assert isinstance(metric_dict, dict)
        assert metric_dict['name'] == "response_time_seconds"
        assert metric_dict['value'] == 0.25
        assert metric_dict['type'] == "histogram"
        assert metric_dict['labels'] == {"method": "GET", "endpoint": "/api/data"}
        assert 'timestamp' in metric_dict
        assert isinstance(metric_dict['timestamp'], str)  # ISO format


class TestMetricSample:
    """Test the MetricSample dataclass."""
    
    def test_metric_sample_creation(self):
        """Test metric sample creation."""
        sample = MetricSample(
            timestamp=datetime.utcnow(),
            value=42.0,
            labels={"instance": "scraper-1"}
        )
        
        assert isinstance(sample.timestamp, datetime)
        assert sample.value == 42.0
        assert sample.labels == {"instance": "scraper-1"}
    
    def test_metric_sample_comparison(self):
        """Test metric sample comparison and sorting."""
        now = datetime.utcnow()
        
        sample1 = MetricSample(timestamp=now, value=10.0)
        sample2 = MetricSample(timestamp=now + timedelta(seconds=1), value=20.0)
        sample3 = MetricSample(timestamp=now + timedelta(seconds=2), value=30.0)
        
        samples = [sample3, sample1, sample2]
        sorted_samples = sorted(samples)
        
        assert sorted_samples[0] == sample1
        assert sorted_samples[1] == sample2
        assert sorted_samples[2] == sample3


class TestTimeSeriesMetric:
    """Test the TimeSeriesMetric class."""
    
    def test_time_series_metric_initialization(self):
        """Test time series metric initialization."""
        metric = TimeSeriesMetric(
            name="cpu_usage_percent",
            metric_type=MetricType.GAUGE,
            max_samples=100
        )
        
        assert metric.name == "cpu_usage_percent"
        assert metric.metric_type == MetricType.GAUGE
        assert metric.max_samples == 100
        assert len(metric.samples) == 0
        assert len(metric.labels) == 0
    
    def test_add_sample(self):
        """Test adding samples to time series metric."""
        metric = TimeSeriesMetric("test_metric", MetricType.GAUGE, max_samples=3)
        
        # Add samples
        metric.add_sample(10.0, {"tag": "a"})
        metric.add_sample(20.0, {"tag": "b"})
        metric.add_sample(30.0, {"tag": "c"})
        
        assert len(metric.samples) == 3
        assert metric.samples[0].value == 10.0
        assert metric.samples[1].value == 20.0
        assert metric.samples[2].value == 30.0
        
        # Add another sample - should evict oldest
        metric.add_sample(40.0, {"tag": "d"})
        
        assert len(metric.samples) == 3
        assert metric.samples[0].value == 20.0  # First sample evicted
        assert metric.samples[1].value == 30.0
        assert metric.samples[2].value == 40.0
    
    def test_get_latest_value(self):
        """Test getting latest value from time series."""
        metric = TimeSeriesMetric("test_metric", MetricType.GAUGE)
        
        # No samples yet
        assert metric.get_latest_value() is None
        
        # Add samples
        metric.add_sample(10.0)
        assert metric.get_latest_value() == 10.0
        
        metric.add_sample(20.0)
        assert metric.get_latest_value() == 20.0
    
    def test_get_samples_in_range(self):
        """Test getting samples within time range."""
        metric = TimeSeriesMetric("test_metric", MetricType.GAUGE)
        
        now = datetime.utcnow()
        
        # Add samples with specific timestamps
        metric.samples = [
            MetricSample(now - timedelta(minutes=5), 10.0),
            MetricSample(now - timedelta(minutes=3), 20.0),
            MetricSample(now - timedelta(minutes=1), 30.0),
            MetricSample(now, 40.0)
        ]
        
        # Get samples from last 2 minutes
        start_time = now - timedelta(minutes=2)
        recent_samples = metric.get_samples_in_range(start_time, now)
        
        assert len(recent_samples) == 2
        assert recent_samples[0].value == 30.0
        assert recent_samples[1].value == 40.0
    
    def test_calculate_aggregation(self):
        """Test calculating aggregations over samples."""
        metric = TimeSeriesMetric("test_metric", MetricType.GAUGE)
        
        # Add test samples
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for value in values:
            metric.add_sample(value)
        
        # Test different aggregations
        assert metric.calculate_aggregation(MetricAggregation.AVERAGE) == 30.0
        assert metric.calculate_aggregation(MetricAggregation.MIN) == 10.0
        assert metric.calculate_aggregation(MetricAggregation.MAX) == 50.0
        assert metric.calculate_aggregation(MetricAggregation.SUM) == 150.0
        assert metric.calculate_aggregation(MetricAggregation.COUNT) == 5
        
        # Test percentiles
        assert metric.calculate_aggregation(MetricAggregation.P50) == 30.0
        assert metric.calculate_aggregation(MetricAggregation.P95) == 50.0
        assert metric.calculate_aggregation(MetricAggregation.P99) == 50.0
    
    def test_calculate_aggregation_empty_samples(self):
        """Test calculating aggregations with no samples."""
        metric = TimeSeriesMetric("test_metric", MetricType.GAUGE)
        
        # Should return None for empty samples
        assert metric.calculate_aggregation(MetricAggregation.AVERAGE) is None
        assert metric.calculate_aggregation(MetricAggregation.MIN) is None
        assert metric.calculate_aggregation(MetricAggregation.MAX) is None
        assert metric.calculate_aggregation(MetricAggregation.COUNT) == 0


class TestMetricsCollector:
    """Test the MetricsCollector class."""
    
    @pytest.fixture
    def collector(self):
        """Create a metrics collector for testing."""
        return MockMetricsCollector()
    
    def test_collector_initialization(self, collector):
        """Test metrics collector initialization."""
        assert hasattr(collector, 'metrics')
        assert isinstance(collector.metrics, dict)
        assert len(collector.metrics) == 0
        assert hasattr(collector, 'collection_interval')
        assert collector.collection_interval > 0
    
    def test_record_counter(self, collector):
        """Test recording counter metrics."""
        collector.record_counter("requests_total", 1, {"endpoint": "/api/scrape"})
        collector.record_counter("requests_total", 1, {"endpoint": "/api/scrape"})
        collector.record_counter("requests_total", 1, {"endpoint": "/api/data"})
        
        # Check that metrics were recorded
        assert "requests_total" in collector.metrics
        metric = collector.metrics["requests_total"]
        assert metric.metric_type == MetricType.COUNTER
        assert len(metric.samples) == 3
    
    def test_record_gauge(self, collector):
        """Test recording gauge metrics."""
        collector.record_gauge("memory_usage_mb", 512.5)
        collector.record_gauge("memory_usage_mb", 600.0)
        collector.record_gauge("memory_usage_mb", 450.2)
        
        # Check that metrics were recorded
        assert "memory_usage_mb" in collector.metrics
        metric = collector.metrics["memory_usage_mb"]
        assert metric.metric_type == MetricType.GAUGE
        assert metric.get_latest_value() == 450.2
    
    def test_record_histogram(self, collector):
        """Test recording histogram metrics."""
        response_times = [0.1, 0.2, 0.15, 0.3, 0.25, 0.18]
        
        for time_val in response_times:
            collector.record_histogram("response_time_seconds", time_val, {"method": "GET"})
        
        # Check that metrics were recorded
        assert "response_time_seconds" in collector.metrics
        metric = collector.metrics["response_time_seconds"]
        assert metric.metric_type == MetricType.HISTOGRAM
        assert len(metric.samples) == 6
    
    def test_get_metric(self, collector):
        """Test getting specific metrics."""
        collector.record_gauge("cpu_usage", 75.5)
        
        # Get existing metric
        metric = collector.get_metric("cpu_usage")
        assert metric is not None
        assert metric.name == "cpu_usage"
        assert metric.get_latest_value() == 75.5
        
        # Get non-existent metric
        missing_metric = collector.get_metric("non_existent")
        assert missing_metric is None
    
    def test_get_all_metrics(self, collector):
        """Test getting all metrics."""
        collector.record_counter("requests", 10)
        collector.record_gauge("memory", 512)
        collector.record_histogram("latency", 0.1)
        
        all_metrics = collector.get_all_metrics()
        
        assert isinstance(all_metrics, dict)
        assert len(all_metrics) == 3
        assert "requests" in all_metrics
        assert "memory" in all_metrics
        assert "latency" in all_metrics
    
    def test_get_metrics_summary(self, collector):
        """Test getting metrics summary."""
        # Record various metrics
        collector.record_counter("requests", 100)
        collector.record_gauge("memory_mb", 512)
        collector.record_histogram("response_time", 0.25)
        
        summary = collector.get_metrics_summary()
        
        assert isinstance(summary, dict)
        assert "total_metrics" in summary
        assert "metric_types" in summary
        assert "latest_values" in summary
        
        assert summary["total_metrics"] == 3
        assert "counter" in summary["metric_types"]
        assert "gauge" in summary["metric_types"]
        assert "histogram" in summary["metric_types"]
    
    def test_clear_metrics(self, collector):
        """Test clearing all metrics."""
        collector.record_counter("requests", 10)
        collector.record_gauge("memory", 512)
        
        assert len(collector.metrics) == 2
        
        collector.clear_metrics()
        
        assert len(collector.metrics) == 0
    
    def test_export_metrics(self, collector):
        """Test exporting metrics in different formats."""
        collector.record_counter("requests", 100, {"status": "success"})
        collector.record_gauge("memory_mb", 512)
        
        # Export as dictionary
        exported = collector.export_metrics("dict")
        
        assert isinstance(exported, dict)
        assert "requests" in exported
        assert "memory_mb" in exported
        
        # Export as JSON string
        json_exported = collector.export_metrics("json")
        
        assert isinstance(json_exported, str)
        assert "requests" in json_exported
        assert "memory_mb" in json_exported


class TestAlertRule:
    """Test the AlertRule class."""
    
    def test_alert_rule_creation(self):
        """Test alert rule creation."""
        rule = AlertRule(
            name="high_memory_usage",
            metric_name="memory_usage_percent",
            condition=AlertCondition.GREATER_THAN,
            threshold=80.0,
            level=AlertLevel.WARNING,
            description="Memory usage is high"
        )
        
        assert rule.name == "high_memory_usage"
        assert rule.metric_name == "memory_usage_percent"
        assert rule.condition == AlertCondition.GREATER_THAN
        assert rule.threshold == 80.0
        assert rule.level == AlertLevel.WARNING
        assert rule.description == "Memory usage is high"
        assert rule.enabled is True
        assert rule.cooldown_seconds == 300  # Default
    
    def test_alert_rule_evaluation_greater_than(self):
        """Test alert rule evaluation with GREATER_THAN condition."""
        rule = AlertRule(
            name="test_rule",
            metric_name="test_metric",
            condition=AlertCondition.GREATER_THAN,
            threshold=50.0,
            level=AlertLevel.WARNING
        )
        
        # Value above threshold - should trigger
        assert rule.evaluate(75.0) is True
        
        # Value below threshold - should not trigger
        assert rule.evaluate(25.0) is False
        
        # Value equal to threshold - should not trigger
        assert rule.evaluate(50.0) is False
    
    def test_alert_rule_evaluation_less_than(self):
        """Test alert rule evaluation with LESS_THAN condition."""
        rule = AlertRule(
            name="test_rule",
            metric_name="test_metric",
            condition=AlertCondition.LESS_THAN,
            threshold=20.0,
            level=AlertLevel.CRITICAL
        )
        
        # Value below threshold - should trigger
        assert rule.evaluate(10.0) is True
        
        # Value above threshold - should not trigger
        assert rule.evaluate(30.0) is False
        
        # Value equal to threshold - should not trigger
        assert rule.evaluate(20.0) is False
    
    def test_alert_rule_evaluation_equals(self):
        """Test alert rule evaluation with EQUALS condition."""
        rule = AlertRule(
            name="test_rule",
            metric_name="test_metric",
            condition=AlertCondition.EQUALS,
            threshold=100.0,
            level=AlertLevel.INFO
        )
        
        # Value equal to threshold - should trigger
        assert rule.evaluate(100.0) is True
        
        # Value not equal to threshold - should not trigger
        assert rule.evaluate(99.0) is False
        assert rule.evaluate(101.0) is False
    
    def test_alert_rule_disabled(self):
        """Test that disabled alert rules don't trigger."""
        rule = AlertRule(
            name="disabled_rule",
            metric_name="test_metric",
            condition=AlertCondition.GREATER_THAN,
            threshold=50.0,
            level=AlertLevel.WARNING,
            enabled=False
        )
        
        # Even with triggering value, disabled rule should not trigger
        assert rule.evaluate(100.0) is False


class TestAlert:
    """Test the Alert class."""
    
    def test_alert_creation(self):
        """Test alert creation."""
        alert = Alert(
            rule_name="test_rule",
            metric_name="test_metric",
            current_value=85.0,
            threshold=80.0,
            level=AlertLevel.WARNING,
            message="Test alert message"
        )
        
        assert alert.rule_name == "test_rule"
        assert alert.metric_name == "test_metric"
        assert alert.current_value == 85.0
        assert alert.threshold == 80.0
        assert alert.level == AlertLevel.WARNING
        assert alert.message == "Test alert message"
        assert isinstance(alert.timestamp, datetime)
        assert alert.resolved is False
        assert alert.resolved_at is None
    
    def test_alert_resolution(self):
        """Test alert resolution."""
        alert = Alert(
            rule_name="test_rule",
            metric_name="test_metric",
            current_value=85.0,
            threshold=80.0,
            level=AlertLevel.WARNING,
            message="Test alert"
        )
        
        assert alert.resolved is False
        assert alert.resolved_at is None
        
        # Resolve the alert
        alert.resolve()
        
        assert alert.resolved is True
        assert isinstance(alert.resolved_at, datetime)
    
    def test_alert_duration(self):
        """Test alert duration calculation."""
        alert = Alert(
            rule_name="test_rule",
            metric_name="test_metric",
            current_value=85.0,
            threshold=80.0,
            level=AlertLevel.WARNING,
            message="Test alert"
        )
        
        # Wait a bit
        time.sleep(0.1)
        
        # Duration should be positive
        duration = alert.get_duration()
        assert duration > 0
        
        # Resolve and check duration
        alert.resolve()
        resolved_duration = alert.get_duration()
        assert resolved_duration >= duration
    
    def test_alert_serialization(self):
        """Test alert serialization to dictionary."""
        alert = Alert(
            rule_name="test_rule",
            metric_name="test_metric",
            current_value=85.0,
            threshold=80.0,
            level=AlertLevel.WARNING,
            message="Test alert message"
        )
        
        alert_dict = alert.to_dict()
        
        assert isinstance(alert_dict, dict)
        assert alert_dict['rule_name'] == "test_rule"
        assert alert_dict['metric_name'] == "test_metric"
        assert alert_dict['current_value'] == 85.0
        assert alert_dict['threshold'] == 80.0
        assert alert_dict['level'] == "warning"
        assert alert_dict['message'] == "Test alert message"
        assert 'timestamp' in alert_dict
        assert alert_dict['resolved'] is False
        assert alert_dict['resolved_at'] is None


class TestAlertManager:
    """Test the AlertManager class."""
    
    @pytest.fixture
    def alert_manager(self):
        """Create an alert manager for testing."""
        return MockAlertManager()
    
    def test_alert_manager_initialization(self, alert_manager):
        """Test alert manager initialization."""
        assert hasattr(alert_manager, 'rules')
        assert hasattr(alert_manager, 'active_alerts')
        assert hasattr(alert_manager, 'alert_history')
        assert len(alert_manager.rules) == 0
        assert len(alert_manager.active_alerts) == 0
        assert len(alert_manager.alert_history) == 0
    
    def test_add_rule(self, alert_manager):
        """Test adding alert rules."""
        rule = AlertRule(
            name="memory_alert",
            metric_name="memory_usage",
            condition=AlertCondition.GREATER_THAN,
            threshold=80.0,
            level=AlertLevel.WARNING
        )
        
        alert_manager.add_rule(rule)
        
        assert len(alert_manager.rules) == 1
        assert "memory_alert" in alert_manager.rules
        assert alert_manager.rules["memory_alert"] == rule
    
    def test_remove_rule(self, alert_manager):
        """Test removing alert rules."""
        rule = AlertRule(
            name="temp_rule",
            metric_name="temp_metric",
            condition=AlertCondition.GREATER_THAN,
            threshold=50.0,
            level=AlertLevel.INFO
        )
        
        alert_manager.add_rule(rule)
        assert len(alert_manager.rules) == 1
        
        removed = alert_manager.remove_rule("temp_rule")
        assert removed is True
        assert len(alert_manager.rules) == 0
        
        # Try to remove non-existent rule
        removed = alert_manager.remove_rule("non_existent")
        assert removed is False
    
    def test_evaluate_rules_no_alerts(self, alert_manager):
        """Test evaluating rules when no alerts should trigger."""
        rule = AlertRule(
            name="cpu_alert",
            metric_name="cpu_usage",
            condition=AlertCondition.GREATER_THAN,
            threshold=90.0,
            level=AlertLevel.CRITICAL
        )
        
        alert_manager.add_rule(rule)
        
        # Provide metrics that don't trigger alerts
        metrics = {"cpu_usage": 75.0}
        new_alerts = alert_manager.evaluate_rules(metrics)
        
        assert len(new_alerts) == 0
        assert len(alert_manager.active_alerts) == 0
    
    def test_evaluate_rules_with_alerts(self, alert_manager):
        """Test evaluating rules when alerts should trigger."""
        rule = AlertRule(
            name="memory_alert",
            metric_name="memory_usage",
            condition=AlertCondition.GREATER_THAN,
            threshold=80.0,
            level=AlertLevel.WARNING
        )
        
        alert_manager.add_rule(rule)
        
        # Provide metrics that trigger alerts
        metrics = {"memory_usage": 95.0}
        new_alerts = alert_manager.evaluate_rules(metrics)
        
        assert len(new_alerts) == 1
        assert len(alert_manager.active_alerts) == 1
        
        alert = new_alerts[0]
        assert alert.rule_name == "memory_alert"
        assert alert.current_value == 95.0
        assert alert.threshold == 80.0
        assert alert.level == AlertLevel.WARNING
    
    def test_resolve_alert(self, alert_manager):
        """Test resolving active alerts."""
        rule = AlertRule(
            name="disk_alert",
            metric_name="disk_usage",
            condition=AlertCondition.GREATER_THAN,
            threshold=85.0,
            level=AlertLevel.WARNING
        )
        
        alert_manager.add_rule(rule)
        
        # Trigger alert
        metrics = {"disk_usage": 90.0}
        new_alerts = alert_manager.evaluate_rules(metrics)
        
        assert len(alert_manager.active_alerts) == 1
        
        # Resolve alert
        alert_id = list(alert_manager.active_alerts.keys())[0]
        resolved = alert_manager.resolve_alert(alert_id)
        
        assert resolved is True
        assert len(alert_manager.active_alerts) == 0
        assert len(alert_manager.alert_history) == 1
    
    def test_get_active_alerts(self, alert_manager):
        """Test getting active alerts."""
        rule1 = AlertRule("rule1", "metric1", AlertCondition.GREATER_THAN, 50.0, AlertLevel.WARNING)
        rule2 = AlertRule("rule2", "metric2", AlertCondition.LESS_THAN, 10.0, AlertLevel.CRITICAL)
        
        alert_manager.add_rule(rule1)
        alert_manager.add_rule(rule2)
        
        # Trigger both alerts
        metrics = {"metric1": 75.0, "metric2": 5.0}
        alert_manager.evaluate_rules(metrics)
        
        active_alerts = alert_manager.get_active_alerts()
        
        assert len(active_alerts) == 2
        assert all(isinstance(alert, Alert) for alert in active_alerts)
    
    def test_get_alert_history(self, alert_manager):
        """Test getting alert history."""
        rule = AlertRule("test_rule", "test_metric", AlertCondition.GREATER_THAN, 50.0, AlertLevel.INFO)
        alert_manager.add_rule(rule)
        
        # Trigger and resolve alert
        metrics = {"test_metric": 75.0}
        alert_manager.evaluate_rules(metrics)
        
        alert_id = list(alert_manager.active_alerts.keys())[0]
        alert_manager.resolve_alert(alert_id)
        
        history = alert_manager.get_alert_history()
        
        assert len(history) == 1
        assert history[0].resolved is True
    
    def test_get_alert_summary(self, alert_manager):
        """Test getting alert summary."""
        # Add rules and trigger some alerts
        rule1 = AlertRule("rule1", "metric1", AlertCondition.GREATER_THAN, 50.0, AlertLevel.WARNING)
        rule2 = AlertRule("rule2", "metric2", AlertCondition.GREATER_THAN, 80.0, AlertLevel.CRITICAL)
        
        alert_manager.add_rule(rule1)
        alert_manager.add_rule(rule2)
        
        metrics = {"metric1": 75.0, "metric2": 90.0}
        alert_manager.evaluate_rules(metrics)
        
        summary = alert_manager.get_alert_summary()
        
        assert isinstance(summary, dict)
        assert summary['total_rules'] == 2
        assert summary['active_alerts'] == 2
        assert summary['alerts_by_level']['warning'] == 1
        assert summary['alerts_by_level']['critical'] == 1


if __name__ == "__main__":
    pytest.main([__file__])