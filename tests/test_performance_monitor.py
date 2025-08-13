"""
Tests for the Performance Monitor module.

This module tests the performance monitoring capabilities including
metrics collection, real-time monitoring, and performance analysis.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from intelligent_web_scraper.monitoring.performance_monitor import (
    PerformanceMonitor,
    PerformanceMetric,
    OperationMetrics,
    BenchmarkResult
)


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""
    
    @pytest.fixture
    def monitor(self):
        """Create a test monitor instance."""
        return PerformanceMonitor()
    
    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor is not None
        assert hasattr(monitor, 'metrics')
        assert hasattr(monitor, 'operation_metrics')
        assert hasattr(monitor, 'benchmark_results')
        assert monitor.is_monitoring is False
    
    def test_start_stop_monitoring(self, monitor):
        """Test starting and stopping monitoring."""
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.is_monitoring is True
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor.is_monitoring is False
    
    def test_record_metric(self, monitor):
        """Test recording performance metrics."""
        monitor.record_metric("response_time", 1.5)
        monitor.record_metric("memory_usage", 512.0)
        
        assert len(monitor.metrics) == 2
        assert "response_time" in monitor.metrics
        assert "memory_usage" in monitor.metrics
        
        response_time_metrics = monitor.metrics["response_time"]
        assert len(response_time_metrics) == 1
        assert response_time_metrics[0].value == 1.5
    
    def test_get_metrics(self, monitor):
        """Test retrieving metrics."""
        # Record some metrics
        monitor.record_metric("cpu_usage", 75.0)
        monitor.record_metric("cpu_usage", 80.0)
        monitor.record_metric("memory_usage", 60.0)
        
        # Get specific metric
        cpu_metrics = monitor.get_metrics("cpu_usage")
        assert len(cpu_metrics) == 2
        assert cpu_metrics[0].value == 75.0
        assert cpu_metrics[1].value == 80.0
        
        # Get all metrics
        all_metrics = monitor.get_all_metrics()
        assert "cpu_usage" in all_metrics
        assert "memory_usage" in all_metrics
    
    def test_calculate_statistics(self, monitor):
        """Test metric statistics calculation."""
        # Record multiple values
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for value in values:
            monitor.record_metric("test_metric", value)
        
        stats = monitor.calculate_statistics("test_metric")
        assert stats["count"] == 5
        assert stats["average"] == 30.0
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["total"] == 150.0
    
    def test_operation_tracking(self, monitor):
        """Test operation performance tracking."""
        operation_id = monitor.start_operation("test_operation")
        assert operation_id is not None
        
        # Simulate some work
        time.sleep(0.1)
        
        monitor.end_operation(operation_id)
        
        # Check operation was recorded
        assert len(monitor.operation_metrics) == 1
        operation = monitor.operation_metrics[0]
        assert operation.operation_name == "test_operation"
        assert operation.duration > 0
    
    def test_benchmark_recording(self, monitor):
        """Test benchmark result recording."""
        benchmark_data = {
            "name": "test_benchmark",
            "score": 85.5,
            "duration": 2.3,
            "details": {"accuracy": 0.95, "speed": "fast"}
        }
        
        monitor.record_benchmark(benchmark_data)
        
        assert len(monitor.benchmark_results) == 1
        benchmark = monitor.benchmark_results[0]
        assert benchmark.name == "test_benchmark"
        assert benchmark.score == 85.5
        assert benchmark.duration == 2.3
    
    def test_performance_summary(self, monitor):
        """Test performance summary generation."""
        # Add some test data
        monitor.record_metric("response_time", 1.5)
        monitor.record_metric("response_time", 2.0)
        monitor.record_metric("memory_usage", 512.0)
        
        operation_id = monitor.start_operation("test_op")
        time.sleep(0.05)
        monitor.end_operation(operation_id)
        
        monitor.record_benchmark({
            "name": "test_benchmark",
            "score": 90.0,
            "duration": 1.0
        })
        
        summary = monitor.get_performance_summary()
        
        assert "metrics" in summary
        assert "operations" in summary
        assert "benchmarks" in summary
        assert len(summary["metrics"]) == 2
        assert len(summary["operations"]) == 1
        assert len(summary["benchmarks"]) == 1
    
    def test_clear_metrics(self, monitor):
        """Test clearing metrics."""
        # Add some data
        monitor.record_metric("test_metric", 100.0)
        operation_id = monitor.start_operation("test_op")
        monitor.end_operation(operation_id)
        monitor.record_benchmark({"name": "test", "score": 80.0})
        
        # Verify data exists
        assert len(monitor.metrics) > 0
        assert len(monitor.operation_metrics) > 0
        assert len(monitor.benchmark_results) > 0
        
        # Clear metrics
        monitor.clear_metrics()
        
        # Verify data is cleared
        assert len(monitor.metrics) == 0
        assert len(monitor.operation_metrics) == 0
        assert len(monitor.benchmark_results) == 0


class TestPerformanceMetric:
    """Test cases for PerformanceMetric."""
    
    def test_metric_creation(self):
        """Test performance metric creation."""
        metric = PerformanceMetric("response_time", 1.5)
        
        assert metric.name == "response_time"
        assert metric.value == 1.5
        assert isinstance(metric.timestamp, datetime)
    
    def test_metric_with_metadata(self):
        """Test metric creation with metadata."""
        metadata = {"url": "https://example.com", "method": "GET"}
        metric = PerformanceMetric("response_time", 2.0, metadata=metadata)
        
        assert metric.metadata == metadata
        assert metric.metadata["url"] == "https://example.com"


class TestOperationMetrics:
    """Test cases for OperationMetrics."""
    
    def test_operation_creation(self):
        """Test operation metrics creation."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=2)
        
        operation = OperationMetrics(
            operation_id="test_123",
            operation_name="test_operation",
            start_time=start_time,
            end_time=end_time
        )
        
        assert operation.operation_id == "test_123"
        assert operation.operation_name == "test_operation"
        assert operation.duration == 2.0
        assert operation.success is True
    
    def test_operation_with_error(self):
        """Test operation metrics with error."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=1)
        
        operation = OperationMetrics(
            operation_id="error_123",
            operation_name="failing_operation",
            start_time=start_time,
            end_time=end_time,
            success=False,
            error_message="Test error"
        )
        
        assert operation.success is False
        assert operation.error_message == "Test error"


class TestBenchmarkResult:
    """Test cases for BenchmarkResult."""
    
    def test_benchmark_creation(self):
        """Test benchmark result creation."""
        benchmark = BenchmarkResult(
            name="test_benchmark",
            score=85.5,
            duration=3.2
        )
        
        assert benchmark.name == "test_benchmark"
        assert benchmark.score == 85.5
        assert benchmark.duration == 3.2
        assert isinstance(benchmark.timestamp, datetime)
    
    def test_benchmark_with_details(self):
        """Test benchmark with additional details."""
        details = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88
        }
        
        benchmark = BenchmarkResult(
            name="ml_benchmark",
            score=90.0,
            duration=5.0,
            details=details
        )
        
        assert benchmark.details == details
        assert benchmark.details["accuracy"] == 0.95


@pytest.mark.asyncio
async def test_async_monitoring():
    """Test asynchronous monitoring capabilities."""
    monitor = PerformanceMonitor()
    
    async def async_operation():
        """Simulate an async operation."""
        operation_id = monitor.start_operation("async_test")
        await asyncio.sleep(0.1)
        monitor.end_operation(operation_id)
        return operation_id
    
    # Run multiple async operations
    tasks = [async_operation() for _ in range(3)]
    operation_ids = await asyncio.gather(*tasks)
    
    assert len(operation_ids) == 3
    assert len(monitor.operation_metrics) == 3
    
    # Verify all operations completed
    for operation in monitor.operation_metrics:
        assert operation.operation_name == "async_test"
        assert operation.duration > 0


@pytest.mark.integration
def test_real_time_monitoring():
    """Test real-time monitoring functionality."""
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        # Simulate real-time metric collection
        for i in range(5):
            monitor.record_metric("real_time_metric", i * 10.0)
            time.sleep(0.02)  # Small delay to simulate real-time
        
        metrics = monitor.get_metrics("real_time_metric")
        assert len(metrics) == 5
        
        # Verify timestamps are in order
        timestamps = [m.timestamp for m in metrics]
        assert timestamps == sorted(timestamps)
        
    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__])