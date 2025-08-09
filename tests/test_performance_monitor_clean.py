"""
Clean unit tests for the performance monitoring system.

This module tests the comprehensive performance monitoring capabilities including
response time tracking, memory usage analysis, throughput measurement, and
performance benchmarking.
"""

import pytest
import time
import statistics
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from intelligent_web_scraper.monitoring.performance_monitor import (
    PerformanceMonitor,
    PerformanceMetric,
    PerformanceBenchmark,
    PerformanceOptimizationReport,
    OperationTracker
)


class TestPerformanceMetric:
    """Test PerformanceMetric data class."""
    
    def test_performance_metric_creation(self):
        """Test creating a performance metric."""
        metric = PerformanceMetric(
            operation_type="test_operation",
            operation_id="test_123",
            response_time_ms=1500.0,
            memory_usage_mb=256.0,
            cpu_usage_percent=45.0,
            success=True
        )
        
        assert metric.operation_type == "test_operation"
        assert metric.operation_id == "test_123"
        assert metric.response_time_ms == 1500.0
        assert metric.memory_usage_mb == 256.0
        assert metric.cpu_usage_percent == 45.0
        assert metric.success is True
        assert metric.error_message is None
    
    def test_performance_metric_to_dict(self):
        """Test converting performance metric to dictionary."""
        metric = PerformanceMetric(
            operation_type="scraping",
            operation_id="scrape_001",
            response_time_ms=2000.0,
            success=False,
            error_message="Connection timeout"
        )
        
        result = metric.to_dict()
        
        assert result['operation_type'] == "scraping"
        assert result['operation_id'] == "scrape_001"
        assert result['response_time_ms'] == 2000.0
        assert result['success'] is False
        assert result['error_message'] == "Connection timeout"
        assert 'timestamp' in result


class TestPerformanceBenchmark:
    """Test PerformanceBenchmark data class."""
    
    def test_benchmark_creation(self):
        """Test creating a performance benchmark."""
        benchmark = PerformanceBenchmark(
            benchmark_name="test_benchmark",
            total_operations=100,
            successful_operations=95,
            failed_operations=5,
            average_response_time_ms=1200.0,
            throughput_ops_per_sec=8.5
        )
        
        assert benchmark.benchmark_name == "test_benchmark"
        assert benchmark.total_operations == 100
        assert benchmark.successful_operations == 95
        assert benchmark.failed_operations == 5
        assert benchmark.average_response_time_ms == 1200.0
        assert benchmark.throughput_ops_per_sec == 8.5
    
    def test_benchmark_to_dict(self):
        """Test converting benchmark to dictionary."""
        benchmark = PerformanceBenchmark(
            benchmark_name="api_benchmark",
            total_operations=50,
            throughput_ops_per_sec=12.3
        )
        
        result = benchmark.to_dict()
        
        assert result['benchmark_name'] == "api_benchmark"
        assert result['total_operations'] == 50
        assert result['throughput_ops_per_sec'] == 12.3
        assert 'timestamp' in result


class TestOperationTracker:
    """Test OperationTracker helper class."""
    
    def test_operation_tracker_creation(self):
        """Test creating an operation tracker."""
        tracker = OperationTracker("op_123", "test_operation")
        
        assert tracker.operation_id == "op_123"
        assert tracker.operation_type == "test_operation"
        assert tracker.success is True
        assert tracker.error_message is None
        assert tracker.metadata == {}
    
    def test_operation_tracker_set_success(self):
        """Test setting operation success status."""
        tracker = OperationTracker("op_456", "scraping")
        
        # Test success
        tracker.set_success(True)
        assert tracker.success is True
        assert tracker.error_message is None
        
        # Test failure
        tracker.set_success(False, "Network error")
        assert tracker.success is False
        assert tracker.error_message == "Network error"
    
    def test_operation_tracker_metadata(self):
        """Test adding metadata to operation tracker."""
        tracker = OperationTracker("op_789", "analysis")
        
        tracker.add_metadata("url", "https://example.com")
        tracker.add_metadata("items_found", 42)
        
        assert tracker.metadata["url"] == "https://example.com"
        assert tracker.metadata["items_found"] == 42


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""
    
    @pytest.fixture
    def monitor(self):
        """Create a performance monitor for testing."""
        return PerformanceMonitor(
            history_size=100,
            benchmark_retention_days=7,
            enable_detailed_tracking=True
        )
    
    def test_monitor_creation(self, monitor):
        """Test creating a performance monitor."""
        assert monitor.history_size == 100
        assert monitor.benchmark_retention_days == 7
        assert monitor.enable_detailed_tracking is True
        assert len(monitor.performance_metrics) == 0
        assert len(monitor.operation_metrics) == 0
        assert len(monitor.benchmark_results) == 0
        assert monitor.is_monitoring is False
    
    def test_start_stop_monitoring(self, monitor):
        """Test starting and stopping monitoring."""
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.is_monitoring is True
        assert monitor.monitoring_thread is not None
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor.is_monitoring is False
    
    def test_record_performance_metric(self, monitor):
        """Test recording performance metrics."""
        metric = PerformanceMetric(
            operation_type="test_op",
            operation_id="test_123",
            response_time_ms=1500.0,
            memory_usage_mb=256.0,
            success=True
        )
        
        monitor.record_performance_metric(metric)
        
        assert len(monitor.performance_metrics) == 1
        assert len(monitor.operation_metrics["test_op"]) == 1
        assert monitor.operation_counters["test_op"] == 1
    
    @patch('intelligent_web_scraper.monitoring.performance_monitor.psutil.Process')
    def test_track_operation_context_manager(self, mock_process, monitor):
        """Test track_operation context manager."""
        # Mock process metrics
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 256 * 1024 * 1024  # 256 MB
        mock_process_instance.cpu_percent.return_value = 45.0
        mock_process.return_value = mock_process_instance
        
        # Use context manager
        with monitor.track_operation("test_operation", "op_001") as tracker:
            time.sleep(0.01)  # Simulate work
            tracker.set_success(True)
        
        # Check that metric was recorded
        assert len(monitor.performance_metrics) == 1
        metric = monitor.performance_metrics[0]
        assert metric.operation_type == "test_operation"
        assert metric.operation_id == "op_001"
        assert metric.success is True
        assert metric.response_time_ms > 0
    
    def test_run_benchmark_sequential(self, monitor):
        """Test running a sequential benchmark."""
        def test_operation():
            time.sleep(0.001)  # Simulate work
            return "success"
        
        benchmark = monitor.run_benchmark(
            benchmark_name="test_benchmark",
            operation_func=test_operation,
            num_operations=5,
            concurrent_operations=1,
            warmup_operations=2
        )
        
        assert benchmark.benchmark_name == "test_benchmark"
        assert benchmark.total_operations == 5
        assert benchmark.successful_operations == 5
        assert benchmark.failed_operations == 0
        assert benchmark.success_rate_percent == 100.0
        assert benchmark.throughput_ops_per_sec > 0
        assert benchmark.average_response_time_ms > 0
    
    def test_get_performance_summary(self, monitor):
        """Test getting performance summary."""
        # Add some test metrics
        for i in range(10):
            metric = PerformanceMetric(
                operation_type="summary_test",
                operation_id=f"op_{i}",
                response_time_ms=1000.0 + i * 100,
                memory_usage_mb=100.0 + i * 10,
                cpu_usage_percent=20.0 + i * 5,
                success=i < 8  # 8 successful, 2 failed
            )
            monitor.record_performance_metric(metric)
        
        summary = monitor.get_performance_summary(hours=1.0)
        
        assert summary['total_operations'] == 10
        assert summary['successful_operations'] == 8
        assert summary['success_rate_percent'] == 80.0
        assert 'summary_test' in summary['operation_types']
    
    def test_set_performance_baseline(self, monitor):
        """Test setting and using performance baselines."""
        baseline_metrics = {
            'response_time_ms': 2000.0,
            'throughput_ops_per_sec': 5.0,
            'memory_usage_mb': 200.0
        }
        
        monitor.set_performance_baseline("baseline_test", baseline_metrics)
        
        # Add current metrics (better performance)
        for i in range(5):
            metric = PerformanceMetric(
                operation_type="baseline_test",
                operation_id=f"baseline_{i}",
                response_time_ms=1500.0,  # Better than baseline
                throughput_ops_per_sec=7.0,  # Better than baseline
                memory_usage_mb=150.0,  # Better than baseline
                success=True
            )
            monitor.record_performance_metric(metric)
        
        comparison = monitor._get_performance_comparison()
        
        assert "baseline_test_response_time_improvement" in comparison
        assert comparison["baseline_test_response_time_improvement"] > 0
    
    def test_generate_optimization_report(self, monitor):
        """Test generating optimization report."""
        # Add metrics with various performance characteristics
        for i in range(15):
            metric = PerformanceMetric(
                operation_type="optimization_test",
                operation_id=f"opt_{i}",
                response_time_ms=2000.0 + i * 200,  # Increasing response times
                memory_usage_mb=300.0 + i * 20,     # Increasing memory usage
                cpu_usage_percent=50.0 + i * 2,     # Increasing CPU usage
                success=i < 12  # Most successful
            )
            monitor.record_performance_metric(metric)
        
        report = monitor.generate_optimization_report(analysis_hours=1.0)
        
        assert report.report_id.startswith("opt_report_")
        assert report.analysis_period_hours == 1.0
        assert report.overall_performance_score >= 0.0
        assert report.overall_performance_score <= 100.0
        assert report.average_response_time_ms > 0
        assert isinstance(report.optimization_recommendations, list)
    
    def test_compare_benchmarks(self, monitor):
        """Test comparing two benchmarks."""
        # Create first benchmark
        def operation1():
            time.sleep(0.002)
            return "result1"
        
        benchmark1 = monitor.run_benchmark(
            "benchmark_1", operation1, num_operations=5, warmup_operations=0
        )
        
        # Create second benchmark (faster)
        def operation2():
            time.sleep(0.001)
            return "result2"
        
        benchmark2 = monitor.run_benchmark(
            "benchmark_2", operation2, num_operations=5, warmup_operations=0
        )
        
        comparison = monitor.compare_benchmarks("benchmark_1", "benchmark_2")
        
        assert 'benchmark1' in comparison
        assert 'benchmark2' in comparison
        assert 'comparison' in comparison
        assert 'response_time_difference_ms' in comparison['comparison']
        assert 'throughput_difference_ops_per_sec' in comparison['comparison']
        assert 'overall_improvement' in comparison['comparison']
    
    def test_get_resource_utilization_trends(self, monitor):
        """Test getting resource utilization trends."""
        # Add metrics over time
        base_time = datetime.utcnow() - timedelta(hours=1)
        
        for i in range(15):
            metric = PerformanceMetric(
                timestamp=base_time + timedelta(minutes=i * 4),
                operation_type="trend_test",
                operation_id=f"trend_{i}",
                response_time_ms=1000.0 + i * 50,
                memory_usage_mb=200.0 + i * 10,
                cpu_usage_percent=30.0 + i * 2,
                success=True
            )
            monitor.record_performance_metric(metric)
        
        trends = monitor.get_resource_utilization_trends(hours=2.0)
        
        assert 'memory_trend' in trends
        assert 'cpu_trend' in trends
        assert 'response_time_trend' in trends
        assert 'interval_minutes' in trends
        assert 'total_data_points' in trends
        
        # Check that trends have data points
        assert len(trends['memory_trend']) > 0
        assert len(trends['cpu_trend']) > 0
        assert len(trends['response_time_trend']) > 0
    
    def test_export_performance_data(self, monitor):
        """Test exporting performance data."""
        # Add some test data
        for i in range(5):
            metric = PerformanceMetric(
                operation_type="export_test",
                operation_id=f"export_{i}",
                response_time_ms=1000.0 + i * 100,
                success=True
            )
            monitor.record_performance_metric(metric)
        
        # Run a benchmark
        def simple_op():
            return "test"
        
        monitor.run_benchmark("export_benchmark", simple_op, num_operations=3)
        
        # Export data
        export_data = monitor.export_performance_data(hours=1.0)
        
        assert export_data['total_metrics'] >= 5  # May include benchmark metrics
        assert len(export_data['metrics']) >= 5  # May include benchmark metrics
        assert len(export_data['benchmarks']) == 1
        assert 'performance_thresholds' in export_data
        assert 'operation_counters' in export_data
    
    def test_calculate_percentile(self, monitor):
        """Test percentile calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        assert monitor._calculate_percentile(values, 50) == 5.5  # Median
        assert monitor._calculate_percentile(values, 90) == 9.1
        assert abs(monitor._calculate_percentile(values, 95) - 9.55) < 0.01  # Allow for floating point precision
        
        # Edge cases
        assert monitor._calculate_percentile([5.0], 50) == 5.0
        assert monitor._calculate_percentile([], 50) == 0.0
    
    def test_calculate_trend(self, monitor):
        """Test trend calculation."""
        # Increasing trend
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        assert monitor._calculate_trend(increasing_values) == "increasing"
        
        # Decreasing trend
        decreasing_values = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        assert monitor._calculate_trend(decreasing_values) == "decreasing"
        
        # Stable trend
        stable_values = [5.0, 5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9, 5.0, 5.0]
        assert monitor._calculate_trend(stable_values) == "stable"
        
        # Edge cases
        assert monitor._calculate_trend([]) == "stable"
        assert monitor._calculate_trend([5.0]) == "stable"


class TestPerformanceOptimizationReport:
    """Test PerformanceOptimizationReport model."""
    
    def test_optimization_report_creation(self):
        """Test creating an optimization report."""
        report = PerformanceOptimizationReport(
            report_id="test_report_123",
            generated_at=datetime.utcnow(),
            analysis_period_hours=24.0,
            overall_performance_score=75.5,
            performance_trend="improving",
            average_response_time_ms=1500.0,
            response_time_trend="decreasing",
            throughput_ops_per_sec=8.2,
            throughput_trend="increasing",
            resource_utilization_percent=65.0,
            identified_bottlenecks=["High memory usage", "Slow database queries"],
            optimization_recommendations=["Implement caching", "Optimize queries"],
            configuration_suggestions={"max_connections": 100, "cache_size": "512MB"}
        )
        
        assert report.report_id == "test_report_123"
        assert report.overall_performance_score == 75.5
        assert report.performance_trend == "improving"
        assert len(report.identified_bottlenecks) == 2
        assert len(report.optimization_recommendations) == 2
        assert "max_connections" in report.configuration_suggestions