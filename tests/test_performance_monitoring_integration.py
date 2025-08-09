"""
Integration tests for performance monitoring system.

This module tests the integration of performance monitoring with the
intelligent web scraper system components.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from intelligent_web_scraper.monitoring import (
    PerformanceMonitor,
    MonitoringDashboard,
    MetricsCollector,
    AlertManager
)
from intelligent_web_scraper.monitoring.performance_monitor import (
    PerformanceMetric,
    PerformanceBenchmark,
    PerformanceOptimizationReport
)


class TestPerformanceMonitoringIntegration:
    """Test performance monitoring integration with other components."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create a performance monitor for testing."""
        return PerformanceMonitor(
            history_size=100,
            enable_detailed_tracking=True
        )
    
    @pytest.fixture
    def metrics_collector(self):
        """Create a metrics collector for testing."""
        return MetricsCollector(
            collection_interval=0.1,
            history_size=50
        )
    
    @pytest.fixture
    def monitoring_dashboard(self):
        """Create a monitoring dashboard for testing."""
        return MonitoringDashboard(
            refresh_rate=1.0,
            enable_sound_alerts=False
        )
    
    def test_performance_monitor_with_metrics_collector(self, performance_monitor, metrics_collector):
        """Test performance monitor integration with metrics collector."""
        # Start metrics collection
        metrics_collector.start_collection()
        
        try:
            # Record some performance metrics
            for i in range(5):
                metric = PerformanceMetric(
                    operation_type="integration_test",
                    operation_id=f"test_{i}",
                    response_time_ms=1000.0 + i * 100,
                    memory_usage_mb=100.0 + i * 10,
                    success=True
                )
                performance_monitor.record_performance_metric(metric)
            
            # Verify metrics were recorded
            assert len(performance_monitor.performance_metrics) == 5
            
            # Get performance summary
            summary = performance_monitor.get_performance_summary(hours=1.0)
            assert summary['total_operations'] == 5
            assert summary['success_rate_percent'] == 100.0
            
            # Verify metrics collector can collect instance metrics
            instance_stats = {
                'instance_id': 'test_instance',
                'status': 'running',
                'requests_processed': 5,
                'successful_requests': 5,
                'failed_requests': 0,
                'success_rate': 100.0,
                'average_response_time': 1200.0,
                'memory_usage_mb': 150.0,
                'cpu_usage_percent': 25.0
            }
            
            instance_metrics = metrics_collector.collect_instance_metrics(instance_stats)
            assert instance_metrics.instance_id == 'test_instance'
            assert instance_metrics.success_rate == 100.0
            
        finally:
            metrics_collector.stop_collection()
    
    def test_performance_monitor_with_dashboard(self, performance_monitor, monitoring_dashboard):
        """Test performance monitor integration with monitoring dashboard."""
        # Add performance callback to connect with dashboard
        dashboard_events = []
        
        def dashboard_callback(event_data):
            dashboard_events.append(event_data)
        
        performance_monitor.add_performance_callback(dashboard_callback)
        
        # Record metrics
        metric = PerformanceMetric(
            operation_type="dashboard_test",
            operation_id="dash_001",
            response_time_ms=2000.0,
            success=True
        )
        performance_monitor.record_performance_metric(metric)
        
        # Verify dashboard received callback
        assert len(dashboard_events) == 1
        assert dashboard_events[0]['type'] == 'metric_recorded'
        assert dashboard_events[0]['metric']['operation_type'] == 'dashboard_test'
        
        # Test dashboard can display performance data
        dashboard_stats = monitoring_dashboard.get_dashboard_stats()
        assert 'is_running' in dashboard_stats
        assert 'metrics_collection_stats' in dashboard_stats
    
    def test_performance_benchmarking_integration(self, performance_monitor):
        """Test performance benchmarking with real operations."""
        def mock_scraping_operation():
            """Mock scraping operation for benchmarking."""
            time.sleep(0.01)  # Simulate work
            return {"status": "success", "items": 10}
        
        # Run benchmark
        benchmark = performance_monitor.run_benchmark(
            benchmark_name="integration_benchmark",
            operation_func=mock_scraping_operation,
            num_operations=5,
            concurrent_operations=1,
            warmup_operations=1
        )
        
        # Verify benchmark results
        assert benchmark.benchmark_name == "integration_benchmark"
        assert benchmark.total_operations == 5
        assert benchmark.successful_operations == 5
        assert benchmark.throughput_ops_per_sec > 0
        assert benchmark.average_response_time_ms > 0
        
        # Verify benchmark was stored
        assert len(performance_monitor.benchmark_results) == 1
        stored_benchmark = performance_monitor.benchmark_results[0]
        assert stored_benchmark.benchmark_name == "integration_benchmark"
    
    def test_optimization_report_generation(self, performance_monitor):
        """Test optimization report generation with realistic data."""
        # Add only slow metrics to ensure we trigger optimization recommendations
        slow_metrics = [
            PerformanceMetric(
                operation_type="slow_operation",
                operation_id=f"slow_{i}",
                response_time_ms=12000.0 + i * 1000,  # Very slow operations (exceeds critical threshold)
                memory_usage_mb=800.0 + i * 100,      # Very high memory usage
                cpu_usage_percent=90.0 + i * 2,       # Very high CPU usage
                success=True
            ) for i in range(5)  # More slow operations to dominate the average
        ]
        
        # Record all metrics
        for metric in slow_metrics:
            performance_monitor.record_performance_metric(metric)
        
        # Generate optimization report
        report = performance_monitor.generate_optimization_report(analysis_hours=1.0)
        
        # Verify report content
        assert isinstance(report, PerformanceOptimizationReport)
        assert report.analysis_period_hours == 1.0
        assert 0 <= report.overall_performance_score <= 100
        assert report.performance_trend in ["improving", "stable", "degrading", "mixed", "unknown"]
        
        # With only slow operations, we should definitely get recommendations
        print(f"Average response time: {report.average_response_time_ms}ms")
        print(f"Performance score: {report.overall_performance_score}")
        print(f"Bottlenecks: {report.identified_bottlenecks}")
        print(f"Issues: {report.performance_issues}")
        print(f"Recommendations: {report.optimization_recommendations}")
        
        # Should identify performance issues with only slow operations
        assert len(report.identified_bottlenecks) > 0 or len(report.optimization_recommendations) > 0
    
    def test_performance_alerts_integration(self, performance_monitor):
        """Test performance alerts integration."""
        alert_events = []
        
        def alert_handler(event_data):
            if event_data.get('type') == 'performance_alert':
                alert_events.append(event_data)
        
        performance_monitor.add_performance_callback(alert_handler)
        
        # Record metric that exceeds thresholds
        critical_metric = PerformanceMetric(
            operation_type="critical_test",
            operation_id="critical_001",
            response_time_ms=12000.0,  # Exceeds critical threshold
            memory_usage_mb=800.0,     # Exceeds warning threshold
            cpu_usage_percent=95.0,    # Exceeds critical threshold
            success=True
        )
        
        performance_monitor.record_performance_metric(critical_metric)
        performance_monitor._check_performance_thresholds()
        
        # Verify alerts were generated
        assert len(alert_events) > 0
        alert_event = alert_events[0]
        assert alert_event['operation_id'] == 'critical_001'
        assert len(alert_event['alerts']) > 0
        
        # Check alert content
        alerts = alert_event['alerts']
        assert any('Critical response time' in alert for alert in alerts)
        assert any('Critical CPU usage' in alert for alert in alerts)
    
    def test_performance_data_export(self, performance_monitor):
        """Test performance data export functionality."""
        # Add some test data
        for i in range(10):
            metric = PerformanceMetric(
                operation_type="export_test",
                operation_id=f"export_{i}",
                response_time_ms=1000.0 + i * 100,
                success=i < 8  # 8 successful, 2 failed
            )
            performance_monitor.record_performance_metric(metric)
        
        # Add a benchmark
        benchmark = PerformanceBenchmark(
            benchmark_name="export_benchmark",
            total_operations=5,
            successful_operations=4,
            failed_operations=1,
            throughput_ops_per_sec=2.5
        )
        performance_monitor.benchmark_results.append(benchmark)
        
        # Export data
        export_data = performance_monitor.export_performance_data(hours=1.0)
        
        # Verify export structure
        assert 'export_timestamp' in export_data
        assert export_data['period_hours'] == 1.0
        assert export_data['total_metrics'] == 10
        assert len(export_data['metrics']) == 10
        assert len(export_data['benchmarks']) == 1
        
        # Verify data content
        assert 'performance_thresholds' in export_data
        assert 'operation_counters' in export_data
        assert export_data['operation_counters']['export_test'] == 10
        
        # Verify benchmark data
        benchmark_data = export_data['benchmarks'][0]
        assert benchmark_data['benchmark_name'] == 'export_benchmark'
        assert benchmark_data['total_operations'] == 5
    
    def test_performance_baseline_comparison(self, performance_monitor):
        """Test performance baseline comparison functionality."""
        # Set baseline
        baseline_metrics = {
            'response_time_ms': 2000.0,
            'throughput_ops_per_sec': 3.0,
            'memory_usage_mb': 200.0
        }
        performance_monitor.set_performance_baseline("baseline_test", baseline_metrics)
        
        # Add current metrics that are better than baseline
        for i in range(5):
            metric = PerformanceMetric(
                operation_type="baseline_test",
                operation_id=f"current_{i}",
                response_time_ms=1500.0 + i * 50,  # Better than baseline
                memory_usage_mb=150.0 + i * 10,    # Better than baseline
                throughput_ops_per_sec=4.0,        # Better than baseline
                success=True
            )
            performance_monitor.record_performance_metric(metric)
        
        # Get comparison
        comparison = performance_monitor._get_performance_comparison()
        
        # Verify comparison results
        assert 'baseline_test_response_time_improvement' in comparison
        assert comparison['baseline_test_response_time_improvement'] > 0  # Should show improvement
        
        # Generate optimization report to see baseline comparison
        report = performance_monitor.generate_optimization_report(analysis_hours=1.0)
        assert len(report.performance_comparison) > 0
    
    @patch('intelligent_web_scraper.monitoring.performance_monitor.psutil.Process')
    def test_system_resource_monitoring(self, mock_process, performance_monitor):
        """Test system resource monitoring integration."""
        # Mock process for consistent testing
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 256 * 1024 * 1024  # 256MB
        mock_process_instance.cpu_percent.return_value = 45.0
        mock_process.return_value = mock_process_instance
        
        # Test resource tracking
        with performance_monitor.track_operation("resource_test", "res_001") as tracker:
            time.sleep(0.01)
            tracker.set_success(True)
        
        # Verify resource data was captured
        assert len(performance_monitor.performance_metrics) == 1
        metric = performance_monitor.performance_metrics[0]
        assert metric.memory_usage_mb == 256.0
        assert metric.cpu_usage_percent == 45.0
    
    def test_concurrent_performance_monitoring(self, performance_monitor):
        """Test performance monitoring under concurrent operations."""
        import threading
        import concurrent.futures
        
        def concurrent_operation(operation_id: int):
            """Simulate concurrent operation."""
            with performance_monitor.track_operation("concurrent_test", f"conc_{operation_id}") as tracker:
                time.sleep(0.01)  # Simulate work
                tracker.set_success(True)
                return operation_id
        
        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(concurrent_operation, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all operations were tracked
        assert len(performance_monitor.performance_metrics) == 10
        assert len(results) == 10
        
        # Verify no data corruption from concurrent access
        operation_ids = [m.operation_id for m in performance_monitor.performance_metrics]
        assert len(set(operation_ids)) == 10  # All unique
        
        # Get summary
        summary = performance_monitor.get_performance_summary(hours=1.0)
        assert summary['total_operations'] == 10
        assert summary['success_rate_percent'] == 100.0


class TestPerformanceMonitoringLifecycle:
    """Test performance monitoring lifecycle management."""
    
    def test_monitor_lifecycle(self):
        """Test complete monitor lifecycle."""
        monitor = PerformanceMonitor()
        
        # Test initial state
        assert not monitor.is_monitoring
        assert monitor.monitoring_thread is None
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.is_monitoring
        assert monitor.monitoring_thread is not None
        assert monitor.monitoring_thread.is_alive()
        
        # Add some data
        metric = PerformanceMetric(
            operation_type="lifecycle_test",
            operation_id="life_001",
            response_time_ms=1000.0,
            success=True
        )
        monitor.record_performance_metric(metric)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.is_monitoring
        
        # Verify data is still accessible
        assert len(monitor.performance_metrics) == 1
        summary = monitor.get_performance_summary(hours=1.0)
        assert summary['total_operations'] == 1
    
    def test_monitor_cleanup(self):
        """Test monitor data cleanup functionality."""
        monitor = PerformanceMonitor(benchmark_retention_days=1)
        
        # Add old benchmark
        old_benchmark = PerformanceBenchmark(
            benchmark_name="old_test",
            timestamp=datetime.utcnow() - timedelta(days=2)  # Older than retention
        )
        monitor.benchmark_results.append(old_benchmark)
        
        # Add recent benchmark
        recent_benchmark = PerformanceBenchmark(
            benchmark_name="recent_test",
            timestamp=datetime.utcnow()
        )
        monitor.benchmark_results.append(recent_benchmark)
        
        assert len(monitor.benchmark_results) == 2
        
        # Run cleanup
        monitor._cleanup_old_data()
        
        # Should only have recent benchmark
        assert len(monitor.benchmark_results) == 1
        assert monitor.benchmark_results[0].benchmark_name == "recent_test"


if __name__ == "__main__":
    pytest.main([__file__])