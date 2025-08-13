"""
Tests for the Enhanced Performance Monitor module.

This module tests the enhanced performance monitoring capabilities
including benchmarking, optimization reporting, and data export.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from intelligent_web_scraper.monitoring.enhanced_performance_monitor import (
    EnhancedPerformanceMonitor,
    PerformanceBenchmark,
    OptimizationReport,
    ResourceUtilizationTrend
)


class TestEnhancedPerformanceMonitor:
    """Test cases for EnhancedPerformanceMonitor."""
    
    @pytest.fixture
    def monitor(self):
        """Create a test monitor instance."""
        return EnhancedPerformanceMonitor()
    
    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor is not None
        assert hasattr(monitor, 'benchmarks')
        assert hasattr(monitor, 'optimization_reports')
    
    def test_performance_summary(self, monitor):
        """Test performance summary generation."""
        # Add some mock metrics
        monitor.record_metric("response_time", 1.5)
        monitor.record_metric("memory_usage", 512.0)
        
        summary = monitor.get_performance_summary()
        assert summary is not None
        assert "response_time" in summary
        assert "memory_usage" in summary
    
    def test_benchmarking(self, monitor):
        """Test benchmarking functionality."""
        benchmark = monitor.create_benchmark("test_benchmark")
        assert benchmark is not None
        assert benchmark.name == "test_benchmark"
        
        # Complete benchmark
        monitor.complete_benchmark(benchmark.id, {"score": 85.5})
        completed = monitor.get_benchmark(benchmark.id)
        assert completed.completed is True
        assert completed.results["score"] == 85.5
    
    def test_optimization_report(self, monitor):
        """Test optimization report generation."""
        # Add some metrics
        monitor.record_metric("cpu_usage", 75.0)
        monitor.record_metric("memory_usage", 60.0)
        
        report = monitor.generate_optimization_report()
        assert report is not None
        assert hasattr(report, 'recommendations')
        assert len(report.recommendations) > 0
    
    def test_resource_trends(self, monitor):
        """Test resource utilization trend analysis."""
        # Record metrics over time
        for i in range(10):
            monitor.record_metric("cpu_usage", 50.0 + i * 2)
            monitor.record_metric("memory_usage", 40.0 + i * 1.5)
        
        trends = monitor.analyze_resource_trends()
        assert trends is not None
        assert "cpu_usage" in trends
        assert "memory_usage" in trends
    
    def test_data_export(self, monitor):
        """Test performance data export."""
        # Add some test data
        monitor.record_metric("test_metric", 100.0)
        benchmark = monitor.create_benchmark("export_test")
        monitor.complete_benchmark(benchmark.id, {"result": "success"})
        
        export_data = monitor.export_performance_data()
        assert export_data is not None
        assert "metrics" in export_data
        assert "benchmarks" in export_data
        assert len(export_data["metrics"]) > 0
        assert len(export_data["benchmarks"]) > 0


class TestPerformanceBenchmark:
    """Test cases for PerformanceBenchmark."""
    
    def test_benchmark_creation(self):
        """Test benchmark creation."""
        benchmark = PerformanceBenchmark("test_benchmark")
        assert benchmark.name == "test_benchmark"
        assert benchmark.completed is False
        assert benchmark.results is None
    
    def test_benchmark_completion(self):
        """Test benchmark completion."""
        benchmark = PerformanceBenchmark("test_benchmark")
        results = {"score": 90.0, "time": 2.5}
        
        benchmark.complete(results)
        assert benchmark.completed is True
        assert benchmark.results == results
        assert benchmark.end_time is not None


class TestOptimizationReport:
    """Test cases for OptimizationReport."""
    
    def test_report_creation(self):
        """Test optimization report creation."""
        metrics = {"cpu_usage": 80.0, "memory_usage": 70.0}
        report = OptimizationReport(metrics)
        
        assert report.metrics == metrics
        assert hasattr(report, 'recommendations')
        assert len(report.recommendations) > 0
    
    def test_report_recommendations(self):
        """Test report recommendation generation."""
        high_cpu_metrics = {"cpu_usage": 95.0, "memory_usage": 50.0}
        report = OptimizationReport(high_cpu_metrics)
        
        cpu_recommendations = [r for r in report.recommendations if "CPU" in r]
        assert len(cpu_recommendations) > 0


class TestResourceUtilizationTrend:
    """Test cases for ResourceUtilizationTrend."""
    
    def test_trend_analysis(self):
        """Test resource utilization trend analysis."""
        # Create sample data points
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(10, 0, -1)]
        values = [50.0 + i * 2 for i in range(10)]
        
        trend = ResourceUtilizationTrend("cpu_usage", timestamps, values)
        
        assert trend.resource_name == "cpu_usage"
        assert len(trend.data_points) == 10
        assert trend.trend_direction in ["increasing", "decreasing", "stable"]
    
    def test_trend_prediction(self):
        """Test trend prediction functionality."""
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(5, 0, -1)]
        values = [10.0, 20.0, 30.0, 40.0, 50.0]  # Clear increasing trend
        
        trend = ResourceUtilizationTrend("memory_usage", timestamps, values)
        prediction = trend.predict_next_value()
        
        assert prediction > 50.0  # Should predict higher value
        assert isinstance(prediction, float)


@pytest.mark.asyncio
async def test_async_monitoring():
    """Test asynchronous monitoring capabilities."""
    monitor = EnhancedPerformanceMonitor()
    
    # Simulate async metric recording
    async def record_metrics():
        for i in range(5):
            monitor.record_metric("async_metric", i * 10.0)
            await asyncio.sleep(0.1)
    
    await record_metrics()
    
    metrics = monitor.get_metrics("async_metric")
    assert len(metrics) == 5
    assert metrics[-1].value == 40.0


if __name__ == "__main__":
    pytest.main([__file__])