#!/usr/bin/env python3
"""
Test script for enhanced performance monitoring functionality.
"""

import time
import statistics
from datetime import datetime, timedelta

from intelligent_web_scraper.monitoring.performance_monitor import (
    PerformanceMonitor,
    PerformanceMetric,
    PerformanceBenchmark,
    PerformanceOptimizationReport
)


def test_enhanced_performance_monitor():
    """Test enhanced performance monitoring features."""
    print("Testing Enhanced Performance Monitor...")
    
    # Create monitor
    monitor = PerformanceMonitor(
        history_size=1000,
        benchmark_retention_days=7,
        enable_detailed_tracking=True
    )
    
    print("✓ Performance monitor created")
    
    # Test 1: Performance Summary
    print("\n1. Testing Performance Summary...")
    
    # Add test metrics
    for i in range(20):
        metric = PerformanceMetric(
            operation_type="test_operation",
            operation_id=f"test_{i}",
            response_time_ms=1000.0 + i * 100,
            memory_usage_mb=100.0 + i * 10,
            cpu_usage_percent=20.0 + i * 2,
            success=i < 16  # 80% success rate
        )
        monitor.record_performance_metric(metric)
    
    summary = monitor.get_performance_summary(hours=1.0)
    
    assert summary['total_operations'] == 20
    assert summary['success_rate_percent'] == 80.0
    assert 'test_operation' in summary['operation_types']
    
    print("✓ Performance summary generated successfully")
    
    # Test 2: Benchmarking
    print("\n2. Testing Benchmarking...")
    
    def test_operation():
        time.sleep(0.001)
        return "benchmark_result"
    
    benchmark = monitor.run_benchmark(
        benchmark_name="test_benchmark",
        operation_func=test_operation,
        num_operations=10,
        concurrent_operations=2,
        warmup_operations=2
    )
    
    assert benchmark.benchmark_name == "test_benchmark"
    assert benchmark.total_operations == 10
    assert benchmark.throughput_ops_per_sec > 0
    
    print("✓ Benchmark completed successfully")
    
    # Test 3: Performance Baselines
    print("\n3. Testing Performance Baselines...")
    
    baseline_metrics = {
        'response_time_ms': 2000.0,
        'throughput_ops_per_sec': 5.0,
        'memory_usage_mb': 200.0
    }
    
    monitor.set_performance_baseline("baseline_test", baseline_metrics)
    
    # Add better performance metrics
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
    
    print("✓ Performance baseline comparison working")
    
    # Test 4: Resource Utilization Trends
    print("\n4. Testing Resource Utilization Trends...")
    
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
    assert len(trends['memory_trend']) > 0
    
    print("✓ Resource utilization trends generated")
    
    # Test 5: Optimization Report
    print("\n5. Testing Optimization Report...")
    
    # Add metrics with performance issues
    for i in range(10):
        metric = PerformanceMetric(
            operation_type="optimization_test",
            operation_id=f"opt_{i}",
            response_time_ms=3000.0 + i * 200,  # High response times
            memory_usage_mb=400.0 + i * 30,     # High memory usage
            cpu_usage_percent=60.0 + i * 3,     # High CPU usage
            success=i < 8  # Some failures
        )
        monitor.record_performance_metric(metric)
    
    report = monitor.generate_optimization_report(analysis_hours=1.0)
    
    assert report.report_id.startswith("opt_report_")
    assert 0.0 <= report.overall_performance_score <= 100.0
    assert report.average_response_time_ms > 0
    # Optimization recommendations may be empty if performance is acceptable
    assert isinstance(report.optimization_recommendations, list)
    
    print("✓ Optimization report generated")
    
    # Test 6: Benchmark Comparison
    print("\n6. Testing Benchmark Comparison...")
    
    # Create second benchmark
    def faster_operation():
        time.sleep(0.0005)  # Faster than first benchmark
        return "faster_result"
    
    benchmark2 = monitor.run_benchmark(
        benchmark_name="faster_benchmark",
        operation_func=faster_operation,
        num_operations=10,
        concurrent_operations=2,
        warmup_operations=2
    )
    
    comparison = monitor.compare_benchmarks("test_benchmark", "faster_benchmark")
    
    assert 'benchmark1' in comparison
    assert 'benchmark2' in comparison
    assert 'comparison' in comparison
    assert 'overall_improvement' in comparison['comparison']
    
    print("✓ Benchmark comparison working")
    
    # Test 7: Data Export
    print("\n7. Testing Data Export...")
    
    export_data = monitor.export_performance_data(hours=1.0)
    
    assert 'export_timestamp' in export_data
    assert 'total_metrics' in export_data
    assert 'benchmarks' in export_data
    assert 'performance_thresholds' in export_data
    assert len(export_data['benchmarks']) == 2  # We created 2 benchmarks
    
    print("✓ Performance data export working")
    
    print("\n🎉 All enhanced performance monitoring tests passed!")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"- Total metrics recorded: {len(monitor.performance_metrics)}")
    print(f"- Total benchmarks: {len(monitor.benchmark_results)}")
    print(f"- Performance score: {report.overall_performance_score:.1f}")
    print(f"- Export data size: {export_data['total_metrics']} metrics")


if __name__ == "__main__":
    test_enhanced_performance_monitor()