#!/usr/bin/env python3
"""
Performance Monitoring Example for Intelligent Web Scraper

This example demonstrates the comprehensive performance monitoring capabilities
including response time tracking, memory usage analysis, throughput measurement,
and performance benchmarking.
"""

import asyncio
import time
import random
from datetime import datetime
from typing import Dict, Any

from intelligent_web_scraper.monitoring.performance_monitor import (
    PerformanceMonitor,
    PerformanceMetric,
    PerformanceBenchmark,
    PerformanceOptimizationReport
)


def simulate_scraping_operation(operation_id: str, complexity: str = "medium") -> Dict[str, Any]:
    """
    Simulate a scraping operation with different complexity levels.
    
    Args:
        operation_id: Unique identifier for the operation
        complexity: Operation complexity (simple, medium, complex)
        
    Returns:
        Dictionary with operation results
    """
    # Simulate different response times based on complexity
    complexity_delays = {
        "simple": (0.1, 0.5),
        "medium": (0.5, 2.0),
        "complex": (2.0, 5.0)
    }
    
    min_delay, max_delay = complexity_delays.get(complexity, (0.5, 2.0))
    delay = random.uniform(min_delay, max_delay)
    
    # Simulate work
    time.sleep(delay)
    
    # Simulate occasional failures
    success = random.random() > 0.1  # 90% success rate
    
    if success:
        return {
            "operation_id": operation_id,
            "status": "success",
            "items_extracted": random.randint(5, 50),
            "pages_processed": random.randint(1, 5),
            "quality_score": random.uniform(70, 95)
        }
    else:
        raise Exception(f"Simulated failure for operation {operation_id}")


def demonstrate_basic_performance_tracking():
    """Demonstrate basic performance tracking with context manager."""
    print("🔍 Demonstrating Basic Performance Tracking")
    print("=" * 50)
    
    # Create performance monitor
    monitor = PerformanceMonitor(
        history_size=1000,
        enable_detailed_tracking=True
    )
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Track several operations
        for i in range(5):
            operation_id = f"basic_op_{i+1}"
            complexity = random.choice(["simple", "medium", "complex"])
            
            print(f"Tracking operation {operation_id} ({complexity})...")
            
            with monitor.track_operation("scraping", operation_id) as tracker:
                try:
                    result = simulate_scraping_operation(operation_id, complexity)
                    tracker.set_success(True)
                    tracker.add_metadata("items_extracted", result["items_extracted"])
                    tracker.add_metadata("complexity", complexity)
                    print(f"  ✅ Success: {result['items_extracted']} items extracted")
                    
                except Exception as e:
                    tracker.set_success(False, str(e))
                    print(f"  ❌ Failed: {e}")
        
        # Get performance summary
        print("\n📊 Performance Summary:")
        summary = monitor.get_performance_summary(hours=1.0)
        
        print(f"  Total Operations: {summary['total_operations']}")
        print(f"  Success Rate: {summary['success_rate_percent']:.1f}%")
        print(f"  Average Response Time: {summary['average_response_time_ms']:.0f}ms")
        print(f"  Throughput: {summary['throughput_ops_per_sec']:.2f} ops/sec")
        
        # Show operation type breakdown
        if summary['operation_types']:
            print("\n  Operation Types:")
            for op_type, stats in summary['operation_types'].items():
                print(f"    {op_type}: {stats['count']} ops, "
                      f"{stats['avg_response_time']:.0f}ms avg")
    
    finally:
        monitor.stop_monitoring()
    
    print()


def demonstrate_performance_benchmarking():
    """Demonstrate performance benchmarking capabilities."""
    print("🏁 Demonstrating Performance Benchmarking")
    print("=" * 50)
    
    monitor = PerformanceMonitor()
    
    def benchmark_operation():
        """Operation to benchmark."""
        return simulate_scraping_operation(f"bench_{random.randint(1000, 9999)}", "medium")
    
    # Run sequential benchmark
    print("Running sequential benchmark...")
    sequential_benchmark = monitor.run_benchmark(
        benchmark_name="sequential_scraping",
        operation_func=benchmark_operation,
        num_operations=10,
        concurrent_operations=1,
        warmup_operations=2
    )
    
    print(f"Sequential Benchmark Results:")
    print(f"  Total Operations: {sequential_benchmark.total_operations}")
    print(f"  Success Rate: {sequential_benchmark.success_rate_percent:.1f}%")
    print(f"  Average Response Time: {sequential_benchmark.average_response_time_ms:.0f}ms")
    print(f"  Throughput: {sequential_benchmark.throughput_ops_per_sec:.2f} ops/sec")
    print(f"  P95 Response Time: {sequential_benchmark.p95_response_time_ms:.0f}ms")
    
    # Run concurrent benchmark
    print("\nRunning concurrent benchmark...")
    concurrent_benchmark = monitor.run_benchmark(
        benchmark_name="concurrent_scraping",
        operation_func=benchmark_operation,
        num_operations=15,
        concurrent_operations=3,
        warmup_operations=0
    )
    
    print(f"Concurrent Benchmark Results:")
    print(f"  Total Operations: {concurrent_benchmark.total_operations}")
    print(f"  Success Rate: {concurrent_benchmark.success_rate_percent:.1f}%")
    print(f"  Average Response Time: {concurrent_benchmark.average_response_time_ms:.0f}ms")
    print(f"  Throughput: {concurrent_benchmark.throughput_ops_per_sec:.2f} ops/sec")
    print(f"  P95 Response Time: {concurrent_benchmark.p95_response_time_ms:.0f}ms")
    
    # Compare benchmarks
    print(f"\n📈 Performance Comparison:")
    throughput_improvement = ((concurrent_benchmark.throughput_ops_per_sec - 
                              sequential_benchmark.throughput_ops_per_sec) / 
                             sequential_benchmark.throughput_ops_per_sec) * 100
    print(f"  Throughput Improvement: {throughput_improvement:+.1f}%")
    
    print()


def demonstrate_optimization_report():
    """Demonstrate optimization report generation."""
    print("📋 Demonstrating Optimization Report")
    print("=" * 50)
    
    monitor = PerformanceMonitor()
    
    # Generate some performance data with varying characteristics
    print("Generating performance data...")
    
    # Add some slow operations
    for i in range(5):
        metric = PerformanceMetric(
            operation_type="slow_scraping",
            operation_id=f"slow_{i}",
            response_time_ms=random.uniform(8000, 12000),  # Very slow
            memory_usage_mb=random.uniform(400, 600),      # High memory
            cpu_usage_percent=random.uniform(70, 90),      # High CPU
            success=True
        )
        monitor.record_performance_metric(metric)
    
    # Add some fast operations
    for i in range(10):
        metric = PerformanceMetric(
            operation_type="fast_scraping",
            operation_id=f"fast_{i}",
            response_time_ms=random.uniform(200, 800),     # Fast
            memory_usage_mb=random.uniform(50, 150),       # Low memory
            cpu_usage_percent=random.uniform(10, 30),      # Low CPU
            success=True
        )
        monitor.record_performance_metric(metric)
    
    # Add some failing operations
    for i in range(3):
        metric = PerformanceMetric(
            operation_type="failing_scraping",
            operation_id=f"fail_{i}",
            response_time_ms=random.uniform(1000, 3000),
            success=False,
            error_message="Connection timeout"
        )
        monitor.record_performance_metric(metric)
    
    # Generate optimization report
    print("Generating optimization report...")
    report = monitor.generate_optimization_report(analysis_hours=1.0)
    
    print(f"\n🎯 Optimization Report:")
    print(f"  Report ID: {report.report_id}")
    print(f"  Analysis Period: {report.analysis_period_hours} hours")
    print(f"  Overall Performance Score: {report.overall_performance_score:.1f}/100")
    print(f"  Performance Trend: {report.performance_trend}")
    print(f"  Average Response Time: {report.average_response_time_ms:.0f}ms")
    print(f"  Throughput: {report.throughput_ops_per_sec:.2f} ops/sec")
    
    if report.identified_bottlenecks:
        print(f"\n🚨 Identified Bottlenecks:")
        for bottleneck in report.identified_bottlenecks:
            print(f"    • {bottleneck}")
    
    if report.performance_issues:
        print(f"\n⚠️  Performance Issues:")
        for issue in report.performance_issues:
            print(f"    • {issue}")
    
    if report.optimization_recommendations:
        print(f"\n💡 Optimization Recommendations:")
        for recommendation in report.optimization_recommendations:
            print(f"    • {recommendation}")
    
    if report.configuration_suggestions:
        print(f"\n⚙️  Configuration Suggestions:")
        for key, value in report.configuration_suggestions.items():
            print(f"    • {key}: {value}")
    
    print()


def demonstrate_performance_callbacks():
    """Demonstrate performance event callbacks."""
    print("📡 Demonstrating Performance Callbacks")
    print("=" * 50)
    
    monitor = PerformanceMonitor()
    
    # Callback to handle performance events
    def performance_event_handler(event_data: Dict[str, Any]):
        event_type = event_data.get('type', 'unknown')
        
        if event_type == 'metric_recorded':
            metric = event_data['metric']
            print(f"  📊 Metric recorded: {metric['operation_type']} - "
                  f"{metric['response_time_ms']:.0f}ms")
        
        elif event_type == 'performance_alert':
            operation_id = event_data['operation_id']
            alerts = event_data['alerts']
            print(f"  🚨 Performance alert for {operation_id}:")
            for alert in alerts:
                print(f"      • {alert}")
    
    # Add callback
    monitor.add_performance_callback(performance_event_handler)
    
    print("Recording metrics with callback notifications...")
    
    # Record some metrics that will trigger callbacks
    for i in range(3):
        # Normal operation
        metric = PerformanceMetric(
            operation_type="callback_test",
            operation_id=f"normal_{i}",
            response_time_ms=random.uniform(500, 1500),
            success=True
        )
        monitor.record_performance_metric(metric)
        time.sleep(0.1)
    
    # Record a metric that exceeds thresholds
    critical_metric = PerformanceMetric(
        operation_type="callback_test",
        operation_id="critical_op",
        response_time_ms=15000,  # Exceeds critical threshold
        memory_usage_mb=1200,    # Exceeds critical threshold
        success=True
    )
    monitor.record_performance_metric(critical_metric)
    
    # Check thresholds to trigger alerts
    monitor._check_performance_thresholds()
    
    print()


def demonstrate_performance_baselines():
    """Demonstrate performance baseline comparison."""
    print("📏 Demonstrating Performance Baselines")
    print("=" * 50)
    
    monitor = PerformanceMonitor()
    
    # Set performance baseline
    baseline_metrics = {
        'response_time_ms': 3000.0,
        'throughput_ops_per_sec': 2.0,
        'memory_usage_mb': 300.0
    }
    monitor.set_performance_baseline("baseline_test", baseline_metrics)
    print(f"Set baseline: {baseline_metrics}")
    
    # Add current metrics that are better than baseline
    print("\nRecording current performance metrics...")
    for i in range(5):
        metric = PerformanceMetric(
            operation_type="baseline_test",
            operation_id=f"current_{i}",
            response_time_ms=random.uniform(1500, 2500),  # Better than baseline
            memory_usage_mb=random.uniform(150, 250),     # Better than baseline
            success=True
        )
        monitor.record_performance_metric(metric)
    
    # Get performance comparison
    comparison = monitor._get_performance_comparison()
    print(f"\n📊 Performance Comparison:")
    for metric_name, improvement in comparison.items():
        print(f"  {metric_name}: {improvement:+.1f}%")
    
    print()


def main():
    """Main demonstration function."""
    print("🕷️ Intelligent Web Scraper - Performance Monitoring Demo")
    print("=" * 60)
    print()
    
    # Run all demonstrations
    demonstrate_basic_performance_tracking()
    demonstrate_performance_benchmarking()
    demonstrate_optimization_report()
    demonstrate_performance_callbacks()
    demonstrate_performance_baselines()
    
    print("✅ Performance monitoring demonstration completed!")
    print("\nKey Features Demonstrated:")
    print("  • Real-time performance tracking with context managers")
    print("  • Comprehensive benchmarking (sequential and concurrent)")
    print("  • Automated optimization report generation")
    print("  • Performance event callbacks and alerts")
    print("  • Baseline comparison and trend analysis")
    print("  • Memory and CPU usage monitoring")
    print("  • Throughput and response time analysis")


if __name__ == "__main__":
    main()