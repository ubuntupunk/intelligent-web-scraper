"""
Performance monitoring system for the Intelligent Web Scraper.

This module provides comprehensive performance monitoring capabilities including
response time tracking, memory usage analysis, throughput measurement, and
performance benchmarking for system optimization.
"""

import asyncio
import threading
import time
import statistics
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
from contextlib import contextmanager

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    operation_type: str = ""
    operation_id: str = ""
    response_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    throughput_ops_per_sec: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'operation_type': self.operation_type,
            'operation_id': self.operation_id,
            'response_time_ms': self.response_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'throughput_ops_per_sec': self.throughput_ops_per_sec,
            'success': self.success,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results."""
    benchmark_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_duration_seconds: float = 0.0
    average_response_time_ms: float = 0.0
    median_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    min_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    average_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    average_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    success_rate_percent: float = 0.0
    error_rate_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'benchmark_name': self.benchmark_name,
            'timestamp': self.timestamp.isoformat(),
            'total_operations': self.total_operations,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'total_duration_seconds': self.total_duration_seconds,
            'average_response_time_ms': self.average_response_time_ms,
            'median_response_time_ms': self.median_response_time_ms,
            'p95_response_time_ms': self.p95_response_time_ms,
            'p99_response_time_ms': self.p99_response_time_ms,
            'min_response_time_ms': self.min_response_time_ms,
            'max_response_time_ms': self.max_response_time_ms,
            'throughput_ops_per_sec': self.throughput_ops_per_sec,
            'average_memory_mb': self.average_memory_mb,
            'peak_memory_mb': self.peak_memory_mb,
            'average_cpu_percent': self.average_cpu_percent,
            'peak_cpu_percent': self.peak_cpu_percent,
            'success_rate_percent': self.success_rate_percent,
            'error_rate_percent': self.error_rate_percent
        }


class PerformanceOptimizationReport(BaseModel):
    """Performance optimization recommendations."""
    
    report_id: str = Field(..., description="Unique report identifier")
    generated_at: datetime = Field(..., description="Report generation timestamp")
    analysis_period_hours: float = Field(..., description="Analysis period in hours")
    
    # Performance summary
    overall_performance_score: float = Field(..., description="Overall performance score (0-100)")
    performance_trend: str = Field(..., description="Performance trend (improving, stable, degrading)")
    
    # Key metrics
    average_response_time_ms: float = Field(..., description="Average response time")
    response_time_trend: str = Field(..., description="Response time trend")
    throughput_ops_per_sec: float = Field(..., description="Current throughput")
    throughput_trend: str = Field(..., description="Throughput trend")
    resource_utilization_percent: float = Field(..., description="Resource utilization")
    
    # Bottlenecks and issues
    identified_bottlenecks: List[str] = Field(default_factory=list, description="Identified bottlenecks")
    performance_issues: List[str] = Field(default_factory=list, description="Performance issues")
    
    # Recommendations
    optimization_recommendations: List[str] = Field(default_factory=list, description="Optimization recommendations")
    configuration_suggestions: Dict[str, Any] = Field(default_factory=dict, description="Configuration suggestions")
    
    # Comparative analysis
    performance_comparison: Dict[str, float] = Field(default_factory=dict, description="Performance comparison with baselines")
    benchmark_results: List[Dict[str, Any]] = Field(default_factory=list, description="Recent benchmark results")


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    This class provides advanced performance monitoring capabilities including:
    - Real-time response time tracking
    - Memory and CPU usage monitoring
    - Throughput measurement and analysis
    - Performance benchmarking
    - Optimization recommendations
    """
    
    def __init__(
        self, 
        history_size: int = 10000,
        benchmark_retention_days: int = 30,
        enable_detailed_tracking: bool = True
    ):
        self.history_size = history_size
        self.benchmark_retention_days = benchmark_retention_days
        self.enable_detailed_tracking = enable_detailed_tracking
        
        # Performance data storage
        self.performance_metrics: deque = deque(maxlen=history_size)
        self.operation_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.benchmark_results: List[PerformanceBenchmark] = []
        
        # Active operation tracking
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self.operation_counters: Dict[str, int] = defaultdict(int)
        
        # Performance baselines
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        # Thread safety
        self.metrics_lock = threading.RLock()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Performance thresholds
        self.performance_thresholds = {
            'response_time_warning_ms': 5000.0,
            'response_time_critical_ms': 10000.0,
            'memory_warning_mb': 500.0,
            'memory_critical_mb': 1000.0,
            'cpu_warning_percent': 80.0,
            'cpu_critical_percent': 95.0,
            'throughput_warning_ops_per_sec': 0.1,
            'success_rate_warning_percent': 90.0,
            'success_rate_critical_percent': 80.0
        }
        
        # Callbacks for performance events
        self.performance_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        logger.info("PerformanceMonitor initialized")
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.is_monitoring:
            logger.warning("Performance monitoring is already running")
            return
        
        self.is_monitoring = True
        self.shutdown_event.clear()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PerformanceMonitoringThread",
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.shutdown_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring and not self.shutdown_event.is_set():
            try:
                # Clean up old data
                self._cleanup_old_data()
                
                # Check for performance issues
                self._check_performance_thresholds()
                
                # Sleep for monitoring interval
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                time.sleep(1.0)
    
    @contextmanager
    def track_operation(
        self, 
        operation_type: str, 
        operation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracking operation performance.
        
        Usage:
            with monitor.track_operation("scraping", "op-123") as tracker:
                # Perform operation
                result = do_scraping()
                tracker.set_success(True)
        """
        if operation_id is None:
            operation_id = f"{operation_type}_{int(time.time() * 1000)}"
        
        # Record operation start
        start_time = time.time()
        start_memory = self._get_current_memory_usage()
        start_cpu = self._get_current_cpu_usage()
        
        # Create operation tracker
        tracker = OperationTracker(operation_id, operation_type)
        
        # Store active operation
        with self.metrics_lock:
            self.active_operations[operation_id] = {
                'operation_type': operation_type,
                'start_time': start_time,
                'start_memory': start_memory,
                'start_cpu': start_cpu,
                'metadata': metadata or {}
            }
        
        try:
            yield tracker
            
        finally:
            # Record operation end
            end_time = time.time()
            end_memory = self._get_current_memory_usage()
            end_cpu = self._get_current_cpu_usage()
            
            # Calculate metrics
            response_time_ms = (end_time - start_time) * 1000
            memory_usage_mb = max(end_memory, start_memory)
            cpu_usage_percent = max(end_cpu, start_cpu)
            
            # Create performance metric
            metric = PerformanceMetric(
                operation_type=operation_type,
                operation_id=operation_id,
                response_time_ms=response_time_ms,
                memory_usage_mb=memory_usage_mb,
                cpu_usage_percent=cpu_usage_percent,
                success=tracker.success,
                error_message=tracker.error_message,
                metadata=metadata or {}
            )
            
            # Store metric
            self.record_performance_metric(metric)
            
            # Clean up active operation
            with self.metrics_lock:
                self.active_operations.pop(operation_id, None)
    
    def record_performance_metric(self, metric: PerformanceMetric) -> None:
        """Record a performance metric."""
        with self.metrics_lock:
            # Add to general metrics
            self.performance_metrics.append(metric)
            
            # Add to operation-specific metrics
            self.operation_metrics[metric.operation_type].append(metric)
            
            # Update operation counter
            self.operation_counters[metric.operation_type] += 1
            
            # Calculate throughput if we have enough data
            if len(self.operation_metrics[metric.operation_type]) >= 2:
                recent_metrics = list(self.operation_metrics[metric.operation_type])[-10:]
                time_span = (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds()
                if time_span > 0:
                    metric.throughput_ops_per_sec = len(recent_metrics) / time_span
        
        # Notify callbacks
        self._notify_performance_callbacks({
            'type': 'metric_recorded',
            'metric': metric.to_dict()
        })
        
        logger.debug(f"Recorded performance metric for {metric.operation_type}: {metric.response_time_ms:.2f}ms")    

    def run_benchmark(
        self, 
        benchmark_name: str,
        operation_func: Callable[[], Any],
        num_operations: int = 100,
        concurrent_operations: int = 1,
        warmup_operations: int = 10
    ) -> PerformanceBenchmark:
        """
        Run a performance benchmark.
        
        Args:
            benchmark_name: Name of the benchmark
            operation_func: Function to benchmark
            num_operations: Number of operations to run
            concurrent_operations: Number of concurrent operations
            warmup_operations: Number of warmup operations
            
        Returns:
            PerformanceBenchmark with results
        """
        logger.info(f"Starting benchmark '{benchmark_name}' with {num_operations} operations")
        
        # Run warmup operations
        if warmup_operations > 0:
            logger.info(f"Running {warmup_operations} warmup operations")
            for _ in range(warmup_operations):
                try:
                    operation_func()
                except Exception as e:
                    logger.warning(f"Warmup operation failed: {e}")
        
        # Collect metrics during benchmark
        benchmark_metrics: List[PerformanceMetric] = []
        start_time = time.time()
        
        if concurrent_operations == 1:
            # Sequential execution
            for i in range(num_operations):
                with self.track_operation(f"benchmark_{benchmark_name}", f"op_{i}") as tracker:
                    try:
                        operation_func()
                        tracker.set_success(True)
                    except Exception as e:
                        tracker.set_success(False, str(e))
                
                # Get the last recorded metric
                if self.performance_metrics:
                    benchmark_metrics.append(self.performance_metrics[-1])
        else:
            # Concurrent execution
            import concurrent.futures
            
            def run_single_operation(op_id: int) -> PerformanceMetric:
                with self.track_operation(f"benchmark_{benchmark_name}", f"op_{op_id}") as tracker:
                    try:
                        operation_func()
                        tracker.set_success(True)
                    except Exception as e:
                        tracker.set_success(False, str(e))
                
                return self.performance_metrics[-1] if self.performance_metrics else None
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_operations) as executor:
                futures = [executor.submit(run_single_operation, i) for i in range(num_operations)]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        metric = future.result()
                        if metric:
                            benchmark_metrics.append(metric)
                    except Exception as e:
                        logger.error(f"Benchmark operation failed: {e}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Calculate benchmark results
        benchmark = self._calculate_benchmark_results(
            benchmark_name, benchmark_metrics, total_duration
        )
        
        # Store benchmark
        with self.metrics_lock:
            self.benchmark_results.append(benchmark)
        
        logger.info(f"Benchmark '{benchmark_name}' completed: {benchmark.throughput_ops_per_sec:.2f} ops/sec")
        
        return benchmark
    
    def _calculate_benchmark_results(
        self, 
        benchmark_name: str, 
        metrics: List[PerformanceMetric], 
        total_duration: float
    ) -> PerformanceBenchmark:
        """Calculate benchmark results from metrics."""
        if not metrics:
            return PerformanceBenchmark(benchmark_name=benchmark_name)
        
        # Extract values
        response_times = [m.response_time_ms for m in metrics]
        memory_usage = [m.memory_usage_mb for m in metrics]
        cpu_usage = [m.cpu_usage_percent for m in metrics]
        successful_ops = [m for m in metrics if m.success]
        failed_ops = [m for m in metrics if not m.success]
        
        # Calculate statistics
        benchmark = PerformanceBenchmark(
            benchmark_name=benchmark_name,
            total_operations=len(metrics),
            successful_operations=len(successful_ops),
            failed_operations=len(failed_ops),
            total_duration_seconds=total_duration,
            throughput_ops_per_sec=len(metrics) / max(total_duration, 0.001)
        )
        
        if response_times:
            benchmark.average_response_time_ms = statistics.mean(response_times)
            benchmark.median_response_time_ms = statistics.median(response_times)
            benchmark.min_response_time_ms = min(response_times)
            benchmark.max_response_time_ms = max(response_times)
            
            # Calculate percentiles
            sorted_times = sorted(response_times)
            benchmark.p95_response_time_ms = self._calculate_percentile(sorted_times, 95)
            benchmark.p99_response_time_ms = self._calculate_percentile(sorted_times, 99)
        
        if memory_usage:
            benchmark.average_memory_mb = statistics.mean(memory_usage)
            benchmark.peak_memory_mb = max(memory_usage)
        
        if cpu_usage:
            benchmark.average_cpu_percent = statistics.mean(cpu_usage)
            benchmark.peak_cpu_percent = max(cpu_usage)
        
        # Calculate rates
        if benchmark.total_operations > 0:
            benchmark.success_rate_percent = (benchmark.successful_operations / benchmark.total_operations) * 100
            benchmark.error_rate_percent = (benchmark.failed_operations / benchmark.total_operations) * 100
        
        return benchmark
    
    def _calculate_percentile(self, sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)
        
        if lower_index == upper_index:
            return sorted_values[lower_index]
        
        # Linear interpolation
        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight 
   
    def generate_optimization_report(self, analysis_hours: float = 24.0) -> PerformanceOptimizationReport:
        """Generate performance optimization report."""
        logger.info(f"Generating performance optimization report for last {analysis_hours} hours")
        
        # Get metrics from analysis period
        cutoff_time = datetime.utcnow() - timedelta(hours=analysis_hours)
        
        with self.metrics_lock:
            recent_metrics = [
                m for m in self.performance_metrics 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            logger.warning("No metrics available for optimization report")
            return PerformanceOptimizationReport(
                report_id=f"opt_report_{int(time.time())}",
                generated_at=datetime.utcnow(),
                analysis_period_hours=analysis_hours,
                overall_performance_score=0.0,
                performance_trend="unknown",
                average_response_time_ms=0.0,
                response_time_trend="unknown",
                throughput_ops_per_sec=0.0,
                throughput_trend="unknown",
                resource_utilization_percent=0.0
            )
        
        # Analyze performance
        analysis = self._analyze_performance_metrics(recent_metrics)
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(analysis)
        
        # Create report
        report = PerformanceOptimizationReport(
            report_id=f"opt_report_{int(time.time())}",
            generated_at=datetime.utcnow(),
            analysis_period_hours=analysis_hours,
            **analysis,
            **recommendations
        )
        
        logger.info(f"Generated optimization report with score: {report.overall_performance_score:.1f}")
        
        return report
    
    def _analyze_performance_metrics(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze performance metrics for optimization report."""
        if not metrics:
            return {}
        
        # Calculate basic statistics
        response_times = [m.response_time_ms for m in metrics]
        memory_usage = [m.memory_usage_mb for m in metrics]
        cpu_usage = [m.cpu_usage_percent for m in metrics]
        successful_ops = [m for m in metrics if m.success]
        
        avg_response_time = statistics.mean(response_times) if response_times else 0.0
        avg_memory = statistics.mean(memory_usage) if memory_usage else 0.0
        avg_cpu = statistics.mean(cpu_usage) if cpu_usage else 0.0
        success_rate = (len(successful_ops) / len(metrics)) * 100 if metrics else 0.0
        
        # Calculate throughput
        if len(metrics) >= 2:
            time_span = (metrics[-1].timestamp - metrics[0].timestamp).total_seconds()
            throughput = len(metrics) / max(time_span, 1.0)
        else:
            throughput = 0.0
        
        # Calculate trends (simplified)
        response_time_trend = self._calculate_trend([m.response_time_ms for m in metrics[-50:]])
        throughput_trend = self._calculate_trend([m.throughput_ops_per_sec for m in metrics[-50:] if m.throughput_ops_per_sec > 0])
        
        # Calculate overall performance score
        performance_score = self._calculate_performance_score(
            avg_response_time, throughput, success_rate, avg_memory, avg_cpu
        )
        
        return {
            'overall_performance_score': performance_score,
            'performance_trend': self._get_overall_trend(response_time_trend, throughput_trend),
            'average_response_time_ms': avg_response_time,
            'response_time_trend': response_time_trend,
            'throughput_ops_per_sec': throughput,
            'throughput_trend': throughput_trend,
            'resource_utilization_percent': (avg_memory / 1000.0 + avg_cpu / 100.0) * 50.0  # Simplified calculation
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from values."""
        if len(values) < 10:
            return "stable"
        
        # Simple trend calculation using first and last quartiles
        first_quarter = values[:len(values)//4]
        last_quarter = values[-len(values)//4:]
        
        if not first_quarter or not last_quarter:
            return "stable"
        
        first_avg = statistics.mean(first_quarter)
        last_avg = statistics.mean(last_quarter)
        
        if first_avg == 0:
            return "stable"
        
        change_percent = ((last_avg - first_avg) / first_avg) * 100
        
        if change_percent > 15:
            return "increasing"
        elif change_percent < -15:
            return "decreasing"
        else:
            return "stable"
    
    def _get_overall_trend(self, response_trend: str, throughput_trend: str) -> str:
        """Get overall performance trend."""
        if response_trend == "decreasing" and throughput_trend == "increasing":
            return "improving"
        elif response_trend == "increasing" and throughput_trend == "decreasing":
            return "degrading"
        elif response_trend == "stable" and throughput_trend == "stable":
            return "stable"
        else:
            return "mixed"
    
    def _calculate_performance_score(
        self, 
        avg_response_time: float, 
        throughput: float, 
        success_rate: float,
        avg_memory: float, 
        avg_cpu: float
    ) -> float:
        """Calculate overall performance score (0-100)."""
        # Response time score (lower is better)
        response_score = max(0, 100 - (avg_response_time / 100.0))  # 10s = 0 points
        
        # Throughput score (higher is better)
        throughput_score = min(100, throughput * 10)  # 10 ops/sec = 100 points
        
        # Success rate score
        success_score = success_rate
        
        # Resource efficiency score (lower usage is better)
        memory_score = max(0, 100 - (avg_memory / 10.0))  # 1GB = 0 points
        cpu_score = max(0, 100 - avg_cpu)  # 100% CPU = 0 points
        
        # Weighted average
        total_score = (
            response_score * 0.3 +
            throughput_score * 0.25 +
            success_score * 0.25 +
            memory_score * 0.1 +
            cpu_score * 0.1
        )
        
        return min(100.0, max(0.0, total_score))   
 
    def _generate_optimization_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        config_suggestions = {}
        bottlenecks = []
        issues = []
        
        avg_response_time = analysis.get('average_response_time_ms', 0)
        throughput = analysis.get('throughput_ops_per_sec', 0)
        performance_score = analysis.get('overall_performance_score', 0)
        
        # Response time analysis
        if avg_response_time > self.performance_thresholds['response_time_critical_ms']:
            bottlenecks.append("Critical response time performance")
            recommendations.append("Investigate slow operations and optimize critical paths")
            config_suggestions['request_timeout'] = min(avg_response_time * 0.8, 30000)
        elif avg_response_time > self.performance_thresholds['response_time_warning_ms']:
            issues.append("Elevated response times detected")
            recommendations.append("Monitor response times and consider optimization")
        
        # Throughput analysis
        if throughput < self.performance_thresholds['throughput_warning_ops_per_sec']:
            bottlenecks.append("Low system throughput")
            recommendations.append("Consider increasing concurrency or optimizing operations")
            config_suggestions['concurrent_instances'] = min(10, max(2, int(1.0 / max(throughput, 0.1))))
        
        # Overall performance
        if performance_score < 50:
            issues.append("Overall performance below acceptable threshold")
            recommendations.append("Comprehensive performance review and optimization needed")
        elif performance_score < 70:
            recommendations.append("Performance optimization opportunities available")
        
        # Memory optimization
        with self.metrics_lock:
            if self.performance_metrics:
                recent_memory = [m.memory_usage_mb for m in list(self.performance_metrics)[-100:]]
                if recent_memory:
                    avg_memory = statistics.mean(recent_memory)
                    if avg_memory > self.performance_thresholds['memory_warning_mb']:
                        bottlenecks.append("High memory usage")
                        recommendations.append("Optimize memory usage and implement caching strategies")
                        config_suggestions['memory_limit_mb'] = int(avg_memory * 1.2)
        
        # Get recent benchmark results for comparison
        recent_benchmarks = []
        if self.benchmark_results:
            recent_benchmarks = [b.to_dict() for b in self.benchmark_results[-5:]]
        
        return {
            'identified_bottlenecks': bottlenecks,
            'performance_issues': issues,
            'optimization_recommendations': recommendations,
            'configuration_suggestions': config_suggestions,
            'performance_comparison': self._get_performance_comparison(),
            'benchmark_results': recent_benchmarks
        }
    
    def _get_performance_comparison(self) -> Dict[str, float]:
        """Get performance comparison with baselines."""
        comparison = {}
        
        # Compare with baselines if available
        for operation_type, baseline in self.performance_baselines.items():
            with self.metrics_lock:
                if operation_type in self.operation_metrics:
                    recent_metrics = list(self.operation_metrics[operation_type])[-100:]
                    if recent_metrics:
                        current_avg_response = statistics.mean([m.response_time_ms for m in recent_metrics])
                        baseline_response = baseline.get('response_time_ms', current_avg_response)
                        
                        if baseline_response > 0:
                            improvement = ((baseline_response - current_avg_response) / baseline_response) * 100
                            comparison[f"{operation_type}_response_time_improvement"] = improvement
                        
                        current_throughput = statistics.mean([m.throughput_ops_per_sec for m in recent_metrics if m.throughput_ops_per_sec > 0])
                        baseline_throughput = baseline.get('throughput_ops_per_sec', current_throughput)
                        
                        if baseline_throughput > 0:
                            throughput_change = ((current_throughput - baseline_throughput) / baseline_throughput) * 100
                            comparison[f"{operation_type}_throughput_change"] = throughput_change
        
        return comparison
    
    def set_performance_baseline(self, operation_type: str, baseline_metrics: Dict[str, float]) -> None:
        """Set performance baseline for comparison."""
        self.performance_baselines[operation_type] = baseline_metrics
        logger.info(f"Set performance baseline for {operation_type}")
    
    def get_performance_summary(self, hours: float = 1.0) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.metrics_lock:
            recent_metrics = [
                m for m in self.performance_metrics 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {
                'period_hours': hours,
                'total_operations': 0,
                'operation_types': {},
                'overall_stats': {},
                'trends': {}
            }
        
        # Calculate overall statistics
        response_times = [m.response_time_ms for m in recent_metrics]
        memory_usage = [m.memory_usage_mb for m in recent_metrics]
        cpu_usage = [m.cpu_usage_percent for m in recent_metrics]
        successful_ops = [m for m in recent_metrics if m.success]
        
        overall_stats = {
            'total_operations': len(recent_metrics),
            'successful_operations': len(successful_ops),
            'success_rate': (len(successful_ops) / len(recent_metrics)) * 100,
            'average_response_time_ms': statistics.mean(response_times) if response_times else 0.0,
            'median_response_time_ms': statistics.median(response_times) if response_times else 0.0,
            'p95_response_time_ms': self._calculate_percentile(sorted(response_times), 95) if response_times else 0.0,
            'average_memory_mb': statistics.mean(memory_usage) if memory_usage else 0.0,
            'peak_memory_mb': max(memory_usage) if memory_usage else 0.0,
            'average_cpu_percent': statistics.mean(cpu_usage) if cpu_usage else 0.0,
            'peak_cpu_percent': max(cpu_usage) if cpu_usage else 0.0
        }
        
        # Calculate per-operation-type statistics
        operation_types = {}
        for op_type in set(m.operation_type for m in recent_metrics):
            op_metrics = [m for m in recent_metrics if m.operation_type == op_type]
            op_response_times = [m.response_time_ms for m in op_metrics]
            op_successful = [m for m in op_metrics if m.success]
            
            operation_types[op_type] = {
                'count': len(op_metrics),
                'success_rate': (len(op_successful) / len(op_metrics)) * 100,
                'average_response_time_ms': statistics.mean(op_response_times) if op_response_times else 0.0,
                'median_response_time_ms': statistics.median(op_response_times) if op_response_times else 0.0,
                'throughput_ops_per_sec': len(op_metrics) / (hours * 3600) if hours > 0 else 0.0
            }
        
        # Calculate trends
        trends = {
            'response_time_trend': self._calculate_trend([m.response_time_ms for m in recent_metrics]),
            'memory_trend': self._calculate_trend([m.memory_usage_mb for m in recent_metrics]),
            'cpu_trend': self._calculate_trend([m.cpu_usage_percent for m in recent_metrics])
        }
        
        return {
            'period_hours': hours,
            'total_operations': len(recent_metrics),
            'operation_types': operation_types,
            'overall_stats': overall_stats,
            'trends': trends,
            'active_operations': len(self.active_operations)
        }
    
    def compare_benchmarks(self, benchmark1_name: str, benchmark2_name: str) -> Dict[str, Any]:
        """Compare two benchmarks."""
        benchmark1 = None
        benchmark2 = None
        
        # Find benchmarks by name
        for benchmark in self.benchmark_results:
            if benchmark.benchmark_name == benchmark1_name:
                benchmark1 = benchmark
            elif benchmark.benchmark_name == benchmark2_name:
                benchmark2 = benchmark
        
        if not benchmark1 or not benchmark2:
            return {
                'error': 'One or both benchmarks not found',
                'available_benchmarks': [b.benchmark_name for b in self.benchmark_results]
            }
        
        # Calculate comparisons
        response_time_diff = benchmark2.average_response_time_ms - benchmark1.average_response_time_ms
        throughput_diff = benchmark2.throughput_ops_per_sec - benchmark1.throughput_ops_per_sec
        success_rate_diff = benchmark2.success_rate_percent - benchmark1.success_rate_percent
        memory_diff = benchmark2.average_memory_mb - benchmark1.average_memory_mb
        
        return {
            'benchmark1': benchmark1.to_dict(),
            'benchmark2': benchmark2.to_dict(),
            'comparison': {
                'response_time_difference_ms': response_time_diff,
                'response_time_improvement_percent': (response_time_diff / benchmark1.average_response_time_ms) * 100 if benchmark1.average_response_time_ms > 0 else 0.0,
                'throughput_difference_ops_per_sec': throughput_diff,
                'throughput_improvement_percent': (throughput_diff / benchmark1.throughput_ops_per_sec) * 100 if benchmark1.throughput_ops_per_sec > 0 else 0.0,
                'success_rate_difference_percent': success_rate_diff,
                'memory_difference_mb': memory_diff,
                'memory_improvement_percent': (memory_diff / benchmark1.average_memory_mb) * 100 if benchmark1.average_memory_mb > 0 else 0.0,
                'overall_improvement': self._calculate_overall_improvement(benchmark1, benchmark2)
            }
        }
    
    def _calculate_overall_improvement(self, benchmark1: PerformanceBenchmark, benchmark2: PerformanceBenchmark) -> str:
        """Calculate overall improvement between benchmarks."""
        improvements = 0
        total_metrics = 0
        
        # Response time (lower is better)
        if benchmark1.average_response_time_ms > 0:
            if benchmark2.average_response_time_ms < benchmark1.average_response_time_ms:
                improvements += 1
            total_metrics += 1
        
        # Throughput (higher is better)
        if benchmark1.throughput_ops_per_sec > 0:
            if benchmark2.throughput_ops_per_sec > benchmark1.throughput_ops_per_sec:
                improvements += 1
            total_metrics += 1
        
        # Success rate (higher is better)
        if benchmark2.success_rate_percent > benchmark1.success_rate_percent:
            improvements += 1
        total_metrics += 1
        
        # Memory usage (lower is better)
        if benchmark1.average_memory_mb > 0:
            if benchmark2.average_memory_mb < benchmark1.average_memory_mb:
                improvements += 1
            total_metrics += 1
        
        if total_metrics == 0:
            return "no_data"
        
        improvement_ratio = improvements / total_metrics
        
        if improvement_ratio >= 0.75:
            return "significant_improvement"
        elif improvement_ratio >= 0.5:
            return "moderate_improvement"
        elif improvement_ratio >= 0.25:
            return "slight_improvement"
        else:
            return "degradation"
    
    def get_resource_utilization_trends(self, hours: float = 24.0) -> Dict[str, List[Dict[str, Any]]]:
        """Get detailed resource utilization trends."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.metrics_lock:
            recent_metrics = [
                m for m in self.performance_metrics 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {'memory_trend': [], 'cpu_trend': [], 'response_time_trend': []}
        
        # Group metrics by time intervals (e.g., 5-minute intervals)
        interval_minutes = 5
        interval_seconds = interval_minutes * 60
        
        # Create time buckets
        start_time = recent_metrics[0].timestamp
        end_time = recent_metrics[-1].timestamp
        total_seconds = (end_time - start_time).total_seconds()
        
        if total_seconds < interval_seconds:
            # If data span is less than interval, use single bucket
            buckets = 1
        else:
            buckets = int(total_seconds / interval_seconds) + 1
        
        memory_trend = []
        cpu_trend = []
        response_time_trend = []
        
        for i in range(buckets):
            bucket_start = start_time + timedelta(seconds=i * interval_seconds)
            bucket_end = bucket_start + timedelta(seconds=interval_seconds)
            
            bucket_metrics = [
                m for m in recent_metrics 
                if bucket_start <= m.timestamp < bucket_end
            ]
            
            if bucket_metrics:
                memory_values = [m.memory_usage_mb for m in bucket_metrics]
                cpu_values = [m.cpu_usage_percent for m in bucket_metrics]
                response_values = [m.response_time_ms for m in bucket_metrics]
                
                memory_trend.append({
                    'timestamp': bucket_start.isoformat(),
                    'average': statistics.mean(memory_values),
                    'min': min(memory_values),
                    'max': max(memory_values),
                    'count': len(memory_values)
                })
                
                cpu_trend.append({
                    'timestamp': bucket_start.isoformat(),
                    'average': statistics.mean(cpu_values),
                    'min': min(cpu_values),
                    'max': max(cpu_values),
                    'count': len(cpu_values)
                })
                
                response_time_trend.append({
                    'timestamp': bucket_start.isoformat(),
                    'average': statistics.mean(response_values),
                    'min': min(response_values),
                    'max': max(response_values),
                    'p95': self._calculate_percentile(sorted(response_values), 95),
                    'count': len(response_values)
                })
        
        return {
            'memory_trend': memory_trend,
            'cpu_trend': cpu_trend,
            'response_time_trend': response_time_trend,
            'interval_minutes': interval_minutes,
            'total_data_points': len(recent_metrics)
        }
    
    def _get_performance_comparison(self) -> Dict[str, float]:
        """Get performance comparison with baselines."""
        comparison = {}
        
        # Compare with baselines if available
        for operation_type, baseline in self.performance_baselines.items():
            with self.metrics_lock:
                if operation_type in self.operation_metrics:
                    recent_metrics = list(self.operation_metrics[operation_type])[-100:]
                    if recent_metrics:
                        current_avg_response = statistics.mean([m.response_time_ms for m in recent_metrics])
                        baseline_response = baseline.get('response_time_ms', current_avg_response)
                        
                        if baseline_response > 0:
                            improvement = ((baseline_response - current_avg_response) / baseline_response) * 100
                            comparison[f"{operation_type}_response_time_improvement"] = improvement
                        
                        current_throughput = statistics.mean([m.throughput_ops_per_sec for m in recent_metrics if m.throughput_ops_per_sec > 0])
                        baseline_throughput = baseline.get('throughput_ops_per_sec', 0)
                        
                        if baseline_throughput > 0 and current_throughput > 0:
                            throughput_change = ((current_throughput - baseline_throughput) / baseline_throughput) * 100
                            comparison[f"{operation_type}_throughput_change"] = throughput_change
        
        return comparison
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            memory_bytes = process.memory_info().rss
            return memory_bytes / 1024 / 1024  # Convert to MB
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def _get_current_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            process = psutil.Process()
            return process.cpu_percent()
        except Exception as e:
            logger.warning(f"Failed to get CPU usage: {e}")
            return 0.0
    
    def _cleanup_old_data(self) -> None:
        """Clean up old performance data."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.benchmark_retention_days)
        
        with self.metrics_lock:
            # Clean up old benchmarks
            self.benchmark_results = [
                b for b in self.benchmark_results 
                if b.timestamp >= cutoff_date
            ]
    
    def _check_performance_thresholds(self) -> None:
        """Check performance thresholds and trigger alerts."""
        if not self.performance_metrics:
            return
        
        # Get recent metrics for threshold checking
        recent_metrics = list(self.performance_metrics)[-10:]
        
        for metric in recent_metrics:
            alerts = []
            
            # Check response time thresholds
            if metric.response_time_ms > self.performance_thresholds['response_time_critical_ms']:
                alerts.append(f"Critical response time: {metric.response_time_ms:.0f}ms")
            elif metric.response_time_ms > self.performance_thresholds['response_time_warning_ms']:
                alerts.append(f"High response time: {metric.response_time_ms:.0f}ms")
            
            # Check memory thresholds
            if metric.memory_usage_mb > self.performance_thresholds['memory_critical_mb']:
                alerts.append(f"Critical memory usage: {metric.memory_usage_mb:.0f}MB")
            elif metric.memory_usage_mb > self.performance_thresholds['memory_warning_mb']:
                alerts.append(f"High memory usage: {metric.memory_usage_mb:.0f}MB")
            
            # Check CPU thresholds
            if metric.cpu_usage_percent > self.performance_thresholds['cpu_critical_percent']:
                alerts.append(f"Critical CPU usage: {metric.cpu_usage_percent:.1f}%")
            elif metric.cpu_usage_percent > self.performance_thresholds['cpu_warning_percent']:
                alerts.append(f"High CPU usage: {metric.cpu_usage_percent:.1f}%")
            
            # Notify callbacks if alerts found
            if alerts:
                self._notify_performance_callbacks({
                    'type': 'performance_alert',
                    'operation_id': metric.operation_id,
                    'operation_type': metric.operation_type,
                    'alerts': alerts,
                    'metric': metric.to_dict()
                })
    
    def _notify_performance_callbacks(self, event_data: Dict[str, Any]) -> None:
        """Notify performance event callbacks."""
        for callback in self.performance_callbacks:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Performance callback failed: {e}")
    
    def add_performance_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for performance events."""
        self.performance_callbacks.append(callback)
        logger.info("Added performance event callback")
    
    def set_performance_baseline(self, operation_type: str, baseline_metrics: Dict[str, float]) -> None:
        """Set performance baseline for an operation type."""
        self.performance_baselines[operation_type] = baseline_metrics.copy()
        logger.info(f"Set performance baseline for {operation_type}")
    
    def get_performance_summary(self, hours: float = 24.0) -> Dict[str, Any]:
        """Get performance summary over time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.metrics_lock:
            recent_metrics = [
                m for m in self.performance_metrics 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {
                'total_operations': 0,
                'successful_operations': 0,
                'failed_operations': 0,
                'success_rate_percent': 0.0,
                'average_response_time_ms': 0.0,
                'min_response_time_ms': 0.0,
                'max_response_time_ms': 0.0,
                'throughput_ops_per_sec': 0.0,
                'operation_types': {}
            }
        
        # Calculate summary statistics
        successful_ops = [m for m in recent_metrics if m.success]
        failed_ops = [m for m in recent_metrics if not m.success]
        response_times = [m.response_time_ms for m in recent_metrics]
        
        # Calculate throughput
        if len(recent_metrics) >= 2:
            time_span = (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds()
            throughput = len(recent_metrics) / max(time_span, 1.0)
        else:
            throughput = 0.0
        
        # Operation type breakdown
        operation_types = {}
        for metric in recent_metrics:
            op_type = metric.operation_type
            if op_type not in operation_types:
                operation_types[op_type] = {
                    'count': 0,
                    'successful': 0,
                    'failed': 0,
                    'avg_response_time': 0.0
                }
            
            operation_types[op_type]['count'] += 1
            if metric.success:
                operation_types[op_type]['successful'] += 1
            else:
                operation_types[op_type]['failed'] += 1
        
        # Calculate averages for each operation type
        for op_type, stats in operation_types.items():
            op_metrics = [m for m in recent_metrics if m.operation_type == op_type]
            if op_metrics:
                stats['avg_response_time'] = statistics.mean([m.response_time_ms for m in op_metrics])
        
        return {
            'total_operations': len(recent_metrics),
            'successful_operations': len(successful_ops),
            'failed_operations': len(failed_ops),
            'success_rate_percent': (len(successful_ops) / len(recent_metrics)) * 100,
            'average_response_time_ms': statistics.mean(response_times),
            'min_response_time_ms': min(response_times),
            'max_response_time_ms': max(response_times),
            'throughput_ops_per_sec': throughput,
            'operation_types': operation_types
        }
    
    def export_performance_data(self, hours: float = 24.0) -> Dict[str, Any]:
        """Export performance data for analysis."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.metrics_lock:
            recent_metrics = [
                m for m in self.performance_metrics 
                if m.timestamp >= cutoff_time
            ]
            
            recent_benchmarks = [
                b for b in self.benchmark_results 
                if b.timestamp >= cutoff_time
            ]
        
        return {
            'export_timestamp': datetime.utcnow().isoformat(),
            'period_hours': hours,
            'total_metrics': len(recent_metrics),
            'metrics': [m.to_dict() for m in recent_metrics],
            'benchmarks': [b.to_dict() for b in recent_benchmarks],
            'performance_thresholds': self.performance_thresholds.copy(),
            'operation_counters': dict(self.operation_counters),
            'performance_baselines': self.performance_baselines.copy()
        }


class OperationTracker:
    """Helper class for tracking operation performance."""
    
    def __init__(self, operation_id: str, operation_type: str):
        self.operation_id = operation_id
        self.operation_type = operation_type
        self.success = True
        self.error_message: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    def set_success(self, success: bool, error_message: Optional[str] = None) -> None:
        """Set operation success status."""
        self.success = success
        self.error_message = error_message
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the operation."""
        self.metadata[key] = value
        logger.info(f"Set performance baseline for {operation_type}")
    
    def get_performance_summary(self, hours: float = 1.0) -> Dict[str, Any]:
        """Get performance summary for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.metrics_lock:
            recent_metrics = [
                m for m in self.performance_metrics 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {
                'period_hours': hours,
                'total_operations': 0,
                'average_response_time_ms': 0.0,
                'throughput_ops_per_sec': 0.0,
                'success_rate_percent': 0.0,
                'active_operations': len(self.active_operations)
            }
        
        # Calculate summary statistics
        response_times = [m.response_time_ms for m in recent_metrics]
        successful_ops = [m for m in recent_metrics if m.success]
        
        # Calculate throughput
        if len(recent_metrics) >= 2:
            time_span = (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds()
            throughput = len(recent_metrics) / max(time_span, 1.0)
        else:
            throughput = 0.0
        
        return {
            'period_hours': hours,
            'total_operations': len(recent_metrics),
            'successful_operations': len(successful_ops),
            'failed_operations': len(recent_metrics) - len(successful_ops),
            'average_response_time_ms': statistics.mean(response_times) if response_times else 0.0,
            'median_response_time_ms': statistics.median(response_times) if response_times else 0.0,
            'min_response_time_ms': min(response_times) if response_times else 0.0,
            'max_response_time_ms': max(response_times) if response_times else 0.0,
            'throughput_ops_per_sec': throughput,
            'success_rate_percent': (len(successful_ops) / len(recent_metrics)) * 100 if recent_metrics else 0.0,
            'active_operations': len(self.active_operations),
            'operation_types': list(set(m.operation_type for m in recent_metrics))
        }  
  
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def _get_current_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            process = psutil.Process()
            return process.cpu_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def _cleanup_old_data(self) -> None:
        """Clean up old performance data."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.benchmark_retention_days)
        
        # Clean up old benchmarks
        self.benchmark_results = [
            b for b in self.benchmark_results 
            if b.timestamp >= cutoff_date
        ]
    
    def _check_performance_thresholds(self) -> None:
        """Check performance thresholds and trigger alerts."""
        if not self.performance_metrics:
            return
        
        # Get recent metrics
        recent_metrics = list(self.performance_metrics)[-10:]
        
        for metric in recent_metrics:
            alerts = []
            
            # Check response time
            if metric.response_time_ms > self.performance_thresholds['response_time_critical_ms']:
                alerts.append(f"Critical response time: {metric.response_time_ms:.2f}ms")
            elif metric.response_time_ms > self.performance_thresholds['response_time_warning_ms']:
                alerts.append(f"High response time: {metric.response_time_ms:.2f}ms")
            
            # Check memory usage
            if metric.memory_usage_mb > self.performance_thresholds['memory_critical_mb']:
                alerts.append(f"Critical memory usage: {metric.memory_usage_mb:.2f}MB")
            elif metric.memory_usage_mb > self.performance_thresholds['memory_warning_mb']:
                alerts.append(f"High memory usage: {metric.memory_usage_mb:.2f}MB")
            
            # Check CPU usage
            if metric.cpu_usage_percent > self.performance_thresholds['cpu_critical_percent']:
                alerts.append(f"Critical CPU usage: {metric.cpu_usage_percent:.1f}%")
            elif metric.cpu_usage_percent > self.performance_thresholds['cpu_warning_percent']:
                alerts.append(f"High CPU usage: {metric.cpu_usage_percent:.1f}%")
            
            # Notify callbacks if alerts found
            if alerts:
                self._notify_performance_callbacks({
                    'type': 'performance_alert',
                    'operation_id': metric.operation_id,
                    'operation_type': metric.operation_type,
                    'alerts': alerts,
                    'metric': metric.to_dict()
                })
    
    def add_performance_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for performance events."""
        self.performance_callbacks.append(callback)
        logger.info("Added performance callback")
    
    def _notify_performance_callbacks(self, event_data: Dict[str, Any]) -> None:
        """Notify performance callbacks."""
        for callback in self.performance_callbacks:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Performance callback failed: {e}")
    
    def export_performance_data(self, hours: float = 24.0) -> Dict[str, Any]:
        """Export performance data for analysis."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.metrics_lock:
            recent_metrics = [
                m.to_dict() for m in self.performance_metrics 
                if m.timestamp >= cutoff_time
            ]
        
        return {
            'export_timestamp': datetime.utcnow().isoformat(),
            'period_hours': hours,
            'total_metrics': len(recent_metrics),
            'metrics': recent_metrics,
            'benchmarks': [b.to_dict() for b in self.benchmark_results],
            'performance_thresholds': self.performance_thresholds,
            'operation_counters': dict(self.operation_counters),
            'active_operations': len(self.active_operations)
        }


class OperationTracker:
    """Helper class for tracking individual operations."""
    
    def __init__(self, operation_id: str, operation_type: str):
        self.operation_id = operation_id
        self.operation_type = operation_type
        self.success = True
        self.error_message: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    def set_success(self, success: bool, error_message: Optional[str] = None) -> None:
        """Set operation success status."""
        self.success = success
        self.error_message = error_message
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the operation."""
        self.metadata[key] = value