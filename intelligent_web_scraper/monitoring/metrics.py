"""
Metrics collection system for monitoring dashboard.

This module provides comprehensive metrics collection and aggregation
capabilities for real-time monitoring and performance analysis.
"""

import asyncio
import threading
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System-level metrics."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    process_count: int = 0
    thread_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_available_mb': self.memory_available_mb,
            'disk_usage_percent': self.disk_usage_percent,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_recv': self.network_bytes_recv,
            'process_count': self.process_count,
            'thread_count': self.thread_count
        }


@dataclass
class InstanceMetrics:
    """Metrics for a specific scraper instance."""
    instance_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: str = "unknown"
    requests_processed: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    success_rate: float = 0.0
    error_rate: float = 0.0
    average_response_time: float = 0.0
    current_memory_mb: float = 0.0
    current_cpu_percent: float = 0.0
    throughput: float = 0.0
    quality_score_avg: float = 0.0
    uptime_seconds: float = 0.0
    idle_time_seconds: float = 0.0
    current_task: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'instance_id': self.instance_id,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status,
            'requests_processed': self.requests_processed,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.success_rate,
            'error_rate': self.error_rate,
            'average_response_time': self.average_response_time,
            'current_memory_mb': self.current_memory_mb,
            'current_cpu_percent': self.current_cpu_percent,
            'throughput': self.throughput,
            'quality_score_avg': self.quality_score_avg,
            'uptime_seconds': self.uptime_seconds,
            'idle_time_seconds': self.idle_time_seconds,
            'current_task': self.current_task
        }


class MetricsAggregator:
    """Aggregates metrics over time windows."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_points: deque = deque(maxlen=window_size)
        self.lock = threading.Lock()
    
    def add_data_point(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add a data point."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        with self.lock:
            self.data_points.append((timestamp, value))
    
    def get_average(self, time_window_seconds: Optional[float] = None) -> float:
        """Get average value over time window."""
        with self.lock:
            if not self.data_points:
                return 0.0
            
            if time_window_seconds is None:
                # Average over all data points
                values = [point[1] for point in self.data_points]
                return sum(values) / len(values)
            
            # Average over time window
            cutoff_time = datetime.utcnow() - timedelta(seconds=time_window_seconds)
            recent_values = [
                point[1] for point in self.data_points 
                if point[0] >= cutoff_time
            ]
            
            if not recent_values:
                return 0.0
            
            return sum(recent_values) / len(recent_values)
    
    def get_trend(self, time_window_seconds: float = 300.0) -> str:
        """Get trend direction (increasing, decreasing, stable)."""
        with self.lock:
            if len(self.data_points) < 2:
                return "stable"
            
            cutoff_time = datetime.utcnow() - timedelta(seconds=time_window_seconds)
            recent_points = [
                point for point in self.data_points 
                if point[0] >= cutoff_time
            ]
            
            if len(recent_points) < 2:
                return "stable"
            
            # Simple trend calculation
            first_half = recent_points[:len(recent_points)//2]
            second_half = recent_points[len(recent_points)//2:]
            
            first_avg = sum(point[1] for point in first_half) / len(first_half)
            second_avg = sum(point[1] for point in second_half) / len(second_half)
            
            diff_percent = ((second_avg - first_avg) / max(first_avg, 0.001)) * 100
            
            if diff_percent > 10:
                return "increasing"
            elif diff_percent < -10:
                return "decreasing"
            else:
                return "stable"
    
    def get_recent_values(self, count: int = 10) -> List[float]:
        """Get recent values."""
        with self.lock:
            recent_points = list(self.data_points)[-count:]
            return [point[1] for point in recent_points]


class MetricsCollector:
    """
    Comprehensive metrics collection system.
    
    This class collects system metrics, instance metrics, and provides
    aggregation and analysis capabilities for the monitoring dashboard.
    """
    
    def __init__(self, collection_interval: float = 1.0, history_size: int = 1000):
        self.collection_interval = collection_interval
        self.history_size = history_size
        
        # Metrics storage
        self.system_metrics_history: deque = deque(maxlen=history_size)
        self.instance_metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        
        # Aggregators for key metrics
        self.aggregators = {
            'system_cpu': MetricsAggregator(),
            'system_memory': MetricsAggregator(),
            'overall_throughput': MetricsAggregator(),
            'overall_success_rate': MetricsAggregator(),
            'overall_error_rate': MetricsAggregator()
        }
        
        # Thread safety
        self.metrics_lock = threading.RLock()
        
        # Collection thread
        self.collection_thread: Optional[threading.Thread] = None
        self.collection_active = threading.Event()
        self.shutdown_event = threading.Event()
        
        # Callbacks for real-time updates
        self.update_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Performance tracking
        self.collection_stats = {
            'collections_performed': 0,
            'collection_errors': 0,
            'last_collection_time': None,
            'average_collection_duration': 0.0
        }
        
        logger.info("MetricsCollector initialized")
    
    def start_collection(self) -> None:
        """Start metrics collection in background thread."""
        if self.collection_thread is None or not self.collection_thread.is_alive():
            self.collection_active.set()
            self.collection_thread = threading.Thread(
                target=self._collection_loop,
                name="MetricsCollectionThread",
                daemon=True
            )
            self.collection_thread.start()
            logger.info("Metrics collection started")
    
    def stop_collection(self) -> None:
        """Stop metrics collection."""
        self.collection_active.clear()
        self.shutdown_event.set()
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)
        
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self) -> None:
        """Main collection loop."""
        while self.collection_active.is_set() and not self.shutdown_event.is_set():
            start_time = time.time()
            
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self._store_system_metrics(system_metrics)
                
                # Update aggregators
                self._update_aggregators(system_metrics)
                
                # Notify callbacks
                self._notify_update_callbacks({
                    'type': 'system_metrics',
                    'data': system_metrics.to_dict()
                })
                
                # Update collection stats
                collection_duration = time.time() - start_time
                self._update_collection_stats(collection_duration, success=True)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                self._update_collection_stats(0.0, success=False)
            
            # Sleep for collection interval
            time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / 1024 / 1024
            memory_available_mb = memory.available / 1024 / 1024
            
            # Disk usage (root partition)
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Process information
            process_count = len(psutil.pids())
            
            # Thread count for current process
            try:
                current_process = psutil.Process()
                thread_count = current_process.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                thread_count = 0
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                process_count=process_count,
                thread_count=thread_count
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # Return default metrics
            return SystemMetrics()
    
    def _store_system_metrics(self, metrics: SystemMetrics) -> None:
        """Store system metrics in history."""
        with self.metrics_lock:
            self.system_metrics_history.append(metrics)
    
    def _update_aggregators(self, system_metrics: SystemMetrics) -> None:
        """Update metric aggregators."""
        timestamp = system_metrics.timestamp
        
        self.aggregators['system_cpu'].add_data_point(
            system_metrics.cpu_percent, timestamp
        )
        self.aggregators['system_memory'].add_data_point(
            system_metrics.memory_percent, timestamp
        )
    
    def _update_collection_stats(self, duration: float, success: bool) -> None:
        """Update collection statistics."""
        self.collection_stats['collections_performed'] += 1
        self.collection_stats['last_collection_time'] = datetime.utcnow()
        
        if not success:
            self.collection_stats['collection_errors'] += 1
        
        # Update average duration
        current_avg = self.collection_stats['average_collection_duration']
        count = self.collection_stats['collections_performed']
        new_avg = ((current_avg * (count - 1)) + duration) / count
        self.collection_stats['average_collection_duration'] = new_avg
    
    def collect_instance_metrics(self, instance_stats: Dict[str, Any]) -> InstanceMetrics:
        """Collect metrics for a specific instance."""
        instance_id = instance_stats.get('instance_id', 'unknown')
        
        metrics = InstanceMetrics(
            instance_id=instance_id,
            status=instance_stats.get('status', 'unknown'),
            requests_processed=instance_stats.get('requests_processed', 0),
            successful_requests=instance_stats.get('successful_requests', 0),
            failed_requests=instance_stats.get('failed_requests', 0),
            success_rate=instance_stats.get('success_rate', 0.0),
            error_rate=instance_stats.get('error_rate', 0.0),
            average_response_time=instance_stats.get('average_response_time', 0.0),
            current_memory_mb=instance_stats.get('memory_usage_mb', 0.0),
            current_cpu_percent=instance_stats.get('cpu_usage_percent', 0.0),
            throughput=instance_stats.get('throughput', 0.0),
            quality_score_avg=instance_stats.get('quality_score_avg', 0.0),
            uptime_seconds=instance_stats.get('uptime', 0.0),
            idle_time_seconds=instance_stats.get('idle_time_seconds', 0.0),
            current_task=instance_stats.get('current_task')
        )
        
        # Store in history
        with self.metrics_lock:
            self.instance_metrics_history[instance_id].append(metrics)
        
        # Notify callbacks
        self._notify_update_callbacks({
            'type': 'instance_metrics',
            'instance_id': instance_id,
            'data': metrics.to_dict()
        })
        
        return metrics
    
    def get_system_metrics_summary(self, time_window_seconds: float = 300.0) -> Dict[str, Any]:
        """Get system metrics summary over time window."""
        with self.metrics_lock:
            if not self.system_metrics_history:
                return {
                    'cpu_percent': 0.0,
                    'memory_percent': 0.0,
                    'memory_used_mb': 0.0,
                    'disk_usage_percent': 0.0,
                    'process_count': 0,
                    'thread_count': 0,
                    'trend': 'stable'
                }
            
            # Get recent metrics
            cutoff_time = datetime.utcnow() - timedelta(seconds=time_window_seconds)
            recent_metrics = [
                m for m in self.system_metrics_history 
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                recent_metrics = [self.system_metrics_history[-1]]
            
            # Calculate averages
            cpu_avg = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            memory_avg = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            memory_used_avg = sum(m.memory_used_mb for m in recent_metrics) / len(recent_metrics)
            disk_avg = sum(m.disk_usage_percent for m in recent_metrics) / len(recent_metrics)
            
            # Get latest values for counts
            latest = recent_metrics[-1]
            
            return {
                'cpu_percent': cpu_avg,
                'memory_percent': memory_avg,
                'memory_used_mb': memory_used_avg,
                'disk_usage_percent': disk_avg,
                'process_count': latest.process_count,
                'thread_count': latest.thread_count,
                'cpu_trend': self.aggregators['system_cpu'].get_trend(time_window_seconds),
                'memory_trend': self.aggregators['system_memory'].get_trend(time_window_seconds),
                'data_points': len(recent_metrics)
            }
    
    def get_instance_metrics_summary(self, instance_id: str, time_window_seconds: float = 300.0) -> Dict[str, Any]:
        """Get instance metrics summary."""
        with self.metrics_lock:
            if instance_id not in self.instance_metrics_history:
                return {}
            
            history = self.instance_metrics_history[instance_id]
            if not history:
                return {}
            
            # Get recent metrics
            cutoff_time = datetime.utcnow() - timedelta(seconds=time_window_seconds)
            recent_metrics = [
                m for m in history 
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                recent_metrics = [history[-1]]
            
            # Calculate summary
            latest = recent_metrics[-1]
            
            return {
                'instance_id': instance_id,
                'current_status': latest.status,
                'current_task': latest.current_task,
                'total_requests': latest.requests_processed,
                'success_rate': latest.success_rate,
                'error_rate': latest.error_rate,
                'average_response_time': latest.average_response_time,
                'current_memory_mb': latest.current_memory_mb,
                'current_cpu_percent': latest.current_cpu_percent,
                'throughput': latest.throughput,
                'quality_score_avg': latest.quality_score_avg,
                'uptime_seconds': latest.uptime_seconds,
                'idle_time_seconds': latest.idle_time_seconds,
                'data_points': len(recent_metrics)
            }
    
    def get_overall_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics across all instances."""
        with self.metrics_lock:
            # Aggregate instance metrics
            all_instances = []
            total_requests = 0
            total_successful = 0
            total_failed = 0
            total_memory = 0.0
            total_cpu = 0.0
            active_instances = 0
            
            for instance_id, history in self.instance_metrics_history.items():
                if history:
                    latest = history[-1]
                    all_instances.append(latest)
                    
                    total_requests += latest.requests_processed
                    total_successful += latest.successful_requests
                    total_failed += latest.failed_requests
                    total_memory += latest.current_memory_mb
                    total_cpu += latest.current_cpu_percent
                    
                    if latest.status in ['running', 'idle']:
                        active_instances += 1
            
            # Calculate rates
            overall_success_rate = (total_successful / max(total_requests, 1)) * 100
            overall_error_rate = (total_failed / max(total_requests, 1)) * 100
            overall_throughput = sum(i.throughput for i in all_instances)
            
            # Update aggregators
            self.aggregators['overall_success_rate'].add_data_point(overall_success_rate)
            self.aggregators['overall_error_rate'].add_data_point(overall_error_rate)
            self.aggregators['overall_throughput'].add_data_point(overall_throughput)
            
            return {
                'total_instances': len(all_instances),
                'active_instances': active_instances,
                'total_requests': total_requests,
                'successful_requests': total_successful,
                'failed_requests': total_failed,
                'overall_success_rate': overall_success_rate,
                'overall_error_rate': overall_error_rate,
                'overall_throughput': overall_throughput,
                'total_memory_mb': total_memory,
                'average_cpu_percent': total_cpu / max(len(all_instances), 1),
                'success_rate_trend': self.aggregators['overall_success_rate'].get_trend(),
                'error_rate_trend': self.aggregators['overall_error_rate'].get_trend(),
                'throughput_trend': self.aggregators['overall_throughput'].get_trend()
            }
    
    def get_performance_trends(self, time_window_seconds: float = 3600.0) -> Dict[str, List[float]]:
        """Get performance trends over time window."""
        trends = {}
        
        # System trends
        trends['cpu_usage'] = self.aggregators['system_cpu'].get_recent_values(60)
        trends['memory_usage'] = self.aggregators['system_memory'].get_recent_values(60)
        trends['success_rate'] = self.aggregators['overall_success_rate'].get_recent_values(60)
        trends['error_rate'] = self.aggregators['overall_error_rate'].get_recent_values(60)
        trends['throughput'] = self.aggregators['overall_throughput'].get_recent_values(60)
        
        return trends
    
    def add_update_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for real-time metric updates."""
        self.update_callbacks.append(callback)
        logger.info("Added metrics update callback")
    
    def _notify_update_callbacks(self, update_data: Dict[str, Any]) -> None:
        """Notify all update callbacks."""
        for callback in self.update_callbacks:
            try:
                callback(update_data)
            except Exception as e:
                logger.error(f"Metrics update callback failed: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get metrics collection statistics."""
        return self.collection_stats.copy()
    
    def reset_metrics(self) -> None:
        """Reset all metrics (for testing)."""
        with self.metrics_lock:
            self.system_metrics_history.clear()
            self.instance_metrics_history.clear()
            
            for aggregator in self.aggregators.values():
                aggregator.data_points.clear()
            
            self.collection_stats = {
                'collections_performed': 0,
                'collection_errors': 0,
                'last_collection_time': None,
                'average_collection_duration': 0.0
            }
            
            logger.info("All metrics reset")