"""
Enhanced Concurrency Manager for coordinating thread pools and async operations.

This module implements advanced concurrency patterns with performance optimizations,
intelligent resource allocation, load balancing, and caching strategies for the
Intelligent Web Scraper system.
"""

import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from contextlib import contextmanager
from collections import deque, defaultdict

from ..config import IntelligentScrapingConfig


logger = logging.getLogger(__name__)


class OperationStatus(Enum):
    """Status enumeration for concurrent operations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ResourceType(Enum):
    """Types of resources managed by the concurrency system."""
    THREAD = "thread"
    ASYNC_TASK = "async_task"
    MEMORY = "memory"
    NETWORK = "network"
    FILE_HANDLE = "file_handle"


@dataclass
class ConcurrentOperation:
    """Represents a concurrent operation being managed."""
    operation_id: str
    operation_type: str
    status: OperationStatus = OperationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    timeout_seconds: float = 300.0
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_duration(self) -> Optional[float]:
        """Get operation duration in seconds."""
        if self.started_at is None:
            return None
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
    
    def is_expired(self) -> bool:
        """Check if operation has exceeded timeout."""
        if self.started_at is None:
            return False
        return (datetime.utcnow() - self.started_at).total_seconds() > self.timeout_seconds


@dataclass
class ResourceUsage:
    """Tracks resource usage for a specific resource type."""
    resource_type: ResourceType
    current_usage: int = 0
    max_usage: int = 0
    peak_usage: int = 0
    total_allocated: int = 0
    total_released: int = 0
    allocation_history: List[datetime] = field(default_factory=list)
    
    def allocate(self, count: int = 1) -> bool:
        """Allocate resources if available."""
        if self.current_usage + count > self.max_usage:
            return False
        
        self.current_usage += count
        self.total_allocated += count
        self.peak_usage = max(self.peak_usage, self.current_usage)
        self.allocation_history.append(datetime.utcnow())
        
        # Keep only last 1000 allocations
        if len(self.allocation_history) > 1000:
            self.allocation_history = self.allocation_history[-1000:]
        
        return True
    
    def release(self, count: int = 1) -> None:
        """Release allocated resources."""
        self.current_usage = max(0, self.current_usage - count)
        self.total_released += count
    
    def get_utilization_rate(self) -> float:
        """Get current utilization rate as percentage."""
        if self.max_usage == 0:
            return 0.0
        return (self.current_usage / self.max_usage) * 100.0


class DynamicThreadPool:
    """Dynamic thread pool that scales based on workload."""
    
    def __init__(self, min_workers: int, max_workers: int, scale_factor: float = 1.5):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_factor = scale_factor
        self.current_workers = min_workers
        self.pending_tasks = 0
        self.completed_tasks = 0
        self.lock = threading.Lock()
        
        # Create initial thread pool
        self.thread_pool = ThreadPoolExecutor(
            max_workers=min_workers,
            thread_name_prefix="DynamicScraper"
        )
        
        # Scaling metrics
        self.last_scale_time = time.time()
        self.scale_cooldown = 30.0  # seconds
        
    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit task with dynamic scaling."""
        with self.lock:
            self.pending_tasks += 1
            
            # Check if we need to scale up
            if self._should_scale_up():
                self._scale_up()
        
        future = self.thread_pool.submit(self._wrapped_execution, fn, *args, **kwargs)
        return future
    
    def _wrapped_execution(self, fn: Callable, *args, **kwargs):
        """Wrapped execution to track completion."""
        try:
            result = fn(*args, **kwargs)
            with self.lock:
                self.completed_tasks += 1
                self.pending_tasks = max(0, self.pending_tasks - 1)
            return result
        except Exception as e:
            with self.lock:
                self.pending_tasks = max(0, self.pending_tasks - 1)
            raise
    
    def _should_scale_up(self) -> bool:
        """Determine if we should scale up the thread pool."""
        if self.current_workers >= self.max_workers:
            return False
        
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        # Scale up if pending tasks exceed current workers
        return self.pending_tasks > self.current_workers * 2
    
    def _scale_up(self) -> None:
        """Scale up the thread pool."""
        new_workers = min(
            self.max_workers,
            int(self.current_workers * self.scale_factor)
        )
        
        if new_workers > self.current_workers:
            logger.info(f"Scaling thread pool up from {self.current_workers} to {new_workers}")
            
            # Create new thread pool with more workers
            old_pool = self.thread_pool
            self.thread_pool = ThreadPoolExecutor(
                max_workers=new_workers,
                thread_name_prefix="DynamicScraper"
            )
            
            self.current_workers = new_workers
            self.last_scale_time = time.time()
            
            # Schedule old pool shutdown
            threading.Thread(
                target=lambda: old_pool.shutdown(wait=True),
                daemon=True
            ).start()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread pool statistics."""
        with self.lock:
            return {
                'current_workers': self.current_workers,
                'min_workers': self.min_workers,
                'max_workers': self.max_workers,
                'pending_tasks': self.pending_tasks,
                'completed_tasks': self.completed_tasks,
                'utilization_rate': (self.pending_tasks / self.current_workers * 100) if self.current_workers > 0 else 0
            }


class LRUCache:
    """Least Recently Used cache for operation results."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order: deque = deque()
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache[key] = value
                self.access_order.remove(key)
                self.access_order.append(key)
            else:
                # Add new
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    lru_key = self.access_order.popleft()
                    del self.cache[lru_key]
                
                self.cache[key] = value
                self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization_rate': (len(self.cache) / self.max_size * 100) if self.max_size > 0 else 0
            }


class ConcurrencyManager:
    """
    Enhanced concurrency manager with performance optimizations and intelligent resource allocation.
    
    This class demonstrates sophisticated patterns for managing concurrent operations
    with advanced performance optimizations, intelligent load balancing, caching strategies,
    and dynamic resource allocation for optimal throughput and resource utilization.
    """
    
    def __init__(self, config: IntelligentScrapingConfig):
        self.config = config
        
        # Enhanced thread pool management with dynamic sizing
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.max_workers,
            thread_name_prefix="IntelligentScraper"
        )
        self.dynamic_thread_pool = DynamicThreadPool(
            min_workers=max(1, config.max_workers // 4),
            max_workers=config.max_workers * 2,
            scale_factor=1.5
        )
        
        # Async task management
        self.async_semaphore = asyncio.Semaphore(config.max_async_tasks)
        self.async_tasks: Dict[str, asyncio.Task] = {}
        
        # Operation tracking with performance optimization
        self.operations: Dict[str, ConcurrentOperation] = {}
        self.operation_lock = threading.RLock()
        self.operation_cache = LRUCache(max_size=10000)
        
        # Enhanced resource management
        self.resources: Dict[ResourceType, ResourceUsage] = {
            ResourceType.THREAD: ResourceUsage(ResourceType.THREAD, max_usage=config.max_workers * 2),
            ResourceType.ASYNC_TASK: ResourceUsage(ResourceType.ASYNC_TASK, max_usage=config.max_async_tasks * 2),
            ResourceType.MEMORY: ResourceUsage(ResourceType.MEMORY, max_usage=2048),  # MB
            ResourceType.NETWORK: ResourceUsage(ResourceType.NETWORK, max_usage=config.max_concurrent_requests * 2),
            ResourceType.FILE_HANDLE: ResourceUsage(ResourceType.FILE_HANDLE, max_usage=200)
        }
        self.resource_lock = threading.Lock()
        
        # Monitoring and cleanup with enhanced metrics
        self.monitoring_enabled = config.enable_monitoring
        self.cleanup_interval = 30.0  # Reduced for better performance
        self.cleanup_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Enhanced performance metrics
        self.metrics = {
            'operations_started': 0,
            'operations_completed': 0,
            'operations_failed': 0,
            'operations_cancelled': 0,
            'operations_timeout': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'peak_concurrent_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'resource_allocation_time': 0.0,
            'performance_optimizations': 0
        }
        self.metrics_lock = threading.Lock()
        self.performance_history = deque(maxlen=1000)
        
        # Start enhanced background services
        self._start_cleanup_thread()
        
        logger.info(f"Initialized Enhanced ConcurrencyManager with dynamic scaling and caching")
    
    def submit_sync_operation_enhanced(
        self, 
        func: Callable, 
        *args, 
        operation_type: str = "sync_operation",
        timeout_seconds: float = 300.0,
        priority: int = 0,
        enable_caching: bool = True,
        **kwargs
    ) -> str:
        """
        Submit a synchronous operation with enhanced performance optimizations.
        """
        operation_id = str(uuid.uuid4())
        
        # Check cache first if enabled
        if enable_caching:
            cache_key = self._generate_cache_key(func, args, kwargs)
            cached_result = self.operation_cache.get(cache_key)
            if cached_result is not None:
                with self.metrics_lock:
                    self.metrics['cache_hits'] += 1
                
                # Create cached operation record
                operation = ConcurrentOperation(
                    operation_id=operation_id,
                    operation_type=operation_type,
                    status=OperationStatus.COMPLETED,
                    result=cached_result,
                    timeout_seconds=timeout_seconds,
                    priority=priority
                )
                operation.completed_at = datetime.utcnow()
                
                with self.operation_lock:
                    self.operations[operation_id] = operation
                
                logger.debug(f"Returned cached result for operation {operation_id}")
                return operation_id
            else:
                with self.metrics_lock:
                    self.metrics['cache_misses'] += 1
        
        # Create operation record
        operation = ConcurrentOperation(
            operation_id=operation_id,
            operation_type=operation_type,
            timeout_seconds=timeout_seconds,
            priority=priority,
            metadata={'args': args, 'kwargs': kwargs, 'enable_caching': enable_caching}
        )
        
        with self.operation_lock:
            self.operations[operation_id] = operation
        
        # Allocate thread resource
        if not self._allocate_resource(ResourceType.THREAD):
            operation.status = OperationStatus.FAILED
            operation.error = "No thread resources available"
            return operation_id
        
        try:
            # Submit to dynamic thread pool for better performance
            future = self.dynamic_thread_pool.submit(
                self._execute_sync_operation_enhanced, 
                operation, func, enable_caching, *args, **kwargs
            )
            
            operation.metadata['future'] = future
            
            # Update metrics
            with self.metrics_lock:
                self.metrics['operations_started'] += 1
                current_ops = len([op for op in self.operations.values() if op.status == OperationStatus.RUNNING])
                self.metrics['peak_concurrent_operations'] = max(
                    self.metrics['peak_concurrent_operations'], 
                    current_ops
                )
            
            logger.debug(f"Submitted enhanced sync operation {operation_id}")
            
        except Exception as e:
            operation.status = OperationStatus.FAILED
            operation.error = str(e)
            self._release_resource(ResourceType.THREAD)
            logger.error(f"Failed to submit enhanced sync operation {operation_id}: {e}")
        
        return operation_id
    
    def _execute_sync_operation_enhanced(
        self, 
        operation: ConcurrentOperation, 
        func: Callable, 
        enable_caching: bool,
        *args, 
        **kwargs
    ) -> Any:
        """Execute a synchronous operation with enhanced performance tracking."""
        operation.status = OperationStatus.RUNNING
        operation.started_at = datetime.utcnow()
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Cache result if enabled
            if enable_caching:
                cache_key = self._generate_cache_key(func, args, kwargs)
                self.operation_cache.put(cache_key, result)
            
            # Update operation
            operation.status = OperationStatus.COMPLETED
            operation.completed_at = datetime.utcnow()
            operation.result = result
            
            # Update metrics and performance history
            duration = operation.get_duration()
            with self.metrics_lock:
                self.metrics['operations_completed'] += 1
                if duration:
                    self.metrics['total_processing_time'] += duration
                    self.metrics['average_processing_time'] = (
                        self.metrics['total_processing_time'] / self.metrics['operations_completed']
                    )
                    
                    # Add to performance history
                    self.performance_history.append({
                        'operation_id': operation.operation_id,
                        'operation_type': operation.operation_type,
                        'duration': duration,
                        'success': True,
                        'timestamp': datetime.utcnow()
                    })
            
            logger.debug(f"Completed enhanced sync operation {operation.operation_id} in {duration:.2f}s")
            return result
            
        except Exception as e:
            operation.status = OperationStatus.FAILED
            operation.completed_at = datetime.utcnow()
            operation.error = str(e)
            
            # Update metrics
            duration = operation.get_duration()
            with self.metrics_lock:
                self.metrics['operations_failed'] += 1
                if duration:
                    self.performance_history.append({
                        'operation_id': operation.operation_id,
                        'operation_type': operation.operation_type,
                        'duration': duration,
                        'success': False,
                        'error': str(e),
                        'timestamp': datetime.utcnow()
                    })
            
            logger.error(f"Enhanced sync operation {operation.operation_id} failed: {e}")
            raise
            
        finally:
            # Release thread resource
            self._release_resource(ResourceType.THREAD)
    
    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function call."""
        func_name = getattr(func, '__name__', str(func))
        args_str = str(args) if args else ""
        kwargs_str = str(sorted(kwargs.items())) if kwargs else ""
        
        key_data = f"{func_name}_{args_str}_{kwargs_str}"
        return f"cache_{hash(key_data)}"
    
    def get_enhanced_performance_metrics(self) -> Dict[str, Any]:
        """Get enhanced performance metrics including optimization data."""
        base_metrics = self.get_performance_metrics()
        
        # Add enhanced metrics
        enhanced_metrics = {
            'cache_performance': {
                'hits': self.metrics.get('cache_hits', 0),
                'misses': self.metrics.get('cache_misses', 0),
                'hit_rate': self._calculate_cache_hit_rate(),
                'cache_stats': self.operation_cache.get_stats()
            },
            'dynamic_pool_stats': self.dynamic_thread_pool.get_stats(),
            'recent_performance': self._get_recent_performance_stats()
        }
        
        return {**base_metrics, 'enhanced_metrics': enhanced_metrics}
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        hits = self.metrics.get('cache_hits', 0)
        misses = self.metrics.get('cache_misses', 0)
        total = hits + misses
        
        if total == 0:
            return 0.0
        
        return (hits / total) * 100.0
    
    def _get_recent_performance_stats(self) -> Dict[str, Any]:
        """Get recent performance statistics from history."""
        if not self.performance_history:
            return {}
        
        recent_window = 300.0  # 5 minutes
        current_time = datetime.utcnow()
        
        recent_operations = [
            op for op in self.performance_history
            if (current_time - op['timestamp']).total_seconds() <= recent_window
        ]
        
        if not recent_operations:
            return {}
        
        successful_ops = [op for op in recent_operations if op['success']]
        failed_ops = [op for op in recent_operations if not op['success']]
        
        return {
            'total_operations': len(recent_operations),
            'successful_operations': len(successful_ops),
            'failed_operations': len(failed_ops),
            'success_rate': (len(successful_ops) / len(recent_operations)) * 100,
            'average_duration': sum(op['duration'] for op in recent_operations) / len(recent_operations),
            'operations_per_second': len(recent_operations) / recent_window
        }
    
    # Legacy and base methods
    def submit_sync_operation(self, func: Callable, *args, **kwargs) -> str:
        """Legacy method - redirects to enhanced version."""
        return self.submit_sync_operation_enhanced(func, *args, **kwargs)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the concurrency manager."""
        with self.metrics_lock:
            metrics = self.metrics.copy()
        
        with self.operation_lock:
            active_operations = len([
                op for op in self.operations.values() 
                if op.status == OperationStatus.RUNNING
            ])
            total_operations = len(self.operations)
        
        metrics.update({
            'active_operations': active_operations,
            'total_operations': total_operations,
            'thread_pool_active': getattr(self.thread_pool, '_threads', 0),
            'async_tasks_active': len(self.async_tasks)
        })
        
        return metrics
    
    def _allocate_resource(self, resource_type: ResourceType, count: int = 1) -> bool:
        """Allocate resources of the specified type."""
        with self.resource_lock:
            resource = self.resources.get(resource_type)
            if resource:
                return resource.allocate(count)
            return False
    
    def _release_resource(self, resource_type: ResourceType, count: int = 1) -> None:
        """Release resources of the specified type."""
        with self.resource_lock:
            resource = self.resources.get(resource_type)
            if resource:
                resource.release(count)
    
    def _start_cleanup_thread(self) -> None:
        """Start the cleanup thread for expired operations."""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                name="ConcurrencyManagerCleanup",
                daemon=True
            )
            self.cleanup_thread.start()
            logger.debug("Started concurrency manager cleanup thread")
    
    def _cleanup_loop(self) -> None:
        """Main cleanup loop for expired operations."""
        while not self.shutdown_event.is_set():
            try:
                self._cleanup_expired_operations()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(5.0)
    
    def _cleanup_expired_operations(self) -> None:
        """Clean up expired and completed operations."""
        current_time = datetime.utcnow()
        cleanup_threshold = current_time - timedelta(hours=1)  # Keep operations for 1 hour
        
        with self.operation_lock:
            expired_operations = []
            
            for operation_id, operation in list(self.operations.items()):
                # Remove old completed operations
                if (operation.status in [OperationStatus.COMPLETED, OperationStatus.FAILED, 
                                       OperationStatus.CANCELLED, OperationStatus.TIMEOUT] and
                    operation.completed_at and operation.completed_at < cleanup_threshold):
                    expired_operations.append(operation_id)
            
            # Remove expired operations
            for operation_id in expired_operations:
                del self.operations[operation_id]
                if operation_id in self.async_tasks:
                    del self.async_tasks[operation_id]
            
            if expired_operations:
                logger.debug(f"Cleaned up {len(expired_operations)} expired operations")
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Shutdown the enhanced concurrency manager and clean up resources."""
        logger.info("Shutting down Enhanced ConcurrencyManager")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Cancel all pending async tasks
        for task in list(self.async_tasks.values()):
            if isinstance(task, asyncio.Task) and not task.done():
                task.cancel()
        
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=wait, timeout=timeout)
        if hasattr(self.dynamic_thread_pool, 'thread_pool'):
            self.dynamic_thread_pool.thread_pool.shutdown(wait=wait, timeout=timeout)
        
        # Wait for cleanup thread
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)
        
        logger.info("Enhanced ConcurrencyManager shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()