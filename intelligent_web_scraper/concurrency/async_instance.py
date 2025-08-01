"""
Async Scraper Instance with advanced async coordination and resource management.

This module provides an enhanced async-first scraper instance that demonstrates
advanced asyncio patterns, resource coordination, and performance optimization
for high-concurrency scraping operations.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Set, Callable, Awaitable
from datetime import datetime, timedelta
from collections import deque
from contextlib import asynccontextmanager
import logging
import weakref

from ..agents.instance_manager import (
    ScraperInstance, 
    InstanceStatus, 
    InstanceMetrics,
    HealthStatus,
    ScrapingTask,
    ScrapingResult
)
from ..config import IntelligentScrapingConfig


logger = logging.getLogger(__name__)


class AsyncResourceManager:
    """Manages async resources with proper cleanup and monitoring."""
    
    def __init__(self, max_resources: int = 100):
        self.max_resources = max_resources
        self.active_resources: Set[str] = set()
        self.resource_semaphore = asyncio.Semaphore(max_resources)
        self.resource_lock = asyncio.Lock()
        self.allocation_history: deque = deque(maxlen=1000)
        
    async def allocate_resource(self, resource_id: str) -> bool:
        """Allocate a resource with async coordination."""
        # Check if we can acquire without blocking
        if self.resource_semaphore.locked() and self.resource_semaphore._value == 0:
            return False
        
        # Try to acquire semaphore
        try:
            await asyncio.wait_for(self.resource_semaphore.acquire(), timeout=0.001)
        except asyncio.TimeoutError:
            # Semaphore is at capacity
            return False
        
        try:
            async with self.resource_lock:
                if resource_id in self.active_resources:
                    self.resource_semaphore.release()
                    return False
                
                self.active_resources.add(resource_id)
                self.allocation_history.append({
                    'resource_id': resource_id,
                    'action': 'allocate',
                    'timestamp': datetime.utcnow()
                })
                
                logger.debug(f"Allocated resource {resource_id}")
                return True
        except Exception:
            # Release semaphore on error
            self.resource_semaphore.release()
            raise
    
    async def release_resource(self, resource_id: str) -> bool:
        """Release a resource with async coordination."""
        async with self.resource_lock:
            if resource_id not in self.active_resources:
                return False
            
            self.active_resources.remove(resource_id)
            self.allocation_history.append({
                'resource_id': resource_id,
                'action': 'release',
                'timestamp': datetime.utcnow()
            })
            
            # Release semaphore permit
            self.resource_semaphore.release()
            
            logger.debug(f"Released resource {resource_id}")
            return True
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource utilization statistics."""
        return {
            'active_resources': len(self.active_resources),
            'max_resources': self.max_resources,
            'utilization_rate': (len(self.active_resources) / self.max_resources) * 100,
            'available_permits': self.resource_semaphore._value,
            'allocation_history_size': len(self.allocation_history)
        }


class AsyncTaskCoordinator:
    """Coordinates async tasks with proper lifecycle management."""
    
    def __init__(self):
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, Any] = {}
        self.task_lock = asyncio.Lock()
        self.completion_events: Dict[str, asyncio.Event] = {}
        
    async def submit_task(
        self, 
        task_id: str, 
        coro: Awaitable, 
        timeout: Optional[float] = None
    ) -> str:
        """Submit an async task for execution."""
        async with self.task_lock:
            if task_id in self.active_tasks:
                raise ValueError(f"Task {task_id} already exists")
            
            # Create completion event
            self.completion_events[task_id] = asyncio.Event()
            
            # Create and start task
            task = asyncio.create_task(self._execute_task(task_id, coro, timeout))
            self.active_tasks[task_id] = task
            
            logger.debug(f"Submitted async task {task_id}")
            return task_id
    
    async def _execute_task(
        self, 
        task_id: str, 
        coro: Awaitable, 
        timeout: Optional[float]
    ) -> Any:
        """Execute task with proper error handling and cleanup."""
        try:
            if timeout:
                result = await asyncio.wait_for(coro, timeout=timeout)
            else:
                result = await coro
            
            # Store result
            async with self.task_lock:
                self.task_results[task_id] = {'success': True, 'result': result, 'error': None}
            
            return result
            
        except asyncio.TimeoutError as e:
            async with self.task_lock:
                self.task_results[task_id] = {'success': False, 'result': None, 'error': 'timeout'}
            raise
            
        except Exception as e:
            async with self.task_lock:
                self.task_results[task_id] = {'success': False, 'result': None, 'error': str(e)}
            raise
            
        finally:
            # Clean up and signal completion
            async with self.task_lock:
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
                
                if task_id in self.completion_events:
                    self.completion_events[task_id].set()
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for a specific task to complete."""
        if task_id not in self.completion_events:
            raise ValueError(f"Task {task_id} not found")
        
        try:
            await asyncio.wait_for(self.completion_events[task_id].wait(), timeout=timeout)
            
            # Get result
            async with self.task_lock:
                result_info = self.task_results.get(task_id)
                if result_info and result_info['success']:
                    return result_info['result']
                elif result_info:
                    raise RuntimeError(f"Task failed: {result_info['error']}")
                else:
                    raise RuntimeError("Task result not available")
                    
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Task {task_id} did not complete within timeout")
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        async with self.task_lock:
            task = self.active_tasks.get(task_id)
            if task and not task.done():
                task.cancel()
                return True
            return False
    
    async def wait_for_all_tasks(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for all active tasks to complete."""
        async with self.task_lock:
            active_task_ids = list(self.active_tasks.keys())
        
        results = {}
        for task_id in active_task_ids:
            try:
                result = await self.wait_for_task(task_id, timeout=timeout)
                results[task_id] = {'success': True, 'result': result}
            except Exception as e:
                results[task_id] = {'success': False, 'error': str(e)}
        
        return results
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get task coordination statistics."""
        return {
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.task_results),
            'pending_completions': len(self.completion_events)
        }


class AsyncPerformanceMonitor:
    """Monitors async performance with real-time metrics."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_history: deque = deque(maxlen=window_size)
        self.current_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'operations_per_second': 0.0,
            'error_rate': 0.0
        }
        self.metrics_lock = asyncio.Lock()
        self.last_calculation = datetime.utcnow()
    
    async def record_operation(
        self, 
        operation_type: str, 
        processing_time: float, 
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an operation for performance monitoring."""
        async with self.metrics_lock:
            record = {
                'operation_type': operation_type,
                'processing_time': processing_time,
                'success': success,
                'timestamp': datetime.utcnow(),
                'metadata': metadata or {}
            }
            
            self.metrics_history.append(record)
            await self._update_current_metrics()
    
    async def _update_current_metrics(self) -> None:
        """Update current performance metrics."""
        if not self.metrics_history:
            return
        
        # Calculate metrics from recent history
        recent_window = list(self.metrics_history)
        total_ops = len(recent_window)
        successful_ops = sum(1 for r in recent_window if r['success'])
        failed_ops = total_ops - successful_ops
        total_time = sum(r['processing_time'] for r in recent_window)
        
        # Calculate time-based metrics
        now = datetime.utcnow()
        time_window = 60.0  # 1 minute window
        recent_ops = [
            r for r in recent_window 
            if (now - r['timestamp']).total_seconds() <= time_window
        ]
        
        ops_per_second = len(recent_ops) / time_window if recent_ops else 0.0
        
        # Update current metrics
        self.current_metrics.update({
            'total_operations': total_ops,
            'successful_operations': successful_ops,
            'failed_operations': failed_ops,
            'total_processing_time': total_time,
            'average_processing_time': total_time / total_ops if total_ops > 0 else 0.0,
            'operations_per_second': ops_per_second,
            'error_rate': (failed_ops / total_ops * 100) if total_ops > 0 else 0.0
        })
        
        self.last_calculation = now
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        async with self.metrics_lock:
            # Ensure metrics are up to date
            if (datetime.utcnow() - self.last_calculation).total_seconds() > 5.0:
                await self._update_current_metrics()
            
            return self.current_metrics.copy()
    
    async def get_performance_trends(self, time_window: float = 300.0) -> Dict[str, List[float]]:
        """Get performance trends over time."""
        async with self.metrics_lock:
            now = datetime.utcnow()
            recent_records = [
                r for r in self.metrics_history
                if (now - r['timestamp']).total_seconds() <= time_window
            ]
            
            # Group by time buckets (30-second intervals)
            bucket_size = 30.0
            buckets = {}
            
            for record in recent_records:
                bucket_time = int((now - record['timestamp']).total_seconds() / bucket_size)
                if bucket_time not in buckets:
                    buckets[bucket_time] = []
                buckets[bucket_time].append(record)
            
            # Calculate trends
            processing_times = []
            success_rates = []
            operation_counts = []
            
            for bucket_records in buckets.values():
                if bucket_records:
                    avg_time = sum(r['processing_time'] for r in bucket_records) / len(bucket_records)
                    success_rate = sum(1 for r in bucket_records if r['success']) / len(bucket_records) * 100
                    
                    processing_times.append(avg_time)
                    success_rates.append(success_rate)
                    operation_counts.append(len(bucket_records))
            
            return {
                'processing_times': processing_times,
                'success_rates': success_rates,
                'operation_counts': operation_counts
            }


class AsyncScraperInstance(ScraperInstance):
    """
    Enhanced async-first scraper instance with advanced coordination.
    
    This class extends the base ScraperInstance with sophisticated async patterns:
    - Advanced async resource management
    - Task coordination and lifecycle management
    - Real-time performance monitoring
    - Async health checking and recovery
    - Proper async cleanup and shutdown
    """
    
    def __init__(self, instance_id: str, config: IntelligentScrapingConfig):
        # Initialize parent class
        super().__init__(instance_id, config)
        
        # Async-specific components
        self.resource_manager = AsyncResourceManager(max_resources=50)
        self.task_coordinator = AsyncTaskCoordinator()
        self.performance_monitor = AsyncPerformanceMonitor()
        
        # Async synchronization primitives
        self.async_lock = asyncio.Lock()
        self.status_condition = asyncio.Condition(self.async_lock)
        self.health_check_event = asyncio.Event()
        
        # Task management
        self.background_tasks: Set[asyncio.Task] = set()
        self.shutdown_event = asyncio.Event()
        
        # Performance optimization
        self.operation_cache: Dict[str, Any] = {}
        self.cache_lock = asyncio.Lock()
        self.cache_ttl = 300.0  # 5 minutes
        
        # Connection pooling
        self.connection_pool = None  # Would be initialized with actual HTTP client
        self.connection_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # Start background monitoring
        self._start_background_tasks()
        
        logger.info(f"Initialized AsyncScraperInstance {instance_id}")
    
    def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks."""
        # Health monitoring task
        health_task = asyncio.create_task(self._health_monitoring_loop())
        self.background_tasks.add(health_task)
        health_task.add_done_callback(self.background_tasks.discard)
        
        # Performance monitoring task
        perf_task = asyncio.create_task(self._performance_monitoring_loop())
        self.background_tasks.add(perf_task)
        perf_task.add_done_callback(self.background_tasks.discard)
        
        # Cache cleanup task
        cache_task = asyncio.create_task(self._cache_cleanup_loop())
        self.background_tasks.add(cache_task)
        cache_task.add_done_callback(self.background_tasks.discard)
    
    async def execute_task_async_enhanced(self, task: ScrapingTask) -> ScrapingResult:
        """Execute task with enhanced async coordination and monitoring."""
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Allocate resources
        resource_id = f"task_{operation_id}"
        if not await self.resource_manager.allocate_resource(resource_id):
            return ScrapingResult(
                task_id=task.task_id,
                instance_id=self.instance_id,
                success=False,
                error="Could not allocate resources"
            )
        
        try:
            async with self.async_lock:
                if self.current_task is not None:
                    raise RuntimeError(f"Instance {self.instance_id} is already executing a task")
                
                self.current_task = task
                await self._set_status_async(InstanceStatus.RUNNING)
                task.started_at = datetime.utcnow()
            
            # Execute with coordination
            result_data = await self._execute_with_coordination(task, operation_id)
            
            processing_time = time.time() - start_time
            quality_score = result_data.get('quality_score', 75.0)
            
            # Update metrics
            await self.performance_monitor.record_operation(
                operation_type='scraping_task',
                processing_time=processing_time,
                success=True,
                metadata={
                    'task_id': task.task_id,
                    'quality_score': quality_score,
                    'items_extracted': len(result_data.get('items', []))
                }
            )
            
            # Create result
            result = ScrapingResult(
                task_id=task.task_id,
                instance_id=self.instance_id,
                success=True,
                data=result_data,
                processing_time=processing_time,
                quality_score=quality_score,
                metadata={
                    'operation_id': operation_id,
                    'resource_stats': self.resource_manager.get_resource_stats()
                }
            )
            
            logger.info(f"AsyncInstance {self.instance_id} completed task {task.task_id} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Record error metrics
            await self.performance_monitor.record_operation(
                operation_type='scraping_task',
                processing_time=processing_time,
                success=False,
                metadata={'task_id': task.task_id, 'error': str(e)}
            )
            
            # Create error result
            result = ScrapingResult(
                task_id=task.task_id,
                instance_id=self.instance_id,
                success=False,
                error=str(e),
                processing_time=processing_time
            )
            
            logger.error(f"AsyncInstance {self.instance_id} failed task {task.task_id}: {e}")
            return result
            
        finally:
            # Clean up
            async with self.async_lock:
                task.completed_at = datetime.utcnow()
                self.task_history.append(task)
                self.current_task = None
                await self._set_status_async(InstanceStatus.IDLE)
            
            # Release resources
            await self.resource_manager.release_resource(resource_id)
    
    async def _execute_with_coordination(self, task: ScrapingTask, operation_id: str) -> Dict[str, Any]:
        """Execute task with full async coordination."""
        # Submit to task coordinator
        task_id = await self.task_coordinator.submit_task(
            f"scraping_{operation_id}",
            self._perform_scraping_work(task),
            timeout=task.timeout_seconds
        )
        
        # Wait for completion
        result = await self.task_coordinator.wait_for_task(task_id)
        return result
    
    async def _perform_scraping_work(self, task: ScrapingTask) -> Dict[str, Any]:
        """Perform the actual scraping work with async patterns."""
        # Check cache first
        cache_key = self._generate_cache_key(task)
        cached_result = await self._get_cached_result(cache_key)
        if cached_result:
            logger.debug(f"Using cached result for task {task.task_id}")
            return cached_result
        
        # Simulate async scraping work
        async with self.connection_semaphore:
            # Simulate network delay
            await asyncio.sleep(0.5 + (task.priority * 0.1))
            
            # Generate mock result
            result = {
                'items': [
                    {'title': f'Async Item {i}', 'content': f'Async content for item {i}'}
                    for i in range(1, 4)
                ],
                'total_found': 3,
                'quality_score': 85.0 + (task.priority * 2),
                'strategy_used': 'async_simulation',
                'processing_metadata': {
                    'async_coordination': True,
                    'resource_utilization': self.resource_manager.get_resource_stats()
                }
            }
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            return result
    
    async def _set_status_async(self, status: InstanceStatus) -> None:
        """Set instance status with async coordination."""
        async with self.status_condition:
            old_status = self.status
            self.status = status
            self.update_activity()
            
            # Notify waiting coroutines
            self.status_condition.notify_all()
            
            logger.debug(f"AsyncInstance {self.instance_id} status changed: {old_status} -> {status}")
    
    async def wait_for_status(self, target_status: InstanceStatus, timeout: Optional[float] = None) -> bool:
        """Wait for instance to reach a specific status."""
        async with self.status_condition:
            try:
                await asyncio.wait_for(
                    self.status_condition.wait_for(lambda: self.status == target_status),
                    timeout=timeout
                )
                return True
            except asyncio.TimeoutError:
                return False
    
    def _generate_cache_key(self, task: ScrapingTask) -> str:
        """Generate cache key for task result."""
        # Simple cache key based on task input
        key_data = f"{task.input_data.get('url', '')}-{task.priority}-{task.operation_type if hasattr(task, 'operation_type') else 'default'}"
        return f"cache_{hash(key_data)}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired."""
        async with self.cache_lock:
            cached_entry = self.operation_cache.get(cache_key)
            if cached_entry:
                # Check if expired
                if (datetime.utcnow() - cached_entry['timestamp']).total_seconds() < self.cache_ttl:
                    return cached_entry['result']
                else:
                    # Remove expired entry
                    del self.operation_cache[cache_key]
            
            return None
    
    async def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache result for future use."""
        async with self.cache_lock:
            self.operation_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.utcnow()
            }
            
            # Limit cache size
            if len(self.operation_cache) > 1000:
                # Remove oldest entries
                sorted_entries = sorted(
                    self.operation_cache.items(),
                    key=lambda x: x[1]['timestamp']
                )
                for key, _ in sorted_entries[:100]:  # Remove oldest 100
                    del self.operation_cache[key]
    
    async def check_health_async(self) -> HealthStatus:
        """Perform async health check with enhanced monitoring."""
        issues = []
        requires_restart = False
        
        try:
            # Get performance metrics
            perf_metrics = await self.performance_monitor.get_performance_metrics()
            
            # Check error rate
            if perf_metrics['error_rate'] > 50.0:
                issues.append(f"High error rate: {perf_metrics['error_rate']:.1f}%")
                if perf_metrics['error_rate'] > 80.0:
                    requires_restart = True
            
            # Check response time
            if perf_metrics['average_processing_time'] > 30.0:
                issues.append(f"High response time: {perf_metrics['average_processing_time']:.1f}s")
            
            # Check resource utilization
            resource_stats = self.resource_manager.get_resource_stats()
            if resource_stats['utilization_rate'] > 90.0:
                issues.append(f"High resource utilization: {resource_stats['utilization_rate']:.1f}%")
            
            # Check task coordination
            task_stats = self.task_coordinator.get_task_stats()
            if task_stats['active_tasks'] > 10:
                issues.append(f"Too many active tasks: {task_stats['active_tasks']}")
            
            # Update health status
            self.health_status = HealthStatus(
                is_healthy=len(issues) == 0,
                requires_restart=requires_restart,
                issues=issues,
                last_check=datetime.utcnow(),
                memory_usage_mb=resource_stats['active_resources'] * 10,  # Estimate
                cpu_usage_percent=min(resource_stats['utilization_rate'], 100.0),
                response_time_ms=perf_metrics['average_processing_time'] * 1000
            )
            
            # Signal health check completion
            self.health_check_event.set()
            self.health_check_event.clear()
            
            return self.health_status
            
        except Exception as e:
            logger.error(f"Async health check failed for instance {self.instance_id}: {e}")
            self.health_status = HealthStatus(
                is_healthy=False,
                requires_restart=True,
                issues=[f"Health check failed: {str(e)}"],
                last_check=datetime.utcnow()
            )
            return self.health_status
    
    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while not self.shutdown_event.is_set():
            try:
                await self.check_health_async()
                await asyncio.sleep(30.0)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _performance_monitoring_loop(self) -> None:
        """Background performance monitoring loop."""
        while not self.shutdown_event.is_set():
            try:
                # Update performance metrics
                await self.performance_monitor.get_performance_metrics()
                await asyncio.sleep(10.0)  # Update every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _cache_cleanup_loop(self) -> None:
        """Background cache cleanup loop."""
        while not self.shutdown_event.is_set():
            try:
                async with self.cache_lock:
                    current_time = datetime.utcnow()
                    expired_keys = [
                        key for key, entry in self.operation_cache.items()
                        if (current_time - entry['timestamp']).total_seconds() > self.cache_ttl
                    ]
                    
                    for key in expired_keys:
                        del self.operation_cache[key]
                    
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                await asyncio.sleep(60.0)  # Cleanup every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
                await asyncio.sleep(5.0)
    
    async def get_async_stats(self) -> Dict[str, Any]:
        """Get comprehensive async statistics."""
        base_stats = self.get_stats()
        
        # Add async-specific stats
        async_stats = {
            'performance_metrics': await self.performance_monitor.get_performance_metrics(),
            'performance_trends': await self.performance_monitor.get_performance_trends(),
            'resource_stats': self.resource_manager.get_resource_stats(),
            'task_coordination': self.task_coordinator.get_task_stats(),
            'cache_stats': {
                'cache_size': len(self.operation_cache),
                'cache_hit_rate': 85.0  # Would calculate actual hit rate
            },
            'background_tasks': len(self.background_tasks),
            'async_locks_active': self.async_lock.locked()
        }
        
        return {**base_stats, 'async_metrics': async_stats}
    
    async def shutdown_async(self) -> None:
        """Shutdown async instance with proper cleanup."""
        logger.info(f"Shutting down AsyncScraperInstance {self.instance_id}")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Cancel all background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for background tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Cancel all coordinated tasks
        await self.task_coordinator.wait_for_all_tasks(timeout=10.0)
        
        # Clean up resources
        for resource_id in list(self.resource_manager.active_resources):
            await self.resource_manager.release_resource(resource_id)
        
        # Clear cache
        async with self.cache_lock:
            self.operation_cache.clear()
        
        # Set final status
        await self._set_status_async(InstanceStatus.STOPPED)
        
        logger.info(f"AsyncScraperInstance {self.instance_id} shutdown complete")
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'background_tasks'):
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()