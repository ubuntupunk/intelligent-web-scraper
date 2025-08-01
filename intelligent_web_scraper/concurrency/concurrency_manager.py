"""
Concurrency Manager for coordinating thread pools and async operations.

This module implements advanced concurrency patterns for managing
thread pools, async operations, and resource coordination in the
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


class ConcurrencyManager:
    """
    Advanced concurrency manager for coordinating thread pools and async operations.
    
    This class demonstrates sophisticated patterns for managing concurrent operations
    with proper resource allocation, monitoring, and coordination between different
    concurrency models (threads, async tasks, processes).
    """
    
    def __init__(self, config: IntelligentScrapingConfig):
        self.config = config
        
        # Thread pool management
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.max_workers,
            thread_name_prefix="IntelligentScraper"
        )
        
        # Async task management
        self.async_semaphore = asyncio.Semaphore(config.max_async_tasks)
        self.async_tasks: Dict[str, asyncio.Task] = {}
        
        # Operation tracking
        self.operations: Dict[str, ConcurrentOperation] = {}
        self.operation_lock = threading.RLock()
        
        # Resource management
        self.resources: Dict[ResourceType, ResourceUsage] = {
            ResourceType.THREAD: ResourceUsage(ResourceType.THREAD, max_usage=config.max_workers),
            ResourceType.ASYNC_TASK: ResourceUsage(ResourceType.ASYNC_TASK, max_usage=config.max_async_tasks),
            ResourceType.MEMORY: ResourceUsage(ResourceType.MEMORY, max_usage=1024),  # MB
            ResourceType.NETWORK: ResourceUsage(ResourceType.NETWORK, max_usage=config.max_concurrent_requests),
            ResourceType.FILE_HANDLE: ResourceUsage(ResourceType.FILE_HANDLE, max_usage=100)
        }
        self.resource_lock = threading.Lock()
        
        # Monitoring and cleanup
        self.monitoring_enabled = config.enable_monitoring
        self.cleanup_interval = 60.0  # seconds
        self.cleanup_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Performance metrics
        self.metrics = {
            'operations_started': 0,
            'operations_completed': 0,
            'operations_failed': 0,
            'operations_cancelled': 0,
            'operations_timeout': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'peak_concurrent_operations': 0
        }
        self.metrics_lock = threading.Lock()
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.info(f"Initialized ConcurrencyManager with {config.max_workers} threads, {config.max_async_tasks} async tasks")
    
    def submit_sync_operation(
        self, 
        func: Callable, 
        *args, 
        operation_type: str = "sync_operation",
        timeout_seconds: float = 300.0,
        priority: int = 0,
        **kwargs
    ) -> str:
        """
        Submit a synchronous operation to the thread pool.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            operation_type: Type identifier for the operation
            timeout_seconds: Timeout for the operation
            priority: Priority level (higher = more priority)
            **kwargs: Keyword arguments for the function
            
        Returns:
            Operation ID for tracking
        """
        operation_id = str(uuid.uuid4())
        
        # Create operation record
        operation = ConcurrentOperation(
            operation_id=operation_id,
            operation_type=operation_type,
            timeout_seconds=timeout_seconds,
            priority=priority,
            metadata={'args': args, 'kwargs': kwargs}
        )
        
        with self.operation_lock:
            self.operations[operation_id] = operation
        
        # Allocate thread resource
        if not self._allocate_resource(ResourceType.THREAD):
            operation.status = OperationStatus.FAILED
            operation.error = "No thread resources available"
            return operation_id
        
        try:
            # Submit to thread pool
            future = self.thread_pool.submit(self._execute_sync_operation, operation, func, *args, **kwargs)
            operation.metadata['future'] = future
            
            # Update metrics
            with self.metrics_lock:
                self.metrics['operations_started'] += 1
                current_ops = len([op for op in self.operations.values() if op.status == OperationStatus.RUNNING])
                self.metrics['peak_concurrent_operations'] = max(
                    self.metrics['peak_concurrent_operations'], 
                    current_ops
                )
            
            logger.debug(f"Submitted sync operation {operation_id} ({operation_type})")
            
        except Exception as e:
            operation.status = OperationStatus.FAILED
            operation.error = str(e)
            self._release_resource(ResourceType.THREAD)
            logger.error(f"Failed to submit sync operation {operation_id}: {e}")
        
        return operation_id
    
    async def submit_async_operation(
        self, 
        coro: Awaitable, 
        operation_type: str = "async_operation",
        timeout_seconds: float = 300.0,
        priority: int = 0
    ) -> str:
        """
        Submit an asynchronous operation for execution.
        
        Args:
            coro: Coroutine to execute
            operation_type: Type identifier for the operation
            timeout_seconds: Timeout for the operation
            priority: Priority level (higher = more priority)
            
        Returns:
            Operation ID for tracking
        """
        operation_id = str(uuid.uuid4())
        
        # Create operation record
        operation = ConcurrentOperation(
            operation_id=operation_id,
            operation_type=operation_type,
            timeout_seconds=timeout_seconds,
            priority=priority
        )
        
        with self.operation_lock:
            self.operations[operation_id] = operation
        
        # Allocate async task resource
        if not self._allocate_resource(ResourceType.ASYNC_TASK):
            operation.status = OperationStatus.FAILED
            operation.error = "No async task resources available"
            return operation_id
        
        try:
            # Create and start async task
            task = asyncio.create_task(self._execute_async_operation(operation, coro))
            self.async_tasks[operation_id] = task
            operation.metadata['task'] = task
            
            # Update metrics
            with self.metrics_lock:
                self.metrics['operations_started'] += 1
                current_ops = len([op for op in self.operations.values() if op.status == OperationStatus.RUNNING])
                self.metrics['peak_concurrent_operations'] = max(
                    self.metrics['peak_concurrent_operations'], 
                    current_ops
                )
            
            logger.debug(f"Submitted async operation {operation_id} ({operation_type})")
            
        except Exception as e:
            operation.status = OperationStatus.FAILED
            operation.error = str(e)
            self._release_resource(ResourceType.ASYNC_TASK)
            logger.error(f"Failed to submit async operation {operation_id}: {e}")
        
        return operation_id
    
    def _execute_sync_operation(self, operation: ConcurrentOperation, func: Callable, *args, **kwargs) -> Any:
        """Execute a synchronous operation with proper tracking."""
        operation.status = OperationStatus.RUNNING
        operation.started_at = datetime.utcnow()
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Update operation
            operation.status = OperationStatus.COMPLETED
            operation.completed_at = datetime.utcnow()
            operation.result = result
            
            # Update metrics
            duration = operation.get_duration()
            with self.metrics_lock:
                self.metrics['operations_completed'] += 1
                if duration:
                    self.metrics['total_processing_time'] += duration
                    self.metrics['average_processing_time'] = (
                        self.metrics['total_processing_time'] / self.metrics['operations_completed']
                    )
            
            logger.debug(f"Completed sync operation {operation.operation_id} in {duration:.2f}s")
            return result
            
        except Exception as e:
            operation.status = OperationStatus.FAILED
            operation.completed_at = datetime.utcnow()
            operation.error = str(e)
            
            with self.metrics_lock:
                self.metrics['operations_failed'] += 1
            
            logger.error(f"Sync operation {operation.operation_id} failed: {e}")
            raise
            
        finally:
            # Release thread resource
            self._release_resource(ResourceType.THREAD)
    
    async def _execute_async_operation(self, operation: ConcurrentOperation, coro: Awaitable) -> Any:
        """Execute an asynchronous operation with proper tracking."""
        async with self.async_semaphore:
            operation.status = OperationStatus.RUNNING
            operation.started_at = datetime.utcnow()
            
            try:
                # Execute with timeout
                result = await asyncio.wait_for(coro, timeout=operation.timeout_seconds)
                
                # Update operation
                operation.status = OperationStatus.COMPLETED
                operation.completed_at = datetime.utcnow()
                operation.result = result
                
                # Update metrics
                duration = operation.get_duration()
                with self.metrics_lock:
                    self.metrics['operations_completed'] += 1
                    if duration:
                        self.metrics['total_processing_time'] += duration
                        self.metrics['average_processing_time'] = (
                            self.metrics['total_processing_time'] / self.metrics['operations_completed']
                        )
                
                logger.debug(f"Completed async operation {operation.operation_id} in {duration:.2f}s")
                return result
                
            except asyncio.TimeoutError:
                operation.status = OperationStatus.TIMEOUT
                operation.completed_at = datetime.utcnow()
                operation.error = f"Operation timed out after {operation.timeout_seconds}s"
                
                with self.metrics_lock:
                    self.metrics['operations_timeout'] += 1
                
                logger.warning(f"Async operation {operation.operation_id} timed out")
                raise
                
            except asyncio.CancelledError:
                operation.status = OperationStatus.CANCELLED
                operation.completed_at = datetime.utcnow()
                operation.error = "Operation was cancelled"
                
                with self.metrics_lock:
                    self.metrics['operations_cancelled'] += 1
                
                logger.info(f"Async operation {operation.operation_id} was cancelled")
                raise
                
            except Exception as e:
                operation.status = OperationStatus.FAILED
                operation.completed_at = datetime.utcnow()
                operation.error = str(e)
                
                with self.metrics_lock:
                    self.metrics['operations_failed'] += 1
                
                logger.error(f"Async operation {operation.operation_id} failed: {e}")
                raise
                
            finally:
                # Clean up task reference
                if operation.operation_id in self.async_tasks:
                    del self.async_tasks[operation.operation_id]
                
                # Release async task resource
                self._release_resource(ResourceType.ASYNC_TASK)
    
    def get_operation_status(self, operation_id: str) -> Optional[ConcurrentOperation]:
        """Get the status of a specific operation."""
        with self.operation_lock:
            return self.operations.get(operation_id)
    
    def cancel_operation(self, operation_id: str) -> bool:
        """Cancel a running operation."""
        with self.operation_lock:
            operation = self.operations.get(operation_id)
            if not operation:
                return False
            
            if operation.status not in [OperationStatus.PENDING, OperationStatus.RUNNING]:
                return False
            
            try:
                # Cancel thread pool operation
                if 'future' in operation.metadata:
                    future = operation.metadata['future']
                    if isinstance(future, Future):
                        cancelled = future.cancel()
                        if cancelled:
                            operation.status = OperationStatus.CANCELLED
                            operation.completed_at = datetime.utcnow()
                            operation.error = "Operation cancelled by user"
                            return True
                
                # Cancel async task
                if operation.operation_id in self.async_tasks:
                    task = self.async_tasks[operation.operation_id]
                    if isinstance(task, asyncio.Task):
                        task.cancel()
                        operation.status = OperationStatus.CANCELLED
                        operation.completed_at = datetime.utcnow()
                        operation.error = "Operation cancelled by user"
                        return True
                
                return False
                
            except Exception as e:
                logger.error(f"Failed to cancel operation {operation_id}: {e}")
                return False
    
    def wait_for_operation(self, operation_id: str, timeout: Optional[float] = None) -> Optional[Any]:
        """Wait for an operation to complete and return its result."""
        operation = self.get_operation_status(operation_id)
        if not operation:
            return None
        
        # If already completed, return result
        if operation.status in [OperationStatus.COMPLETED, OperationStatus.FAILED, OperationStatus.CANCELLED]:
            return operation.result if operation.status == OperationStatus.COMPLETED else None
        
        # Wait for thread pool operation
        if 'future' in operation.metadata:
            future = operation.metadata['future']
            if isinstance(future, Future):
                try:
                    return future.result(timeout=timeout)
                except Exception:
                    return None
        
        # For async operations, we can't wait synchronously
        # The caller should use await on the async method instead
        return None
    
    async def wait_for_async_operation(self, operation_id: str) -> Optional[Any]:
        """Wait for an async operation to complete and return its result."""
        if operation_id in self.async_tasks:
            task = self.async_tasks[operation_id]
            try:
                return await task
            except Exception:
                return None
        
        return None
    
    def wait_for_all_operations(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for all pending operations to complete."""
        results = {}
        
        with self.operation_lock:
            pending_operations = [
                op for op in self.operations.values() 
                if op.status in [OperationStatus.PENDING, OperationStatus.RUNNING]
            ]
        
        # Wait for thread pool operations
        futures = []
        for operation in pending_operations:
            if 'future' in operation.metadata:
                futures.append((operation.operation_id, operation.metadata['future']))
        
        if futures:
            for operation_id, future in futures:
                try:
                    result = future.result(timeout=timeout)
                    results[operation_id] = result
                except Exception as e:
                    results[operation_id] = None
                    logger.error(f"Operation {operation_id} failed: {e}")
        
        return results
    
    @contextmanager
    def resource_allocation(self, resource_type: ResourceType, count: int = 1):
        """Context manager for resource allocation and automatic cleanup."""
        allocated = self._allocate_resource(resource_type, count)
        if not allocated:
            raise RuntimeError(f"Could not allocate {count} {resource_type.value} resources")
        
        try:
            yield
        finally:
            self._release_resource(resource_type, count)
    
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
    
    def get_resource_usage(self) -> Dict[str, Dict[str, Any]]:
        """Get current resource usage statistics."""
        with self.resource_lock:
            return {
                resource_type.value: {
                    'current_usage': resource.current_usage,
                    'max_usage': resource.max_usage,
                    'peak_usage': resource.peak_usage,
                    'utilization_rate': resource.get_utilization_rate(),
                    'total_allocated': resource.total_allocated,
                    'total_released': resource.total_released
                }
                for resource_type, resource in self.resources.items()
            }
    
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
            'thread_pool_active': self.thread_pool._threads,
            'async_tasks_active': len(self.async_tasks)
        })
        
        return metrics
    
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
                
                # Cancel truly expired running operations
                elif operation.status == OperationStatus.RUNNING and operation.is_expired():
                    self.cancel_operation(operation_id)
                    logger.warning(f"Cancelled expired operation {operation_id}")
            
            # Remove expired operations
            for operation_id in expired_operations:
                del self.operations[operation_id]
                if operation_id in self.async_tasks:
                    del self.async_tasks[operation_id]
            
            if expired_operations:
                logger.debug(f"Cleaned up {len(expired_operations)} expired operations")
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Shutdown the concurrency manager and clean up resources."""
        logger.info("Shutting down ConcurrencyManager")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Cancel all pending async tasks
        for task in list(self.async_tasks.values()):
            if isinstance(task, asyncio.Task) and not task.done():
                task.cancel()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=wait, timeout=timeout)
        
        # Wait for cleanup thread
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)
        
        logger.info("ConcurrencyManager shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()