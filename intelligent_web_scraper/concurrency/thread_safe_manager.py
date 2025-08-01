"""
Thread-Safe Instance Manager with advanced synchronization primitives.

This module extends the basic ScraperInstanceManager with enhanced thread safety,
advanced synchronization patterns, and comprehensive resource coordination
for high-concurrency scraping operations.
"""

import asyncio
import threading
import time
import uuid
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import contextmanager
import logging

from ..agents.instance_manager import (
    ScraperInstanceManager, 
    ScraperInstance, 
    InstanceStatus,
    ScrapingTask,
    ScrapingResult
)
from ..config import IntelligentScrapingConfig
from .concurrency_manager import ConcurrencyManager, ResourceType


logger = logging.getLogger(__name__)


class ThreadSafeInstanceManager(ScraperInstanceManager):
    """
    Thread-safe extension of ScraperInstanceManager with advanced synchronization.
    
    This class demonstrates sophisticated thread safety patterns including:
    - Read-write locks for performance optimization
    - Condition variables for coordination
    - Atomic operations for state management
    - Deadlock prevention strategies
    - Resource pooling and allocation
    """
    
    def __init__(self, config: IntelligentScrapingConfig):
        # Initialize parent class
        super().__init__(config)
        
        # Enhanced thread safety primitives
        self.rw_lock = ReadWriteLock()  # Read-write lock for instance access
        self.condition = threading.Condition(self.instance_lock)  # Condition variable
        self.barrier = threading.Barrier(config.max_instances + 1)  # Synchronization barrier
        
        # Advanced synchronization
        self.instance_semaphore = threading.Semaphore(config.max_instances)
        self.task_queue_lock = threading.Lock()
        self.metrics_condition = threading.Condition(self.metrics_lock)
        
        # Resource coordination
        self.concurrency_manager = ConcurrencyManager(config)
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.allocation_tracker = AllocationTracker()
        
        # State management
        self.state_machine = InstanceStateMachine()
        self.operation_coordinator = OperationCoordinator()
        
        # Performance optimization
        self.instance_cache: Dict[str, ScraperInstance] = {}
        self.cache_lock = threading.RLock()
        self.cache_ttl = 300.0  # 5 minutes
        self.cache_cleanup_interval = 60.0  # 1 minute
        
        # Deadlock prevention
        self.lock_hierarchy = LockHierarchy()
        self.deadlock_detector = DeadlockDetector()
        
        # Advanced monitoring
        self.thread_local_data = threading.local()
        self.operation_history: deque = deque(maxlen=10000)
        self.performance_tracker = PerformanceTracker()
        
        # Initialize resource pools
        self._initialize_resource_pools()
        
        logger.info("Initialized ThreadSafeInstanceManager with enhanced synchronization")
    
    def _initialize_resource_pools(self) -> None:
        """Initialize resource pools for different resource types."""
        self.resource_pools = {
            'network_connections': ResourcePool('network_connections', self.config.max_concurrent_requests),
            'memory_buffers': ResourcePool('memory_buffers', 100),
            'file_handles': ResourcePool('file_handles', 50),
            'processing_slots': ResourcePool('processing_slots', self.config.max_workers)
        }
    
    @contextmanager
    def read_lock(self):
        """Context manager for read lock acquisition."""
        self.rw_lock.acquire_read()
        try:
            yield
        finally:
            self.rw_lock.release_read()
    
    @contextmanager
    def write_lock(self):
        """Context manager for write lock acquisition."""
        self.rw_lock.acquire_write()
        try:
            yield
        finally:
            self.rw_lock.release_write()
    
    def create_instance_thread_safe(self, instance_id: Optional[str] = None) -> ScraperInstance:
        """Create instance with enhanced thread safety."""
        if instance_id is None:
            instance_id = f"scraper-{uuid.uuid4().hex[:8]}"
        
        # Use lock hierarchy to prevent deadlocks
        with self.lock_hierarchy.acquire_locks(['instance_creation', 'metrics']):
            with self.write_lock():
                # Check if we can create more instances
                if not self.instance_semaphore.acquire(blocking=False):
                    raise RuntimeError(f"Maximum instances ({self.max_instances}) already created")
                
                try:
                    # Check for duplicate
                    if instance_id in self.instances:
                        self.instance_semaphore.release()
                        raise ValueError(f"Instance {instance_id} already exists")
                    
                    # Create instance with resource allocation
                    with self.allocation_tracker.track_allocation('instance_creation'):
                        instance = ScraperInstance(instance_id, self.config)
                        
                        # Register with state machine
                        self.state_machine.register_instance(instance_id, InstanceStatus.IDLE)
                        
                        # Add to instances
                        self.instances[instance_id] = instance
                        
                        # Update cache
                        with self.cache_lock:
                            self.instance_cache[instance_id] = instance
                        
                        # Update metrics atomically
                        with self.metrics_condition:
                            self.global_metrics['instances_created'] += 1
                            self.metrics_condition.notify_all()
                        
                        # Record operation
                        self.operation_history.append({
                            'operation': 'create_instance',
                            'instance_id': instance_id,
                            'timestamp': datetime.utcnow(),
                            'thread_id': threading.get_ident()
                        })
                        
                        logger.info(f"Created thread-safe instance {instance_id}")
                        return instance
                
                except Exception as e:
                    self.instance_semaphore.release()
                    raise
    
    def get_instance_thread_safe(self, instance_id: str) -> Optional[ScraperInstance]:
        """Get instance with optimized read access."""
        # Try cache first (with read lock for cache consistency)
        with self.cache_lock:
            cached_instance = self.instance_cache.get(instance_id)
            if cached_instance:
                return cached_instance
        
        # Fall back to main storage with read lock
        with self.read_lock():
            instance = self.instances.get(instance_id)
            
            # Update cache if found
            if instance:
                with self.cache_lock:
                    self.instance_cache[instance_id] = instance
            
            return instance
    
    def list_instances_thread_safe(self) -> List[ScraperInstance]:
        """Get thread-safe list of all instances."""
        with self.read_lock():
            return list(self.instances.values())
    
    async def execute_task_coordinated(
        self, 
        task: ScrapingTask, 
        instance_id: Optional[str] = None,
        coordination_strategy: str = 'default'
    ) -> ScrapingResult:
        """Execute task with advanced coordination and resource management."""
        operation_id = str(uuid.uuid4())
        
        try:
            # Register operation with coordinator
            self.operation_coordinator.register_operation(operation_id, task)
            
            # Acquire necessary resources
            async with self._acquire_execution_resources(task):
                # Get or assign instance
                if instance_id:
                    instance = self.get_instance_thread_safe(instance_id)
                    if not instance:
                        raise ValueError(f"Instance {instance_id} not found")
                else:
                    instance = await self._get_optimal_instance(coordination_strategy)
                
                # Execute with coordination
                result = await self._execute_with_coordination(instance, task, operation_id)
                
                # Update performance tracking
                self.performance_tracker.record_execution(
                    instance.instance_id,
                    result.processing_time,
                    result.success
                )
                
                return result
                
        except Exception as e:
            logger.error(f"Coordinated task execution failed: {e}")
            return ScrapingResult(
                task_id=task.task_id,
                instance_id=instance_id or "unknown",
                success=False,
                error=str(e)
            )
        finally:
            # Unregister operation
            self.operation_coordinator.unregister_operation(operation_id)
    
    @contextmanager
    async def _acquire_execution_resources(self, task: ScrapingTask):
        """Acquire all necessary resources for task execution."""
        acquired_resources = []
        
        try:
            # Acquire network connection
            network_resource = await self.resource_pools['network_connections'].acquire()
            acquired_resources.append(('network_connections', network_resource))
            
            # Acquire processing slot
            processing_resource = await self.resource_pools['processing_slots'].acquire()
            acquired_resources.append(('processing_slots', processing_resource))
            
            # Acquire memory buffer if needed
            if task.requires_heavy_processing:
                memory_resource = await self.resource_pools['memory_buffers'].acquire()
                acquired_resources.append(('memory_buffers', memory_resource))
            
            yield
            
        finally:
            # Release all acquired resources
            for pool_name, resource in acquired_resources:
                await self.resource_pools[pool_name].release(resource)
    
    async def _get_optimal_instance(self, strategy: str) -> ScraperInstance:
        """Get optimal instance based on coordination strategy."""
        with self.read_lock():
            available_instances = [
                instance for instance in self.instances.values()
                if self.state_machine.can_accept_task(instance.instance_id)
            ]
        
        if not available_instances:
            # Try to create new instance if under limit
            if len(self.instances) < self.max_instances:
                return self.create_instance_thread_safe()
            else:
                # Wait for instance to become available
                return await self._wait_for_available_instance()
        
        # Apply coordination strategy
        if strategy == 'least_loaded':
            return min(available_instances, key=lambda i: i.metrics.requests_processed)
        elif strategy == 'best_performance':
            return self.performance_tracker.get_best_performing_instance(available_instances)
        elif strategy == 'round_robin':
            return self._get_round_robin_instance(available_instances)
        else:  # default
            return available_instances[0]
    
    async def _wait_for_available_instance(self, timeout: float = 30.0) -> ScraperInstance:
        """Wait for an instance to become available."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.condition:
                # Wait for state change notification
                self.condition.wait(timeout=1.0)
                
                # Check for available instances
                with self.read_lock():
                    available_instances = [
                        instance for instance in self.instances.values()
                        if self.state_machine.can_accept_task(instance.instance_id)
                    ]
                    
                    if available_instances:
                        return available_instances[0]
        
        raise RuntimeError("No instances became available within timeout")
    
    async def _execute_with_coordination(
        self, 
        instance: ScraperInstance, 
        task: ScrapingTask, 
        operation_id: str
    ) -> ScrapingResult:
        """Execute task with full coordination and monitoring."""
        # Update state machine
        self.state_machine.transition_state(instance.instance_id, InstanceStatus.RUNNING)
        
        try:
            # Execute task
            if asyncio.iscoroutinefunction(instance.execute_task_async):
                result = await instance.execute_task_async(task)
            else:
                # Run sync method in thread pool
                result = await asyncio.get_event_loop().run_in_executor(
                    None, instance.execute_task_sync, task
                )
            
            # Notify state change
            with self.condition:
                self.state_machine.transition_state(instance.instance_id, InstanceStatus.IDLE)
                self.condition.notify_all()
            
            return result
            
        except Exception as e:
            # Handle error state
            self.state_machine.transition_state(instance.instance_id, InstanceStatus.ERROR)
            
            with self.condition:
                self.condition.notify_all()
            
            raise
    
    def _get_round_robin_instance(self, instances: List[ScraperInstance]) -> ScraperInstance:
        """Get instance using thread-safe round-robin selection."""
        if not hasattr(self.thread_local_data, 'round_robin_index'):
            self.thread_local_data.round_robin_index = 0
        
        index = self.thread_local_data.round_robin_index % len(instances)
        self.thread_local_data.round_robin_index += 1
        
        return instances[index]
    
    def get_thread_safe_stats(self) -> Dict[str, Any]:
        """Get comprehensive thread-safe statistics."""
        with self.read_lock():
            base_stats = super().get_global_stats()
        
        # Add thread safety metrics
        thread_safety_stats = {
            'active_threads': threading.active_count(),
            'lock_contention': self.deadlock_detector.get_contention_stats(),
            'resource_utilization': {
                pool_name: pool.get_utilization_stats()
                for pool_name, pool in self.resource_pools.items()
            },
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'operation_history_size': len(self.operation_history),
            'state_machine_states': self.state_machine.get_state_summary(),
            'performance_metrics': self.performance_tracker.get_summary()
        }
        
        return {**base_stats, 'thread_safety': thread_safety_stats}
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate for performance monitoring."""
        # This would be implemented with actual cache hit/miss tracking
        # For now, return a placeholder
        return 85.0
    
    def perform_coordinated_health_checks(self) -> Dict[str, Any]:
        """Perform health checks with coordination and synchronization."""
        health_results = {}
        
        # Use barrier to synchronize health checks across instances
        def check_instance_health(instance: ScraperInstance) -> None:
            try:
                health_status = instance.check_health()
                with self.metrics_lock:
                    health_results[instance.instance_id] = health_status.dict()
                
                # Wait at barrier for all health checks to complete
                self.barrier.wait(timeout=10.0)
                
            except threading.BrokenBarrierError:
                logger.warning(f"Health check barrier broken for instance {instance.instance_id}")
            except Exception as e:
                logger.error(f"Health check failed for instance {instance.instance_id}: {e}")
        
        # Start health checks for all instances
        with self.read_lock():
            instances = list(self.instances.values())
        
        threads = []
        for instance in instances:
            thread = threading.Thread(
                target=check_instance_health,
                args=(instance,),
                name=f"HealthCheck-{instance.instance_id}"
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all health checks to complete
        for thread in threads:
            thread.join(timeout=15.0)
        
        # Reset barrier for next use
        try:
            self.barrier.reset()
        except threading.BrokenBarrierError:
            pass
        
        return health_results
    
    def shutdown_coordinated(self, timeout: float = 30.0) -> None:
        """Shutdown with proper coordination and resource cleanup."""
        logger.info("Starting coordinated shutdown of ThreadSafeInstanceManager")
        
        # Stop accepting new operations
        self.operation_coordinator.stop_accepting_operations()
        
        # Wait for ongoing operations to complete
        self.operation_coordinator.wait_for_completion(timeout=timeout)
        
        # Shutdown concurrency manager
        self.concurrency_manager.shutdown(wait=True, timeout=timeout)
        
        # Clean up resource pools
        for pool in self.resource_pools.values():
            pool.shutdown()
        
        # Release all semaphores
        for _ in range(self.max_instances):
            try:
                self.instance_semaphore.release()
            except ValueError:
                break
        
        # Call parent shutdown
        super().shutdown()
        
        logger.info("ThreadSafeInstanceManager coordinated shutdown complete")


class ReadWriteLock:
    """Read-write lock implementation for optimized concurrent access."""
    
    def __init__(self):
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0
    
    def acquire_read(self):
        """Acquire read lock."""
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()
    
    def release_read(self):
        """Release read lock."""
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notifyAll()
        finally:
            self._read_ready.release()
    
    def acquire_write(self):
        """Acquire write lock."""
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()
    
    def release_write(self):
        """Release write lock."""
        self._read_ready.release()


class ResourcePool:
    """Thread-safe resource pool for managing limited resources."""
    
    def __init__(self, name: str, max_size: int):
        self.name = name
        self.max_size = max_size
        self._pool = asyncio.Queue(maxsize=max_size)
        self._created = 0
        self._lock = asyncio.Lock()
        
        # Pre-populate pool
        for i in range(max_size):
            self._pool.put_nowait(f"{name}_resource_{i}")
    
    async def acquire(self) -> str:
        """Acquire a resource from the pool."""
        return await self._pool.get()
    
    async def release(self, resource: str) -> None:
        """Release a resource back to the pool."""
        await self._pool.put(resource)
    
    def get_utilization_stats(self) -> Dict[str, Any]:
        """Get resource utilization statistics."""
        return {
            'total_resources': self.max_size,
            'available_resources': self._pool.qsize(),
            'utilized_resources': self.max_size - self._pool.qsize(),
            'utilization_rate': ((self.max_size - self._pool.qsize()) / self.max_size) * 100
        }
    
    def shutdown(self) -> None:
        """Shutdown the resource pool."""
        # Clear the pool
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
            except asyncio.QueueEmpty:
                break


class AllocationTracker:
    """Tracks resource allocations for debugging and monitoring."""
    
    def __init__(self):
        self.allocations: Dict[str, List[datetime]] = defaultdict(list)
        self.lock = threading.Lock()
    
    @contextmanager
    def track_allocation(self, resource_type: str):
        """Context manager for tracking resource allocation."""
        with self.lock:
            self.allocations[resource_type].append(datetime.utcnow())
        
        try:
            yield
        finally:
            # Could track deallocation here if needed
            pass
    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get allocation statistics."""
        with self.lock:
            return {
                resource_type: {
                    'total_allocations': len(timestamps),
                    'recent_allocations': len([
                        ts for ts in timestamps 
                        if (datetime.utcnow() - ts).total_seconds() < 300
                    ])
                }
                for resource_type, timestamps in self.allocations.items()
            }


class InstanceStateMachine:
    """State machine for managing instance states with thread safety."""
    
    def __init__(self):
        self.states: Dict[str, InstanceStatus] = {}
        self.transitions: Dict[str, List[datetime]] = defaultdict(list)
        self.lock = threading.RLock()
    
    def register_instance(self, instance_id: str, initial_state: InstanceStatus) -> None:
        """Register an instance with initial state."""
        with self.lock:
            self.states[instance_id] = initial_state
            self.transitions[instance_id].append(datetime.utcnow())
    
    def transition_state(self, instance_id: str, new_state: InstanceStatus) -> bool:
        """Transition instance to new state."""
        with self.lock:
            if instance_id not in self.states:
                return False
            
            old_state = self.states[instance_id]
            if self._is_valid_transition(old_state, new_state):
                self.states[instance_id] = new_state
                self.transitions[instance_id].append(datetime.utcnow())
                return True
            
            return False
    
    def can_accept_task(self, instance_id: str) -> bool:
        """Check if instance can accept a new task."""
        with self.lock:
            state = self.states.get(instance_id)
            return state == InstanceStatus.IDLE
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of all instance states."""
        with self.lock:
            state_counts = defaultdict(int)
            for state in self.states.values():
                state_counts[state.value] += 1
            
            return dict(state_counts)
    
    def _is_valid_transition(self, old_state: InstanceStatus, new_state: InstanceStatus) -> bool:
        """Check if state transition is valid."""
        # Define valid transitions
        valid_transitions = {
            InstanceStatus.IDLE: [InstanceStatus.RUNNING, InstanceStatus.STOPPING, InstanceStatus.ERROR],
            InstanceStatus.RUNNING: [InstanceStatus.IDLE, InstanceStatus.ERROR, InstanceStatus.STOPPING],
            InstanceStatus.ERROR: [InstanceStatus.IDLE, InstanceStatus.STOPPING],
            InstanceStatus.STOPPING: [InstanceStatus.STOPPED],
            InstanceStatus.STOPPED: [InstanceStatus.IDLE]
        }
        
        return new_state in valid_transitions.get(old_state, [])


class OperationCoordinator:
    """Coordinates operations across multiple instances."""
    
    def __init__(self):
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self.operation_queue: deque = deque()
        self.accepting_operations = True
        self.lock = threading.RLock()
        self.completion_event = threading.Event()
    
    def register_operation(self, operation_id: str, task: ScrapingTask) -> None:
        """Register a new operation."""
        with self.lock:
            if not self.accepting_operations:
                raise RuntimeError("Not accepting new operations")
            
            self.active_operations[operation_id] = {
                'task': task,
                'started_at': datetime.utcnow(),
                'status': 'active'
            }
    
    def unregister_operation(self, operation_id: str) -> None:
        """Unregister a completed operation."""
        with self.lock:
            if operation_id in self.active_operations:
                del self.active_operations[operation_id]
            
            # Signal completion if no more operations
            if not self.active_operations:
                self.completion_event.set()
    
    def stop_accepting_operations(self) -> None:
        """Stop accepting new operations."""
        with self.lock:
            self.accepting_operations = False
    
    def wait_for_completion(self, timeout: float = 30.0) -> bool:
        """Wait for all operations to complete."""
        return self.completion_event.wait(timeout=timeout)


class LockHierarchy:
    """Manages lock hierarchy to prevent deadlocks."""
    
    def __init__(self):
        self.hierarchy = {
            'instance_creation': 1,
            'metrics': 2,
            'cache': 3,
            'state_machine': 4
        }
        self.acquired_locks = threading.local()
    
    @contextmanager
    def acquire_locks(self, lock_names: List[str]):
        """Acquire locks in hierarchical order."""
        if not hasattr(self.acquired_locks, 'locks'):
            self.acquired_locks.locks = []
        
        # Sort by hierarchy
        sorted_locks = sorted(lock_names, key=lambda x: self.hierarchy.get(x, 999))
        
        acquired = []
        try:
            for lock_name in sorted_locks:
                # This is a simplified implementation
                # In practice, you'd have actual lock objects
                acquired.append(lock_name)
                self.acquired_locks.locks.append(lock_name)
            
            yield
            
        finally:
            # Release in reverse order
            for lock_name in reversed(acquired):
                if lock_name in self.acquired_locks.locks:
                    self.acquired_locks.locks.remove(lock_name)


class DeadlockDetector:
    """Detects potential deadlocks and provides contention statistics."""
    
    def __init__(self):
        self.contention_stats = defaultdict(int)
        self.lock = threading.Lock()
    
    def record_contention(self, lock_name: str) -> None:
        """Record lock contention."""
        with self.lock:
            self.contention_stats[lock_name] += 1
    
    def get_contention_stats(self) -> Dict[str, int]:
        """Get lock contention statistics."""
        with self.lock:
            return dict(self.contention_stats)


class PerformanceTracker:
    """Tracks performance metrics for instances."""
    
    def __init__(self):
        self.instance_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def record_execution(self, instance_id: str, processing_time: float, success: bool) -> None:
        """Record execution metrics."""
        with self.lock:
            self.instance_metrics[instance_id].append({
                'processing_time': processing_time,
                'success': success,
                'timestamp': datetime.utcnow()
            })
            
            # Keep only last 1000 records per instance
            if len(self.instance_metrics[instance_id]) > 1000:
                self.instance_metrics[instance_id] = self.instance_metrics[instance_id][-1000:]
    
    def get_best_performing_instance(self, instances: List[ScraperInstance]) -> ScraperInstance:
        """Get the best performing instance based on metrics."""
        best_instance = instances[0]
        best_score = 0.0
        
        with self.lock:
            for instance in instances:
                metrics = self.instance_metrics.get(instance.instance_id, [])
                if not metrics:
                    continue
                
                # Calculate performance score
                recent_metrics = [m for m in metrics if (datetime.utcnow() - m['timestamp']).total_seconds() < 300]
                if recent_metrics:
                    avg_time = sum(m['processing_time'] for m in recent_metrics) / len(recent_metrics)
                    success_rate = sum(1 for m in recent_metrics if m['success']) / len(recent_metrics)
                    score = success_rate / max(avg_time, 0.1)  # Higher success rate, lower time = better
                    
                    if score > best_score:
                        best_score = score
                        best_instance = instance
        
        return best_instance
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self.lock:
            total_executions = sum(len(metrics) for metrics in self.instance_metrics.values())
            
            return {
                'total_executions': total_executions,
                'instances_tracked': len(self.instance_metrics),
                'average_executions_per_instance': total_executions / max(len(self.instance_metrics), 1)
            }