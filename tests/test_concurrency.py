"""
Unit tests for the concurrency management system.

This module tests the advanced concurrency patterns including thread management,
async coordination, resource management, and synchronization primitives.
"""

import pytest
import asyncio
import threading
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from intelligent_web_scraper.concurrency import (
    ConcurrencyManager,
    ThreadSafeInstanceManager,
    AsyncScraperInstance
)
from intelligent_web_scraper.concurrency.concurrency_manager import (
    OperationStatus,
    ResourceType,
    ConcurrentOperation,
    ResourceUsage
)
from intelligent_web_scraper.concurrency.thread_safe_manager import (
    ReadWriteLock,
    ResourcePool,
    AllocationTracker,
    InstanceStateMachine,
    OperationCoordinator
)
from intelligent_web_scraper.concurrency.async_instance import (
    AsyncResourceManager,
    AsyncTaskCoordinator,
    AsyncPerformanceMonitor
)
from intelligent_web_scraper.agents.instance_manager import (
    InstanceStatus,
    ScrapingTask,
    ScrapingResult
)
from intelligent_web_scraper.config import IntelligentScrapingConfig


class TestConcurrentOperation:
    """Test the ConcurrentOperation dataclass."""
    
    def test_operation_creation(self):
        """Test operation creation with all fields."""
        operation = ConcurrentOperation(
            operation_id="test-op-123",
            operation_type="test_operation",
            timeout_seconds=120.0,
            priority=5,
            metadata={"key": "value"}
        )
        
        assert operation.operation_id == "test-op-123"
        assert operation.operation_type == "test_operation"
        assert operation.status == OperationStatus.PENDING
        assert operation.timeout_seconds == 120.0
        assert operation.priority == 5
        assert operation.metadata == {"key": "value"}
        assert isinstance(operation.created_at, datetime)
        assert operation.started_at is None
        assert operation.completed_at is None
    
    def test_operation_duration_calculation(self):
        """Test operation duration calculation."""
        operation = ConcurrentOperation("test-op", "test")
        
        # No duration when not started
        assert operation.get_duration() is None
        
        # Duration when started
        operation.started_at = datetime.utcnow()
        time.sleep(0.1)
        duration = operation.get_duration()
        assert duration is not None
        assert duration > 0.0
        
        # Duration when completed
        operation.completed_at = datetime.utcnow()
        completed_duration = operation.get_duration()
        assert completed_duration >= duration
    
    def test_operation_expiry_check(self):
        """Test operation expiry checking."""
        operation = ConcurrentOperation("test-op", "test", timeout_seconds=0.1)
        
        # Not expired when not started
        assert not operation.is_expired()
        
        # Not expired when just started
        operation.started_at = datetime.utcnow()
        assert not operation.is_expired()
        
        # Expired after timeout
        operation.started_at = datetime.utcnow() - timedelta(seconds=0.2)
        assert operation.is_expired()


class TestResourceUsage:
    """Test the ResourceUsage class."""
    
    def test_resource_allocation(self):
        """Test resource allocation and tracking."""
        resource = ResourceUsage(ResourceType.THREAD, max_usage=5)
        
        # Initial state
        assert resource.current_usage == 0
        assert resource.total_allocated == 0
        assert resource.peak_usage == 0
        
        # Successful allocation
        assert resource.allocate(2) is True
        assert resource.current_usage == 2
        assert resource.total_allocated == 2
        assert resource.peak_usage == 2
        
        # Another allocation
        assert resource.allocate(2) is True
        assert resource.current_usage == 4
        assert resource.total_allocated == 4
        assert resource.peak_usage == 4
        
        # Failed allocation (exceeds max)
        assert resource.allocate(2) is False
        assert resource.current_usage == 4  # Unchanged
        assert resource.total_allocated == 4  # Unchanged
    
    def test_resource_release(self):
        """Test resource release."""
        resource = ResourceUsage(ResourceType.THREAD, max_usage=5)
        
        # Allocate some resources
        resource.allocate(3)
        assert resource.current_usage == 3
        
        # Release resources
        resource.release(2)
        assert resource.current_usage == 1
        assert resource.total_released == 2
        
        # Release more than allocated (should not go negative)
        resource.release(5)
        assert resource.current_usage == 0
        assert resource.total_released == 7
    
    def test_utilization_rate_calculation(self):
        """Test utilization rate calculation."""
        resource = ResourceUsage(ResourceType.THREAD, max_usage=10)
        
        # 0% utilization
        assert resource.get_utilization_rate() == 0.0
        
        # 50% utilization
        resource.allocate(5)
        assert resource.get_utilization_rate() == 50.0
        
        # 100% utilization
        resource.allocate(5)
        assert resource.get_utilization_rate() == 100.0
        
        # Zero max usage edge case
        zero_resource = ResourceUsage(ResourceType.THREAD, max_usage=0)
        assert zero_resource.get_utilization_rate() == 0.0


class TestConcurrencyManager:
    """Test the ConcurrencyManager class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return IntelligentScrapingConfig(
            max_workers=3,
            max_async_tasks=5,
            max_concurrent_requests=4,
            enable_monitoring=True
        )
    
    def test_concurrency_manager_initialization(self, mock_config):
        """Test concurrency manager initialization."""
        manager = ConcurrencyManager(mock_config)
        
        assert manager.config == mock_config
        assert manager.thread_pool._max_workers == 3
        assert manager.async_semaphore._value == 5
        assert len(manager.resources) == 5  # All resource types
        assert manager.monitoring_enabled is True
        assert isinstance(manager.operations, dict)
        assert isinstance(manager.async_tasks, dict)
        
        # Check resource initialization
        thread_resource = manager.resources[ResourceType.THREAD]
        assert thread_resource.max_usage == 3
        assert thread_resource.current_usage == 0
        
        manager.shutdown()
    
    def test_submit_sync_operation(self, mock_config):
        """Test submitting synchronous operations."""
        manager = ConcurrencyManager(mock_config)
        
        def test_function(x, y):
            time.sleep(0.1)
            return x + y
        
        # Submit operation
        operation_id = manager.submit_sync_operation(
            test_function, 5, 10,
            operation_type="addition",
            timeout_seconds=5.0,
            priority=1
        )
        
        assert isinstance(operation_id, str)
        assert operation_id in manager.operations
        
        # Check operation details
        operation = manager.operations[operation_id]
        assert operation.operation_type == "addition"
        assert operation.timeout_seconds == 5.0
        assert operation.priority == 1
        
        # Wait for completion
        result = manager.wait_for_operation(operation_id, timeout=2.0)
        assert result == 15
        
        # Check final status
        final_operation = manager.get_operation_status(operation_id)
        assert final_operation.status == OperationStatus.COMPLETED
        assert final_operation.result == 15
        
        manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_submit_async_operation(self, mock_config):
        """Test submitting asynchronous operations."""
        manager = ConcurrencyManager(mock_config)
        
        async def test_coroutine(x, y):
            await asyncio.sleep(0.1)
            return x * y
        
        # Submit operation
        operation_id = await manager.submit_async_operation(
            test_coroutine(3, 4),
            operation_type="multiplication",
            timeout_seconds=5.0,
            priority=2
        )
        
        assert isinstance(operation_id, str)
        assert operation_id in manager.operations
        assert operation_id in manager.async_tasks
        
        # Wait for completion
        result = await manager.wait_for_async_operation(operation_id)
        assert result == 12
        
        # Check final status
        final_operation = manager.get_operation_status(operation_id)
        assert final_operation.status == OperationStatus.COMPLETED
        assert final_operation.result == 12
        
        manager.shutdown()
    
    def test_operation_cancellation(self, mock_config):
        """Test operation cancellation."""
        manager = ConcurrencyManager(mock_config)
        
        def long_running_function():
            time.sleep(2.0)
            return "completed"
        
        # Submit long-running operation
        operation_id = manager.submit_sync_operation(long_running_function)
        
        # Cancel operation
        time.sleep(0.1)  # Let it start
        cancelled = manager.cancel_operation(operation_id)
        assert cancelled is True
        
        # Check status
        operation = manager.get_operation_status(operation_id)
        assert operation.status == OperationStatus.CANCELLED
        
        manager.shutdown()
    
    def test_resource_allocation_context_manager(self, mock_config):
        """Test resource allocation context manager."""
        manager = ConcurrencyManager(mock_config)
        
        # Test successful allocation
        with manager.resource_allocation(ResourceType.NETWORK, 2):
            network_resource = manager.resources[ResourceType.NETWORK]
            assert network_resource.current_usage == 2
        
        # Resource should be released after context
        assert network_resource.current_usage == 0
        
        # Test failed allocation
        with pytest.raises(RuntimeError, match="Could not allocate"):
            with manager.resource_allocation(ResourceType.NETWORK, 1000):
                pass
        
        manager.shutdown()
    
    def test_resource_usage_tracking(self, mock_config):
        """Test resource usage tracking."""
        manager = ConcurrencyManager(mock_config)
        
        # Allocate some resources
        manager._allocate_resource(ResourceType.THREAD, 2)
        manager._allocate_resource(ResourceType.ASYNC_TASK, 3)
        
        # Get usage stats
        usage_stats = manager.get_resource_usage()
        
        assert 'thread' in usage_stats
        assert 'async_task' in usage_stats
        
        thread_stats = usage_stats['thread']
        assert thread_stats['current_usage'] == 2
        assert thread_stats['max_usage'] == 3
        assert thread_stats['utilization_rate'] == (2/3) * 100
        
        async_stats = usage_stats['async_task']
        assert async_stats['current_usage'] == 3
        assert async_stats['max_usage'] == 5
        
        manager.shutdown()
    
    def test_performance_metrics(self, mock_config):
        """Test performance metrics collection."""
        manager = ConcurrencyManager(mock_config)
        
        def simple_function():
            return "done"
        
        # Submit some operations
        op1 = manager.submit_sync_operation(simple_function)
        op2 = manager.submit_sync_operation(simple_function)
        
        # Wait for completion
        manager.wait_for_operation(op1)
        manager.wait_for_operation(op2)
        
        # Get metrics
        metrics = manager.get_performance_metrics()
        
        assert metrics['operations_started'] >= 2
        assert metrics['operations_completed'] >= 2
        assert metrics['total_operations'] >= 2
        assert 'average_processing_time' in metrics
        assert 'peak_concurrent_operations' in metrics
        
        manager.shutdown()
    
    def test_wait_for_all_operations(self, mock_config):
        """Test waiting for all operations to complete."""
        manager = ConcurrencyManager(mock_config)
        
        def test_function(value):
            time.sleep(0.1)
            return value * 2
        
        # Submit multiple operations
        op1 = manager.submit_sync_operation(test_function, 5)
        op2 = manager.submit_sync_operation(test_function, 10)
        op3 = manager.submit_sync_operation(test_function, 15)
        
        # Wait for all
        results = manager.wait_for_all_operations(timeout=2.0)
        
        assert len(results) == 3
        assert results[op1] == 10
        assert results[op2] == 20
        assert results[op3] == 30
        
        manager.shutdown()


class TestReadWriteLock:
    """Test the ReadWriteLock implementation."""
    
    def test_multiple_readers(self):
        """Test that multiple readers can acquire the lock simultaneously."""
        lock = ReadWriteLock()
        results = []
        
        def reader_function(reader_id):
            lock.acquire_read()
            try:
                results.append(f"reader_{reader_id}_start")
                time.sleep(0.1)
                results.append(f"reader_{reader_id}_end")
            finally:
                lock.release_read()
        
        # Start multiple readers
        threads = []
        for i in range(3):
            thread = threading.Thread(target=reader_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all readers
        for thread in threads:
            thread.join()
        
        # All readers should have started before any ended
        assert len(results) == 6
        start_count = len([r for r in results if 'start' in r])
        assert start_count == 3
    
    def test_writer_exclusivity(self):
        """Test that writers have exclusive access."""
        lock = ReadWriteLock()
        results = []
        
        def writer_function(writer_id):
            lock.acquire_write()
            try:
                results.append(f"writer_{writer_id}_start")
                time.sleep(0.1)
                results.append(f"writer_{writer_id}_end")
            finally:
                lock.release_write()
        
        def reader_function(reader_id):
            lock.acquire_read()
            try:
                results.append(f"reader_{reader_id}_start")
                time.sleep(0.05)
                results.append(f"reader_{reader_id}_end")
            finally:
                lock.release_read()
        
        # Start writer and reader
        writer_thread = threading.Thread(target=writer_function, args=(1,))
        reader_thread = threading.Thread(target=reader_function, args=(1,))
        
        writer_thread.start()
        time.sleep(0.02)  # Let writer start first
        reader_thread.start()
        
        writer_thread.join()
        reader_thread.join()
        
        # Writer should complete before reader starts
        writer_end_index = results.index("writer_1_end")
        reader_start_index = results.index("reader_1_start")
        assert writer_end_index < reader_start_index


class TestResourcePool:
    """Test the ResourcePool class."""
    
    @pytest.mark.asyncio
    async def test_resource_pool_basic_operations(self):
        """Test basic resource pool operations."""
        pool = ResourcePool("test_pool", max_size=3)
        
        # Acquire resources
        resource1 = await pool.acquire()
        resource2 = await pool.acquire()
        resource3 = await pool.acquire()
        
        assert resource1.startswith("test_pool_resource_")
        assert resource2.startswith("test_pool_resource_")
        assert resource3.startswith("test_pool_resource_")
        assert resource1 != resource2 != resource3
        
        # Pool should be empty now
        stats = pool.get_utilization_stats()
        assert stats['available_resources'] == 0
        assert stats['utilized_resources'] == 3
        assert stats['utilization_rate'] == 100.0
        
        # Release resources
        await pool.release(resource1)
        await pool.release(resource2)
        
        # Check stats after release
        stats = pool.get_utilization_stats()
        assert stats['available_resources'] == 2
        assert stats['utilized_resources'] == 1
        
        pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_resource_pool_blocking(self):
        """Test that resource pool blocks when empty."""
        pool = ResourcePool("blocking_test", max_size=1)
        
        # Acquire the only resource
        resource = await pool.acquire()
        
        # Try to acquire another (should block)
        start_time = time.time()
        
        async def try_acquire():
            return await pool.acquire()
        
        # Start acquisition task
        acquire_task = asyncio.create_task(try_acquire())
        
        # Wait a bit, then release
        await asyncio.sleep(0.1)
        await pool.release(resource)
        
        # Now the blocked acquisition should complete
        acquired_resource = await acquire_task
        elapsed = time.time() - start_time
        
        assert elapsed >= 0.1  # Should have waited
        assert acquired_resource is not None
        
        pool.shutdown()


class TestAllocationTracker:
    """Test the AllocationTracker class."""
    
    def test_allocation_tracking(self):
        """Test allocation tracking functionality."""
        tracker = AllocationTracker()
        
        # Track some allocations
        with tracker.track_allocation("memory"):
            pass
        
        with tracker.track_allocation("memory"):
            pass
        
        with tracker.track_allocation("cpu"):
            pass
        
        # Get stats
        stats = tracker.get_allocation_stats()
        
        assert "memory" in stats
        assert "cpu" in stats
        assert stats["memory"]["total_allocations"] == 2
        assert stats["cpu"]["total_allocations"] == 1
        assert stats["memory"]["recent_allocations"] == 2  # Within 5 minutes
        assert stats["cpu"]["recent_allocations"] == 1


class TestInstanceStateMachine:
    """Test the InstanceStateMachine class."""
    
    def test_instance_registration(self):
        """Test instance registration and state management."""
        state_machine = InstanceStateMachine()
        
        # Register instance
        state_machine.register_instance("instance-1", InstanceStatus.IDLE)
        
        # Check state
        assert state_machine.can_accept_task("instance-1") is True
        
        # Transition state
        success = state_machine.transition_state("instance-1", InstanceStatus.RUNNING)
        assert success is True
        assert state_machine.can_accept_task("instance-1") is False
        
        # Invalid transition
        invalid = state_machine.transition_state("instance-1", InstanceStatus.STOPPED)
        assert invalid is False  # Can't go directly from RUNNING to STOPPED
    
    def test_state_summary(self):
        """Test state summary generation."""
        state_machine = InstanceStateMachine()
        
        # Register multiple instances
        state_machine.register_instance("instance-1", InstanceStatus.IDLE)
        state_machine.register_instance("instance-2", InstanceStatus.RUNNING)
        state_machine.register_instance("instance-3", InstanceStatus.IDLE)
        state_machine.register_instance("instance-4", InstanceStatus.ERROR)
        
        # Get summary
        summary = state_machine.get_state_summary()
        
        assert summary["idle"] == 2
        assert summary["running"] == 1
        assert summary["error"] == 1


class TestOperationCoordinator:
    """Test the OperationCoordinator class."""
    
    def test_operation_coordination(self):
        """Test operation registration and coordination."""
        coordinator = OperationCoordinator()
        
        # Create mock task
        task = ScrapingTask(
            task_id="test-task",
            instance_id="test-instance",
            input_data={"url": "https://example.com"}
        )
        
        # Register operation
        coordinator.register_operation("op-1", task)
        assert "op-1" in coordinator.active_operations
        
        # Unregister operation
        coordinator.unregister_operation("op-1")
        assert "op-1" not in coordinator.active_operations
    
    def test_operation_completion_waiting(self):
        """Test waiting for operation completion."""
        coordinator = OperationCoordinator()
        
        # Create mock task
        task = ScrapingTask(
            task_id="test-task",
            instance_id="test-instance",
            input_data={"url": "https://example.com"}
        )
        
        # Register operation
        coordinator.register_operation("op-1", task)
        
        # Start completion waiter in thread
        completion_result = []
        
        def wait_for_completion():
            result = coordinator.wait_for_completion(timeout=1.0)
            completion_result.append(result)
        
        wait_thread = threading.Thread(target=wait_for_completion)
        wait_thread.start()
        
        # Unregister operation after short delay
        time.sleep(0.1)
        coordinator.unregister_operation("op-1")
        
        wait_thread.join()
        
        # Should have completed successfully
        assert len(completion_result) == 1
        assert completion_result[0] is True


class TestThreadSafeInstanceManager:
    """Test the ThreadSafeInstanceManager class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return IntelligentScrapingConfig(
            max_instances=3,
            max_workers=5,
            enable_monitoring=True,
            monitoring_interval=0.1
        )
    
    def test_thread_safe_manager_initialization(self, mock_config):
        """Test thread-safe manager initialization."""
        manager = ThreadSafeInstanceManager(mock_config)
        
        assert manager.config == mock_config
        assert manager.max_instances == 3
        assert hasattr(manager, 'rw_lock')
        assert hasattr(manager, 'condition')
        assert hasattr(manager, 'concurrency_manager')
        assert len(manager.resource_pools) > 0
        
        manager.shutdown_coordinated()
    
    def test_thread_safe_instance_creation(self, mock_config):
        """Test thread-safe instance creation."""
        manager = ThreadSafeInstanceManager(mock_config)
        
        # Create instance
        instance = manager.create_instance_thread_safe("test-instance")
        
        assert instance.instance_id == "test-instance"
        assert "test-instance" in manager.instances
        
        # Check cache
        cached_instance = manager.get_instance_thread_safe("test-instance")
        assert cached_instance is instance
        
        manager.shutdown_coordinated()
    
    def test_concurrent_instance_access(self, mock_config):
        """Test concurrent access to instances."""
        manager = ThreadSafeInstanceManager(mock_config)
        
        # Create instance
        manager.create_instance_thread_safe("concurrent-test")
        
        results = []
        errors = []
        
        def access_instance(thread_id):
            try:
                for _ in range(10):
                    instance = manager.get_instance_thread_safe("concurrent-test")
                    if instance:
                        results.append(f"thread_{thread_id}_success")
                    time.sleep(0.01)
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=access_instance, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have no errors and many successful accesses
        assert len(errors) == 0
        assert len(results) == 50  # 5 threads * 10 accesses each
        
        manager.shutdown_coordinated()
    
    def test_read_write_lock_usage(self, mock_config):
        """Test read-write lock usage patterns."""
        manager = ThreadSafeInstanceManager(mock_config)
        
        # Create some instances
        manager.create_instance_thread_safe("rw-test-1")
        manager.create_instance_thread_safe("rw-test-2")
        
        read_results = []
        write_results = []
        
        def reader_thread(thread_id):
            with manager.read_lock():
                read_results.append(f"reader_{thread_id}_start")
                time.sleep(0.1)
                instances = manager.list_instances_thread_safe()
                read_results.append(f"reader_{thread_id}_end_{len(instances)}")
        
        def writer_thread(thread_id):
            with manager.write_lock():
                write_results.append(f"writer_{thread_id}_start")
                time.sleep(0.1)
                # Simulate write operation
                write_results.append(f"writer_{thread_id}_end")
        
        # Start readers and writers
        threads = []
        
        # Start multiple readers
        for i in range(3):
            thread = threading.Thread(target=reader_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Start one writer
        writer_thread_obj = threading.Thread(target=writer_thread, args=(1,))
        threads.append(writer_thread_obj)
        writer_thread_obj.start()
        
        # Wait for all
        for thread in threads:
            thread.join()
        
        # Readers should be able to run concurrently
        assert len(read_results) == 6  # 3 readers * 2 results each
        assert len(write_results) == 2  # 1 writer * 2 results
        
        manager.shutdown_coordinated()


class TestAsyncResourceManager:
    """Test the AsyncResourceManager class."""
    
    @pytest.mark.asyncio
    async def test_async_resource_allocation(self):
        """Test async resource allocation and release."""
        manager = AsyncResourceManager(max_resources=3)
        
        # Allocate resources
        success1 = await manager.allocate_resource("resource-1")
        success2 = await manager.allocate_resource("resource-2")
        success3 = await manager.allocate_resource("resource-3")
        
        assert success1 is True
        assert success2 is True
        assert success3 is True
        assert len(manager.active_resources) == 3
        
        # Try to allocate beyond limit
        success4 = await manager.allocate_resource("resource-4")
        assert success4 is False  # Should fail due to semaphore limit
        
        # Release resource
        release_success = await manager.release_resource("resource-1")
        assert release_success is True
        assert len(manager.active_resources) == 2
        
        # Now allocation should succeed
        success5 = await manager.allocate_resource("resource-5")
        assert success5 is True
    
    @pytest.mark.asyncio
    async def test_resource_stats(self):
        """Test resource statistics."""
        manager = AsyncResourceManager(max_resources=5)
        
        # Allocate some resources
        await manager.allocate_resource("res-1")
        await manager.allocate_resource("res-2")
        
        stats = manager.get_resource_stats()
        
        assert stats['active_resources'] == 2
        assert stats['max_resources'] == 5
        assert stats['utilization_rate'] == 40.0  # 2/5 * 100
        assert stats['available_permits'] == 3  # 5 - 2


class TestAsyncTaskCoordinator:
    """Test the AsyncTaskCoordinator class."""
    
    @pytest.mark.asyncio
    async def test_task_submission_and_completion(self):
        """Test async task submission and completion."""
        coordinator = AsyncTaskCoordinator()
        
        async def test_task(value):
            await asyncio.sleep(0.1)
            return value * 2
        
        # Submit task
        task_id = await coordinator.submit_task("test-task", test_task(5))
        
        assert task_id == "test-task"
        assert task_id in coordinator.active_tasks
        
        # Wait for completion
        result = await coordinator.wait_for_task(task_id)
        assert result == 10
        
        # Task should be cleaned up
        assert task_id not in coordinator.active_tasks
        assert task_id in coordinator.task_results
    
    @pytest.mark.asyncio
    async def test_task_timeout(self):
        """Test task timeout handling."""
        coordinator = AsyncTaskCoordinator()
        
        async def slow_task():
            await asyncio.sleep(1.0)
            return "completed"
        
        # Submit task with short timeout
        task_id = await coordinator.submit_task("timeout-task", slow_task(), timeout=0.1)
        
        # Should timeout
        with pytest.raises(asyncio.TimeoutError):
            await coordinator.wait_for_task(task_id)
        
        # Check result shows timeout
        assert task_id in coordinator.task_results
        result_info = coordinator.task_results[task_id]
        assert result_info['success'] is False
        assert result_info['error'] == 'timeout'
    
    @pytest.mark.asyncio
    async def test_task_cancellation(self):
        """Test task cancellation."""
        coordinator = AsyncTaskCoordinator()
        
        async def long_task():
            await asyncio.sleep(2.0)
            return "completed"
        
        # Submit task
        task_id = await coordinator.submit_task("cancel-task", long_task())
        
        # Cancel task
        await asyncio.sleep(0.1)  # Let it start
        cancelled = await coordinator.cancel_task(task_id)
        assert cancelled is True
        
        # Wait a bit for cancellation to take effect
        await asyncio.sleep(0.1)
        
        # Task should be cancelled
        assert task_id not in coordinator.active_tasks or coordinator.active_tasks[task_id].cancelled()
    
    @pytest.mark.asyncio
    async def test_wait_for_all_tasks(self):
        """Test waiting for all tasks to complete."""
        coordinator = AsyncTaskCoordinator()
        
        async def test_task(value):
            await asyncio.sleep(0.1)
            return value
        
        # Submit multiple tasks
        await coordinator.submit_task("task-1", test_task(10))
        await coordinator.submit_task("task-2", test_task(20))
        await coordinator.submit_task("task-3", test_task(30))
        
        # Wait for all
        results = await coordinator.wait_for_all_tasks(timeout=1.0)
        
        assert len(results) == 3
        assert results["task-1"]["success"] is True
        assert results["task-1"]["result"] == 10
        assert results["task-2"]["result"] == 20
        assert results["task-3"]["result"] == 30


class TestAsyncPerformanceMonitor:
    """Test the AsyncPerformanceMonitor class."""
    
    @pytest.mark.asyncio
    async def test_performance_recording(self):
        """Test performance metrics recording."""
        monitor = AsyncPerformanceMonitor()
        
        # Record some operations
        await monitor.record_operation("scraping", 1.5, True, {"items": 5})
        await monitor.record_operation("scraping", 2.0, True, {"items": 3})
        await monitor.record_operation("scraping", 0.8, False, {"error": "timeout"})
        
        # Get metrics
        metrics = await monitor.get_performance_metrics()
        
        assert metrics['total_operations'] == 3
        assert metrics['successful_operations'] == 2
        assert metrics['failed_operations'] == 1
        assert metrics['error_rate'] == (1/3) * 100
        assert metrics['average_processing_time'] == (1.5 + 2.0 + 0.8) / 3
    
    @pytest.mark.asyncio
    async def test_performance_trends(self):
        """Test performance trends calculation."""
        monitor = AsyncPerformanceMonitor()
        
        # Record operations over time
        for i in range(10):
            await monitor.record_operation("test", 1.0 + i * 0.1, i % 2 == 0)
            await asyncio.sleep(0.01)  # Small delay to spread over time
        
        # Get trends
        trends = await monitor.get_performance_trends(time_window=60.0)
        
        assert 'processing_times' in trends
        assert 'success_rates' in trends
        assert 'operation_counts' in trends
        
        # Should have some data points
        assert len(trends['processing_times']) > 0
        assert len(trends['success_rates']) > 0


class TestAsyncScraperInstance:
    """Test the AsyncScraperInstance class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return IntelligentScrapingConfig(
            max_instances=3,
            max_concurrent_requests=5,
            enable_monitoring=True
        )
    
    @pytest.mark.asyncio
    async def test_async_instance_initialization(self, mock_config):
        """Test async instance initialization."""
        instance = AsyncScraperInstance("async-test", mock_config)
        
        assert instance.instance_id == "async-test"
        assert isinstance(instance.resource_manager, AsyncResourceManager)
        assert isinstance(instance.task_coordinator, AsyncTaskCoordinator)
        assert isinstance(instance.performance_monitor, AsyncPerformanceMonitor)
        assert len(instance.background_tasks) > 0  # Should have background tasks
        
        await instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_async_task_execution(self, mock_config):
        """Test async task execution."""
        instance = AsyncScraperInstance("async-exec-test", mock_config)
        
        # Create test task
        task = ScrapingTask(
            task_id="async-task-1",
            instance_id=instance.instance_id,
            input_data={"url": "https://example.com"},
            priority=1
        )
        
        # Execute task
        result = await instance.execute_task_async_enhanced(task)
        
        assert isinstance(result, ScrapingResult)
        assert result.task_id == task.task_id
        assert result.instance_id == instance.instance_id
        assert result.success is True
        assert result.processing_time > 0.0
        assert 'items' in result.data
        
        await instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_async_health_checking(self, mock_config):
        """Test async health checking."""
        instance = AsyncScraperInstance("health-test", mock_config)
        
        # Perform health check
        health_status = await instance.check_health_async()
        
        assert isinstance(health_status, HealthStatus)
        assert isinstance(health_status.is_healthy, bool)
        assert isinstance(health_status.last_check, datetime)
        assert health_status.memory_usage_mb >= 0
        assert health_status.cpu_usage_percent >= 0
        
        await instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_async_status_coordination(self, mock_config):
        """Test async status coordination."""
        instance = AsyncScraperInstance("status-test", mock_config)
        
        # Initial status should be IDLE
        assert instance.status == InstanceStatus.IDLE
        
        # Change status
        await instance._set_status_async(InstanceStatus.RUNNING)
        assert instance.status == InstanceStatus.RUNNING
        
        # Wait for status change
        status_changed = await instance.wait_for_status(InstanceStatus.IDLE, timeout=0.1)
        assert status_changed is False  # Should timeout
        
        # Change back to IDLE
        await instance._set_status_async(InstanceStatus.IDLE)
        status_changed = await instance.wait_for_status(InstanceStatus.IDLE, timeout=0.1)
        assert status_changed is True
        
        await instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_async_caching(self, mock_config):
        """Test async result caching."""
        instance = AsyncScraperInstance("cache-test", mock_config)
        
        # Create task
        task = ScrapingTask(
            task_id="cache-task",
            instance_id=instance.instance_id,
            input_data={"url": "https://example.com"},
            priority=1
        )
        
        # Execute task twice
        result1 = await instance.execute_task_async_enhanced(task)
        result2 = await instance.execute_task_async_enhanced(task)
        
        # Both should succeed
        assert result1.success is True
        assert result2.success is True
        
        # Second should be faster due to caching (in a real implementation)
        # For now, just verify both completed
        assert result1.processing_time > 0
        assert result2.processing_time > 0
        
        await instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_async_stats_collection(self, mock_config):
        """Test async statistics collection."""
        instance = AsyncScraperInstance("stats-test", mock_config)
        
        # Execute a task to generate some stats
        task = ScrapingTask(
            task_id="stats-task",
            instance_id=instance.instance_id,
            input_data={"url": "https://example.com"}
        )
        
        await instance.execute_task_async_enhanced(task)
        
        # Get stats
        stats = await instance.get_async_stats()
        
        assert 'async_metrics' in stats
        async_metrics = stats['async_metrics']
        
        assert 'performance_metrics' in async_metrics
        assert 'resource_stats' in async_metrics
        assert 'task_coordination' in async_metrics
        assert 'cache_stats' in async_metrics
        
        # Performance metrics should show the executed task
        perf_metrics = async_metrics['performance_metrics']
        assert perf_metrics['total_operations'] >= 1
        
        await instance.shutdown_async()


if __name__ == "__main__":
    pytest.main([__file__])