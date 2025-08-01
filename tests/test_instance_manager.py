"""
Unit tests for the ScraperInstanceManager and ScraperInstance classes.

This module tests the instance management system including lifecycle management,
monitoring capabilities, thread safety, and performance tracking.
"""

import pytest
import asyncio
import threading
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from intelligent_web_scraper.agents.instance_manager import (
    ScraperInstanceManager,
    ScraperInstance,
    InstanceStatus,
    InstanceMetrics,
    HealthStatus,
    ScrapingTask,
    ScrapingResult
)
from intelligent_web_scraper.config import IntelligentScrapingConfig


class TestInstanceMetrics:
    """Test the InstanceMetrics class."""
    
    def test_metrics_initialization(self):
        """Test that metrics initialize with correct default values."""
        metrics = InstanceMetrics()
        
        assert metrics.requests_processed == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.total_processing_time == 0.0
        assert metrics.average_response_time == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.success_rate == 0.0
        assert metrics.throughput == 0.0
        assert len(metrics.memory_usage_history) == 0
        assert len(metrics.cpu_usage_history) == 0
        assert len(metrics.quality_scores) == 0
        assert isinstance(metrics.last_updated, datetime)
    
    def test_metrics_update(self):
        """Test metrics update functionality."""
        metrics = InstanceMetrics()
        
        # Update with new data
        new_metrics = {
            'requests': 5,
            'successful': 4,
            'failed': 1,
            'processing_time': 10.0,
            'memory_mb': 100.0,
            'cpu_percent': 25.0,
            'quality_score': 85.0
        }
        
        metrics.update(new_metrics)
        
        # Verify updates
        assert metrics.requests_processed == 5
        assert metrics.successful_requests == 4
        assert metrics.failed_requests == 1
        assert metrics.total_processing_time == 10.0
        assert len(metrics.memory_usage_history) == 1
        assert metrics.memory_usage_history[0] == 100.0
        assert len(metrics.cpu_usage_history) == 1
        assert metrics.cpu_usage_history[0] == 25.0
        assert len(metrics.quality_scores) == 1
        assert metrics.quality_scores[0] == 85.0
        
        # Verify calculated metrics
        assert metrics.success_rate == 80.0  # 4/5 * 100
        assert metrics.error_rate == 20.0    # 1/5 * 100
        assert metrics.average_response_time == 2.0  # 10.0/5
        assert metrics.throughput == 0.5  # 5/10.0
    
    def test_metrics_record_success(self):
        """Test recording successful operations."""
        metrics = InstanceMetrics()
        
        metrics.record_success(2.5, 90.0)
        
        assert metrics.requests_processed == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.total_processing_time == 2.5
        assert metrics.success_rate == 100.0
        assert metrics.error_rate == 0.0
        assert len(metrics.quality_scores) == 1
        assert metrics.quality_scores[0] == 90.0
    
    def test_metrics_record_error(self):
        """Test recording failed operations."""
        metrics = InstanceMetrics()
        
        metrics.record_error(1.5)
        
        assert metrics.requests_processed == 1
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 1
        assert metrics.total_processing_time == 1.5
        assert metrics.success_rate == 0.0
        assert metrics.error_rate == 100.0
    
    def test_metrics_history_limit(self):
        """Test that history lists maintain size limits."""
        metrics = InstanceMetrics()
        
        # Add more than 100 entries
        for i in range(150):
            metrics.update({
                'memory_mb': float(i),
                'cpu_percent': float(i),
                'quality_score': float(i)
            })
        
        # Verify limits are maintained
        assert len(metrics.memory_usage_history) == 100
        assert len(metrics.cpu_usage_history) == 100
        assert len(metrics.quality_scores) == 100
        
        # Verify oldest entries were removed
        assert metrics.memory_usage_history[0] == 50.0  # Should start from 50
        assert metrics.memory_usage_history[-1] == 149.0  # Should end at 149


class TestHealthStatus:
    """Test the HealthStatus class."""
    
    def test_health_status_creation(self):
        """Test health status creation with all fields."""
        health = HealthStatus(
            is_healthy=True,
            requires_restart=False,
            issues=["Minor warning"],
            memory_usage_mb=75.0,
            cpu_usage_percent=20.0,
            response_time_ms=150.0
        )
        
        assert health.is_healthy is True
        assert health.requires_restart is False
        assert len(health.issues) == 1
        assert health.issues[0] == "Minor warning"
        assert health.memory_usage_mb == 75.0
        assert health.cpu_usage_percent == 20.0
        assert health.response_time_ms == 150.0
        assert isinstance(health.last_check, datetime)
    
    def test_health_status_defaults(self):
        """Test health status with default values."""
        health = HealthStatus(is_healthy=False)
        
        assert health.is_healthy is False
        assert health.requires_restart is False
        assert len(health.issues) == 0
        assert health.memory_usage_mb == 0.0
        assert health.cpu_usage_percent == 0.0
        assert health.response_time_ms == 0.0


class TestScrapingTask:
    """Test the ScrapingTask dataclass."""
    
    def test_task_creation(self):
        """Test scraping task creation."""
        task_id = str(uuid.uuid4())
        instance_id = "test-instance"
        input_data = {"url": "https://example.com", "max_results": 10}
        
        task = ScrapingTask(
            task_id=task_id,
            instance_id=instance_id,
            input_data=input_data,
            priority=5,
            requires_heavy_processing=True,
            timeout_seconds=600.0
        )
        
        assert task.task_id == task_id
        assert task.instance_id == instance_id
        assert task.input_data == input_data
        assert task.priority == 5
        assert task.requires_heavy_processing is True
        assert task.timeout_seconds == 600.0
        assert isinstance(task.created_at, datetime)
        assert task.started_at is None
        assert task.completed_at is None
    
    def test_task_defaults(self):
        """Test task creation with default values."""
        task = ScrapingTask(
            task_id="test-task",
            instance_id="test-instance",
            input_data={}
        )
        
        assert task.priority == 0
        assert task.requires_heavy_processing is False
        assert task.timeout_seconds == 300.0


class TestScrapingResult:
    """Test the ScrapingResult dataclass."""
    
    def test_result_creation_success(self):
        """Test successful result creation."""
        result = ScrapingResult(
            task_id="test-task",
            instance_id="test-instance",
            success=True,
            data={"items": [{"title": "Test"}]},
            processing_time=2.5,
            quality_score=85.0,
            metadata={"source": "test"}
        )
        
        assert result.task_id == "test-task"
        assert result.instance_id == "test-instance"
        assert result.success is True
        assert result.data == {"items": [{"title": "Test"}]}
        assert result.error is None
        assert result.processing_time == 2.5
        assert result.quality_score == 85.0
        assert result.metadata == {"source": "test"}
        assert isinstance(result.completed_at, datetime)
    
    def test_result_creation_error(self):
        """Test error result creation."""
        result = ScrapingResult(
            task_id="test-task",
            instance_id="test-instance",
            success=False,
            error="Connection timeout",
            processing_time=1.0
        )
        
        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.processing_time == 1.0
        assert result.quality_score == 0.0
        assert len(result.data) == 0
        assert len(result.metadata) == 0


class TestScraperInstance:
    """Test the ScraperInstance class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return IntelligentScrapingConfig(
            max_instances=5,
            enable_monitoring=True,
            monitoring_interval=1.0
        )
    
    def test_instance_initialization(self, mock_config):
        """Test that instance initializes correctly."""
        instance_id = "test-instance-123"
        instance = ScraperInstance(instance_id, mock_config)
        
        assert instance.instance_id == instance_id
        assert instance.config == mock_config
        assert instance.status == InstanceStatus.IDLE
        assert isinstance(instance.created_at, datetime)
        assert isinstance(instance.last_activity, datetime)
        assert isinstance(instance.metrics, InstanceMetrics)
        assert isinstance(instance.health_status, HealthStatus)
        assert instance.current_task is None
        assert len(instance.task_history) == 0
        assert instance.health_status.is_healthy is True
    
    def test_instance_uptime_calculation(self, mock_config):
        """Test uptime calculation."""
        instance = ScraperInstance("test-instance", mock_config)
        
        # Wait a small amount of time
        time.sleep(0.1)
        
        uptime = instance.get_uptime()
        assert uptime > 0.0
        assert uptime < 1.0  # Should be less than 1 second
    
    def test_instance_idle_time_calculation(self, mock_config):
        """Test idle time calculation."""
        instance = ScraperInstance("test-instance", mock_config)
        
        # Wait and update activity
        time.sleep(0.1)
        instance.update_activity()
        
        # Check idle time is small
        idle_time = instance.get_idle_time()
        assert idle_time < 0.1
    
    def test_instance_status_management(self, mock_config):
        """Test status management with thread safety."""
        instance = ScraperInstance("test-instance", mock_config)
        
        # Test status changes
        assert instance.status == InstanceStatus.IDLE
        
        instance.set_status(InstanceStatus.RUNNING)
        assert instance.status == InstanceStatus.RUNNING
        
        instance.set_status(InstanceStatus.ERROR)
        assert instance.status == InstanceStatus.ERROR
        
        instance.set_status(InstanceStatus.STOPPED)
        assert instance.status == InstanceStatus.STOPPED
    
    @patch('psutil.Process')
    def test_system_metrics_collection(self, mock_process, mock_config):
        """Test system metrics collection."""
        # Mock psutil.Process
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100MB in bytes
        
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value = mock_memory_info
        mock_process_instance.cpu_percent.return_value = 25.0
        mock_process_instance.num_threads.return_value = 5
        
        mock_process.return_value = mock_process_instance
        
        instance = ScraperInstance("test-instance", mock_config)
        metrics = instance.collect_system_metrics()
        
        assert metrics['memory_mb'] == 100.0
        assert metrics['cpu_percent'] == 25.0
        assert metrics['thread_count'] == 5
        assert 'uptime_seconds' in metrics
        assert 'idle_time_seconds' in metrics
    
    def test_metrics_update(self, mock_config):
        """Test metrics update functionality."""
        instance = ScraperInstance("test-instance", mock_config)
        
        # Update metrics
        operation_metrics = {
            'requests': 3,
            'successful': 2,
            'failed': 1,
            'processing_time': 5.0,
            'quality_score': 80.0
        }
        
        with patch.object(instance, 'collect_system_metrics') as mock_system:
            mock_system.return_value = {
                'memory_mb': 50.0,
                'cpu_percent': 15.0,
                'thread_count': 3,
                'uptime_seconds': 120.0,
                'idle_time_seconds': 10.0
            }
            
            instance.update_metrics(operation_metrics)
        
        # Verify metrics were updated
        assert instance.metrics.requests_processed == 3
        assert instance.metrics.successful_requests == 2
        assert instance.metrics.failed_requests == 1
        assert instance.metrics.total_processing_time == 5.0
        assert len(instance.metrics.quality_scores) == 1
        assert instance.metrics.quality_scores[0] == 80.0
        assert len(instance.metrics.memory_usage_history) == 1
        assert instance.metrics.memory_usage_history[0] == 50.0
    
    def test_health_check(self, mock_config):
        """Test health check functionality."""
        instance = ScraperInstance("test-instance", mock_config)
        
        with patch.object(instance, 'collect_system_metrics') as mock_system:
            # Test healthy instance
            mock_system.return_value = {
                'memory_mb': 100.0,  # Under 500MB threshold
                'cpu_percent': 30.0,  # Under 80% threshold
                'uptime_seconds': 120.0,
                'idle_time_seconds': 10.0
            }
            
            health = instance.check_health()
            
            assert health.is_healthy is True
            assert health.requires_restart is False
            assert len(health.issues) == 0
            assert health.memory_usage_mb == 100.0
            assert health.cpu_usage_percent == 30.0
    
    def test_health_check_high_memory(self, mock_config):
        """Test health check with high memory usage."""
        instance = ScraperInstance("test-instance", mock_config)
        
        with patch.object(instance, 'collect_system_metrics') as mock_system:
            # Test high memory usage
            mock_system.return_value = {
                'memory_mb': 1200.0,  # Over 1GB critical threshold
                'cpu_percent': 30.0,
                'uptime_seconds': 120.0,
                'idle_time_seconds': 10.0
            }
            
            health = instance.check_health()
            
            assert health.is_healthy is False
            assert health.requires_restart is True
            assert len(health.issues) > 0
            assert "High memory usage" in health.issues[0]
    
    def test_health_check_high_cpu(self, mock_config):
        """Test health check with high CPU usage."""
        instance = ScraperInstance("test-instance", mock_config)
        
        with patch.object(instance, 'collect_system_metrics') as mock_system:
            # Test high CPU usage
            mock_system.return_value = {
                'memory_mb': 100.0,
                'cpu_percent': 90.0,  # Over 80% threshold
                'uptime_seconds': 120.0,
                'idle_time_seconds': 10.0
            }
            
            health = instance.check_health()
            
            assert health.is_healthy is False
            assert health.requires_restart is False  # High CPU doesn't require restart
            assert len(health.issues) > 0
            assert "High CPU usage" in health.issues[0]
    
    def test_health_check_stuck_instance(self, mock_config):
        """Test health check for stuck instance."""
        instance = ScraperInstance("test-instance", mock_config)
        instance.set_status(InstanceStatus.RUNNING)  # Set as running
        
        with patch.object(instance, 'collect_system_metrics') as mock_system:
            # Test stuck instance (running but idle for too long)
            mock_system.return_value = {
                'memory_mb': 100.0,
                'cpu_percent': 30.0,
                'uptime_seconds': 120.0,
                'idle_time_seconds': 400.0  # Over 300s threshold
            }
            
            health = instance.check_health()
            
            assert health.is_healthy is False
            assert health.requires_restart is True
            assert len(health.issues) > 0
            assert "appears stuck" in health.issues[0]
    
    def test_health_check_high_error_rate(self, mock_config):
        """Test health check with high error rate."""
        instance = ScraperInstance("test-instance", mock_config)
        
        # Set high error rate
        instance.metrics.error_rate = 90.0
        
        with patch.object(instance, 'collect_system_metrics') as mock_system:
            mock_system.return_value = {
                'memory_mb': 100.0,
                'cpu_percent': 30.0,
                'uptime_seconds': 120.0,
                'idle_time_seconds': 10.0
            }
            
            health = instance.check_health()
            
            assert health.is_healthy is False
            assert health.requires_restart is True
            assert len(health.issues) > 0
            assert "High error rate" in health.issues[0]
    
    def test_task_execution_sync(self, mock_config):
        """Test synchronous task execution."""
        instance = ScraperInstance("test-instance", mock_config)
        
        task = ScrapingTask(
            task_id="test-task",
            instance_id=instance.instance_id,
            input_data={"url": "https://example.com"},
            priority=1
        )
        
        # Execute task
        result = instance.execute_task_sync(task)
        
        # Verify result
        assert isinstance(result, ScrapingResult)
        assert result.task_id == task.task_id
        assert result.instance_id == instance.instance_id
        assert result.success is True
        assert result.processing_time > 0.0
        assert result.quality_score > 0.0
        assert 'items' in result.data
        
        # Verify instance state
        assert instance.status == InstanceStatus.IDLE
        assert instance.current_task is None
        assert len(instance.task_history) == 1
        assert instance.task_history[0] == task
        
        # Verify metrics were updated
        assert instance.metrics.requests_processed == 1
        assert instance.metrics.successful_requests == 1
        assert instance.metrics.failed_requests == 0
    
    @pytest.mark.asyncio
    async def test_task_execution_async(self, mock_config):
        """Test asynchronous task execution."""
        instance = ScraperInstance("test-instance", mock_config)
        
        task = ScrapingTask(
            task_id="test-task-async",
            instance_id=instance.instance_id,
            input_data={"url": "https://example.com"},
            priority=2
        )
        
        # Execute task asynchronously
        result = await instance.execute_task_async(task)
        
        # Verify result
        assert isinstance(result, ScrapingResult)
        assert result.task_id == task.task_id
        assert result.instance_id == instance.instance_id
        assert result.success is True
        assert result.processing_time > 0.0
        assert result.quality_score > 0.0
        assert 'items' in result.data
        
        # Verify instance state
        assert instance.status == InstanceStatus.IDLE
        assert instance.current_task is None
        assert len(instance.task_history) == 1
    
    def test_task_execution_error_handling(self, mock_config):
        """Test task execution error handling."""
        instance = ScraperInstance("test-instance", mock_config)
        
        # Mock the simulation method to raise an error
        with patch.object(instance, '_simulate_scraping_work_sync') as mock_work:
            mock_work.side_effect = Exception("Simulated error")
            
            task = ScrapingTask(
                task_id="error-task",
                instance_id=instance.instance_id,
                input_data={"url": "https://example.com"}
            )
            
            result = instance.execute_task_sync(task)
            
            # Verify error result
            assert result.success is False
            assert result.error == "Simulated error"
            assert result.processing_time > 0.0
            assert result.quality_score == 0.0
            
            # Verify metrics were updated for error
            assert instance.metrics.requests_processed == 1
            assert instance.metrics.successful_requests == 0
            assert instance.metrics.failed_requests == 1
    
    def test_concurrent_task_execution_prevention(self, mock_config):
        """Test that concurrent task execution is prevented."""
        instance = ScraperInstance("test-instance", mock_config)
        
        # Set a current task
        task1 = ScrapingTask(
            task_id="task1",
            instance_id=instance.instance_id,
            input_data={}
        )
        instance.current_task = task1
        instance.set_status(InstanceStatus.RUNNING)
        
        # Try to execute another task
        task2 = ScrapingTask(
            task_id="task2",
            instance_id=instance.instance_id,
            input_data={}
        )
        
        with pytest.raises(RuntimeError, match="already executing a task"):
            instance.execute_task_sync(task2)
    
    def test_instance_stop(self, mock_config):
        """Test instance stop functionality."""
        instance = ScraperInstance("test-instance", mock_config)
        
        # Set instance as running
        instance.set_status(InstanceStatus.RUNNING)
        
        # Stop instance
        instance.stop()
        
        assert instance.status == InstanceStatus.STOPPED
    
    def test_instance_restart(self, mock_config):
        """Test instance restart functionality."""
        instance = ScraperInstance("test-instance", mock_config)
        
        # Add some metrics and set error status
        instance.metrics.requests_processed = 10
        instance.metrics.failed_requests = 5
        instance.set_status(InstanceStatus.ERROR)
        instance.health_status.is_healthy = False
        
        # Restart instance
        instance.restart()
        
        # Verify restart
        assert instance.status == InstanceStatus.IDLE
        assert instance.health_status.is_healthy is True
        assert instance.current_task is None
        assert instance.metrics.requests_processed == 0  # Metrics reset
        assert instance.metrics.failed_requests == 0
    
    def test_get_stats(self, mock_config):
        """Test getting comprehensive instance statistics."""
        instance = ScraperInstance("test-instance", mock_config)
        
        # Add some metrics
        instance.metrics.requests_processed = 15
        instance.metrics.successful_requests = 12
        instance.metrics.failed_requests = 3
        instance.metrics.quality_scores = [80.0, 85.0, 90.0]
        
        with patch.object(instance, 'collect_system_metrics') as mock_system:
            mock_system.return_value = {
                'memory_mb': 75.0,
                'cpu_percent': 20.0,
                'thread_count': 4,
                'uptime_seconds': 300.0,
                'idle_time_seconds': 30.0
            }
            
            stats = instance.get_stats()
        
        # Verify stats structure
        assert stats['instance_id'] == instance.instance_id
        assert stats['status'] == instance.status.value
        assert stats['uptime'] > 0.0
        assert stats['requests_processed'] == 15
        assert stats['success_rate'] == 80.0  # 12/15 * 100
        assert stats['error_rate'] == 20.0   # 3/15 * 100
        assert stats['memory_usage_mb'] == 75.0
        assert stats['cpu_usage_percent'] == 20.0
        assert isinstance(stats['last_activity'], datetime)
        assert stats['current_task'] is None
        assert 'health_status' in stats
        assert stats['quality_score_avg'] == 85.0  # (80+85+90)/3


class TestScraperInstanceManager:
    """Test the ScraperInstanceManager class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return IntelligentScrapingConfig(
            max_instances=3,
            enable_monitoring=True,
            monitoring_interval=0.1  # Fast interval for testing
        )
    
    def test_manager_initialization(self, mock_config):
        """Test that manager initializes correctly."""
        manager = ScraperInstanceManager(mock_config)
        
        assert manager.config == mock_config
        assert manager.max_instances == 3
        assert len(manager.instances) == 0
        assert manager.monitoring_enabled is True
        assert manager.monitoring_interval == 0.1
        assert manager.task_distribution_strategy == 'round_robin'
        assert manager._round_robin_index == 0
        assert isinstance(manager.global_metrics, dict)
    
    def test_create_instance_sync(self, mock_config):
        """Test synchronous instance creation."""
        manager = ScraperInstanceManager(mock_config)
        
        # Create instance
        instance = manager.create_instance("test-instance")
        
        assert isinstance(instance, ScraperInstance)
        assert instance.instance_id == "test-instance"
        assert len(manager.instances) == 1
        assert "test-instance" in manager.instances
        assert manager.global_metrics['instances_created'] == 1
    
    def test_create_instance_auto_id(self, mock_config):
        """Test instance creation with auto-generated ID."""
        manager = ScraperInstanceManager(mock_config)
        
        instance = manager.create_instance()
        
        assert isinstance(instance, ScraperInstance)
        assert instance.instance_id.startswith("scraper-")
        assert len(manager.instances) == 1
    
    def test_create_instance_max_limit(self, mock_config):
        """Test that max instance limit is enforced."""
        manager = ScraperInstanceManager(mock_config)
        
        # Create max instances
        for i in range(mock_config.max_instances):
            manager.create_instance(f"instance-{i}")
        
        # Try to create one more
        with pytest.raises(RuntimeError, match="Maximum instances"):
            manager.create_instance("overflow-instance")
    
    def test_create_duplicate_instance(self, mock_config):
        """Test that duplicate instance IDs are prevented."""
        manager = ScraperInstanceManager(mock_config)
        
        # Create first instance
        manager.create_instance("duplicate-test")
        
        # Try to create duplicate
        with pytest.raises(ValueError, match="already exists"):
            manager.create_instance("duplicate-test")
    
    @pytest.mark.asyncio
    async def test_create_instance_async(self, mock_config):
        """Test asynchronous instance creation."""
        manager = ScraperInstanceManager(mock_config)
        
        instance = await manager.create_instance_async("async-instance")
        
        assert isinstance(instance, ScraperInstance)
        assert instance.instance_id == "async-instance"
        assert len(manager.instances) == 1
        assert manager.global_metrics['instances_created'] == 1
    
    def test_get_instance(self, mock_config):
        """Test getting instance by ID."""
        manager = ScraperInstanceManager(mock_config)
        
        # Create instance
        created_instance = manager.create_instance("get-test")
        
        # Get instance
        retrieved_instance = manager.get_instance("get-test")
        
        assert retrieved_instance is created_instance
        assert retrieved_instance.instance_id == "get-test"
        
        # Test non-existent instance
        assert manager.get_instance("non-existent") is None
    
    def test_list_instances(self, mock_config):
        """Test listing all instances."""
        manager = ScraperInstanceManager(mock_config)
        
        # Create multiple instances
        instance1 = manager.create_instance("list-test-1")
        instance2 = manager.create_instance("list-test-2")
        
        instances = manager.list_instances()
        
        assert len(instances) == 2
        assert instance1 in instances
        assert instance2 in instances
    
    def test_get_available_instance_round_robin(self, mock_config):
        """Test getting available instance with round-robin distribution."""
        manager = ScraperInstanceManager(mock_config)
        
        # Create multiple idle instances
        instance1 = manager.create_instance("rr-test-1")
        instance2 = manager.create_instance("rr-test-2")
        instance3 = manager.create_instance("rr-test-3")
        
        # All should be idle by default
        assert instance1.status == InstanceStatus.IDLE
        assert instance2.status == InstanceStatus.IDLE
        assert instance3.status == InstanceStatus.IDLE
        
        # Get instances in round-robin order
        available1 = manager.get_available_instance()
        available2 = manager.get_available_instance()
        available3 = manager.get_available_instance()
        available4 = manager.get_available_instance()  # Should wrap around
        
        assert available1 == instance1
        assert available2 == instance2
        assert available3 == instance3
        assert available4 == instance1  # Wrapped around
    
    def test_get_available_instance_least_loaded(self, mock_config):
        """Test getting available instance with least-loaded distribution."""
        manager = ScraperInstanceManager(mock_config)
        manager.task_distribution_strategy = 'least_loaded'
        
        # Create instances with different loads
        instance1 = manager.create_instance("ll-test-1")
        instance2 = manager.create_instance("ll-test-2")
        
        # Set different request counts
        instance1.metrics.requests_processed = 10
        instance2.metrics.requests_processed = 5
        
        # Should get the less loaded instance
        available = manager.get_available_instance()
        assert available == instance2
    
    def test_get_available_instance_no_available(self, mock_config):
        """Test getting available instance when none are available."""
        manager = ScraperInstanceManager(mock_config)
        
        # Create instance and set as running
        instance = manager.create_instance("busy-test")
        instance.set_status(InstanceStatus.RUNNING)
        
        # Should return None
        available = manager.get_available_instance()
        assert available is None
    
    @pytest.mark.asyncio
    async def test_execute_task_with_specific_instance(self, mock_config):
        """Test executing task on specific instance."""
        manager = ScraperInstanceManager(mock_config)
        
        # Create instance
        instance = manager.create_instance("specific-test")
        
        # Create task
        task = ScrapingTask(
            task_id="specific-task",
            instance_id=instance.instance_id,
            input_data={"url": "https://example.com"}
        )
        
        # Execute task
        result = await manager.execute_task(task, instance.instance_id)
        
        assert isinstance(result, ScrapingResult)
        assert result.success is True
        assert result.instance_id == instance.instance_id
        assert manager.global_metrics['total_requests'] == 1
        assert manager.global_metrics['successful_requests'] == 1
    
    @pytest.mark.asyncio
    async def test_execute_task_auto_assign(self, mock_config):
        """Test executing task with automatic instance assignment."""
        manager = ScraperInstanceManager(mock_config)
        
        # Create instance
        instance = manager.create_instance("auto-test")
        
        # Create task
        task = ScrapingTask(
            task_id="auto-task",
            instance_id="",  # Will be assigned automatically
            input_data={"url": "https://example.com"}
        )
        
        # Execute task
        result = await manager.execute_task(task)
        
        assert isinstance(result, ScrapingResult)
        assert result.success is True
        assert result.instance_id == instance.instance_id
    
    @pytest.mark.asyncio
    async def test_execute_task_create_instance_if_needed(self, mock_config):
        """Test that new instance is created if none available."""
        manager = ScraperInstanceManager(mock_config)
        
        # No instances created yet
        assert len(manager.instances) == 0
        
        # Create task
        task = ScrapingTask(
            task_id="create-task",
            instance_id="",
            input_data={"url": "https://example.com"}
        )
        
        # Execute task - should create instance automatically
        result = await manager.execute_task(task)
        
        assert isinstance(result, ScrapingResult)
        assert result.success is True
        assert len(manager.instances) == 1
    
    @pytest.mark.asyncio
    async def test_execute_task_max_instances_reached(self, mock_config):
        """Test task execution when max instances reached and all busy."""
        manager = ScraperInstanceManager(mock_config)
        
        # Create max instances and set all as running
        for i in range(mock_config.max_instances):
            instance = manager.create_instance(f"busy-{i}")
            instance.set_status(InstanceStatus.RUNNING)
        
        # Create task
        task = ScrapingTask(
            task_id="overflow-task",
            instance_id="",
            input_data={"url": "https://example.com"}
        )
        
        # Should raise error
        with pytest.raises(RuntimeError, match="No available instances"):
            await manager.execute_task(task)
    
    @pytest.mark.asyncio
    async def test_execute_task_nonexistent_instance(self, mock_config):
        """Test executing task on non-existent instance."""
        manager = ScraperInstanceManager(mock_config)
        
        task = ScrapingTask(
            task_id="nonexistent-task",
            instance_id="nonexistent",
            input_data={}
        )
        
        with pytest.raises(ValueError, match="not found"):
            await manager.execute_task(task, "nonexistent")
    
    def test_remove_instance(self, mock_config):
        """Test removing an instance."""
        manager = ScraperInstanceManager(mock_config)
        
        # Create instance
        instance = manager.create_instance("remove-test")
        assert len(manager.instances) == 1
        
        # Remove instance
        result = manager.remove_instance("remove-test")
        
        assert result is True
        assert len(manager.instances) == 0
        assert instance.status == InstanceStatus.STOPPED
    
    def test_remove_nonexistent_instance(self, mock_config):
        """Test removing non-existent instance."""
        manager = ScraperInstanceManager(mock_config)
        
        result = manager.remove_instance("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_restart_instance(self, mock_config):
        """Test restarting an instance."""
        manager = ScraperInstanceManager(mock_config)
        
        # Create instance and set some state
        instance = manager.create_instance("restart-test")
        instance.set_status(InstanceStatus.ERROR)
        instance.metrics.requests_processed = 10
        
        # Restart instance
        result = await manager.restart_instance("restart-test")
        
        assert result is True
        assert instance.status == InstanceStatus.IDLE
        assert instance.metrics.requests_processed == 0  # Metrics reset
        assert manager.global_metrics['instances_restarted'] == 1
    
    @pytest.mark.asyncio
    async def test_restart_nonexistent_instance(self, mock_config):
        """Test restarting non-existent instance."""
        manager = ScraperInstanceManager(mock_config)
        
        result = await manager.restart_instance("nonexistent")
        assert result is False
    
    def test_perform_health_checks(self, mock_config):
        """Test performing health checks on all instances."""
        manager = ScraperInstanceManager(mock_config)
        
        # Create instances
        instance1 = manager.create_instance("health-test-1")
        instance2 = manager.create_instance("health-test-2")
        
        # Mock health check results
        with patch.object(instance1, 'check_health') as mock_health1, \
             patch.object(instance2, 'check_health') as mock_health2:
            
            mock_health1.return_value = HealthStatus(is_healthy=True)
            mock_health2.return_value = HealthStatus(is_healthy=False, requires_restart=True)
            
            health_results = manager.perform_health_checks()
        
        assert len(health_results) == 2
        assert health_results["health-test-1"].is_healthy is True
        assert health_results["health-test-2"].is_healthy is False
        assert health_results["health-test-2"].requires_restart is True
        assert manager.global_metrics['health_checks_performed'] == 2
    
    def test_start_stop_monitoring(self, mock_config):
        """Test starting and stopping monitoring thread."""
        manager = ScraperInstanceManager(mock_config)
        
        # Start monitoring
        manager.start_monitoring()
        
        assert manager.monitoring_active.is_set()
        assert manager.monitoring_thread is not None
        assert manager.monitoring_thread.is_alive()
        
        # Stop monitoring
        manager.stop_monitoring()
        
        assert not manager.monitoring_active.is_set()
        assert manager.shutdown_event.is_set()
    
    def test_monitoring_disabled(self, mock_config):
        """Test behavior when monitoring is disabled."""
        mock_config.enable_monitoring = False
        manager = ScraperInstanceManager(mock_config)
        
        # Start monitoring should do nothing
        manager.start_monitoring()
        
        assert manager.monitoring_thread is None
    
    def test_get_global_stats(self, mock_config):
        """Test getting global statistics."""
        manager = ScraperInstanceManager(mock_config)
        
        # Create instances with different states
        instance1 = manager.create_instance("stats-test-1")
        instance2 = manager.create_instance("stats-test-2")
        instance3 = manager.create_instance("stats-test-3")
        
        instance1.set_status(InstanceStatus.RUNNING)
        instance2.set_status(InstanceStatus.IDLE)
        instance3.set_status(InstanceStatus.ERROR)
        
        # Set some global metrics
        manager.global_metrics.update({
            'total_requests': 100,
            'successful_requests': 85,
            'failed_requests': 15,
            'total_processing_time': 250.0
        })
        
        stats = manager.get_global_stats()
        
        assert stats['total_instances'] == 3
        assert stats['active_instances'] == 1
        assert stats['idle_instances'] == 1
        assert stats['error_instances'] == 1
        assert stats['max_instances'] == mock_config.max_instances
        assert stats['total_requests'] == 100
        assert stats['successful_requests'] == 85
        assert stats['failed_requests'] == 15
        assert stats['success_rate'] == 85.0
        assert stats['error_rate'] == 15.0
        assert stats['average_processing_time'] == 2.5  # 250/100
        assert stats['monitoring_enabled'] is True
    
    def test_get_detailed_report(self, mock_config):
        """Test getting detailed report."""
        manager = ScraperInstanceManager(mock_config)
        
        # Create instance
        instance = manager.create_instance("report-test")
        
        report = manager.get_detailed_report()
        
        assert 'global_statistics' in report
        assert 'instance_statistics' in report
        assert 'health_status' in report
        assert 'configuration' in report
        assert 'report_generated_at' in report
        
        # Verify structure
        assert len(report['instance_statistics']) == 1
        assert len(report['health_status']) == 1
        assert report['configuration']['max_instances'] == mock_config.max_instances
        assert report['configuration']['monitoring_enabled'] is True
    
    def test_context_manager(self, mock_config):
        """Test using manager as context manager."""
        with ScraperInstanceManager(mock_config) as manager:
            # Create instance
            instance = manager.create_instance("context-test")
            assert len(manager.instances) == 1
            
            # Monitoring should be started
            if manager.monitoring_enabled:
                assert manager.monitoring_active.is_set()
        
        # After context exit, should be shutdown
        assert len(manager.instances) == 0
        assert not manager.monitoring_active.is_set()
    
    def test_shutdown(self, mock_config):
        """Test manager shutdown."""
        manager = ScraperInstanceManager(mock_config)
        
        # Create instances and start monitoring
        instance1 = manager.create_instance("shutdown-test-1")
        instance2 = manager.create_instance("shutdown-test-2")
        manager.start_monitoring()
        
        assert len(manager.instances) == 2
        assert manager.monitoring_active.is_set()
        
        # Shutdown
        manager.shutdown()
        
        # Verify shutdown
        assert len(manager.instances) == 0
        assert not manager.monitoring_active.is_set()
        assert instance1.status == InstanceStatus.STOPPED
        assert instance2.status == InstanceStatus.STOPPED


if __name__ == "__main__":
    pytest.main([__file__])