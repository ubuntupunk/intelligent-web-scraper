"""
Unit tests for enhanced concurrency manager with performance optimizations.
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from intelligent_web_scraper.concurrency.concurrency_manager import (
    DynamicThreadPool,
    LRUCache,
    ConcurrentOperation,
    OperationStatus,
    ResourceType,
    ResourceUsage,
    ConcurrencyManager
)
from intelligent_web_scraper.config import IntelligentScrapingConfig


class TestDynamicThreadPool:
    """Test dynamic thread pool scaling functionality."""
    
    def test_initialization(self):
        """Test dynamic thread pool initialization."""
        pool = DynamicThreadPool(min_workers=2, max_workers=10, scale_factor=1.5)
        
        assert pool.min_workers == 2
        assert pool.max_workers == 10
        assert pool.scale_factor == 1.5
        assert pool.current_workers == 2
        assert pool.pending_tasks == 0
        assert pool.completed_tasks == 0
    
    def test_task_submission(self):
        """Test task submission and tracking."""
        pool = DynamicThreadPool(min_workers=2, max_workers=10)
        
        def simple_task():
            time.sleep(0.1)
            return "completed"
        
        future = pool.submit(simple_task)
        result = future.result(timeout=1.0)
        
        assert result == "completed"
        assert pool.completed_tasks == 1
    
    def test_get_stats(self):
        """Test thread pool statistics."""
        pool = DynamicThreadPool(min_workers=2, max_workers=10)
        
        stats = pool.get_stats()
        
        assert 'current_workers' in stats
        assert 'min_workers' in stats
        assert 'max_workers' in stats
        assert 'pending_tasks' in stats
        assert 'completed_tasks' in stats
        assert 'utilization_rate' in stats


class TestLRUCache:
    """Test LRU cache functionality."""
    
    def test_initialization(self):
        """Test LRU cache initialization."""
        cache = LRUCache(max_size=100)
        
        assert cache.max_size == 100
        assert len(cache.cache) == 0
        assert len(cache.access_order) == 0
    
    def test_put_and_get(self):
        """Test putting and getting values from cache."""
        cache = LRUCache(max_size=3)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("nonexistent") is None
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = LRUCache(max_size=2)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_clear(self):
        """Test cache clearing."""
        cache = LRUCache(max_size=10)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.clear()
        
        assert len(cache.cache) == 0
        assert len(cache.access_order) == 0
        assert cache.get("key1") is None
    
    def test_get_stats(self):
        """Test cache statistics."""
        cache = LRUCache(max_size=10)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        stats = cache.get_stats()
        
        assert stats['size'] == 2
        assert stats['max_size'] == 10
        assert stats['utilization_rate'] == 20.0


class TestResourceUsage:
    """Test resource usage tracking."""
    
    def test_initialization(self):
        """Test resource usage initialization."""
        resource = ResourceUsage(ResourceType.THREAD, max_usage=10)
        
        assert resource.resource_type == ResourceType.THREAD
        assert resource.current_usage == 0
        assert resource.max_usage == 10
        assert resource.peak_usage == 0
        assert resource.total_allocated == 0
        assert resource.total_released == 0
    
    def test_resource_allocation(self):
        """Test resource allocation."""
        resource = ResourceUsage(ResourceType.THREAD, max_usage=10)
        
        # Test successful allocation
        assert resource.allocate(3) == True
        assert resource.current_usage == 3
        assert resource.total_allocated == 3
        assert resource.peak_usage == 3
        
        # Test allocation at capacity
        assert resource.allocate(7) == True
        assert resource.current_usage == 10
        
        # Test allocation beyond capacity
        assert resource.allocate(1) == False
        assert resource.current_usage == 10
    
    def test_resource_release(self):
        """Test resource release."""
        resource = ResourceUsage(ResourceType.THREAD, max_usage=10)
        
        resource.allocate(5)
        resource.release(2)
        
        assert resource.current_usage == 3
        assert resource.total_released == 2
        
        # Test release more than allocated
        resource.release(10)
        assert resource.current_usage == 0
    
    def test_utilization_rate(self):
        """Test utilization rate calculation."""
        resource = ResourceUsage(ResourceType.THREAD, max_usage=10)
        
        # Test 0% utilization
        assert resource.get_utilization_rate() == 0.0
        
        # Test 50% utilization
        resource.allocate(5)
        assert resource.get_utilization_rate() == 50.0
        
        # Test 100% utilization
        resource.allocate(5)
        assert resource.get_utilization_rate() == 100.0


class TestEnhancedConcurrencyManager:
    """Test enhanced concurrency manager functionality."""
    
    def test_initialization(self):
        """Test enhanced concurrency manager initialization."""
        config = IntelligentScrapingConfig()
        manager = ConcurrencyManager(config)
        
        assert manager.config == config
        assert hasattr(manager, 'dynamic_thread_pool')
        assert hasattr(manager, 'operation_cache')
        assert hasattr(manager, 'resources')
        assert hasattr(manager, 'performance_history')
    
    def test_cache_key_generation(self):
        """Test cache key generation for function calls."""
        config = IntelligentScrapingConfig()
        manager = ConcurrencyManager(config)
        
        def test_func(arg1, arg2, kwarg1=None):
            return f"{arg1}_{arg2}_{kwarg1}"
        
        key1 = manager._generate_cache_key(test_func, ("a", "b"), {"kwarg1": "c"})
        key2 = manager._generate_cache_key(test_func, ("a", "b"), {"kwarg1": "c"})
        key3 = manager._generate_cache_key(test_func, ("x", "y"), {"kwarg1": "z"})
        
        # Same inputs should generate same key
        assert key1 == key2
        
        # Different inputs should generate different keys
        assert key1 != key3
    
    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        config = IntelligentScrapingConfig()
        manager = ConcurrencyManager(config)
        
        # Test with no cache activity
        assert manager._calculate_cache_hit_rate() == 0.0
        
        # Test with cache activity
        manager.metrics['cache_hits'] = 80
        manager.metrics['cache_misses'] = 20
        
        hit_rate = manager._calculate_cache_hit_rate()
        assert hit_rate == 80.0
    
    def test_enhanced_operation_submission(self):
        """Test enhanced operation submission with caching."""
        config = IntelligentScrapingConfig()
        manager = ConcurrencyManager(config)
        
        def test_operation(x, y):
            return x + y
        
        # First submission should execute and cache
        op_id1 = manager.submit_sync_operation_enhanced(
            test_operation, 5, 3, 
            operation_type="test_op",
            enable_caching=True
        )
        
        # Wait for completion
        time.sleep(0.1)
        
        # Second submission with same args should hit cache
        op_id2 = manager.submit_sync_operation_enhanced(
            test_operation, 5, 3,
            operation_type="test_op", 
            enable_caching=True
        )
        
        assert op_id1 != op_id2
        assert manager.metrics['cache_hits'] >= 1
    
    def test_enhanced_performance_metrics(self):
        """Test enhanced performance metrics collection."""
        config = IntelligentScrapingConfig()
        manager = ConcurrencyManager(config)
        
        enhanced_metrics = manager.get_enhanced_performance_metrics()
        
        assert 'enhanced_metrics' in enhanced_metrics
        assert 'cache_performance' in enhanced_metrics['enhanced_metrics']
        assert 'dynamic_pool_stats' in enhanced_metrics['enhanced_metrics']
        assert 'recent_performance' in enhanced_metrics['enhanced_metrics']
    
    def test_recent_performance_stats(self):
        """Test recent performance statistics calculation."""
        config = IntelligentScrapingConfig()
        manager = ConcurrencyManager(config)
        
        # Add some recent performance data
        current_time = datetime.utcnow()
        
        # Add successful operations
        for i in range(8):
            manager.performance_history.append({
                'operation_id': f"success_{i}",
                'operation_type': "test_task",
                'duration': 2.0,
                'success': True,
                'timestamp': current_time - timedelta(seconds=i * 10)
            })
        
        # Add failed operations
        for i in range(2):
            manager.performance_history.append({
                'operation_id': f"failed_{i}",
                'operation_type': "test_task",
                'duration': 5.0,
                'success': False,
                'timestamp': current_time - timedelta(seconds=i * 15)
            })
        
        stats = manager._get_recent_performance_stats()
        
        assert stats['total_operations'] == 10
        assert stats['successful_operations'] == 8
        assert stats['failed_operations'] == 2
        assert stats['success_rate'] == 80.0
        assert 'average_duration' in stats
        assert 'operations_per_second' in stats
    
    def test_resource_allocation_and_release(self):
        """Test resource allocation and release."""
        config = IntelligentScrapingConfig()
        manager = ConcurrencyManager(config)
        
        # Test thread resource allocation
        assert manager._allocate_resource(ResourceType.THREAD, 2) == True
        
        thread_resource = manager.resources[ResourceType.THREAD]
        assert thread_resource.current_usage == 2
        
        # Test resource release
        manager._release_resource(ResourceType.THREAD, 1)
        assert thread_resource.current_usage == 1
    
    def test_shutdown(self):
        """Test graceful shutdown."""
        config = IntelligentScrapingConfig()
        manager = ConcurrencyManager(config)
        
        # Submit a quick operation
        def quick_op():
            return "done"
        
        op_id = manager.submit_sync_operation_enhanced(quick_op)
        
        # Shutdown should complete without errors
        manager.shutdown(wait=True, timeout=5.0)
        
        assert manager.shutdown_event.is_set()


if __name__ == "__main__":
    pytest.main([__file__])