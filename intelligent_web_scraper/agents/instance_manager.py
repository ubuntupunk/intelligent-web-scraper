"""
Scraper Instance Management System.

This module implements comprehensive instance management and monitoring
for the Intelligent Web Scraper, demonstrating advanced patterns for
managing concurrent operations with proper thread safety and monitoring.
"""

import asyncio
import threading
import time
import uuid
import os
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import logging

from pydantic import BaseModel, Field

from ..config import IntelligentScrapingConfig


logger = logging.getLogger(__name__)


class InstanceStatus(Enum):
    """Status enumeration for scraper instances."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"


class HealthStatus(BaseModel):
    """Health status for a scraper instance."""
    is_healthy: bool = Field(..., description="Whether the instance is healthy")
    requires_restart: bool = Field(default=False, description="Whether restart is needed")
    issues: List[str] = Field(default_factory=list, description="List of health issues")
    last_check: datetime = Field(default_factory=datetime.utcnow, description="Last health check time")
    memory_usage_mb: float = Field(default=0.0, description="Current memory usage in MB")
    cpu_usage_percent: float = Field(default=0.0, description="Current CPU usage percentage")
    response_time_ms: float = Field(default=0.0, description="Last response time in milliseconds")


class InstanceMetrics(BaseModel):
    """Comprehensive metrics for a scraper instance."""
    requests_processed: int = Field(default=0, description="Total requests processed")
    successful_requests: int = Field(default=0, description="Number of successful requests")
    failed_requests: int = Field(default=0, description="Number of failed requests")
    total_processing_time: float = Field(default=0.0, description="Total processing time in seconds")
    average_response_time: float = Field(default=0.0, description="Average response time in seconds")
    memory_usage_history: List[float] = Field(default_factory=list, description="Memory usage history")
    cpu_usage_history: List[float] = Field(default_factory=list, description="CPU usage history")
    error_rate: float = Field(default=0.0, description="Current error rate percentage")
    success_rate: float = Field(default=0.0, description="Current success rate percentage")
    throughput: float = Field(default=0.0, description="Requests per second")
    quality_scores: List[float] = Field(default_factory=list, description="Quality scores history")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last metrics update")
    
    def update(self, new_metrics: Dict[str, Any]) -> None:
        """Update metrics with new data."""
        # Update counters
        if 'requests' in new_metrics:
            self.requests_processed += new_metrics['requests']
        if 'successful' in new_metrics:
            self.successful_requests += new_metrics['successful']
        if 'failed' in new_metrics:
            self.failed_requests += new_metrics['failed']
        if 'processing_time' in new_metrics:
            self.total_processing_time += new_metrics['processing_time']
        
        # Update resource usage history (keep last 100 entries)
        if 'memory_mb' in new_metrics:
            self.memory_usage_history.append(new_metrics['memory_mb'])
            if len(self.memory_usage_history) > 100:
                self.memory_usage_history.pop(0)
        
        if 'cpu_percent' in new_metrics:
            self.cpu_usage_history.append(new_metrics['cpu_percent'])
            if len(self.cpu_usage_history) > 100:
                self.cpu_usage_history.pop(0)
        
        if 'quality_score' in new_metrics:
            self.quality_scores.append(new_metrics['quality_score'])
            if len(self.quality_scores) > 100:
                self.quality_scores.pop(0)
        
        # Calculate derived metrics
        self._calculate_derived_metrics()
        self.last_updated = datetime.utcnow()
    
    def _calculate_derived_metrics(self) -> None:
        """Calculate derived metrics from raw data."""
        # Calculate rates
        total_requests = self.requests_processed
        if total_requests > 0:
            self.success_rate = (self.successful_requests / total_requests) * 100
            self.error_rate = (self.failed_requests / total_requests) * 100
        
        # Calculate average response time
        if self.requests_processed > 0:
            self.average_response_time = self.total_processing_time / self.requests_processed
        
        # Calculate throughput (requests per second over last minute)
        # This is a simplified calculation - in production would use time windows
        if self.total_processing_time > 0:
            self.throughput = self.requests_processed / max(self.total_processing_time, 1.0)
    
    def record_success(self, processing_time: float, quality_score: float = 0.0) -> None:
        """Record a successful operation."""
        self.update({
            'requests': 1,
            'successful': 1,
            'processing_time': processing_time,
            'quality_score': quality_score
        })
    
    def record_error(self, processing_time: float) -> None:
        """Record a failed operation."""
        self.update({
            'requests': 1,
            'failed': 1,
            'processing_time': processing_time
        })


@dataclass
class ScrapingTask:
    """Represents a scraping task to be executed by an instance."""
    task_id: str
    instance_id: str
    input_data: Dict[str, Any]
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    requires_heavy_processing: bool = False
    timeout_seconds: float = 300.0  # 5 minutes default timeout


@dataclass
class ScrapingResult:
    """Result of a scraping operation."""
    task_id: str
    instance_id: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time: float = 0.0
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.utcnow)


class ScraperInstance:
    """
    Represents a single scraper instance with lifecycle management and monitoring.
    
    This class demonstrates advanced patterns for managing individual scraper
    instances with proper resource tracking, health monitoring, and thread safety.
    """
    
    def __init__(self, instance_id: str, config: IntelligentScrapingConfig):
        self.instance_id = instance_id
        self.config = config
        self.status = InstanceStatus.INITIALIZING
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        # Metrics and monitoring
        self.metrics = InstanceMetrics()
        self.health_status = HealthStatus(is_healthy=True)
        
        # Task management
        self.current_task: Optional[ScrapingTask] = None
        self.task_history: deque = deque(maxlen=100)
        
        # Thread safety
        self.lock = threading.RLock()
        self.status_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        
        # Resource tracking
        try:
            self.process = psutil.Process()
        except Exception:
            self.process = None
        self.thread_id = threading.get_ident()
        self.process_id = os.getpid()
        
        # Async coordination
        self.async_lock = None
        self.task_queue = None
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self.async_lock = asyncio.Lock()
                self.task_queue = asyncio.Queue(maxsize=10)
        except RuntimeError:
            # No event loop running
            pass
        
        # Initialize as idle
        self.status = InstanceStatus.IDLE
        
        logger.info(f"Initialized scraper instance {instance_id}")
    
    def get_uptime(self) -> float:
        """Get instance uptime in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    def get_idle_time(self) -> float:
        """Get time since last activity in seconds."""
        return (datetime.utcnow() - self.last_activity).total_seconds()
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        with self.status_lock:
            self.last_activity = datetime.utcnow()
    
    def set_status(self, status: InstanceStatus) -> None:
        """Set instance status with thread safety."""
        with self.status_lock:
            old_status = self.status
            self.status = status
            self.update_activity()
            logger.debug(f"Instance {self.instance_id} status changed: {old_status} -> {status}")
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics for this instance."""
        # Always return basic metrics
        basic_metrics = {
            'memory_mb': 0.0,
            'cpu_percent': 0.0,
            'thread_count': 0,
            'uptime_seconds': self.get_uptime(),
            'idle_time_seconds': self.get_idle_time()
        }
        
        # Try to get system metrics if psutil is available
        if self.process is None:
            return basic_metrics
            
        try:
            # Get memory usage
            memory_info = self.process.memory_info()
            basic_metrics['memory_mb'] = memory_info.rss / 1024 / 1024  # Convert to MB
            
            # Get CPU usage (non-blocking)
            basic_metrics['cpu_percent'] = self.process.cpu_percent(interval=None)
            
            # Get thread count
            basic_metrics['thread_count'] = self.process.num_threads()
            
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError, Exception) as e:
            logger.warning(f"Failed to collect system metrics for instance {self.instance_id}: {e}")
            # Return basic metrics with defaults
            
        return basic_metrics
    
    def update_metrics(self, operation_metrics: Dict[str, Any]) -> None:
        """Update instance metrics with new operation data."""
        with self.metrics_lock:
            # Add system metrics
            system_metrics = self.collect_system_metrics()
            combined_metrics = {**operation_metrics, **system_metrics}
            
            # Update metrics
            self.metrics.update(combined_metrics)
    
    def check_health(self) -> HealthStatus:
        """Perform health check on the instance."""
        issues = []
        requires_restart = False
        
        try:
            # Collect current metrics
            system_metrics = self.collect_system_metrics()
            
            # Check memory usage
            memory_mb = system_metrics['memory_mb']
            if memory_mb > 500:  # 500MB threshold
                issues.append(f"High memory usage: {memory_mb:.1f}MB")
                if memory_mb > 1000:  # 1GB critical threshold
                    requires_restart = True
            
            # Check CPU usage
            cpu_percent = system_metrics['cpu_percent']
            if cpu_percent > 80:  # 80% threshold
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            # Check if instance is stuck
            idle_time = system_metrics['idle_time_seconds']
            if self.status == InstanceStatus.RUNNING and idle_time > 300:  # 5 minutes
                issues.append(f"Instance appears stuck (idle for {idle_time:.1f}s)")
                requires_restart = True
            
            # Check error rate
            if self.metrics.error_rate > 50:  # 50% error rate
                issues.append(f"High error rate: {self.metrics.error_rate:.1f}%")
                if self.metrics.error_rate > 80:  # 80% critical
                    requires_restart = True
            
            # Update health status
            self.health_status = HealthStatus(
                is_healthy=len(issues) == 0,
                requires_restart=requires_restart,
                issues=issues,
                last_check=datetime.utcnow(),
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent,
                response_time_ms=self.metrics.average_response_time * 1000
            )
            
            return self.health_status
            
        except Exception as e:
            logger.error(f"Health check failed for instance {self.instance_id}: {e}")
            self.health_status = HealthStatus(
                is_healthy=False,
                requires_restart=True,
                issues=[f"Health check failed: {str(e)}"],
                last_check=datetime.utcnow()
            )
            return self.health_status
    
    async def execute_task_async(self, task: ScrapingTask) -> ScrapingResult:
        """Execute a scraping task asynchronously."""
        if self.async_lock is None:
            raise RuntimeError("Async operations not supported - no event loop")
        
        async with self.async_lock:
            if self.current_task is not None:
                raise RuntimeError(f"Instance {self.instance_id} is already executing a task")
            
            self.current_task = task
            self.set_status(InstanceStatus.RUNNING)
            task.started_at = datetime.utcnow()
            
            logger.info(f"Instance {self.instance_id} starting task {task.task_id}")
        
        start_time = time.time()
        
        try:
            # Simulate scraping work (in production, would call actual scraper)
            result_data = await self._simulate_scraping_work(task)
            
            processing_time = time.time() - start_time
            quality_score = result_data.get('quality_score', 75.0)
            
            # Update metrics
            self.update_metrics({
                'requests': 1,
                'successful': 1,
                'processing_time': processing_time,
                'quality_score': quality_score
            })
            
            # Create result
            result = ScrapingResult(
                task_id=task.task_id,
                instance_id=self.instance_id,
                success=True,
                data=result_data,
                processing_time=processing_time,
                quality_score=quality_score,
                metadata={
                    'instance_uptime': self.get_uptime(),
                    'memory_usage_mb': self.collect_system_metrics()['memory_mb']
                }
            )
            
            logger.info(f"Instance {self.instance_id} completed task {task.task_id} in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Update error metrics
            self.update_metrics({
                'requests': 1,
                'failed': 1,
                'processing_time': processing_time
            })
            
            # Create error result
            result = ScrapingResult(
                task_id=task.task_id,
                instance_id=self.instance_id,
                success=False,
                error=str(e),
                processing_time=processing_time,
                metadata={
                    'instance_uptime': self.get_uptime(),
                    'memory_usage_mb': self.collect_system_metrics()['memory_mb']
                }
            )
            
            logger.error(f"Instance {self.instance_id} failed task {task.task_id}: {e}")
            
            return result
            
        finally:
            # Clean up
            async with self.async_lock:
                task.completed_at = datetime.utcnow()
                self.task_history.append(task)
                self.current_task = None
                self.set_status(InstanceStatus.IDLE)
    
    def execute_task_sync(self, task: ScrapingTask) -> ScrapingResult:
        """Execute a scraping task synchronously."""
        with self.lock:
            if self.current_task is not None:
                raise RuntimeError(f"Instance {self.instance_id} is already executing a task")
            
            self.current_task = task
            self.set_status(InstanceStatus.RUNNING)
            task.started_at = datetime.utcnow()
            
            logger.info(f"Instance {self.instance_id} starting task {task.task_id}")
        
        start_time = time.time()
        
        try:
            # Simulate scraping work (in production, would call actual scraper)
            result_data = self._simulate_scraping_work_sync(task)
            
            processing_time = time.time() - start_time
            quality_score = result_data.get('quality_score', 75.0)
            
            # Update metrics
            self.update_metrics({
                'requests': 1,
                'successful': 1,
                'processing_time': processing_time,
                'quality_score': quality_score
            })
            
            # Create result
            result = ScrapingResult(
                task_id=task.task_id,
                instance_id=self.instance_id,
                success=True,
                data=result_data,
                processing_time=processing_time,
                quality_score=quality_score,
                metadata={
                    'instance_uptime': self.get_uptime(),
                    'memory_usage_mb': self.collect_system_metrics()['memory_mb']
                }
            )
            
            logger.info(f"Instance {self.instance_id} completed task {task.task_id} in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Update error metrics
            self.update_metrics({
                'requests': 1,
                'failed': 1,
                'processing_time': processing_time
            })
            
            # Create error result
            result = ScrapingResult(
                task_id=task.task_id,
                instance_id=self.instance_id,
                success=False,
                error=str(e),
                processing_time=processing_time,
                metadata={
                    'instance_uptime': self.get_uptime(),
                    'memory_usage_mb': self.collect_system_metrics()['memory_mb']
                }
            )
            
            logger.error(f"Instance {self.instance_id} failed task {task.task_id}: {e}")
            
            return result
            
        finally:
            # Clean up
            with self.lock:
                task.completed_at = datetime.utcnow()
                self.task_history.append(task)
                self.current_task = None
                self.set_status(InstanceStatus.IDLE)
    
    async def _simulate_scraping_work(self, task: ScrapingTask) -> Dict[str, Any]:
        """Simulate scraping work for testing purposes."""
        # Simulate processing time
        await asyncio.sleep(0.5 + (task.priority * 0.1))
        
        # Return mock result
        return {
            'items': [
                {'title': f'Item {i}', 'content': f'Content for item {i}'}
                for i in range(1, 4)
            ],
            'total_found': 3,
            'quality_score': 80.0 + (task.priority * 5),
            'strategy_used': 'simulation'
        }
    
    def _simulate_scraping_work_sync(self, task: ScrapingTask) -> Dict[str, Any]:
        """Simulate scraping work synchronously for testing purposes."""
        # Simulate processing time
        time.sleep(0.5 + (task.priority * 0.1))
        
        # Return mock result
        return {
            'items': [
                {'title': f'Item {i}', 'content': f'Content for item {i}'}
                for i in range(1, 4)
            ],
            'total_found': 3,
            'quality_score': 80.0 + (task.priority * 5),
            'strategy_used': 'simulation'
        }
    
    def stop(self) -> None:
        """Stop the instance gracefully."""
        with self.status_lock:
            if self.status == InstanceStatus.RUNNING:
                self.set_status(InstanceStatus.STOPPING)
                # In production, would cancel current task
                logger.info(f"Stopping instance {self.instance_id}")
            
            self.set_status(InstanceStatus.STOPPED)
    
    def restart(self) -> None:
        """Restart the instance."""
        logger.info(f"Restarting instance {self.instance_id}")
        self.stop()
        
        # Reset metrics and status
        with self.metrics_lock:
            self.metrics = InstanceMetrics()
        
        self.health_status = HealthStatus(is_healthy=True)
        self.current_task = None
        self.set_status(InstanceStatus.IDLE)
        
        logger.info(f"Instance {self.instance_id} restarted successfully")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for this instance."""
        with self.metrics_lock:
            system_metrics = self.collect_system_metrics()
            
            return {
                'instance_id': self.instance_id,
                'status': self.status.value,
                'uptime': self.get_uptime(),
                'requests_processed': self.metrics.requests_processed,
                'success_rate': self.metrics.success_rate,
                'error_rate': self.metrics.error_rate,
                'average_response_time': self.metrics.average_response_time,
                'memory_usage_mb': system_metrics['memory_mb'],
                'cpu_usage_percent': system_metrics['cpu_percent'],
                'last_activity': self.last_activity,
                'current_task': self.current_task.task_id if self.current_task else None,
                'health_status': self.health_status.dict(),
                'throughput': self.metrics.throughput,
                'quality_score_avg': sum(self.metrics.quality_scores) / len(self.metrics.quality_scores) if self.metrics.quality_scores else 0.0
            }


class ScraperInstanceManager:
    """
    Manages multiple scraper instances with comprehensive monitoring and lifecycle management.
    
    This class demonstrates advanced patterns for managing concurrent scraper instances
    with proper thread safety, resource management, and real-time monitoring capabilities.
    """
    
    def __init__(self, config: IntelligentScrapingConfig):
        self.config = config
        self.max_instances = config.max_instances
        
        # Instance management
        self.instances: Dict[str, ScraperInstance] = {}
        self.instance_pool = asyncio.Queue(maxsize=self.max_instances)
        
        # Thread safety
        self.instance_lock = threading.RLock()
        self.metrics_lock = threading.Lock()
        self.status_lock = threading.Lock()
        
        # Async coordination
        self.async_lock = None
        self.instance_semaphore = None
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self.async_lock = asyncio.Lock()
                self.instance_semaphore = asyncio.Semaphore(self.max_instances)
        except RuntimeError:
            # No event loop running
            pass
        
        # Monitoring
        self.monitoring_enabled = config.enable_monitoring
        self.monitoring_interval = config.monitoring_interval
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = threading.Event()
        self.shutdown_event = threading.Event()
        
        # Metrics collection
        self.global_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0,
            'instances_created': 0,
            'instances_restarted': 0,
            'health_checks_performed': 0
        }
        
        # Task queue for load balancing
        self.task_queue = asyncio.Queue(maxsize=1000) if self._has_event_loop() else None
        self.task_distribution_strategy = 'round_robin'  # or 'least_loaded', 'random'
        self._round_robin_index = 0
        
        logger.info(f"Initialized ScraperInstanceManager with max_instances={self.max_instances}")
    
    def _has_event_loop(self) -> bool:
        """Check if there's an active event loop."""
        try:
            asyncio.get_event_loop()
            return True
        except RuntimeError:
            return False
    
    def create_instance(self, instance_id: Optional[str] = None) -> ScraperInstance:
        """Create a new scraper instance synchronously."""
        if instance_id is None:
            instance_id = f"scraper-{uuid.uuid4().hex[:8]}"
        
        with self.instance_lock:
            if len(self.instances) >= self.max_instances:
                raise RuntimeError(f"Maximum instances ({self.max_instances}) already created")
            
            if instance_id in self.instances:
                raise ValueError(f"Instance {instance_id} already exists")
            
            # Create new instance
            instance = ScraperInstance(instance_id, self.config)
            self.instances[instance_id] = instance
            
            # Update global metrics
            with self.metrics_lock:
                self.global_metrics['instances_created'] += 1
            
            logger.info(f"Created instance {instance_id} ({len(self.instances)}/{self.max_instances})")
            
            return instance
    
    async def create_instance_async(self, instance_id: Optional[str] = None) -> ScraperInstance:
        """Create a new scraper instance asynchronously."""
        if self.async_lock is None:
            raise RuntimeError("Async operations not supported - no event loop")
        
        if instance_id is None:
            instance_id = f"scraper-{uuid.uuid4().hex[:8]}"
        
        async with self.async_lock:
            async with self.instance_semaphore:
                with self.instance_lock:
                    if instance_id in self.instances:
                        raise ValueError(f"Instance {instance_id} already exists")
                
                # Create instance in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                instance = await loop.run_in_executor(
                    None, 
                    self._create_instance_sync, 
                    instance_id
                )
                
                # Register instance
                with self.instance_lock:
                    self.instances[instance_id] = instance
                
                # Start monitoring for this instance
                if self.monitoring_enabled:
                    await self._start_instance_monitoring(instance)
                
                logger.info(f"Created async instance {instance_id} ({len(self.instances)}/{self.max_instances})")
                
                return instance
    
    def _create_instance_sync(self, instance_id: str) -> ScraperInstance:
        """Create instance synchronously in thread pool."""
        instance = ScraperInstance(instance_id, self.config)
        
        # Update global metrics
        with self.metrics_lock:
            self.global_metrics['instances_created'] += 1
        
        return instance
    
    def get_instance(self, instance_id: str) -> Optional[ScraperInstance]:
        """Get an instance by ID."""
        with self.instance_lock:
            return self.instances.get(instance_id)
    
    def list_instances(self) -> List[ScraperInstance]:
        """Get list of all instances."""
        with self.instance_lock:
            return list(self.instances.values())
    
    def get_available_instance(self) -> Optional[ScraperInstance]:
        """Get an available instance for task execution."""
        with self.instance_lock:
            available_instances = [
                instance for instance in self.instances.values()
                if instance.status == InstanceStatus.IDLE
            ]
            
            if not available_instances:
                return None
            
            # Apply distribution strategy
            if self.task_distribution_strategy == 'round_robin':
                instance = available_instances[self._round_robin_index % len(available_instances)]
                self._round_robin_index += 1
                return instance
            
            elif self.task_distribution_strategy == 'least_loaded':
                # Find instance with lowest request count
                return min(available_instances, key=lambda i: i.metrics.requests_processed)
            
            elif self.task_distribution_strategy == 'random':
                import random
                return random.choice(available_instances)
            
            else:
                return available_instances[0]
    
    async def execute_task(self, task: ScrapingTask, instance_id: Optional[str] = None) -> ScrapingResult:
        """Execute a task on a specific instance or find available one."""
        # Get target instance
        if instance_id:
            instance = self.get_instance(instance_id)
            if not instance:
                raise ValueError(f"Instance {instance_id} not found")
        else:
            instance = self.get_available_instance()
            if not instance:
                # Try to create a new instance if under limit
                if len(self.instances) < self.max_instances:
                    instance = await self.create_instance_async()
                else:
                    raise RuntimeError("No available instances and maximum limit reached")
        
        # Execute task
        try:
            if self._has_event_loop():
                result = await instance.execute_task_async(task)
            else:
                result = instance.execute_task_sync(task)
            
            # Update global metrics
            with self.metrics_lock:
                self.global_metrics['total_requests'] += 1
                if result.success:
                    self.global_metrics['successful_requests'] += 1
                else:
                    self.global_metrics['failed_requests'] += 1
                self.global_metrics['total_processing_time'] += result.processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed on instance {instance.instance_id}: {e}")
            
            # Update error metrics
            with self.metrics_lock:
                self.global_metrics['total_requests'] += 1
                self.global_metrics['failed_requests'] += 1
            
            # Return error result
            return ScrapingResult(
                task_id=task.task_id,
                instance_id=instance.instance_id,
                success=False,
                error=str(e),
                processing_time=0.0
            )
    
    def remove_instance(self, instance_id: str) -> bool:
        """Remove an instance from management."""
        with self.instance_lock:
            instance = self.instances.get(instance_id)
            if not instance:
                return False
            
            # Stop the instance
            instance.stop()
            
            # Remove from management
            del self.instances[instance_id]
            
            logger.info(f"Removed instance {instance_id} ({len(self.instances)}/{self.max_instances})")
            return True
    
    async def restart_instance(self, instance_id: str) -> bool:
        """Restart a specific instance."""
        instance = self.get_instance(instance_id)
        if not instance:
            return False
        
        try:
            # Restart the instance
            instance.restart()
            
            # Update global metrics
            with self.metrics_lock:
                self.global_metrics['instances_restarted'] += 1
            
            logger.info(f"Restarted instance {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart instance {instance_id}: {e}")
            return False
    
    def perform_health_checks(self) -> Dict[str, HealthStatus]:
        """Perform health checks on all instances."""
        health_results = {}
        
        with self.instance_lock:
            instances = list(self.instances.values())
        
        for instance in instances:
            try:
                health_status = instance.check_health()
                health_results[instance.instance_id] = health_status
                
                # Handle unhealthy instances
                if health_status.requires_restart:
                    logger.warning(f"Instance {instance.instance_id} requires restart: {health_status.issues}")
                    # Schedule restart (in production, might want to be more careful about timing)
                    threading.Thread(
                        target=self._restart_instance_async,
                        args=(instance.instance_id,),
                        daemon=True
                    ).start()
                
            except Exception as e:
                logger.error(f"Health check failed for instance {instance.instance_id}: {e}")
                health_results[instance.instance_id] = HealthStatus(
                    is_healthy=False,
                    requires_restart=True,
                    issues=[f"Health check error: {str(e)}"]
                )
        
        # Update global metrics
        with self.metrics_lock:
            self.global_metrics['health_checks_performed'] += len(health_results)
        
        return health_results
    
    def _restart_instance_async(self, instance_id: str) -> None:
        """Restart instance in background thread."""
        try:
            if self._has_event_loop():
                # Run in new event loop for thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.restart_instance(instance_id))
                loop.close()
            else:
                # Synchronous restart
                instance = self.get_instance(instance_id)
                if instance:
                    instance.restart()
                    with self.metrics_lock:
                        self.global_metrics['instances_restarted'] += 1
        except Exception as e:
            logger.error(f"Background restart failed for instance {instance_id}: {e}")
    
    def start_monitoring(self) -> None:
        """Start the monitoring thread for all instances."""
        if not self.monitoring_enabled:
            logger.info("Monitoring is disabled")
            return
        
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_active.set()
            self.shutdown_event.clear()
            
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="InstanceManagerMonitoring",
                daemon=True
            )
            self.monitoring_thread.start()
            
            logger.info("Started instance monitoring thread")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_active.clear()
            self.shutdown_event.set()
            
            # Wait for thread to finish
            self.monitoring_thread.join(timeout=5.0)
            
            logger.info("Stopped instance monitoring thread")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in dedicated thread."""
        logger.info("Instance monitoring loop started")
        
        while self.monitoring_active.is_set() and not self.shutdown_event.is_set():
            try:
                # Perform health checks
                health_results = self.perform_health_checks()
                
                # Log summary
                healthy_count = sum(1 for h in health_results.values() if h.is_healthy)
                total_count = len(health_results)
                
                if total_count > 0:
                    logger.debug(f"Health check summary: {healthy_count}/{total_count} instances healthy")
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)  # Back off on error
        
        logger.info("Instance monitoring loop stopped")
    
    async def _start_instance_monitoring(self, instance: ScraperInstance) -> None:
        """Start monitoring for a specific instance."""
        # This would set up instance-specific monitoring
        # For now, we rely on the global monitoring loop
        pass
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics for all instances."""
        with self.instance_lock:
            instance_count = len(self.instances)
            active_instances = sum(1 for i in self.instances.values() if i.status == InstanceStatus.RUNNING)
            idle_instances = sum(1 for i in self.instances.values() if i.status == InstanceStatus.IDLE)
            error_instances = sum(1 for i in self.instances.values() if i.status == InstanceStatus.ERROR)
        
        with self.metrics_lock:
            total_requests = self.global_metrics['total_requests']
            success_rate = (self.global_metrics['successful_requests'] / total_requests * 100) if total_requests > 0 else 0.0
            error_rate = (self.global_metrics['failed_requests'] / total_requests * 100) if total_requests > 0 else 0.0
            avg_processing_time = (self.global_metrics['total_processing_time'] / total_requests) if total_requests > 0 else 0.0
            
            return {
                'total_instances': instance_count,
                'active_instances': active_instances,
                'idle_instances': idle_instances,
                'error_instances': error_instances,
                'max_instances': self.max_instances,
                'total_requests': total_requests,
                'successful_requests': self.global_metrics['successful_requests'],
                'failed_requests': self.global_metrics['failed_requests'],
                'success_rate': success_rate,
                'error_rate': error_rate,
                'average_processing_time': avg_processing_time,
                'total_processing_time': self.global_metrics['total_processing_time'],
                'instances_created': self.global_metrics['instances_created'],
                'instances_restarted': self.global_metrics['instances_restarted'],
                'health_checks_performed': self.global_metrics['health_checks_performed'],
                'monitoring_enabled': self.monitoring_enabled,
                'monitoring_active': self.monitoring_active.is_set() if self.monitoring_thread else False
            }
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """Get detailed report including all instance statistics."""
        global_stats = self.get_global_stats()
        
        # Get individual instance stats
        instance_stats = []
        with self.instance_lock:
            for instance in self.instances.values():
                instance_stats.append(instance.get_stats())
        
        # Get health status for all instances
        health_results = {}
        for instance_id, instance in self.instances.items():
            health_results[instance_id] = instance.health_status.dict()
        
        return {
            'global_statistics': global_stats,
            'instance_statistics': instance_stats,
            'health_status': health_results,
            'configuration': {
                'max_instances': self.max_instances,
                'monitoring_enabled': self.monitoring_enabled,
                'monitoring_interval': self.monitoring_interval,
                'task_distribution_strategy': self.task_distribution_strategy
            },
            'report_generated_at': datetime.utcnow().isoformat()
        }
    
    def shutdown(self) -> None:
        """Shutdown the instance manager and all instances."""
        logger.info("Shutting down ScraperInstanceManager")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Stop all instances
        with self.instance_lock:
            for instance in self.instances.values():
                try:
                    instance.stop()
                except Exception as e:
                    logger.error(f"Error stopping instance {instance.instance_id}: {e}")
            
            self.instances.clear()
        
        logger.info("ScraperInstanceManager shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()