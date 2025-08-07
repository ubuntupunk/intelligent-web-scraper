#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard Demo

This example demonstrates the comprehensive monitoring dashboard with
live updating Rich interface, real-time metrics display, and visual alerts.
"""

import asyncio
import time
import threading
from datetime import datetime
from typing import Dict, Any
import random

from intelligent_web_scraper.monitoring import (
    MonitoringDashboard, 
    AlertManager, 
    AlertLevel, 
    MetricsCollector
)


class MockScrapingSystem:
    """Mock scraping system to generate realistic monitoring data."""
    
    def __init__(self, dashboard: MonitoringDashboard):
        self.dashboard = dashboard
        self.instances = {}
        self.running = False
        self.simulation_thread = None
        
        # Initialize mock instances
        for i in range(3):
            instance_id = f"scraper-{i+1:03d}"
            self.instances[instance_id] = {
                'instance_id': instance_id,
                'status': 'idle',
                'requests_processed': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'success_rate': 100.0,
                'error_rate': 0.0,
                'average_response_time': 0.0,
                'memory_usage_mb': random.uniform(50, 100),
                'cpu_usage_percent': random.uniform(5, 15),
                'throughput': 0.0,
                'quality_score_avg': 0.0,
                'uptime': 0.0,
                'idle_time_seconds': 0.0,
                'current_task': None
            }
    
    def start_simulation(self):
        """Start the mock scraping simulation."""
        self.running = True
        self.simulation_thread = threading.Thread(
            target=self._simulation_loop,
            name="MockScrapingSimulation",
            daemon=True
        )
        self.simulation_thread.start()
        print("🚀 Mock scraping simulation started")
    
    def stop_simulation(self):
        """Stop the mock scraping simulation."""
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)
        print("⏹️  Mock scraping simulation stopped")
    
    def _simulation_loop(self):
        """Main simulation loop."""
        cycle_count = 0
        
        while self.running:
            cycle_count += 1
            
            # Simulate different scenarios based on cycle
            if cycle_count < 10:
                self._simulate_normal_operation()
            elif cycle_count < 20:
                self._simulate_high_load()
            elif cycle_count < 25:
                self._simulate_error_conditions()
            elif cycle_count < 35:
                self._simulate_recovery()
            else:
                # Reset cycle
                cycle_count = 0
                self._simulate_normal_operation()
            
            # Update dashboard with instance data
            instance_stats = list(self.instances.values())
            self.dashboard.update_instance_data(instance_stats)
            
            # Sleep for simulation interval
            time.sleep(2.0)
    
    def _simulate_normal_operation(self):
        """Simulate normal scraping operation."""
        for instance_id, instance in self.instances.items():
            # Randomly assign tasks
            if random.random() < 0.3 and instance['status'] == 'idle':
                instance['status'] = 'running'
                instance['current_task'] = random.choice([
                    'scraping_products',
                    'extracting_articles',
                    'collecting_reviews',
                    'gathering_listings'
                ])
                instance['idle_time_seconds'] = 0.0
            elif random.random() < 0.2 and instance['status'] == 'running':
                instance['status'] = 'idle'
                instance['current_task'] = None
                instance['idle_time_seconds'] = 0.0
            
            # Update metrics for running instances
            if instance['status'] == 'running':
                # Simulate successful requests
                new_requests = random.randint(1, 5)
                new_successful = random.randint(int(new_requests * 0.8), new_requests)
                new_failed = new_requests - new_successful
                
                instance['requests_processed'] += new_requests
                instance['successful_requests'] += new_successful
                instance['failed_requests'] += new_failed
                
                # Update rates
                total_requests = instance['requests_processed']
                if total_requests > 0:
                    instance['success_rate'] = (instance['successful_requests'] / total_requests) * 100
                    instance['error_rate'] = (instance['failed_requests'] / total_requests) * 100
                
                # Update performance metrics
                instance['average_response_time'] = random.uniform(1.0, 3.0)
                instance['memory_usage_mb'] = random.uniform(100, 300)
                instance['cpu_usage_percent'] = random.uniform(20, 60)
                instance['throughput'] = random.uniform(2.0, 8.0)
                instance['quality_score_avg'] = random.uniform(80, 95)
                
            else:
                # Idle instance
                instance['idle_time_seconds'] += 2.0
                instance['memory_usage_mb'] = random.uniform(50, 120)
                instance['cpu_usage_percent'] = random.uniform(5, 20)
                instance['throughput'] = 0.0
            
            # Update uptime
            instance['uptime'] += 2.0
    
    def _simulate_high_load(self):
        """Simulate high load conditions."""
        for instance_id, instance in self.instances.items():
            # All instances should be running under high load
            instance['status'] = 'running'
            instance['current_task'] = 'high_volume_scraping'
            instance['idle_time_seconds'] = 0.0
            
            # Higher resource usage
            instance['memory_usage_mb'] = random.uniform(400, 800)
            instance['cpu_usage_percent'] = random.uniform(70, 95)
            instance['average_response_time'] = random.uniform(3.0, 8.0)
            instance['throughput'] = random.uniform(10.0, 20.0)
            
            # Slightly lower quality due to high load
            instance['quality_score_avg'] = random.uniform(70, 85)
            
            # More requests but similar success rate
            new_requests = random.randint(5, 15)
            new_successful = random.randint(int(new_requests * 0.75), int(new_requests * 0.9))
            new_failed = new_requests - new_successful
            
            instance['requests_processed'] += new_requests
            instance['successful_requests'] += new_successful
            instance['failed_requests'] += new_failed
            
            # Update rates
            total_requests = instance['requests_processed']
            if total_requests > 0:
                instance['success_rate'] = (instance['successful_requests'] / total_requests) * 100
                instance['error_rate'] = (instance['failed_requests'] / total_requests) * 100
            
            instance['uptime'] += 2.0
    
    def _simulate_error_conditions(self):
        """Simulate error conditions that should trigger alerts."""
        for instance_id, instance in self.instances.items():
            # Simulate various error conditions
            if instance_id == 'scraper-001':
                # Memory leak simulation
                instance['memory_usage_mb'] = min(instance['memory_usage_mb'] + 100, 1500)
                instance['status'] = 'running'
                instance['current_task'] = 'memory_intensive_task'
                
            elif instance_id == 'scraper-002':
                # High error rate simulation
                instance['status'] = 'error'
                instance['current_task'] = 'failing_task'
                
                # Add more failed requests
                new_requests = random.randint(3, 8)
                new_failed = random.randint(int(new_requests * 0.6), new_requests)
                new_successful = new_requests - new_failed
                
                instance['requests_processed'] += new_requests
                instance['successful_requests'] += new_successful
                instance['failed_requests'] += new_failed
                
            else:
                # Unresponsive instance
                instance['status'] = 'idle'
                instance['current_task'] = None
                instance['idle_time_seconds'] += 60.0  # Long idle time
                instance['memory_usage_mb'] = random.uniform(50, 100)
                instance['cpu_usage_percent'] = random.uniform(5, 15)
            
            # Update rates
            total_requests = instance['requests_processed']
            if total_requests > 0:
                instance['success_rate'] = (instance['successful_requests'] / total_requests) * 100
                instance['error_rate'] = (instance['failed_requests'] / total_requests) * 100
            
            instance['uptime'] += 2.0
    
    def _simulate_recovery(self):
        """Simulate recovery from error conditions."""
        for instance_id, instance in self.instances.items():
            # Gradually recover from error conditions
            if instance['status'] == 'error':
                instance['status'] = 'running'
                instance['current_task'] = 'recovery_task'
            
            # Reduce memory usage gradually
            if instance['memory_usage_mb'] > 300:
                instance['memory_usage_mb'] = max(
                    instance['memory_usage_mb'] - 50, 
                    random.uniform(100, 300)
                )
            
            # Reduce idle time
            if instance['idle_time_seconds'] > 60:
                instance['idle_time_seconds'] = max(
                    instance['idle_time_seconds'] - 30,
                    random.uniform(0, 30)
                )
            
            # Improve success rates
            new_requests = random.randint(2, 6)
            new_successful = random.randint(int(new_requests * 0.9), new_requests)
            new_failed = new_requests - new_successful
            
            instance['requests_processed'] += new_requests
            instance['successful_requests'] += new_successful
            instance['failed_requests'] += new_failed
            
            # Update rates
            total_requests = instance['requests_processed']
            if total_requests > 0:
                instance['success_rate'] = (instance['successful_requests'] / total_requests) * 100
                instance['error_rate'] = (instance['failed_requests'] / total_requests) * 100
            
            # Normal resource usage
            instance['cpu_usage_percent'] = random.uniform(20, 50)
            instance['average_response_time'] = random.uniform(1.5, 3.5)
            instance['throughput'] = random.uniform(3.0, 7.0)
            instance['quality_score_avg'] = random.uniform(85, 95)
            
            instance['uptime'] += 2.0


def create_demo_alerts(dashboard: MonitoringDashboard):
    """Create some demo alerts for demonstration."""
    print("📢 Creating demo alerts...")
    
    # Create alerts of different levels
    dashboard.alert_manager.create_custom_alert(
        title="Demo Info Alert",
        message="This is an informational alert for demonstration purposes",
        level=AlertLevel.INFO,
        source="demo"
    )
    
    dashboard.alert_manager.create_custom_alert(
        title="Demo Warning Alert", 
        message="This is a warning alert showing potential issues",
        level=AlertLevel.WARNING,
        source="demo"
    )
    
    dashboard.alert_manager.create_custom_alert(
        title="Demo Error Alert",
        message="This is an error alert indicating a problem that needs attention",
        level=AlertLevel.ERROR,
        source="demo"
    )


def demonstrate_dashboard_features(dashboard: MonitoringDashboard):
    """Demonstrate various dashboard features."""
    print("\n🎯 Dashboard Features Demo")
    print("=" * 50)
    
    # Show dashboard stats
    stats = dashboard.get_dashboard_stats()
    print(f"Dashboard running: {stats['is_running']}")
    print(f"Refresh rate: {stats['refresh_rate']} Hz")
    print(f"Compact mode: {stats['compact_mode']}")
    
    # Show alert stats
    alert_stats = dashboard.alert_manager.get_alert_stats()
    print(f"Total alerts: {alert_stats['total_alerts']}")
    print(f"Active alerts: {alert_stats['active_alerts']}")
    print(f"Critical alerts: {alert_stats['critical_alerts']}")
    
    # Show metrics collection stats
    collection_stats = dashboard.metrics_collector.get_collection_stats()
    print(f"Collections performed: {collection_stats['collections_performed']}")
    print(f"Collection errors: {collection_stats['collection_errors']}")
    
    print("\n💡 Dashboard Controls:")
    print("- Press 'q' to quit the dashboard")
    print("- Press 'c' to toggle compact mode")
    print("- Press 'r' to refresh display")
    print("- The dashboard updates automatically every 0.5 seconds")


def main():
    """Main demonstration function."""
    print("🎛️  Real-time Monitoring Dashboard Demo")
    print("=" * 60)
    print("This demo shows the comprehensive monitoring dashboard with:")
    print("- Live updating Rich interface")
    print("- Real-time metrics display with tables and charts")
    print("- Visual alert notifications with color coding")
    print("- System resource monitoring")
    print("- Instance performance tracking")
    print("- Interactive status indicators")
    print()
    
    # Create dashboard with demo settings
    dashboard = MonitoringDashboard(
        refresh_rate=2.0,  # 2 updates per second
        enable_sound_alerts=False,  # Disable sound for demo
        max_history=50
    )
    
    # Create mock scraping system
    mock_system = MockScrapingSystem(dashboard)
    
    try:
        print("🚀 Starting monitoring dashboard...")
        
        # Start dashboard
        dashboard.start()
        
        # Create demo alerts
        create_demo_alerts(dashboard)
        
        # Start mock system simulation
        mock_system.start_simulation()
        
        # Show dashboard features
        demonstrate_dashboard_features(dashboard)
        
        print("\n🎭 Running simulation phases:")
        print("1. Normal operation (20 seconds)")
        print("2. High load conditions (20 seconds)")
        print("3. Error conditions with alerts (10 seconds)")
        print("4. Recovery phase (20 seconds)")
        print("5. Loop back to normal operation")
        print()
        print("📊 Watch the dashboard for:")
        print("- Instance status changes")
        print("- Resource usage fluctuations")
        print("- Alert notifications")
        print("- Performance trends")
        print()
        print("Press Ctrl+C to stop the demo")
        
        # Keep demo running
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n\n⏹️  Demo interrupted by user")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        
    finally:
        print("\n🧹 Cleaning up...")
        
        # Stop mock system
        mock_system.stop_simulation()
        
        # Stop dashboard
        dashboard.stop()
        
        print("✅ Demo cleanup complete")
        print("\n🎉 Thank you for trying the monitoring dashboard demo!")
        print("The dashboard provides comprehensive real-time monitoring")
        print("for intelligent web scraping operations with visual alerts")
        print("and performance tracking.")


if __name__ == "__main__":
    main()