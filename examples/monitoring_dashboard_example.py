#!/usr/bin/env python3
"""
Example script demonstrating the MonitoringDashboard functionality.

This script shows how to use the real-time monitoring dashboard with
live metrics display, alert system, and interactive features.
"""

import asyncio
import time
import random
from datetime import datetime
from typing import Dict, Any, List

from intelligent_web_scraper.monitoring.dashboard import MonitoringDashboard
from intelligent_web_scraper.monitoring.alerts import AlertLevel


def generate_mock_instance_data(instance_count: int = 3) -> List[Dict[str, Any]]:
    """Generate mock instance data for demonstration."""
    instances = []
    
    for i in range(instance_count):
        instance_id = f"scraper-{i+1:03d}"
        
        # Simulate varying performance
        base_success_rate = random.uniform(80, 98)
        base_memory = random.uniform(100, 300)
        base_cpu = random.uniform(10, 60)
        
        # Occasionally simulate problems
        if random.random() < 0.2:  # 20% chance of issues
            base_success_rate = random.uniform(50, 80)
            base_memory = random.uniform(400, 800)
            base_cpu = random.uniform(70, 95)
        
        instance_data = {
            'instance_id': instance_id,
            'status': random.choice(['running', 'idle', 'running', 'running']),  # Bias toward running
            'requests_processed': random.randint(50, 500),
            'successful_requests': int(base_success_rate * random.randint(50, 500) / 100),
            'failed_requests': random.randint(1, 20),
            'success_rate': base_success_rate,
            'error_rate': 100 - base_success_rate,
            'average_response_time': random.uniform(0.5, 3.0),
            'memory_usage_mb': base_memory,
            'cpu_usage_percent': base_cpu,
            'throughput': random.uniform(1.0, 10.0),
            'quality_score_avg': random.uniform(70, 95),
            'uptime': random.uniform(300, 7200),  # 5 minutes to 2 hours
            'idle_time_seconds': random.uniform(0, 60),
            'current_task': random.choice([
                'scraping_products', 
                'analyzing_content', 
                'extracting_data', 
                None, 
                'processing_results'
            ])
        }
        
        instances.append(instance_data)
    
    return instances


def simulate_system_load():
    """Simulate varying system load for demonstration."""
    # Create load patterns
    time_factor = time.time() % 60  # 60-second cycle
    
    # CPU usage with sine wave pattern
    cpu_base = 30 + 20 * abs(time_factor - 30) / 30
    cpu_noise = random.uniform(-10, 10)
    cpu_percent = max(0, min(100, cpu_base + cpu_noise))
    
    # Memory usage with gradual increase
    memory_base = 40 + (time_factor / 60) * 30
    memory_noise = random.uniform(-5, 5)
    memory_percent = max(0, min(100, memory_base + memory_noise))
    
    # Occasionally spike resources to trigger alerts
    if random.random() < 0.1:  # 10% chance
        cpu_percent = random.uniform(85, 98)
        memory_percent = random.uniform(80, 95)
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'memory_used_mb': memory_percent * 40,  # Assume 4GB total
        'disk_usage_percent': random.uniform(45, 75),
        'process_count': random.randint(120, 180),
        'thread_count': random.randint(20, 40)
    }


async def run_dashboard_demo():
    """Run the monitoring dashboard demonstration."""
    print("🕷️ Starting Intelligent Web Scraper Monitoring Dashboard Demo")
    print("=" * 60)
    print("This demo will show:")
    print("• Real-time metrics display")
    print("• Live instance monitoring")
    print("• Alert system with notifications")
    print("• Performance trends and charts")
    print("• Interactive dashboard features")
    print("=" * 60)
    print("Press Ctrl+C to stop the demo")
    print()
    
    # Create dashboard
    dashboard = MonitoringDashboard(
        refresh_rate=2.0,  # 2 FPS for smooth updates
        enable_sound_alerts=False,  # Disable sound for demo
        max_history=100
    )
    
    try:
        # Start the dashboard
        dashboard.start()
        
        # Let the dashboard initialize
        await asyncio.sleep(1)
        
        print("Dashboard started! Generating demo data...")
        
        # Simulation loop
        iteration = 0
        while True:
            iteration += 1
            
            # Generate mock instance data
            instance_data = generate_mock_instance_data(3)
            
            # Update dashboard with instance data
            dashboard.update_instance_data(instance_data)
            
            # Simulate system metrics updates
            system_metrics = simulate_system_load()
            system_update = {
                'type': 'system_metrics',
                'data': system_metrics
            }
            dashboard._handle_metrics_update(system_update)
            
            # Occasionally create custom alerts for demonstration
            if iteration % 20 == 0:  # Every 20 iterations (~40 seconds)
                alert_types = [
                    ("Demo Alert: High Traffic", "Simulated high traffic scenario", AlertLevel.INFO),
                    ("Demo Alert: Performance Warning", "Simulated performance degradation", AlertLevel.WARNING),
                    ("Demo Alert: Resource Alert", "Simulated resource constraint", AlertLevel.ERROR)
                ]
                
                title, message, level = random.choice(alert_types)
                dashboard.create_test_alert(level)
            
            # Occasionally acknowledge alerts
            if iteration % 30 == 0:  # Every 30 iterations (~60 seconds)
                active_alerts = dashboard.alert_manager.get_active_alerts()
                if active_alerts and random.random() < 0.5:  # 50% chance
                    dashboard.acknowledge_all_alerts()
            
            # Toggle compact mode occasionally for demonstration
            if iteration % 50 == 0:  # Every 50 iterations
                dashboard.toggle_compact_mode()
                print(f"Toggled compact mode: {'ON' if dashboard.compact_mode else 'OFF'}")
            
            # Print stats periodically
            if iteration % 25 == 0:  # Every 25 iterations (~50 seconds)
                stats = dashboard.get_dashboard_stats()
                print(f"Dashboard Stats - Iteration {iteration}:")
                print(f"  • Running: {stats['is_running']}")
                print(f"  • Compact Mode: {stats['compact_mode']}")
                print(f"  • Active Alerts: {len(dashboard.alert_manager.get_active_alerts())}")
                print(f"  • Collections: {stats['metrics_collection_stats'].get('collections_performed', 0)}")
                print()
            
            # Wait for next update
            await asyncio.sleep(2.0)  # Update every 2 seconds
    
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    
    finally:
        print("\n🔄 Stopping dashboard...")
        dashboard.stop()
        
        # Export final state
        final_state = dashboard.export_current_state()
        print(f"📊 Final dashboard state exported with {len(final_state.get('active_alerts', []))} alerts")
        
        print("✅ Demo completed!")


def run_simple_dashboard_test():
    """Run a simple dashboard test without asyncio."""
    print("🧪 Running Simple Dashboard Test")
    print("=" * 40)
    
    # Create dashboard
    dashboard = MonitoringDashboard(
        refresh_rate=5.0,
        enable_sound_alerts=False,
        max_history=20
    )
    
    try:
        # Test dashboard components without starting full display
        print("Testing dashboard components...")
        
        # Test header creation
        header = dashboard._create_header()
        print("✅ Header created successfully")
        
        # Test instances table
        instances_table = dashboard._create_instances_table()
        print("✅ Instances table created successfully")
        
        # Test system metrics panel
        system_panel = dashboard._create_system_metrics_panel()
        print("✅ System metrics panel created successfully")
        
        # Test alerts panel
        alerts_panel = dashboard._create_alerts_panel()
        print("✅ Alerts panel created successfully")
        
        # Test trends panel
        trends_panel = dashboard._create_trends_panel()
        print("✅ Trends panel created successfully")
        
        # Test footer
        footer = dashboard._create_footer()
        print("✅ Footer created successfully")
        
        # Test alert creation
        dashboard.create_test_alert(AlertLevel.INFO)
        dashboard.create_test_alert(AlertLevel.WARNING)
        dashboard.create_test_alert(AlertLevel.ERROR)
        
        active_alerts = dashboard.alert_manager.get_active_alerts()
        print(f"✅ Created {len(active_alerts)} test alerts")
        
        # Test instance data update
        test_instances = generate_mock_instance_data(2)
        dashboard.update_instance_data(test_instances)
        print("✅ Updated instance data successfully")
        
        # Test metrics update
        test_metrics = {
            'type': 'system_metrics',
            'data': simulate_system_load()
        }
        dashboard._handle_metrics_update(test_metrics)
        print("✅ Updated system metrics successfully")
        
        # Test dashboard stats
        stats = dashboard.get_dashboard_stats()
        print(f"✅ Dashboard stats retrieved: {len(stats)} fields")
        
        # Test state export
        state = dashboard.export_current_state()
        print(f"✅ Dashboard state exported: {len(state)} sections")
        
        # Test compact mode toggle
        dashboard.toggle_compact_mode()
        print(f"✅ Compact mode toggled: {dashboard.compact_mode}")
        
        # Test alert acknowledgment
        dashboard.acknowledge_all_alerts()
        acknowledged_count = len([a for a in active_alerts if a.acknowledged])
        print(f"✅ Acknowledged {acknowledged_count} alerts")
        
        print("\n🎉 All dashboard components tested successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    
    finally:
        dashboard.stop()
        print("🔄 Dashboard stopped")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run simple test mode
        run_simple_dashboard_test()
    else:
        # Run full interactive demo
        try:
            asyncio.run(run_dashboard_demo())
        except KeyboardInterrupt:
            print("\nDemo stopped by user")
        except Exception as e:
            print(f"Demo failed: {e}")
            sys.exit(1)