"""
Integration tests for monitoring components.

This module tests the integration between MonitoringDashboard, AlertManager,
and MetricsCollector to ensure they work together correctly.
"""

import pytest
import time
from datetime import datetime
from unittest.mock import Mock, patch

from intelligent_web_scraper.monitoring.dashboard import MonitoringDashboard
from intelligent_web_scraper.monitoring.alerts import AlertManager, AlertLevel, Alert, AlertType
from intelligent_web_scraper.monitoring.metrics import MetricsCollector, SystemMetrics


class TestMonitoringIntegration:
    """Integration tests for monitoring components."""
    
    @pytest.fixture
    def dashboard(self):
        """Create a dashboard for integration testing."""
        return MonitoringDashboard(
            console=Mock(),
            refresh_rate=20.0,
            enable_sound_alerts=False,
            max_history=10
        )
    
    def test_alert_manager_initialization(self, dashboard):
        """Test that AlertManager is properly initialized."""
        assert dashboard.alert_manager is not None
        assert isinstance(dashboard.alert_manager, AlertManager)
        assert dashboard.alert_manager.enable_sound is False
        assert dashboard.alert_manager.max_alerts == 10
    
    def test_metrics_collector_initialization(self, dashboard):
        """Test that MetricsCollector is properly initialized."""
        assert dashboard.metrics_collector is not None
        assert isinstance(dashboard.metrics_collector, MetricsCollector)
        assert dashboard.metrics_collector.collection_interval == 1.0
        assert dashboard.metrics_collector.history_size == 10
    
    def test_alert_creation_and_display(self, dashboard):
        """Test creating alerts and displaying them in dashboard."""
        # Create a test alert
        alert = Alert(
            alert_id="test_alert_1",
            level=AlertLevel.WARNING,
            alert_type=AlertType.PERFORMANCE,
            title="Test Performance Alert",
            message="This is a test alert for performance monitoring",
            source="test_integration"
        )
        
        # Add alert to manager
        dashboard.alert_manager.add_alert(alert)
        
        # Verify alert was added
        active_alerts = dashboard.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].alert_id == "test_alert_1"
        assert active_alerts[0].level == AlertLevel.WARNING
        
        # Test alert panel creation
        alerts_panel = dashboard._create_alerts_panel()
        assert alerts_panel is not None
    
    def test_metrics_collection_and_display(self, dashboard):
        """Test metrics collection and display in dashboard."""
        # Mock system metrics
        mock_system_metrics = SystemMetrics(
            cpu_percent=45.5,
            memory_percent=67.8,
            memory_used_mb=2048.0,
            memory_available_mb=1024.0,
            disk_usage_percent=55.2,
            process_count=150,
            thread_count=25
        )
        
        # Store metrics in collector
        dashboard.metrics_collector._store_system_metrics(mock_system_metrics)
        
        # Test system metrics panel creation
        system_panel = dashboard._create_system_metrics_panel()
        assert system_panel is not None
        
        # Verify metrics summary
        summary = dashboard.metrics_collector.get_system_metrics_summary()
        assert 'cpu_percent' in summary
        assert 'memory_percent' in summary
    
    def test_alert_rule_evaluation(self, dashboard):
        """Test alert rule evaluation with metrics."""
        # Create metrics that should trigger alerts
        high_memory_metrics = {
            'memory_usage_mb': 1200.0,  # Above critical threshold (1000MB)
            'cpu_percent': 95.0,        # High CPU
            'error_rate': 60.0,         # High error rate
            'success_rate': 30.0,       # Low success rate
            'idle_time_seconds': 400.0  # Instance unresponsive
        }
        
        # Evaluate rules
        triggered_alerts = dashboard.alert_manager.evaluate_rules(
            high_memory_metrics, 
            source="integration_test"
        )
        
        # Should trigger multiple alerts
        assert len(triggered_alerts) > 0
        
        # Add alerts to manager
        for alert in triggered_alerts:
            dashboard.alert_manager.add_alert(alert)
        
        # Verify alerts were added
        active_alerts = dashboard.alert_manager.get_active_alerts()
        assert len(active_alerts) >= len(triggered_alerts)
    
    def test_dashboard_layout_creation(self, dashboard):
        """Test complete dashboard layout creation."""
        # Add some test data
        dashboard.alert_manager.create_custom_alert(
            title="Layout Test Alert",
            message="Testing dashboard layout",
            level=AlertLevel.INFO,
            source="layout_test"
        )
        
        # Create dashboard layout
        layout = dashboard._create_dashboard_layout()
        assert layout is not None
        
        # Test individual components
        header = dashboard._create_header()
        assert header is not None
        
        instances_table = dashboard._create_instances_table()
        assert instances_table is not None
        
        system_panel = dashboard._create_system_metrics_panel()
        assert system_panel is not None
        
        alerts_panel = dashboard._create_alerts_panel()
        assert alerts_panel is not None
        
        trends_panel = dashboard._create_trends_panel()
        assert trends_panel is not None
        
        footer = dashboard._create_footer()
        assert footer is not None
    
    def test_real_time_updates_integration(self, dashboard):
        """Test real-time updates between components."""
        # Start dashboard (mocked to avoid actual display)
        with patch.object(dashboard, '_start_dashboard_display'):
            dashboard.start()
            
            try:
                # Simulate system metrics update that triggers alerts
                update_data = {
                    'type': 'system_metrics',
                    'data': {
                        'memory_used_mb': 1500.0,  # Critical memory usage
                        'cpu_percent': 85.0,
                        'disk_usage_percent': 75.0
                    }
                }
                
                # Handle the update
                dashboard._handle_metrics_update(update_data)
                
                # Verify data was stored
                assert 'type' in dashboard.current_data
                assert dashboard.current_data['type'] == 'system_metrics'
                
                # Check if alerts were triggered
                active_alerts = dashboard.alert_manager.get_active_alerts()
                # Should have at least one alert for critical memory usage
                critical_alerts = [
                    alert for alert in active_alerts 
                    if alert.level == AlertLevel.CRITICAL
                ]
                assert len(critical_alerts) > 0
                
            finally:
                dashboard.stop()
    
    def test_instance_data_integration(self, dashboard):
        """Test instance data integration with monitoring."""
        # Mock instance data
        instance_stats = [
            {
                'instance_id': 'scraper-001',
                'status': 'running',
                'requests_processed': 100,
                'successful_requests': 85,
                'failed_requests': 15,
                'success_rate': 85.0,
                'error_rate': 15.0,
                'average_response_time': 2.5,
                'memory_usage_mb': 200.0,
                'cpu_usage_percent': 30.0,
                'throughput': 5.0,
                'quality_score_avg': 80.0,
                'uptime': 3600.0,
                'idle_time_seconds': 10.0,
                'current_task': 'scraping_products'
            },
            {
                'instance_id': 'scraper-002',
                'status': 'idle',
                'requests_processed': 50,
                'successful_requests': 48,
                'failed_requests': 2,
                'success_rate': 96.0,
                'error_rate': 4.0,
                'average_response_time': 1.8,
                'memory_usage_mb': 150.0,
                'cpu_usage_percent': 15.0,
                'throughput': 2.0,
                'quality_score_avg': 90.0,
                'uptime': 1800.0,
                'idle_time_seconds': 300.0,
                'current_task': None
            }
        ]
        
        # Update dashboard with instance data
        dashboard.update_instance_data(instance_stats)
        
        # Verify metrics were collected
        instance_1_summary = dashboard.metrics_collector.get_instance_metrics_summary('scraper-001')
        assert instance_1_summary is not None
        if instance_1_summary:  # May be empty if no data collected yet
            assert instance_1_summary.get('instance_id') == 'scraper-001'
        
        # Test instances table creation with data
        with patch.object(dashboard, '_get_instances_data', return_value=instance_stats):
            instances_table = dashboard._create_instances_table()
            assert instances_table is not None
    
    def test_dashboard_state_export(self, dashboard):
        """Test exporting dashboard state."""
        # Add some test data
        dashboard.alert_manager.create_custom_alert(
            title="Export Test Alert",
            message="Testing state export",
            level=AlertLevel.ERROR,
            source="export_test"
        )
        
        # Mock some metrics
        with patch.object(dashboard.metrics_collector, 'get_system_metrics_summary') as mock_system, \
             patch.object(dashboard.metrics_collector, 'get_overall_metrics') as mock_overall, \
             patch.object(dashboard.metrics_collector, 'get_performance_trends') as mock_trends:
            
            mock_system.return_value = {'cpu_percent': 50.0, 'memory_percent': 60.0}
            mock_overall.return_value = {'total_instances': 2, 'active_instances': 1}
            mock_trends.return_value = {'cpu_usage': [10, 20, 30], 'memory_usage': [40, 50, 60]}
            
            # Export state
            state = dashboard.export_current_state()
            
            # Verify export structure
            assert 'timestamp' in state
            assert 'system_metrics' in state
            assert 'overall_metrics' in state
            assert 'active_alerts' in state
            assert 'performance_trends' in state
            assert 'dashboard_stats' in state
            
            # Verify alert was included
            assert len(state['active_alerts']) == 1
            assert state['active_alerts'][0]['title'] == "Export Test Alert"
    
    def test_compact_mode_integration(self, dashboard):
        """Test compact mode functionality."""
        # Test toggle
        initial_mode = dashboard.compact_mode
        dashboard.toggle_compact_mode()
        assert dashboard.compact_mode != initial_mode
        
        # Test compact body creation
        with patch.object(dashboard.metrics_collector, 'get_overall_metrics') as mock_overall, \
             patch.object(dashboard.metrics_collector, 'get_system_metrics_summary') as mock_system:
            
            mock_overall.return_value = {
                'active_instances': 2,
                'total_instances': 3,
                'overall_success_rate': 92.5,
                'overall_throughput': 15.7
            }
            
            mock_system.return_value = {
                'cpu_percent': 35.2,
                'memory_percent': 68.9,
                'process_count': 120
            }
            
            # Create compact body
            compact_body = dashboard._create_compact_body()
            assert compact_body is not None
    
    def test_alert_acknowledgment_integration(self, dashboard):
        """Test alert acknowledgment functionality."""
        # Create multiple test alerts
        for i in range(3):
            dashboard.alert_manager.create_custom_alert(
                title=f"Test Alert {i+1}",
                message=f"Test message {i+1}",
                level=AlertLevel.WARNING,
                source="ack_test"
            )
        
        # Verify alerts were created
        active_alerts = dashboard.alert_manager.get_active_alerts()
        assert len(active_alerts) == 3
        
        # Acknowledge all alerts
        dashboard.acknowledge_all_alerts()
        
        # Verify all alerts were acknowledged
        for alert in active_alerts:
            assert alert.acknowledged
    
    def test_performance_monitoring_integration(self, dashboard):
        """Test performance monitoring and trends."""
        # Start metrics collection (mocked)
        with patch.object(dashboard.metrics_collector, 'start_collection'), \
             patch.object(dashboard.metrics_collector, 'stop_collection'):
            
            dashboard.metrics_collector.start_collection()
            
            try:
                # Simulate performance data
                performance_trends = {
                    'cpu_usage': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
                    'memory_usage': [50, 52, 54, 56, 58, 60, 62, 64, 66, 68],
                    'success_rate': [95, 94, 96, 97, 95, 93, 94, 96, 98, 97],
                    'error_rate': [5, 6, 4, 3, 5, 7, 6, 4, 2, 3],
                    'throughput': [10, 12, 11, 13, 15, 14, 16, 18, 17, 19]
                }
                
                with patch.object(dashboard.metrics_collector, 'get_performance_trends', return_value=performance_trends):
                    trends_panel = dashboard._create_trends_panel()
                    assert trends_panel is not None
                
            finally:
                dashboard.metrics_collector.stop_collection()


if __name__ == "__main__":
    pytest.main([__file__])