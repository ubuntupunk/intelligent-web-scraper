"""
Unit tests for the monitoring dashboard functionality.

This module tests the MonitoringDashboard class and its components including
real-time metrics display, alert system, and dashboard functionality.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from rich.console import Console

from intelligent_web_scraper.monitoring.dashboard import MonitoringDashboard
from intelligent_web_scraper.monitoring.alerts import AlertLevel, Alert, AlertType
from intelligent_web_scraper.monitoring.metrics import MetricsCollector


class TestMonitoringDashboard:
    """Test cases for MonitoringDashboard class."""
    
    @pytest.fixture
    def mock_console(self):
        """Create a mock console for testing."""
        return Mock(spec=Console)
    
    @pytest.fixture
    def dashboard(self, mock_console):
        """Create a MonitoringDashboard instance for testing."""
        return MonitoringDashboard(
            console=mock_console,
            refresh_rate=10.0,
            enable_sound_alerts=False,
            max_history=50
        )
    
    def test_dashboard_initialization(self, dashboard, mock_console):
        """Test dashboard initialization."""
        assert dashboard.console == mock_console
        assert dashboard.refresh_rate == 10.0
        assert dashboard.enable_sound_alerts is False
        assert dashboard.max_history == 50
        assert not dashboard.is_running
        assert dashboard.live_display is None
        assert isinstance(dashboard.metrics_collector, MetricsCollector)
    
    def test_dashboard_start_stop(self, dashboard):
        """Test dashboard start and stop functionality."""
        # Test start
        with patch.object(dashboard.metrics_collector, 'start_collection') as mock_metrics_start, \
             patch.object(dashboard.alert_manager, 'start_processing') as mock_alert_start, \
             patch.object(dashboard, '_start_dashboard_display') as mock_display_start:
            
            dashboard.start()
            
            assert dashboard.is_running
            assert not dashboard.shutdown_event.is_set()
            mock_metrics_start.assert_called_once()
            mock_alert_start.assert_called_once()
            mock_display_start.assert_called_once()
        
        # Test stop
        with patch.object(dashboard.metrics_collector, 'stop_collection') as mock_metrics_stop, \
             patch.object(dashboard.alert_manager, 'stop_processing') as mock_alert_stop:
            
            dashboard.stop()
            
            assert not dashboard.is_running
            assert dashboard.shutdown_event.is_set()
            mock_metrics_stop.assert_called_once()
            mock_alert_stop.assert_called_once()
    
    def test_create_header(self, dashboard):
        """Test header creation."""
        mock_metrics = {
            'active_instances': 3,
            'total_instances': 5,
            'overall_success_rate': 85.5,
            'overall_throughput': 12.34
        }
        
        with patch.object(dashboard.metrics_collector, 'get_overall_metrics', return_value=mock_metrics):
            header = dashboard._create_header()
            assert header is not None
    
    def test_create_instances_table_empty(self, dashboard):
        """Test instances table creation with no instances."""
        with patch.object(dashboard, '_get_instances_data', return_value=[]):
            table_panel = dashboard._create_instances_table()
            assert table_panel is not None
    
    def test_create_system_metrics_panel(self, dashboard):
        """Test system metrics panel creation."""
        mock_system_metrics = {
            'cpu_percent': 45.5,
            'memory_percent': 67.8,
            'memory_used_mb': 2048.0,
            'disk_usage_percent': 55.2,
            'process_count': 150,
            'thread_count': 25,
            'cpu_trend': 'increasing',
            'memory_trend': 'stable'
        }
        
        with patch.object(dashboard.metrics_collector, 'get_system_metrics_summary', return_value=mock_system_metrics):
            panel = dashboard._create_system_metrics_panel()
            assert panel is not None
    
    def test_get_resource_color(self, dashboard):
        """Test resource color determination."""
        assert dashboard._get_resource_color(50.0, 70.0, 90.0) == 'green'
        assert dashboard._get_resource_color(75.0, 70.0, 90.0) == 'yellow'
        assert dashboard._get_resource_color(95.0, 70.0, 90.0) == 'red'
    
    def test_get_trend_arrow(self, dashboard):
        """Test trend arrow indicators."""
        assert dashboard._get_trend_arrow('increasing') == '↗️'
        assert dashboard._get_trend_arrow('decreasing') == '↘️'
        assert dashboard._get_trend_arrow('stable') == '→'
    
    def test_create_mini_chart(self, dashboard):
        """Test mini chart creation."""
        data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        chart = dashboard._create_mini_chart(data, width=10)
        
        assert len(chart) == 10
        assert all(c in "█▆▄▂─" for c in chart)
    
    def test_handle_metrics_update(self, dashboard):
        """Test handling metrics updates."""
        update_data = {
            'type': 'system_metrics',
            'data': {
                'memory_used_mb': 800.0,
                'cpu_percent': 75.0,
                'disk_usage_percent': 85.0
            }
        }
        
        with patch.object(dashboard.alert_manager, 'evaluate_rules') as mock_evaluate, \
             patch.object(dashboard.alert_manager, 'add_alert') as mock_add_alert:
            
            mock_evaluate.return_value = []
            dashboard._handle_metrics_update(update_data)
            
            assert 'type' in dashboard.current_data
            mock_evaluate.assert_called_once()
    
    def test_toggle_compact_mode(self, dashboard):
        """Test toggling compact mode."""
        initial_mode = dashboard.compact_mode
        dashboard.toggle_compact_mode()
        assert dashboard.compact_mode != initial_mode
    
    def test_get_dashboard_stats(self, dashboard):
        """Test getting dashboard statistics."""
        with patch.object(dashboard.metrics_collector, 'get_collection_stats', return_value={}), \
             patch.object(dashboard.alert_manager, 'get_alert_stats', return_value={}):
            
            stats = dashboard.get_dashboard_stats()
            
            assert 'is_running' in stats
            assert 'refresh_rate' in stats
            assert 'compact_mode' in stats


class TestMonitoringDashboardIntegration:
    """Integration tests for monitoring dashboard."""
    
    @pytest.fixture
    def dashboard(self):
        """Create dashboard for integration testing."""
        return MonitoringDashboard(
            console=Mock(spec=Console),
            refresh_rate=20.0,
            enable_sound_alerts=False,
            max_history=10
        )
    
    def test_dashboard_lifecycle(self, dashboard):
        """Test complete dashboard lifecycle."""
        dashboard.start()
        assert dashboard.is_running
        
        time.sleep(0.1)
        
        instance_data = [{
            'instance_id': 'test-instance',
            'memory_usage_mb': 100.0,
            'error_rate': 10.0,
            'success_rate': 90.0,
            'idle_time_seconds': 5.0
        }]
        
        dashboard.update_instance_data(instance_data)
        dashboard.create_test_alert(AlertLevel.WARNING)
        
        stats = dashboard.get_dashboard_stats()
        assert stats['is_running']
        
        dashboard.stop()
        assert not dashboard.is_running


if __name__ == "__main__":
    pytest.main([__file__])