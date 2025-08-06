"""
Real-time monitoring dashboard for the Intelligent Web Scraper.

This module provides a comprehensive monitoring dashboard with live updating
Rich interface, real-time metrics display, and visual alert notifications.
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import logging

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich.columns import Columns
from rich.align import Align
from rich.rule import Rule
from rich.tree import Tree
from rich.status import Status
from rich import box

from .alerts import AlertManager, AlertLevel, Alert
from .metrics import MetricsCollector, SystemMetrics, InstanceMetrics


logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """
    Real-time monitoring dashboard with live updating Rich interface.
    
    This class provides comprehensive real-time monitoring capabilities including:
    - Live updating metrics display with tables and charts
    - Visual alert notifications with color coding
    - System resource monitoring
    - Instance performance tracking
    - Interactive status indicators
    """
    
    def __init__(
        self, 
        console: Optional[Console] = None,
        refresh_rate: float = 2.0,
        enable_sound_alerts: bool = False,
        max_history: int = 100
    ):
        self.console = console or Console()
        self.refresh_rate = refresh_rate
        self.enable_sound_alerts = enable_sound_alerts
        self.max_history = max_history
        
        # Core components
        self.metrics_collector = MetricsCollector(
            collection_interval=1.0,
            history_size=max_history
        )
        self.alert_manager = AlertManager(
            max_alerts=max_history,
            enable_sound=enable_sound_alerts
        )
        
        # Dashboard state
        self.is_running = False
        self.live_display: Optional[Live] = None
        self.dashboard_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Layout components
        self.layout = Layout()
        self.current_data: Dict[str, Any] = {}
        
        # Display preferences
        self.show_system_metrics = True
        self.show_instance_details = True
        self.show_alerts = True
        self.show_performance_trends = True
        self.compact_mode = False
        
        # Color scheme
        self.colors = {
            'success': 'green',
            'warning': 'yellow',
            'error': 'red',
            'critical': 'bright_red',
            'info': 'blue',
            'neutral': 'white',
            'accent': 'cyan',
            'dim': 'dim'
        }
        
        # Status indicators
        self.status_indicators = {
            'running': '🟢',
            'idle': '🟡',
            'error': '🔴',
            'stopped': '⚫',
            'initializing': '🔵',
            'paused': '🟠'
        }
        
        # Initialize alert callbacks
        self.alert_manager.add_notification_callback(self._handle_alert_notification)
        
        # Setup metrics callbacks
        self.metrics_collector.add_update_callback(self._handle_metrics_update)
        
        logger.info("MonitoringDashboard initialized")
    
    def start(self) -> None:
        """Start the monitoring dashboard."""
        if self.is_running:
            logger.warning("Dashboard is already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Start alert processing
        self.alert_manager.start_processing()
        
        # Start dashboard display
        self._start_dashboard_display()
        
        logger.info("Monitoring dashboard started")
    
    def stop(self) -> None:
        """Stop the monitoring dashboard."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop live display
        if self.live_display:
            self.live_display.stop()
        
        # Stop dashboard thread
        if self.dashboard_thread and self.dashboard_thread.is_alive():
            self.dashboard_thread.join(timeout=5.0)
        
        # Stop components
        self.metrics_collector.stop_collection()
        self.alert_manager.stop_processing()
        
        logger.info("Monitoring dashboard stopped")
    
    def _start_dashboard_display(self) -> None:
        """Start the live dashboard display."""
        self.dashboard_thread = threading.Thread(
            target=self._dashboard_loop,
            name="DashboardDisplayThread",
            daemon=True
        )
        self.dashboard_thread.start()
    
    def _dashboard_loop(self) -> None:
        """Main dashboard display loop."""
        try:
            with Live(
                self._create_dashboard_layout(),
                console=self.console,
                refresh_per_second=self.refresh_rate,
                screen=True
            ) as live:
                self.live_display = live
                
                while self.is_running and not self.shutdown_event.is_set():
                    try:
                        # Update dashboard layout
                        updated_layout = self._create_dashboard_layout()
                        live.update(updated_layout)
                        
                        # Sleep for refresh interval
                        time.sleep(1.0 / self.refresh_rate)
                        
                    except Exception as e:
                        logger.error(f"Error updating dashboard: {e}")
                        time.sleep(1.0)
                
        except Exception as e:
            logger.error(f"Dashboard display error: {e}")
        finally:
            self.live_display = None
    
    def _create_dashboard_layout(self) -> Layout:
        """Create the main dashboard layout."""
        # Create main layout
        layout = Layout()
        
        # Split into header, body, and footer
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Create header
        layout["header"].update(self._create_header())
        
        # Create body layout
        if self.compact_mode:
            layout["body"].update(self._create_compact_body())
        else:
            layout["body"].split_row(
                Layout(name="left", ratio=2),
                Layout(name="right", ratio=1)
            )
            
            # Left side: instances and system metrics
            layout["body"]["left"].split_column(
                Layout(name="instances"),
                Layout(name="system", size=8)
            )
            
            # Right side: alerts and trends
            layout["body"]["right"].split_column(
                Layout(name="alerts"),
                Layout(name="trends")
            )
            
            # Update sections
            layout["body"]["left"]["instances"].update(self._create_instances_table())
            layout["body"]["left"]["system"].update(self._create_system_metrics_panel())
            layout["body"]["right"]["alerts"].update(self._create_alerts_panel())
            layout["body"]["right"]["trends"].update(self._create_trends_panel())
        
        # Create footer
        layout["footer"].update(self._create_footer())
        
        return layout
    
    def _create_header(self) -> Panel:
        """Create dashboard header."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get overall metrics
        overall_metrics = self.metrics_collector.get_overall_metrics()
        
        # Create status text
        status_text = Text()
        status_text.append("🕷️ Intelligent Web Scraper Dashboard", style="bold cyan")
        status_text.append(f" | {current_time}", style="dim")
        
        # Add overall stats
        if overall_metrics:
            status_text.append(f" | Instances: {overall_metrics.get('active_instances', 0)}/{overall_metrics.get('total_instances', 0)}")
            status_text.append(f" | Success Rate: {overall_metrics.get('overall_success_rate', 0):.1f}%")
            status_text.append(f" | Throughput: {overall_metrics.get('overall_throughput', 0):.2f} req/s")
        
        return Panel(
            Align.center(status_text),
            style="bright_blue",
            box=box.ROUNDED
        )
    
    def _create_instances_table(self) -> Panel:
        """Create instances monitoring table."""
        table = Table(
            title="🔧 Scraper Instances",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED
        )
        
        # Add columns
        table.add_column("ID", style="cyan", width=12)
        table.add_column("Status", justify="center", width=10)
        table.add_column("Task", style="yellow", width=15)
        table.add_column("Requests", justify="right", width=8)
        table.add_column("Success %", justify="right", width=9)
        table.add_column("Avg Time", justify="right", width=8)
        table.add_column("Memory", justify="right", width=8)
        table.add_column("CPU %", justify="right", width=6)
        
        # Get instance data
        instances_data = self._get_instances_data()
        
        if not instances_data:
            table.add_row(
                "No instances", "—", "—", "—", "—", "—", "—", "—",
                style="dim"
            )
        else:
            for instance in instances_data:
                # Status with indicator
                status_text = f"{self.status_indicators.get(instance['status'], '❓')} {instance['status']}"
                
                # Color code based on status and performance
                row_style = self._get_instance_row_style(instance)
                
                # Format values
                success_rate = f"{instance.get('success_rate', 0):.1f}%"
                avg_time = f"{instance.get('average_response_time', 0):.2f}s"
                memory = f"{instance.get('current_memory_mb', 0):.0f}MB"
                cpu = f"{instance.get('current_cpu_percent', 0):.1f}%"
                
                table.add_row(
                    instance['instance_id'][:10] + "..." if len(instance['instance_id']) > 10 else instance['instance_id'],
                    status_text,
                    instance.get('current_task', 'None')[:13] + "..." if instance.get('current_task') and len(instance.get('current_task', '')) > 13 else instance.get('current_task', 'None'),
                    str(instance.get('requests_processed', 0)),
                    success_rate,
                    avg_time,
                    memory,
                    cpu,
                    style=row_style
                )
        
        return Panel(table, border_style="blue")
    
    def _create_system_metrics_panel(self) -> Panel:
        """Create system metrics panel."""
        system_metrics = self.metrics_collector.get_system_metrics_summary()
        
        if not system_metrics:
            return Panel(
                Text("System metrics not available", style="dim"),
                title="💻 System Resources",
                border_style="yellow"
            )
        
        # Create metrics display
        metrics_text = Text()
        
        # CPU
        cpu_percent = system_metrics.get('cpu_percent', 0)
        cpu_color = self._get_resource_color(cpu_percent, 70, 90)
        cpu_trend = system_metrics.get('cpu_trend', 'stable')
        cpu_arrow = self._get_trend_arrow(cpu_trend)
        metrics_text.append(f"CPU: {cpu_percent:.1f}% {cpu_arrow}\n", style=cpu_color)
        
        # Memory
        memory_percent = system_metrics.get('memory_percent', 0)
        memory_mb = system_metrics.get('memory_used_mb', 0)
        memory_color = self._get_resource_color(memory_percent, 70, 85)
        memory_trend = system_metrics.get('memory_trend', 'stable')
        memory_arrow = self._get_trend_arrow(memory_trend)
        metrics_text.append(f"Memory: {memory_percent:.1f}% ({memory_mb:.0f}MB) {memory_arrow}\n", style=memory_color)
        
        # Disk
        disk_percent = system_metrics.get('disk_usage_percent', 0)
        disk_color = self._get_resource_color(disk_percent, 80, 90)
        metrics_text.append(f"Disk: {disk_percent:.1f}%\n", style=disk_color)
        
        # Processes and threads
        process_count = system_metrics.get('process_count', 0)
        thread_count = system_metrics.get('thread_count', 0)
        metrics_text.append(f"Processes: {process_count}\n", style="white")
        metrics_text.append(f"Threads: {thread_count}", style="white")
        
        return Panel(
            metrics_text,
            title="💻 System Resources",
            border_style="green"
        )
    
    def _create_alerts_panel(self) -> Panel:
        """Create alerts panel."""
        active_alerts = self.alert_manager.get_active_alerts()
        alert_stats = self.alert_manager.get_alert_stats()
        
        if not active_alerts:
            content = Text("✅ No active alerts", style="green")
        else:
            content = Text()
            
            # Show alert summary
            critical_count = len([a for a in active_alerts if a.level == AlertLevel.CRITICAL])
            error_count = len([a for a in active_alerts if a.level == AlertLevel.ERROR])
            warning_count = len([a for a in active_alerts if a.level == AlertLevel.WARNING])
            
            if critical_count > 0:
                content.append(f"🚨 {critical_count} Critical\n", style="bright_red bold")
            if error_count > 0:
                content.append(f"❌ {error_count} Errors\n", style="red")
            if warning_count > 0:
                content.append(f"⚠️  {warning_count} Warnings\n", style="yellow")
            
            content.append("\nRecent Alerts:\n", style="bold")
            
            # Show recent alerts (max 5)
            for alert in active_alerts[:5]:
                age = alert.get_age_seconds()
                age_str = f"{age:.0f}s" if age < 60 else f"{age/60:.0f}m"
                
                alert_color = self._get_alert_color(alert.level)
                alert_icon = self._get_alert_icon(alert.level)
                
                content.append(f"{alert_icon} ", style=alert_color)
                content.append(f"{alert.title[:20]}... ({age_str})\n", style=alert_color)
        
        # Panel color based on highest alert level
        panel_color = "green"
        if active_alerts:
            highest_level = max(alert.level for alert in active_alerts)
            panel_color = self._get_alert_color(highest_level)
        
        return Panel(
            content,
            title=f"🚨 Alerts ({len(active_alerts)})",
            border_style=panel_color
        )
    
    def _create_trends_panel(self) -> Panel:
        """Create performance trends panel."""
        trends = self.metrics_collector.get_performance_trends()
        overall_metrics = self.metrics_collector.get_overall_metrics()
        
        content = Text()
        
        if overall_metrics:
            # Success rate trend
            success_trend = overall_metrics.get('success_rate_trend', 'stable')
            success_arrow = self._get_trend_arrow(success_trend)
            success_color = 'green' if success_trend != 'decreasing' else 'red'
            content.append(f"Success Rate: {success_arrow}\n", style=success_color)
            
            # Error rate trend
            error_trend = overall_metrics.get('error_rate_trend', 'stable')
            error_arrow = self._get_trend_arrow(error_trend)
            error_color = 'red' if error_trend == 'increasing' else 'green'
            content.append(f"Error Rate: {error_arrow}\n", style=error_color)
            
            # Throughput trend
            throughput_trend = overall_metrics.get('throughput_trend', 'stable')
            throughput_arrow = self._get_trend_arrow(throughput_trend)
            throughput_color = 'green' if throughput_trend == 'increasing' else 'yellow'
            content.append(f"Throughput: {throughput_arrow}\n", style=throughput_color)
        
        # Add mini charts (simplified)
        if trends:
            content.append("\nMini Charts:\n", style="bold")
            
            # CPU usage mini chart
            cpu_data = trends.get('cpu_usage', [])[-10:]  # Last 10 points
            if cpu_data:
                cpu_chart = self._create_mini_chart(cpu_data, width=20)
                content.append(f"CPU: {cpu_chart}\n", style="cyan")
            
            # Memory usage mini chart
            memory_data = trends.get('memory_usage', [])[-10:]
            if memory_data:
                memory_chart = self._create_mini_chart(memory_data, width=20)
                content.append(f"MEM: {memory_chart}\n", style="magenta")
        
        return Panel(
            content,
            title="📈 Performance Trends",
            border_style="cyan"
        )
    
    def _create_compact_body(self) -> Panel:
        """Create compact dashboard body."""
        # Get key metrics
        overall_metrics = self.metrics_collector.get_overall_metrics()
        system_metrics = self.metrics_collector.get_system_metrics_summary()
        active_alerts = self.alert_manager.get_active_alerts()
        
        content = Text()
        
        # System overview
        content.append("SYSTEM OVERVIEW\n", style="bold cyan")
        if system_metrics:
            content.append(f"CPU: {system_metrics.get('cpu_percent', 0):.1f}% | ", style="white")
            content.append(f"Memory: {system_metrics.get('memory_percent', 0):.1f}% | ", style="white")
            content.append(f"Processes: {system_metrics.get('process_count', 0)}\n\n", style="white")
        
        # Instance overview
        content.append("INSTANCES\n", style="bold cyan")
        if overall_metrics:
            content.append(f"Active: {overall_metrics.get('active_instances', 0)}/{overall_metrics.get('total_instances', 0)} | ", style="white")
            content.append(f"Success: {overall_metrics.get('overall_success_rate', 0):.1f}% | ", style="white")
            content.append(f"Throughput: {overall_metrics.get('overall_throughput', 0):.2f} req/s\n\n", style="white")
        
        # Alerts
        content.append("ALERTS\n", style="bold cyan")
        if active_alerts:
            critical = len([a for a in active_alerts if a.level == AlertLevel.CRITICAL])
            errors = len([a for a in active_alerts if a.level == AlertLevel.ERROR])
            warnings = len([a for a in active_alerts if a.level == AlertLevel.WARNING])
            
            if critical > 0:
                content.append(f"Critical: {critical} ", style="bright_red")
            if errors > 0:
                content.append(f"Errors: {errors} ", style="red")
            if warnings > 0:
                content.append(f"Warnings: {warnings}", style="yellow")
        else:
            content.append("No active alerts", style="green")
        
        return Panel(
            content,
            title="📊 Dashboard Overview",
            border_style="blue"
        )
    
    def _create_footer(self) -> Panel:
        """Create dashboard footer."""
        # Get collection stats
        collection_stats = self.metrics_collector.get_collection_stats()
        alert_stats = self.alert_manager.get_alert_stats()
        
        footer_text = Text()
        footer_text.append("Press 'q' to quit | 'c' for compact mode | 'r' to refresh", style="dim")
        
        # Add stats
        if collection_stats.get('last_collection_time'):
            last_collection = collection_stats['last_collection_time']
            age = (datetime.utcnow() - last_collection).total_seconds()
            footer_text.append(f" | Last update: {age:.0f}s ago", style="dim")
        
        footer_text.append(f" | Collections: {collection_stats.get('collections_performed', 0)}", style="dim")
        footer_text.append(f" | Alerts: {alert_stats.get('total_alerts', 0)}", style="dim")
        
        return Panel(
            Align.center(footer_text),
            style="dim"
        )
    
    def _get_instances_data(self) -> List[Dict[str, Any]]:
        """Get current instances data."""
        # This would normally come from the instance manager
        # For now, return mock data or empty list
        instances_data = []
        
        # Try to get data from metrics collector
        overall_metrics = self.metrics_collector.get_overall_metrics()
        if overall_metrics and overall_metrics.get('total_instances', 0) > 0:
            # Generate mock instance data for demonstration
            for i in range(overall_metrics.get('total_instances', 0)):
                instance_id = f"scraper-{i+1:03d}"
                instance_data = self.metrics_collector.get_instance_metrics_summary(instance_id)
                
                if instance_data:
                    instances_data.append(instance_data)
                else:
                    # Mock data for demonstration
                    instances_data.append({
                        'instance_id': instance_id,
                        'status': 'idle',
                        'current_task': None,
                        'requests_processed': 0,
                        'success_rate': 0.0,
                        'error_rate': 0.0,
                        'average_response_time': 0.0,
                        'current_memory_mb': 50.0,
                        'current_cpu_percent': 5.0,
                        'throughput': 0.0,
                        'quality_score_avg': 0.0,
                        'uptime_seconds': 0.0,
                        'idle_time_seconds': 0.0
                    })
        
        return instances_data
    
    def _get_instance_row_style(self, instance: Dict[str, Any]) -> str:
        """Get row style based on instance status and performance."""
        status = instance.get('status', 'unknown')
        error_rate = instance.get('error_rate', 0)
        success_rate = instance.get('success_rate', 100)
        
        if status == 'error':
            return 'red'
        elif error_rate > 50:
            return 'yellow'
        elif success_rate < 80:
            return 'yellow'
        elif status == 'running':
            return 'green'
        else:
            return 'white'
    
    def _get_resource_color(self, value: float, warning_threshold: float, critical_threshold: float) -> str:
        """Get color based on resource usage thresholds."""
        if value >= critical_threshold:
            return 'red'
        elif value >= warning_threshold:
            return 'yellow'
        else:
            return 'green'
    
    def _get_trend_arrow(self, trend: str) -> str:
        """Get arrow indicator for trend."""
        if trend == 'increasing':
            return '↗️'
        elif trend == 'decreasing':
            return '↘️'
        else:
            return '→'
    
    def _get_alert_color(self, level: AlertLevel) -> str:
        """Get color for alert level."""
        color_map = {
            AlertLevel.INFO: 'blue',
            AlertLevel.WARNING: 'yellow',
            AlertLevel.ERROR: 'red',
            AlertLevel.CRITICAL: 'bright_red'
        }
        return color_map.get(level, 'white')
    
    def _get_alert_icon(self, level: AlertLevel) -> str:
        """Get icon for alert level."""
        icon_map = {
            AlertLevel.INFO: 'ℹ️',
            AlertLevel.WARNING: '⚠️',
            AlertLevel.ERROR: '❌',
            AlertLevel.CRITICAL: '🚨'
        }
        return icon_map.get(level, '❓')
    
    def _create_mini_chart(self, data: List[float], width: int = 20) -> str:
        """Create a simple ASCII mini chart."""
        if not data or len(data) < 2:
            return "─" * width
        
        # Normalize data to chart width
        min_val = min(data)
        max_val = max(data)
        
        if max_val == min_val:
            return "─" * width
        
        # Create simple bar chart
        chart = ""
        for i in range(min(len(data), width)):
            normalized = (data[i] - min_val) / (max_val - min_val)
            if normalized > 0.75:
                chart += "█"
            elif normalized > 0.5:
                chart += "▆"
            elif normalized > 0.25:
                chart += "▄"
            else:
                chart += "▂"
        
        # Pad to width
        while len(chart) < width:
            chart += "─"
        
        return chart
    
    def _handle_alert_notification(self, alert: Alert) -> None:
        """Handle alert notifications for dashboard updates."""
        # This callback is called when new alerts are created
        # The dashboard will automatically show them in the next refresh
        logger.info(f"Dashboard received alert notification: {alert.title}")
    
    def _handle_metrics_update(self, update_data: Dict[str, Any]) -> None:
        """Handle metrics updates for real-time display."""
        # Store current data for dashboard display
        self.current_data.update(update_data)
        
        # Evaluate alert rules if this is system metrics
        if update_data.get('type') == 'system_metrics':
            system_data = update_data.get('data', {})
            
            # Convert to format expected by alert manager
            metrics_for_alerts = {
                'memory_usage_mb': system_data.get('memory_used_mb', 0),
                'cpu_percent': system_data.get('cpu_percent', 0),
                'disk_usage_percent': system_data.get('disk_usage_percent', 0)
            }
            
            # Evaluate alert rules
            triggered_alerts = self.alert_manager.evaluate_rules(
                metrics_for_alerts, 
                source="system"
            )
            
            # Add triggered alerts
            for alert in triggered_alerts:
                self.alert_manager.add_alert(alert)
    
    def update_instance_data(self, instance_stats: List[Dict[str, Any]]) -> None:
        """Update dashboard with instance data."""
        for stats in instance_stats:
            # Collect metrics for this instance
            self.metrics_collector.collect_instance_metrics(stats)
            
            # Evaluate instance-specific alerts
            instance_metrics = {
                'memory_usage_mb': stats.get('memory_usage_mb', 0),
                'error_rate': stats.get('error_rate', 0),
                'success_rate': stats.get('success_rate', 100),
                'idle_time_seconds': stats.get('idle_time_seconds', 0)
            }
            
            triggered_alerts = self.alert_manager.evaluate_rules(
                instance_metrics,
                source=f"instance_{stats.get('instance_id', 'unknown')}"
            )
            
            for alert in triggered_alerts:
                self.alert_manager.add_alert(alert)
    
    def toggle_compact_mode(self) -> None:
        """Toggle compact display mode."""
        self.compact_mode = not self.compact_mode
        logger.info(f"Compact mode: {'enabled' if self.compact_mode else 'disabled'}")
    
    def acknowledge_all_alerts(self) -> None:
        """Acknowledge all active alerts."""
        active_alerts = self.alert_manager.get_active_alerts()
        for alert in active_alerts:
            self.alert_manager.acknowledge_alert(alert.alert_id)
        logger.info(f"Acknowledged {len(active_alerts)} alerts")
    
    def create_test_alert(self, level: AlertLevel = AlertLevel.WARNING) -> None:
        """Create a test alert for demonstration."""
        self.alert_manager.create_custom_alert(
            title="Test Alert",
            message="This is a test alert for dashboard demonstration",
            level=level,
            source="dashboard_test"
        )
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        return {
            'is_running': self.is_running,
            'refresh_rate': self.refresh_rate,
            'compact_mode': self.compact_mode,
            'metrics_collection_stats': self.metrics_collector.get_collection_stats(),
            'alert_stats': self.alert_manager.get_alert_stats(),
            'display_preferences': {
                'show_system_metrics': self.show_system_metrics,
                'show_instance_details': self.show_instance_details,
                'show_alerts': self.show_alerts,
                'show_performance_trends': self.show_performance_trends
            }
        }
    
    def export_current_state(self) -> Dict[str, Any]:
        """Export current dashboard state for analysis."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system_metrics': self.metrics_collector.get_system_metrics_summary(),
            'overall_metrics': self.metrics_collector.get_overall_metrics(),
            'active_alerts': [alert.to_dict() for alert in self.alert_manager.get_active_alerts()],
            'performance_trends': self.metrics_collector.get_performance_trends(),
            'dashboard_stats': self.get_dashboard_stats()
        }