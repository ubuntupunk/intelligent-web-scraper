"""
Alert system for monitoring dashboard.

This module provides comprehensive alerting capabilities with visual
notifications and sound alerts for critical issues.
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import logging

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    ERROR_RATE = "error_rate"
    INSTANCE_HEALTH = "instance_health"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class Alert:
    """Represents a monitoring alert."""
    alert_id: str
    level: AlertLevel
    alert_type: AlertType
    title: str
    message: str
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def acknowledge(self) -> None:
        """Acknowledge the alert."""
        self.acknowledged = True
        self.metadata['acknowledged_at'] = datetime.utcnow()
    
    def resolve(self) -> None:
        """Mark the alert as resolved."""
        self.resolved = True
        self.metadata['resolved_at'] = datetime.utcnow()
    
    def get_age_seconds(self) -> float:
        """Get alert age in seconds."""
        return (datetime.utcnow() - self.timestamp).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'level': self.level.value,
            'alert_type': self.alert_type.value,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged,
            'resolved': self.resolved,
            'age_seconds': self.get_age_seconds(),
            'metadata': self.metadata
        }


class AlertRule(BaseModel):
    """Defines conditions for triggering alerts."""
    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Human-readable rule name")
    alert_type: AlertType = Field(..., description="Type of alert this rule generates")
    level: AlertLevel = Field(..., description="Severity level for triggered alerts")
    condition: str = Field(..., description="Condition expression to evaluate")
    threshold: float = Field(..., description="Threshold value for triggering")
    duration_seconds: float = Field(default=60.0, description="How long condition must persist")
    cooldown_seconds: float = Field(default=300.0, description="Cooldown period between alerts")
    enabled: bool = Field(default=True, description="Whether the rule is active")
    last_triggered: Optional[datetime] = Field(None, description="When rule was last triggered")
    
    def should_trigger(self, current_value: float) -> bool:
        """Check if the rule should trigger based on current value."""
        if not self.enabled:
            return False
        
        # Check cooldown period
        if self.last_triggered:
            cooldown_elapsed = (datetime.utcnow() - self.last_triggered).total_seconds()
            if cooldown_elapsed < self.cooldown_seconds:
                return False
        
        # Evaluate condition (simplified - in production would use proper expression parser)
        if self.condition == "greater_than":
            return current_value > self.threshold
        elif self.condition == "less_than":
            return current_value < self.threshold
        elif self.condition == "equals":
            return abs(current_value - self.threshold) < 0.001
        else:
            return False


class AlertManager:
    """
    Manages alerts with visual notifications and sound alerts.
    
    This class provides comprehensive alert management including
    rule evaluation, notification delivery, and alert lifecycle management.
    """
    
    def __init__(self, max_alerts: int = 1000, enable_sound: bool = False):
        self.max_alerts = max_alerts
        self.enable_sound = enable_sound
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=max_alerts)
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Thread safety
        self.alerts_lock = threading.RLock()
        self.rules_lock = threading.RLock()
        
        # Notification callbacks
        self.notification_callbacks: List[Callable[[Alert], None]] = []
        
        # Background processing
        self.processing_thread: Optional[threading.Thread] = None
        self.processing_active = threading.Event()
        self.shutdown_event = threading.Event()
        
        # Alert statistics
        self.stats = {
            'total_alerts': 0,
            'alerts_by_level': {level.value: 0 for level in AlertLevel},
            'alerts_by_type': {alert_type.value: 0 for alert_type in AlertType},
            'acknowledged_alerts': 0,
            'resolved_alerts': 0
        }
        
        # Initialize default rules
        self._initialize_default_rules()
        
        logger.info("AlertManager initialized")
    
    def _initialize_default_rules(self) -> None:
        """Initialize default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                alert_type=AlertType.RESOURCE,
                level=AlertLevel.WARNING,
                condition="greater_than",
                threshold=500.0,  # 500MB
                duration_seconds=30.0,
                cooldown_seconds=120.0
            ),
            AlertRule(
                rule_id="critical_memory_usage",
                name="Critical Memory Usage",
                alert_type=AlertType.RESOURCE,
                level=AlertLevel.CRITICAL,
                condition="greater_than",
                threshold=1000.0,  # 1GB
                duration_seconds=10.0,
                cooldown_seconds=60.0
            ),
            AlertRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                alert_type=AlertType.ERROR_RATE,
                level=AlertLevel.ERROR,
                condition="greater_than",
                threshold=50.0,  # 50%
                duration_seconds=60.0,
                cooldown_seconds=180.0
            ),
            AlertRule(
                rule_id="low_success_rate",
                name="Low Success Rate",
                alert_type=AlertType.PERFORMANCE,
                level=AlertLevel.WARNING,
                condition="less_than",
                threshold=80.0,  # 80%
                duration_seconds=120.0,
                cooldown_seconds=300.0
            ),
            AlertRule(
                rule_id="instance_unresponsive",
                name="Instance Unresponsive",
                alert_type=AlertType.INSTANCE_HEALTH,
                level=AlertLevel.CRITICAL,
                condition="greater_than",
                threshold=300.0,  # 5 minutes idle
                duration_seconds=0.0,  # Immediate
                cooldown_seconds=60.0
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        with self.rules_lock:
            self.alert_rules[rule.rule_id] = rule
            logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        with self.rules_lock:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                logger.info(f"Removed alert rule: {rule_id}")
                return True
            return False
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get an alert rule by ID."""
        with self.rules_lock:
            return self.alert_rules.get(rule_id)
    
    def list_rules(self) -> List[AlertRule]:
        """Get all alert rules."""
        with self.rules_lock:
            return list(self.alert_rules.values())
    
    def evaluate_rules(self, metrics: Dict[str, float], source: str = "system") -> List[Alert]:
        """Evaluate all rules against current metrics."""
        triggered_alerts = []
        
        with self.rules_lock:
            for rule in self.alert_rules.values():
                # Get metric value for this rule
                metric_key = self._get_metric_key_for_rule(rule)
                if metric_key not in metrics:
                    continue
                
                current_value = metrics[metric_key]
                
                # Check if rule should trigger
                if rule.should_trigger(current_value):
                    alert = self._create_alert_from_rule(rule, current_value, source)
                    triggered_alerts.append(alert)
                    
                    # Update rule last triggered time
                    rule.last_triggered = datetime.utcnow()
        
        return triggered_alerts
    
    def _get_metric_key_for_rule(self, rule: AlertRule) -> str:
        """Get the metric key that corresponds to an alert rule."""
        # Map rule types to metric keys
        metric_mapping = {
            "high_memory_usage": "memory_usage_mb",
            "critical_memory_usage": "memory_usage_mb",
            "high_error_rate": "error_rate",
            "low_success_rate": "success_rate",
            "instance_unresponsive": "idle_time_seconds"
        }
        return metric_mapping.get(rule.rule_id, "unknown")
    
    def _create_alert_from_rule(self, rule: AlertRule, current_value: float, source: str) -> Alert:
        """Create an alert from a triggered rule."""
        import uuid
        
        alert_id = f"{rule.rule_id}_{uuid.uuid4().hex[:8]}"
        
        # Create descriptive message
        message = f"{rule.name}: Current value {current_value:.2f} exceeds threshold {rule.threshold:.2f}"
        
        alert = Alert(
            alert_id=alert_id,
            level=rule.level,
            alert_type=rule.alert_type,
            title=rule.name,
            message=message,
            source=source,
            metadata={
                'rule_id': rule.rule_id,
                'current_value': current_value,
                'threshold': rule.threshold,
                'condition': rule.condition
            }
        )
        
        return alert
    
    def add_alert(self, alert: Alert) -> None:
        """Add a new alert."""
        with self.alerts_lock:
            # Add to active alerts
            self.active_alerts[alert.alert_id] = alert
            
            # Add to history
            self.alert_history.append(alert)
            
            # Update statistics
            self.stats['total_alerts'] += 1
            self.stats['alerts_by_level'][alert.level.value] += 1
            self.stats['alerts_by_type'][alert.alert_type.value] += 1
            
            logger.warning(f"New {alert.level.value} alert: {alert.title}")
        
        # Send notifications
        self._send_notifications(alert)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self.alerts_lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.acknowledge()
                self.stats['acknowledged_alerts'] += 1
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self.alerts_lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolve()
                self.stats['resolved_alerts'] += 1
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert resolved: {alert_id}")
                return True
            return False
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by level."""
        with self.alerts_lock:
            alerts = list(self.active_alerts.values())
            
            if level:
                alerts = [alert for alert in alerts if alert.level == level]
            
            # Sort by timestamp (newest first)
            alerts.sort(key=lambda a: a.timestamp, reverse=True)
            
            return alerts
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        with self.alerts_lock:
            history = list(self.alert_history)
            return history[-limit:] if limit else history
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self.alerts_lock:
            stats = self.stats.copy()
            stats['active_alerts'] = len(self.active_alerts)
            stats['unacknowledged_alerts'] = len([
                alert for alert in self.active_alerts.values() 
                if not alert.acknowledged
            ])
            stats['critical_alerts'] = len([
                alert for alert in self.active_alerts.values()
                if alert.level == AlertLevel.CRITICAL
            ])
            return stats
    
    def add_notification_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a notification callback function."""
        self.notification_callbacks.append(callback)
        logger.info("Added notification callback")
    
    def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        # Call notification callbacks
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")
        
        # Play sound for critical alerts
        if self.enable_sound and alert.level == AlertLevel.CRITICAL:
            self._play_alert_sound()
    
    def _play_alert_sound(self) -> None:
        """Play alert sound (simplified implementation)."""
        try:
            # In a real implementation, would use a proper audio library
            # For now, just log the sound alert
            logger.warning("🔊 CRITICAL ALERT SOUND")
            
            # Could use system bell or audio file
            # import winsound  # Windows
            # winsound.Beep(1000, 500)
            
            # Or use cross-platform solution
            # import pygame
            # pygame.mixer.init()
            # pygame.mixer.Sound("alert.wav").play()
            
        except Exception as e:
            logger.error(f"Failed to play alert sound: {e}")
    
    def start_processing(self) -> None:
        """Start background alert processing."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_active.set()
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                name="AlertProcessingThread",
                daemon=True
            )
            self.processing_thread.start()
            logger.info("Alert processing started")
    
    def stop_processing(self) -> None:
        """Stop background alert processing."""
        self.processing_active.clear()
        self.shutdown_event.set()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        logger.info("Alert processing stopped")
    
    def _processing_loop(self) -> None:
        """Background processing loop for alert management."""
        while self.processing_active.is_set() and not self.shutdown_event.is_set():
            try:
                # Clean up old resolved alerts
                self._cleanup_old_alerts()
                
                # Auto-resolve alerts that are no longer relevant
                self._auto_resolve_alerts()
                
                # Sleep for processing interval
                time.sleep(10.0)
                
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                time.sleep(5.0)
    
    def _cleanup_old_alerts(self) -> None:
        """Clean up old resolved alerts."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)  # Keep for 24 hours
        
        with self.alerts_lock:
            # Remove old alerts from history
            while (self.alert_history and 
                   self.alert_history[0].timestamp < cutoff_time and 
                   self.alert_history[0].resolved):
                self.alert_history.popleft()
    
    def _auto_resolve_alerts(self) -> None:
        """Auto-resolve alerts that are no longer relevant."""
        # This would contain logic to automatically resolve alerts
        # when conditions are no longer met
        pass
    
    def clear_all_alerts(self) -> None:
        """Clear all active alerts (for testing/reset)."""
        with self.alerts_lock:
            self.active_alerts.clear()
            logger.info("All active alerts cleared")
    
    def create_custom_alert(
        self, 
        title: str, 
        message: str, 
        level: AlertLevel = AlertLevel.INFO,
        source: str = "custom",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create a custom alert."""
        import uuid
        
        alert = Alert(
            alert_id=f"custom_{uuid.uuid4().hex[:8]}",
            level=level,
            alert_type=AlertType.CUSTOM,
            title=title,
            message=message,
            source=source,
            metadata=metadata or {}
        )
        
        self.add_alert(alert)
        return alert