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
       