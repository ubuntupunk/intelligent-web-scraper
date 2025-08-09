"""
Scraping Results Context Provider.

This module provides scraping results context for result processing,
demonstrating how to inject dynamic scraping statistics and quality
metrics into agent prompts for enhanced decision-making.
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase


class ScrapedItem:
    """Represents a single scraped item with quality metrics."""
    
    def __init__(self, data: Dict[str, Any], quality_score: float = 0.0, source_url: str = ""):
        self.data = data
        self.quality_score = quality_score
        self.source_url = source_url
        self.scraped_at = datetime.utcnow()
        self.field_completeness: Dict[str, float] = {}
        self.validation_errors: List[str] = []
        self.extraction_metadata: Dict[str, Any] = {}
    
    def add_validation_error(self, error: str) -> None:
        """Add a validation error for this item."""
        self.validation_errors.append(error)
    
    def set_field_completeness(self, field: str, completeness: float) -> None:
        """Set completeness score for a specific field."""
        self.field_completeness[field] = completeness
    
    def get_overall_completeness(self) -> float:
        """Calculate overall completeness score."""
        if not self.field_completeness:
            return 0.0
        return sum(self.field_completeness.values()) / len(self.field_completeness)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "data": self.data,
            "quality_score": self.quality_score,
            "source_url": self.source_url,
            "scraped_at": self.scraped_at.isoformat(),
            "field_completeness": self.field_completeness,
            "validation_errors": self.validation_errors,
            "extraction_metadata": self.extraction_metadata,
            "overall_completeness": self.get_overall_completeness()
        }


class ExtractionStatistics:
    """Statistics about the extraction process."""
    
    def __init__(self):
        self.total_items_found = 0
        self.total_items_extracted = 0
        self.total_items_validated = 0
        self.total_processing_time = 0.0
        self.pages_processed = 0
        self.extraction_rate = 0.0  # items per second
        self.success_rate = 0.0  # percentage of successful extractions
        self.validation_rate = 0.0  # percentage of items passing validation
        self.average_quality_score = 0.0
        self.field_extraction_rates: Dict[str, float] = {}
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.processing_times: List[float] = []
        self.quality_distribution: Dict[str, int] = defaultdict(int)  # quality ranges
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
    
    def start_extraction(self) -> None:
        """Mark the start of extraction process."""
        self.started_at = datetime.utcnow()
    
    def complete_extraction(self) -> None:
        """Mark the completion of extraction process."""
        self.completed_at = datetime.utcnow()
        self._calculate_derived_metrics()
    
    def add_processing_time(self, time_seconds: float) -> None:
        """Add a processing time measurement."""
        self.processing_times.append(time_seconds)
        self.total_processing_time += time_seconds
    
    def add_error(self, error_type: str) -> None:
        """Add an error count."""
        self.error_counts[error_type] += 1
    
    def add_quality_score(self, score: float) -> None:
        """Add a quality score to the distribution."""
        if score >= 90:
            self.quality_distribution["excellent"] += 1
        elif score >= 70:
            self.quality_distribution["good"] += 1
        elif score >= 50:
            self.quality_distribution["fair"] += 1
        else:
            self.quality_distribution["poor"] += 1
    
    def _calculate_derived_metrics(self) -> None:
        """Calculate derived metrics from collected data."""
        if self.total_items_found > 0:
            self.success_rate = (self.total_items_extracted / self.total_items_found) * 100
        
        if self.total_items_extracted > 0:
            self.validation_rate = (self.total_items_validated / self.total_items_extracted) * 100
        
        if self.total_processing_time > 0:
            self.extraction_rate = self.total_items_extracted / self.total_processing_time
        
        if self.processing_times:
            self.average_quality_score = sum(self.processing_times) / len(self.processing_times)
    
    def get_duration(self) -> Optional[timedelta]:
        """Get the total duration of extraction process."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return datetime.utcnow() - self.started_at
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        duration = self.get_duration()
        return {
            "total_items_found": self.total_items_found,
            "total_items_extracted": self.total_items_extracted,
            "total_items_validated": self.total_items_validated,
            "total_processing_time": self.total_processing_time,
            "pages_processed": self.pages_processed,
            "extraction_rate": self.extraction_rate,
            "success_rate": self.success_rate,
            "validation_rate": self.validation_rate,
            "average_quality_score": self.average_quality_score,
            "field_extraction_rates": self.field_extraction_rates,
            "error_counts": dict(self.error_counts),
            "quality_distribution": dict(self.quality_distribution),
            "duration_seconds": duration.total_seconds() if duration else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class QualityMetrics:
    """Quality metrics for scraped data."""
    
    def __init__(self):
        self.overall_quality_score = 0.0
        self.completeness_score = 0.0
        self.accuracy_score = 0.0
        self.consistency_score = 0.0
        self.field_quality_scores: Dict[str, float] = {}
        self.quality_trends: deque = deque(maxlen=100)  # Last 100 quality measurements
        self.quality_threshold_violations = 0
        self.quality_improvements = 0
        self.quality_degradations = 0
        self.last_quality_check = datetime.utcnow()
    
    def update_quality_score(self, score: float) -> None:
        """Update overall quality score and track trends."""
        previous_score = self.overall_quality_score
        self.overall_quality_score = score
        self.quality_trends.append({
            "timestamp": datetime.utcnow().isoformat(),
            "score": score
        })
        
        # Track quality changes
        if previous_score > 0:
            if score > previous_score:
                self.quality_improvements += 1
            elif score < previous_score:
                self.quality_degradations += 1
        
        self.last_quality_check = datetime.utcnow()
    
    def update_field_quality(self, field: str, score: float) -> None:
        """Update quality score for a specific field."""
        self.field_quality_scores[field] = score
    
    def add_threshold_violation(self) -> None:
        """Record a quality threshold violation."""
        self.quality_threshold_violations += 1
    
    def get_quality_trend(self, window_size: int = 10) -> str:
        """Get quality trend over recent measurements."""
        if len(self.quality_trends) < window_size:
            return "insufficient_data"
        
        recent_scores = [item["score"] for item in list(self.quality_trends)[-window_size:]]
        early_avg = sum(recent_scores[:window_size//2]) / (window_size//2)
        late_avg = sum(recent_scores[window_size//2:]) / (window_size//2)
        
        if late_avg > early_avg + 5:
            return "improving"
        elif late_avg < early_avg - 5:
            return "declining"
        else:
            return "stable"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "overall_quality_score": self.overall_quality_score,
            "completeness_score": self.completeness_score,
            "accuracy_score": self.accuracy_score,
            "consistency_score": self.consistency_score,
            "field_quality_scores": self.field_quality_scores,
            "quality_trend": self.get_quality_trend(),
            "quality_threshold_violations": self.quality_threshold_violations,
            "quality_improvements": self.quality_improvements,
            "quality_degradations": self.quality_degradations,
            "last_quality_check": self.last_quality_check.isoformat(),
            "recent_quality_scores": list(self.quality_trends)[-10:]  # Last 10 scores
        }


class ScrapingSession:
    """Represents a complete scraping session with results and metadata."""
    
    def __init__(self, session_id: str, target_url: str, strategy_used: str):
        self.session_id = session_id
        self.target_url = target_url
        self.strategy_used = strategy_used
        self.started_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        self.items: List[ScrapedItem] = []
        self.statistics = ExtractionStatistics()
        self.quality_metrics = QualityMetrics()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.session_metadata: Dict[str, Any] = {}
        self.status = "running"  # running, completed, failed, cancelled
    
    def add_item(self, item: ScrapedItem) -> None:
        """Add a scraped item to the session."""
        self.items.append(item)
        self.quality_metrics.update_quality_score(item.quality_score)
        self.statistics.add_quality_score(item.quality_score)
    
    def add_error(self, error: str, error_type: str = "general") -> None:
        """Add an error to the session."""
        self.errors.append(error)
        self.statistics.add_error(error_type)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the session."""
        self.warnings.append(warning)
    
    def complete_session(self, status: str = "completed") -> None:
        """Mark the session as completed."""
        self.completed_at = datetime.utcnow()
        self.status = status
        self.statistics.complete_extraction()
    
    def get_duration(self) -> timedelta:
        """Get session duration."""
        end_time = self.completed_at or datetime.utcnow()
        return end_time - self.started_at
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the scraping session."""
        duration = self.get_duration()
        return {
            "session_id": self.session_id,
            "target_url": self.target_url,
            "strategy_used": self.strategy_used,
            "status": self.status,
            "duration_seconds": duration.total_seconds(),
            "items_scraped": len(self.items),
            "average_quality": self.quality_metrics.overall_quality_score,
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "success_rate": self.statistics.success_rate,
            "extraction_rate": self.statistics.extraction_rate
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "target_url": self.target_url,
            "strategy_used": self.strategy_used,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "items": [item.to_dict() for item in self.items],
            "statistics": self.statistics.to_dict(),
            "quality_metrics": self.quality_metrics.to_dict(),
            "errors": self.errors,
            "warnings": self.warnings,
            "session_metadata": self.session_metadata,
            "summary": self.get_summary()
        }


class ScrapingResultsContextProvider(SystemPromptContextProviderBase):
    """
    Provides scraping results context for result processing.
    
    This context provider demonstrates how to inject dynamic scraping
    statistics, quality metrics, and extraction results into agent
    prompts for enhanced result processing and decision-making.
    """
    
    def __init__(self, title: str = "Scraping Results Context"):
        super().__init__(title=title)
        self.current_session: Optional[ScrapingSession] = None
        self.session_history: List[ScrapingSession] = []
        self.results: List[ScrapedItem] = []
        self.quality_metrics = QualityMetrics()
        self.extraction_statistics = ExtractionStatistics()
        self.aggregated_metrics: Dict[str, Any] = {}
        self.max_history_size = 50  # Keep last 50 sessions
        self.context_cache: Dict[str, str] = {}
        self.cache_ttl_seconds = 300  # 5 minutes cache
        self.last_cache_update = datetime.utcnow()
    
    def start_new_session(self, session_id: str, target_url: str, strategy_used: str) -> ScrapingSession:
        """
        Start a new scraping session.
        
        Args:
            session_id: Unique identifier for the session
            target_url: URL being scraped
            strategy_used: Scraping strategy being used
            
        Returns:
            New ScrapingSession instance
        """
        session = ScrapingSession(session_id, target_url, strategy_used)
        self.current_session = session
        
        # Initialize statistics
        self.extraction_statistics = ExtractionStatistics()
        self.extraction_statistics.start_extraction()
        
        # Clear cache
        self._clear_context_cache()
        
        return session
    
    def complete_current_session(self, status: str = "completed") -> None:
        """
        Complete the current scraping session.
        
        Args:
            status: Final status of the session
        """
        if self.current_session:
            self.current_session.complete_session(status)
            
            # Add to history
            self.session_history.append(self.current_session)
            
            # Maintain history size limit
            if len(self.session_history) > self.max_history_size:
                self.session_history = self.session_history[-self.max_history_size:]
            
            # Update aggregated metrics
            self._update_aggregated_metrics()
            
            # Clear cache
            self._clear_context_cache()
    
    def add_scraped_item(self, item: ScrapedItem) -> None:
        """
        Add a scraped item to the current session.
        
        Args:
            item: Scraped item to add
        """
        self.results.append(item)
        self.quality_metrics.update_quality_score(item.quality_score)
        self.extraction_statistics.add_quality_score(item.quality_score)
        
        if self.current_session:
            self.current_session.add_item(item)
        
        # Clear cache to force refresh
        self._clear_context_cache()
    
    def add_extraction_error(self, error: str, error_type: str = "extraction") -> None:
        """
        Add an extraction error.
        
        Args:
            error: Error message
            error_type: Type of error for categorization
        """
        self.extraction_statistics.add_error(error_type)
        
        if self.current_session:
            self.current_session.add_error(error, error_type)
        
        # Clear cache
        self._clear_context_cache()
    
    def update_extraction_statistics(self, **kwargs) -> None:
        """
        Update extraction statistics.
        
        Args:
            **kwargs: Statistics to update
        """
        for key, value in kwargs.items():
            if hasattr(self.extraction_statistics, key):
                setattr(self.extraction_statistics, key, value)
        
        # Clear cache
        self._clear_context_cache()
    
    def set_results(self, items: List[Dict[str, Any]], quality_scores: List[float]) -> None:
        """
        Set scraping results with quality scores.
        
        Args:
            items: List of scraped items as dictionaries
            quality_scores: List of quality scores corresponding to items
        """
        self.results.clear()
        
        # Convert items to ScrapedItem objects
        for i, item in enumerate(items):
            quality_score = quality_scores[i] if i < len(quality_scores) else 0.0
            scraped_item = ScrapedItem(
                data=item,
                quality_score=quality_score,
                source_url=getattr(self.current_session, 'target_url', '')
            )
            self.results.append(scraped_item)
            
            # Update quality metrics
            self.quality_metrics.update_quality_score(quality_score)
            self.extraction_statistics.add_quality_score(quality_score)
            
            # Add to current session if exists
            if self.current_session:
                self.current_session.add_item(scraped_item)
        
        # Clear cache to force refresh
        self._clear_context_cache()
    
    def set_operation_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Set operation metadata for the current session.
        
        Args:
            metadata: Metadata dictionary to set
        """
        if self.current_session:
            self.current_session.metadata.update(metadata)
        
        # Clear cache
        self._clear_context_cache()
    
    def get_info(self) -> str:
        """
        Return formatted scraping results information for agent context.
        
        This method demonstrates how to format complex scraping results
        and statistics into clear, actionable context for AI agents.
        """
        # Check cache first
        cache_key = "main_context"
        if self._is_cache_valid(cache_key):
            return self.context_cache[cache_key]
        
        context_parts = []
        
        # Header with current session info
        if self.current_session:
            session = self.current_session
            duration = session.get_duration()
            context_parts.append(f"## Current Scraping Session: {session.session_id}")
            context_parts.append(f"**Target URL:** {session.target_url}")
            context_parts.append(f"**Strategy:** {session.strategy_used}")
            context_parts.append(f"**Status:** {session.status}")
            context_parts.append(f"**Duration:** {duration.total_seconds():.1f} seconds")
            context_parts.append(f"**Items Scraped:** {len(session.items)}")
            context_parts.append("")
        else:
            context_parts.append("## Scraping Results Context")
            context_parts.append("**Status:** No active scraping session")
            context_parts.append("")
        
        # Current results summary
        if self.results:
            context_parts.append("### Current Results Summary")
            context_parts.append(f"**Total Items:** {len(self.results)}")
            context_parts.append(f"**Average Quality Score:** {self.quality_metrics.overall_quality_score:.1f}%")
            context_parts.append(f"**Quality Trend:** {self.quality_metrics.get_quality_trend()}")
            
            # Quality distribution
            quality_dist = self._get_quality_distribution()
            if quality_dist:
                context_parts.append("**Quality Distribution:**")
                for quality_level, count in quality_dist.items():
                    percentage = (count / len(self.results)) * 100
                    context_parts.append(f"  - {quality_level.title()}: {count} items ({percentage:.1f}%)")
            
            context_parts.append("")
        
        # Extraction statistics
        if self.extraction_statistics.total_items_extracted > 0:
            context_parts.append("### Extraction Statistics")
            stats = self.extraction_statistics
            context_parts.append(f"**Items Found:** {stats.total_items_found}")
            context_parts.append(f"**Items Extracted:** {stats.total_items_extracted}")
            context_parts.append(f"**Success Rate:** {stats.success_rate:.1f}%")
            context_parts.append(f"**Extraction Rate:** {stats.extraction_rate:.2f} items/second")
            context_parts.append(f"**Pages Processed:** {stats.pages_processed}")
            
            if stats.error_counts:
                context_parts.append("**Error Summary:**")
                for error_type, count in stats.error_counts.items():
                    context_parts.append(f"  - {error_type.title()}: {count} errors")
            
            context_parts.append("")
        
        # Field-level analysis
        field_analysis = self._get_field_analysis()
        if field_analysis:
            context_parts.append("### Field-Level Analysis")
            for field_name, analysis in field_analysis.items():
                context_parts.append(f"**{field_name.title()}:**")
                context_parts.append(f"  - Extraction Rate: {analysis['extraction_rate']:.1f}%")
                context_parts.append(f"  - Average Quality: {analysis['avg_quality']:.1f}%")
                context_parts.append(f"  - Completeness: {analysis['completeness']:.1f}%")
            context_parts.append("")
        
        # Recent session history
        if self.session_history:
            context_parts.append("### Recent Session History")
            recent_sessions = self.session_history[-5:]  # Last 5 sessions
            for session in recent_sessions:
                summary = session.get_summary()
                context_parts.append(
                    f"- **{session.session_id}**: {summary['items_scraped']} items, "
                    f"{summary['average_quality']:.1f}% quality, "
                    f"{summary['success_rate']:.1f}% success rate"
                )
            context_parts.append("")
        
        # Performance insights and recommendations
        insights = self._generate_performance_insights()
        if insights:
            context_parts.append("### Performance Insights")
            for insight in insights:
                context_parts.append(f"- {insight}")
            context_parts.append("")
        
        # Recommendations for improvement
        recommendations = self._generate_improvement_recommendations()
        if recommendations:
            context_parts.append("### Recommendations")
            for rec in recommendations:
                context_parts.append(f"- {rec}")
        
        context_text = "\n".join(context_parts)
        
        # Cache the result
        self.context_cache[cache_key] = context_text
        self.last_cache_update = datetime.utcnow()
        
        return context_text
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a programmatic summary of scraping results.
        
        Returns:
            Dictionary containing results summary
        """
        if not self.results:
            return {"total_items": 0, "average_quality": 0.0, "status": "no_results"}
        
        quality_scores = [item.quality_score for item in self.results]
        
        return {
            "total_items": len(self.results),
            "average_quality": sum(quality_scores) / len(quality_scores),
            "min_quality": min(quality_scores),
            "max_quality": max(quality_scores),
            "quality_distribution": self._get_quality_distribution(),
            "field_analysis": self._get_field_analysis(),
            "extraction_statistics": self.extraction_statistics.to_dict(),
            "quality_metrics": self.quality_metrics.to_dict(),
            "current_session": self.current_session.get_summary() if self.current_session else None,
            "session_count": len(self.session_history)
        }
    
    def get_quality_report(self) -> Dict[str, Any]:
        """
        Get a detailed quality report.
        
        Returns:
            Dictionary containing quality analysis
        """
        if not self.results:
            return {"status": "no_data"}
        
        # Calculate quality metrics
        quality_scores = [item.quality_score for item in self.results]
        completeness_scores = [item.get_overall_completeness() for item in self.results]
        
        # Identify quality issues
        low_quality_items = [item for item in self.results if item.quality_score < 50]
        validation_errors = []
        for item in self.results:
            validation_errors.extend(item.validation_errors)
        
        return {
            "overall_quality": {
                "average_score": sum(quality_scores) / len(quality_scores),
                "median_score": sorted(quality_scores)[len(quality_scores) // 2],
                "quality_range": max(quality_scores) - min(quality_scores),
                "standard_deviation": self._calculate_std_dev(quality_scores)
            },
            "completeness_analysis": {
                "average_completeness": sum(completeness_scores) / len(completeness_scores),
                "items_with_missing_fields": len([s for s in completeness_scores if s < 100]),
                "field_completeness": self._get_field_completeness_analysis()
            },
            "quality_issues": {
                "low_quality_count": len(low_quality_items),
                "low_quality_percentage": (len(low_quality_items) / len(self.results)) * 100,
                "validation_errors": len(validation_errors),
                "common_errors": self._get_common_validation_errors()
            },
            "recommendations": self._generate_quality_improvement_recommendations()
        }
    
    def _get_quality_distribution(self) -> Dict[str, int]:
        """Get distribution of quality scores."""
        distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        
        for item in self.results:
            score = item.quality_score
            if score >= 90:
                distribution["excellent"] += 1
            elif score >= 70:
                distribution["good"] += 1
            elif score >= 50:
                distribution["fair"] += 1
            else:
                distribution["poor"] += 1
        
        return distribution
    
    def _get_field_analysis(self) -> Dict[str, Dict[str, float]]:
        """Analyze extraction performance by field."""
        if not self.results:
            return {}
        
        field_stats = defaultdict(lambda: {"extracted": 0, "total": 0, "quality_sum": 0, "completeness_sum": 0})
        
        for item in self.results:
            for field_name in item.data.keys():
                field_stats[field_name]["total"] += 1
                if item.data[field_name]:  # Field has content
                    field_stats[field_name]["extracted"] += 1
                    field_stats[field_name]["quality_sum"] += item.quality_score
                
                if field_name in item.field_completeness:
                    field_stats[field_name]["completeness_sum"] += item.field_completeness[field_name]
        
        # Calculate rates and averages
        analysis = {}
        for field_name, stats in field_stats.items():
            total = stats["total"]
            extracted = stats["extracted"]
            
            analysis[field_name] = {
                "extraction_rate": (extracted / total * 100) if total > 0 else 0,
                "avg_quality": (stats["quality_sum"] / extracted) if extracted > 0 else 0,
                "completeness": (stats["completeness_sum"] / total) if total > 0 else 0
            }
        
        return analysis
    
    def _get_field_completeness_analysis(self) -> Dict[str, float]:
        """Analyze field completeness across all items."""
        if not self.results:
            return {}
        
        field_completeness = defaultdict(list)
        
        for item in self.results:
            for field_name, completeness in item.field_completeness.items():
                field_completeness[field_name].append(completeness)
        
        # Calculate average completeness per field
        analysis = {}
        for field_name, completeness_scores in field_completeness.items():
            analysis[field_name] = sum(completeness_scores) / len(completeness_scores)
        
        return analysis
    
    def _get_common_validation_errors(self) -> List[Dict[str, Any]]:
        """Get most common validation errors."""
        error_counts = defaultdict(int)
        
        for item in self.results:
            for error in item.validation_errors:
                error_counts[error] += 1
        
        # Sort by frequency and return top 5
        common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return [{"error": error, "count": count} for error, count in common_errors]
    
    def _generate_performance_insights(self) -> List[str]:
        """Generate performance insights based on current data."""
        insights = []
        
        if not self.results:
            return ["No scraping results available for analysis"]
        
        # Quality insights
        avg_quality = self.quality_metrics.overall_quality_score
        if avg_quality > 80:
            insights.append(f"High quality extraction achieved ({avg_quality:.1f}% average)")
        elif avg_quality < 50:
            insights.append(f"Low quality extraction detected ({avg_quality:.1f}% average) - review extraction strategy")
        
        # Performance insights
        if self.extraction_statistics.extraction_rate > 0:
            rate = self.extraction_statistics.extraction_rate
            if rate > 5:
                insights.append(f"High extraction rate achieved ({rate:.1f} items/second)")
            elif rate < 1:
                insights.append(f"Low extraction rate detected ({rate:.1f} items/second) - consider optimization")
        
        # Error insights
        if self.extraction_statistics.error_counts:
            total_errors = sum(self.extraction_statistics.error_counts.values())
            if total_errors > len(self.results) * 0.1:  # More than 10% error rate
                insights.append(f"High error rate detected ({total_errors} errors) - review extraction logic")
        
        # Trend insights
        trend = self.quality_metrics.get_quality_trend()
        if trend == "improving":
            insights.append("Quality trend is improving over recent extractions")
        elif trend == "declining":
            insights.append("Quality trend is declining - investigate potential issues")
        
        return insights
    
    def _generate_improvement_recommendations(self) -> List[str]:
        """Generate recommendations for improving scraping results."""
        recommendations = []
        
        if not self.results:
            return ["Start scraping to generate recommendations"]
        
        # Quality-based recommendations
        avg_quality = self.quality_metrics.overall_quality_score
        if avg_quality < 70:
            recommendations.append("Consider refining extraction selectors to improve quality scores")
        
        # Field-based recommendations
        field_analysis = self._get_field_analysis()
        for field_name, analysis in field_analysis.items():
            if analysis["extraction_rate"] < 80:
                recommendations.append(f"Improve extraction for '{field_name}' field (current rate: {analysis['extraction_rate']:.1f}%)")
        
        # Error-based recommendations
        if self.extraction_statistics.error_counts:
            most_common_error = max(self.extraction_statistics.error_counts.items(), key=lambda x: x[1])
            recommendations.append(f"Address '{most_common_error[0]}' errors ({most_common_error[1]} occurrences)")
        
        # Performance recommendations
        if self.extraction_statistics.extraction_rate < 2:
            recommendations.append("Consider optimizing extraction logic for better performance")
        
        return recommendations
    
    def _generate_quality_improvement_recommendations(self) -> List[str]:
        """Generate specific quality improvement recommendations."""
        recommendations = []
        
        if not self.results:
            return recommendations
        
        # Calculate quality metrics directly to avoid recursion
        quality_scores = [item.quality_score for item in self.results]
        completeness_scores = [item.get_overall_completeness() for item in self.results]
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        avg_completeness = sum(completeness_scores) / len(completeness_scores)
        quality_range = max(quality_scores) - min(quality_scores)
        
        # Count validation errors
        validation_errors = sum(len(item.validation_errors) for item in self.results)
        low_quality_items = len([item for item in self.results if item.quality_score < 50])
        low_quality_percentage = (low_quality_items / len(self.results)) * 100
        
        # Overall quality recommendations
        if avg_quality < 60:
            recommendations.append("Overall quality is low - review and improve extraction selectors")
        
        if quality_range > 40:
            recommendations.append("High quality variance detected - standardize extraction approach")
        
        # Completeness recommendations
        if avg_completeness < 80:
            recommendations.append("Low field completeness - add fallback selectors for missing fields")
        
        # Error-based recommendations
        if validation_errors > 0:
            recommendations.append("Address validation errors to improve data quality")
        
        if low_quality_percentage > 20:
            recommendations.append("High percentage of low-quality items - review extraction strategy")
        
        return recommendations
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of values."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _update_aggregated_metrics(self) -> None:
        """Update aggregated metrics from session history."""
        if not self.session_history:
            return
        
        total_items = sum(len(session.items) for session in self.session_history)
        total_quality = sum(session.quality_metrics.overall_quality_score * len(session.items) 
                          for session in self.session_history)
        
        self.aggregated_metrics = {
            "total_sessions": len(self.session_history),
            "total_items_scraped": total_items,
            "average_quality_across_sessions": total_quality / total_items if total_items > 0 else 0,
            "average_items_per_session": total_items / len(self.session_history),
            "most_successful_strategy": self._get_most_successful_strategy(),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def _get_most_successful_strategy(self) -> str:
        """Determine the most successful scraping strategy."""
        if not self.session_history:
            return "unknown"
        
        strategy_performance = defaultdict(lambda: {"sessions": 0, "total_quality": 0, "total_items": 0})
        
        for session in self.session_history:
            strategy = session.strategy_used
            strategy_performance[strategy]["sessions"] += 1
            strategy_performance[strategy]["total_quality"] += session.quality_metrics.overall_quality_score
            strategy_performance[strategy]["total_items"] += len(session.items)
        
        # Find strategy with best average quality
        best_strategy = "unknown"
        best_score = 0
        
        for strategy, perf in strategy_performance.items():
            avg_quality = perf["total_quality"] / perf["sessions"]
            if avg_quality > best_score:
                best_score = avg_quality
                best_strategy = strategy
        
        return best_strategy
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid."""
        if cache_key not in self.context_cache:
            return False
        
        age_seconds = (datetime.utcnow() - self.last_cache_update).total_seconds()
        return age_seconds < self.cache_ttl_seconds
    
    def _clear_context_cache(self) -> None:
        """Clear the context cache."""
        self.context_cache.clear()
        self.last_cache_update = datetime.utcnow()
    
    def clear_results(self) -> None:
        """Clear all results and reset metrics."""
        self.results.clear()
        self.quality_metrics = QualityMetrics()
        self.extraction_statistics = ExtractionStatistics()
        self.current_session = None
        self._clear_context_cache()
    
    def export_results(self, format_type: str = "dict") -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Export results in specified format.
        
        Args:
            format_type: Export format ("dict", "list", "summary")
            
        Returns:
            Exported results in requested format
        """
        if format_type == "list":
            return [item.to_dict() for item in self.results]
        elif format_type == "summary":
            return self.get_results_summary()
        else:  # dict format
            return {
                "results": [item.to_dict() for item in self.results],
                "statistics": self.extraction_statistics.to_dict(),
                "quality_metrics": self.quality_metrics.to_dict(),
                "current_session": self.current_session.to_dict() if self.current_session else None,
                "session_history": [session.get_summary() for session in self.session_history],
                "aggregated_metrics": self.aggregated_metrics
            }
    
    def get_session_by_id(self, session_id: str) -> Optional[ScrapingSession]:
        """
        Get a session by its ID.
        
        Args:
            session_id: ID of the session to retrieve
            
        Returns:
            ScrapingSession if found, None otherwise
        """
        if self.current_session and self.current_session.session_id == session_id:
            return self.current_session
        
        for session in self.session_history:
            if session.session_id == session_id:
                return session
        
        return None
    
    def get_context_for_session(self, session_id: str) -> str:
        """
        Get context information for a specific session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Formatted context string for the session
        """
        session = self.get_session_by_id(session_id)
        if not session:
            return f"Session {session_id} not found"
        
        summary = session.get_summary()
        context_parts = [
            f"## Session Context: {session_id}",
            f"**Target URL:** {session.target_url}",
            f"**Strategy:** {session.strategy_used}",
            f"**Status:** {session.status}",
            f"**Duration:** {summary['duration_seconds']:.1f} seconds",
            f"**Items Scraped:** {summary['items_scraped']}",
            f"**Average Quality:** {summary['average_quality']:.1f}%",
            f"**Success Rate:** {summary['success_rate']:.1f}%",
            f"**Errors:** {summary['errors_count']}",
            f"**Warnings:** {summary['warnings_count']}",
            ""
        ]
        
        if session.errors:
            context_parts.append("### Errors Encountered")
            for error in session.errors[-5:]:  # Last 5 errors
                context_parts.append(f"- {error}")
            context_parts.append("")
        
        if session.warnings:
            context_parts.append("### Warnings")
            for warning in session.warnings[-3:]:  # Last 3 warnings
                context_parts.append(f"- {warning}")
        
        return "\n".join(context_parts)