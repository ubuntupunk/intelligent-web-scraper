"""
Scraping Results Context Provider.

This module provides scraping results context for result processing,
demonstrating advanced context patterns for data analysis and reporting.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase


class ScrapedItem:
    """Represents a single scraped item with quality metrics."""
    
    def __init__(self, data: Dict[str, Any], quality_score: float = 0.0):
        self.data = data
        self.quality_score = quality_score
        self.scraped_at = datetime.utcnow()
        self.source_url: Optional[str] = None
        self.extraction_method: Optional[str] = None
        self.validation_errors: List[str] = []
    
    def add_validation_error(self, error: str) -> None:
        """Add a validation error to this item."""
        self.validation_errors.append(error)
    
    def is_valid(self) -> bool:
        """Check if this item is valid (no validation errors)."""
        return len(self.validation_errors) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "data": self.data,
            "quality_score": self.quality_score,
            "scraped_at": self.scraped_at.isoformat(),
            "source_url": self.source_url,
            "extraction_method": self.extraction_method,
            "validation_errors": self.validation_errors,
            "is_valid": self.is_valid()
        }


class ExtractionStatistics:
    """Statistics about the extraction process."""
    
    def __init__(self):
        self.total_items_found: int = 0
        self.total_items_extracted: int = 0
        self.total_items_valid: int = 0
        self.average_quality_score: float = 0.0
        self.extraction_success_rate: float = 0.0
        self.validation_success_rate: float = 0.0
        self.processing_time_seconds: float = 0.0
        self.pages_processed: int = 0
        self.errors_encountered: List[str] = []
        self.field_completion_rates: Dict[str, float] = {}
    
    def calculate_rates(self, items: List[ScrapedItem]) -> None:
        """Calculate success rates based on scraped items."""
        if not items:
            return
        
        self.total_items_extracted = len(items)
        self.total_items_valid = sum(1 for item in items if item.is_valid())
        
        if self.total_items_found > 0:
            self.extraction_success_rate = (self.total_items_extracted / self.total_items_found) * 100.0
        
        if self.total_items_extracted > 0:
            self.validation_success_rate = (self.total_items_valid / self.total_items_extracted) * 100.0
            self.average_quality_score = sum(item.quality_score for item in items) / len(items)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_items_found": self.total_items_found,
            "total_items_extracted": self.total_items_extracted,
            "total_items_valid": self.total_items_valid,
            "average_quality_score": self.average_quality_score,
            "extraction_success_rate": self.extraction_success_rate,
            "validation_success_rate": self.validation_success_rate,
            "processing_time_seconds": self.processing_time_seconds,
            "pages_processed": self.pages_processed,
            "errors_encountered": self.errors_encountered,
            "field_completion_rates": self.field_completion_rates
        }


class QualityMetrics:
    """Quality metrics for scraped data."""
    
    def __init__(self):
        self.completeness_score: float = 0.0
        self.accuracy_score: float = 0.0
        self.consistency_score: float = 0.0
        self.overall_quality_score: float = 0.0
        self.field_quality_scores: Dict[str, float] = {}
        self.quality_distribution: Dict[str, int] = {
            "excellent": 0,  # 90-100%
            "good": 0,       # 70-89%
            "fair": 0,       # 50-69%
            "poor": 0        # 0-49%
        }
    
    def calculate_distribution(self, items: List[ScrapedItem]) -> None:
        """Calculate quality score distribution."""
        self.quality_distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        
        for item in items:
            score = item.quality_score
            if score >= 90:
                self.quality_distribution["excellent"] += 1
            elif score >= 70:
                self.quality_distribution["good"] += 1
            elif score >= 50:
                self.quality_distribution["fair"] += 1
            else:
                self.quality_distribution["poor"] += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "completeness_score": self.completeness_score,
            "accuracy_score": self.accuracy_score,
            "consistency_score": self.consistency_score,
            "overall_quality_score": self.overall_quality_score,
            "field_quality_scores": self.field_quality_scores,
            "quality_distribution": self.quality_distribution
        }


class ScrapingResultsContextProvider(SystemPromptContextProviderBase):
    """
    Provides scraping results context for result processing.
    
    This context provider demonstrates advanced patterns for analyzing
    and presenting scraping results to agents for further processing,
    validation, and optimization recommendations.
    """
    
    def __init__(self, title: str = "Scraping Results Context"):
        super().__init__(title=title)
        self.results: List[ScrapedItem] = []
        self.extraction_statistics = ExtractionStatistics()
        self.quality_metrics = QualityMetrics()
        self.operation_metadata: Dict[str, Any] = {}
        self.analysis_timestamp = datetime.utcnow()
    
    def set_results(self, items: List[Dict[str, Any]], quality_scores: Optional[List[float]] = None) -> None:
        """
        Set the scraping results for context.
        
        Args:
            items: List of scraped data items
            quality_scores: Optional list of quality scores for each item
        """
        self.results = []
        quality_scores = quality_scores or [0.0] * len(items)
        
        for i, item_data in enumerate(items):
            quality_score = quality_scores[i] if i < len(quality_scores) else 0.0
            scraped_item = ScrapedItem(item_data, quality_score)
            self.results.append(scraped_item)
        
        # Update statistics and metrics
        self._update_statistics()
        self._update_quality_metrics()
        self.analysis_timestamp = datetime.utcnow()
    
    def add_result(self, item_data: Dict[str, Any], quality_score: float = 0.0) -> None:
        """
        Add a single result to the context.
        
        Args:
            item_data: Scraped data item
            quality_score: Quality score for the item
        """
        scraped_item = ScrapedItem(item_data, quality_score)
        self.results.append(scraped_item)
        
        # Update statistics and metrics
        self._update_statistics()
        self._update_quality_metrics()
    
    def set_operation_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Set metadata about the scraping operation.
        
        Args:
            metadata: Operation metadata (URL, strategy, timing, etc.)
        """
        self.operation_metadata = metadata
    
    def get_info(self) -> str:
        """
        Return formatted scraping results information for agent context.
        
        This method demonstrates how to present complex result analysis
        in a clear, actionable format for AI agents.
        """
        if not self.results:
            return self._get_no_results_context()
        
        context_parts = []
        
        # Header with summary
        context_parts.append("## Scraping Results Analysis")
        context_parts.append(f"**Analysis Time:** {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        context_parts.append(f"**Total Items:** {len(self.results)}")
        context_parts.append(f"**Average Quality:** {self.quality_metrics.overall_quality_score:.1f}%")
        context_parts.append("")
        
        # Operation metadata
        if self.operation_metadata:
            context_parts.append("### Operation Details")
            for key, value in self.operation_metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    context_parts.append(f"- **{key.title()}:** {value}")
            context_parts.append("")
        
        # Extraction statistics
        stats = self.extraction_statistics
        context_parts.append("### Extraction Statistics")
        context_parts.append(f"- **Items Found:** {stats.total_items_found}")
        context_parts.append(f"- **Items Extracted:** {stats.total_items_extracted}")
        context_parts.append(f"- **Items Valid:** {stats.total_items_valid}")
        context_parts.append(f"- **Extraction Success Rate:** {stats.extraction_success_rate:.1f}%")
        context_parts.append(f"- **Validation Success Rate:** {stats.validation_success_rate:.1f}%")
        context_parts.append(f"- **Processing Time:** {stats.processing_time_seconds:.2f} seconds")
        context_parts.append(f"- **Pages Processed:** {stats.pages_processed}")
        context_parts.append("")
        
        # Quality metrics
        quality = self.quality_metrics
        context_parts.append("### Quality Analysis")
        context_parts.append(f"- **Overall Quality:** {quality.overall_quality_score:.1f}%")
        context_parts.append(f"- **Completeness:** {quality.completeness_score:.1f}%")
        context_parts.append(f"- **Accuracy:** {quality.accuracy_score:.1f}%")
        context_parts.append(f"- **Consistency:** {quality.consistency_score:.1f}%")
        context_parts.append("")
        
        # Quality distribution
        context_parts.append("### Quality Distribution")
        dist = quality.quality_distribution
        total_items = sum(dist.values())
        if total_items > 0:
            context_parts.append(f"- **Excellent (90-100%):** {dist['excellent']} items ({dist['excellent']/total_items*100:.1f}%)")
            context_parts.append(f"- **Good (70-89%):** {dist['good']} items ({dist['good']/total_items*100:.1f}%)")
            context_parts.append(f"- **Fair (50-69%):** {dist['fair']} items ({dist['fair']/total_items*100:.1f}%)")
            context_parts.append(f"- **Poor (0-49%):** {dist['poor']} items ({dist['poor']/total_items*100:.1f}%)")
        context_parts.append("")
        
        # Field completion rates
        if stats.field_completion_rates:
            context_parts.append("### Field Completion Rates")
            for field, rate in sorted(stats.field_completion_rates.items(), key=lambda x: x[1], reverse=True):
                context_parts.append(f"- **{field.title()}:** {rate:.1f}%")
            context_parts.append("")
        
        # Sample data
        context_parts.append("### Sample Results")
        sample_size = min(3, len(self.results))
        best_results = sorted(self.results, key=lambda x: x.quality_score, reverse=True)[:sample_size]
        
        for i, item in enumerate(best_results, 1):
            context_parts.append(f"**Sample {i} (Quality: {item.quality_score:.1f}%)**")
            # Show first few fields of the item
            sample_fields = list(item.data.items())[:3]
            for key, value in sample_fields:
                if isinstance(value, str) and len(value) > 50:
                    value = value[:47] + "..."
                context_parts.append(f"- {key}: {value}")
            context_parts.append("")
        
        # Errors and issues
        if stats.errors_encountered:
            context_parts.append("### Errors Encountered")
            for error in stats.errors_encountered[:5]:  # Show first 5 errors
                context_parts.append(f"- {error}")
            if len(stats.errors_encountered) > 5:
                context_parts.append(f"- ... and {len(stats.errors_encountered) - 5} more errors")
            context_parts.append("")
        
        # Recommendations
        context_parts.append("### Recommendations")
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            context_parts.append(f"- {rec}")
        
        return "\n".join(context_parts)
    
    def _get_no_results_context(self) -> str:
        """Return context when no results are available."""
        return """## Scraping Results Context

**Status:** No scraping results available

### Analysis Pending
- Results will be analyzed once scraping operation completes
- Quality metrics will be calculated automatically
- Recommendations will be generated based on actual data

### Expected Analysis
- Extraction success rates and validation metrics
- Quality score distribution and field completion rates
- Sample data preview and error analysis
- Optimization recommendations for future operations"""
    
    def _update_statistics(self) -> None:
        """Update extraction statistics based on current results."""
        if not self.results:
            return
        
        stats = self.extraction_statistics
        stats.total_items_extracted = len(self.results)
        stats.total_items_valid = sum(1 for item in self.results if item.is_valid())
        
        if self.results:
            stats.average_quality_score = sum(item.quality_score for item in self.results) / len(self.results)
        
        # Calculate field completion rates
        if self.results:
            all_fields = set()
            for item in self.results:
                all_fields.update(item.data.keys())
            
            field_completion = {}
            for field in all_fields:
                completed = sum(1 for item in self.results if field in item.data and item.data[field])
                field_completion[field] = (completed / len(self.results)) * 100.0
            
            stats.field_completion_rates = field_completion
        
        # Update success rates
        stats.calculate_rates(self.results)
    
    def _update_quality_metrics(self) -> None:
        """Update quality metrics based on current results."""
        if not self.results:
            return
        
        quality = self.quality_metrics
        
        # Calculate overall quality score
        if self.results:
            quality.overall_quality_score = sum(item.quality_score for item in self.results) / len(self.results)
        
        # Calculate quality distribution
        quality.calculate_distribution(self.results)
        
        # Estimate component scores (in a real implementation, these would be calculated properly)
        quality.completeness_score = quality.overall_quality_score * 0.9  # Assume high completeness
        quality.accuracy_score = quality.overall_quality_score * 1.1  # Assume good accuracy
        quality.consistency_score = quality.overall_quality_score * 0.8  # Assume moderate consistency
        
        # Ensure scores don't exceed 100%
        quality.completeness_score = min(100.0, quality.completeness_score)
        quality.accuracy_score = min(100.0, quality.accuracy_score)
        quality.consistency_score = min(100.0, quality.consistency_score)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results analysis."""
        recommendations = []
        
        if not self.results:
            return ["Complete scraping operation to generate recommendations"]
        
        stats = self.extraction_statistics
        quality = self.quality_metrics
        
        # Quality-based recommendations
        if quality.overall_quality_score < 50:
            recommendations.append("Low quality scores detected - review extraction selectors and validation rules")
        elif quality.overall_quality_score < 70:
            recommendations.append("Moderate quality scores - consider refining extraction strategies")
        else:
            recommendations.append("Good quality scores achieved - current strategy is effective")
        
        # Success rate recommendations
        if stats.validation_success_rate < 80:
            recommendations.append("High validation failure rate - review data validation criteria")
        
        # Field completion recommendations
        if stats.field_completion_rates:
            low_completion_fields = [
                field for field, rate in stats.field_completion_rates.items() 
                if rate < 50
            ]
            if low_completion_fields:
                recommendations.append(
                    f"Low completion rates for fields: {', '.join(low_completion_fields[:3])} - "
                    "consider improving selectors or making fields optional"
                )
        
        # Error-based recommendations
        if stats.errors_encountered:
            recommendations.append(f"{len(stats.errors_encountered)} errors encountered - review error log for optimization opportunities")
        
        # Performance recommendations
        if stats.processing_time_seconds > 30:
            recommendations.append("Long processing time - consider optimizing request delays or implementing concurrent processing")
        
        # Distribution-based recommendations
        if quality.quality_distribution["poor"] > len(self.results) * 0.2:  # More than 20% poor quality
            recommendations.append("High percentage of poor quality items - consider stricter filtering or improved extraction")
        
        return recommendations or ["Results look good - no specific optimizations needed"]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for programmatic use.
        
        Returns:
            Dictionary containing key statistics
        """
        return {
            "total_items": len(self.results),
            "average_quality": self.quality_metrics.overall_quality_score,
            "extraction_success_rate": self.extraction_statistics.extraction_success_rate,
            "validation_success_rate": self.extraction_statistics.validation_success_rate,
            "processing_time": self.extraction_statistics.processing_time_seconds,
            "error_count": len(self.extraction_statistics.errors_encountered),
            "quality_distribution": self.quality_metrics.quality_distribution
        }
    
    def get_top_quality_items(self, limit: int = 5) -> List[ScrapedItem]:
        """
        Get the highest quality items.
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of top quality scraped items
        """
        return sorted(self.results, key=lambda x: x.quality_score, reverse=True)[:limit]
    
    def get_failed_items(self) -> List[ScrapedItem]:
        """
        Get items that failed validation.
        
        Returns:
            List of items with validation errors
        """
        return [item for item in self.results if not item.is_valid()]
    
    def clear_results(self) -> None:
        """Clear all results and reset statistics."""
        self.results = []
        self.extraction_statistics = ExtractionStatistics()
        self.quality_metrics = QualityMetrics()
        self.operation_metadata = {}
        self.analysis_timestamp = datetime.utcnow()