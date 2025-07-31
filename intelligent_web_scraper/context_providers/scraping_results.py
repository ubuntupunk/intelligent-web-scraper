"""
Scraping Results Context Provider.

This module provides scraping results context for result processing,
demonstrating context provider patterns in atomic-agents.
"""

from typing import List, Dict, Any, Optional
from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase


class ScrapingResultsContextProvider(SystemPromptContextProviderBase):
    """Provides scraping results context for result processing."""
    
    def __init__(self, title: str = "Scraping Results Context"):
        super().__init__(title=title)
        self.results: List[Dict[str, Any]] = []
        self.quality_metrics: Dict[str, float] = {}
        self.extraction_statistics: Dict[str, int] = {}
        self.processing_metadata: Optional[Dict[str, Any]] = None
    
    def set_results(self, results: List[Dict[str, Any]]) -> None:
        """Set the scraping results."""
        self.results = results
        self._calculate_statistics()
    
    def set_quality_metrics(self, metrics: Dict[str, float]) -> None:
        """Set quality metrics for the results."""
        self.quality_metrics = metrics
    
    def set_processing_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set processing metadata."""
        self.processing_metadata = metadata
    
    def _calculate_statistics(self) -> None:
        """Calculate extraction statistics from results."""
        if not self.results:
            self.extraction_statistics = {}
            return
        
        self.extraction_statistics = {
            "total_items": len(self.results),
            "unique_fields": len(set().union(*(item.keys() for item in self.results))),
            "avg_fields_per_item": sum(len(item) for item in self.results) / len(self.results),
            "items_with_missing_data": sum(1 for item in self.results if any(v is None or v == "" for v in item.values()))
        }
    
    def get_info(self) -> str:
        """Return formatted scraping results information."""
        if not self.results and not self.quality_metrics:
            return "No scraping results available."
        
        info_parts = []
        
        # Extraction statistics
        if self.extraction_statistics:
            info_parts.append("Extraction Statistics:")
            info_parts.append(f"- Total Items: {self.extraction_statistics.get('total_items', 0)}")
            info_parts.append(f"- Unique Fields: {self.extraction_statistics.get('unique_fields', 0)}")
            info_parts.append(f"- Avg Fields per Item: {self.extraction_statistics.get('avg_fields_per_item', 0):.1f}")
            info_parts.append(f"- Items with Missing Data: {self.extraction_statistics.get('items_with_missing_data', 0)}")
        
        # Quality metrics
        if self.quality_metrics:
            info_parts.append("\nQuality Metrics:")
            for metric, value in self.quality_metrics.items():
                if isinstance(value, float):
                    info_parts.append(f"- {metric.replace('_', ' ').title()}: {value:.1%}")
                else:
                    info_parts.append(f"- {metric.replace('_', ' ').title()}: {value}")
        
        # Processing metadata
        if self.processing_metadata:
            info_parts.append("\nProcessing Information:")
            processing_time = self.processing_metadata.get("processing_time", 0)
            pages_processed = self.processing_metadata.get("pages_processed", 0)
            info_parts.append(f"- Processing Time: {processing_time:.2f}s")
            info_parts.append(f"- Pages Processed: {pages_processed}")
            
            if "errors" in self.processing_metadata:
                error_count = len(self.processing_metadata["errors"])
                info_parts.append(f"- Errors Encountered: {error_count}")
        
        # Sample data preview
        if self.results:
            info_parts.append("\nSample Data Fields:")
            sample_item = self.results[0]
            field_names = list(sample_item.keys())[:5]  # Show first 5 fields
            info_parts.append(f"- Available Fields: {', '.join(field_names)}")
            if len(sample_item) > 5:
                info_parts.append(f"- ... and {len(sample_item) - 5} more fields")
        
        return "\n".join(info_parts) if info_parts else "Scraping results processed but no detailed information available."