"""
Website Analysis Context Provider.

This module provides dynamic website analysis context to agents,
demonstrating advanced context injection patterns in atomic-agents.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase


class ContentPattern:
    """Represents a content pattern found on a website."""
    
    def __init__(self, pattern_type: str, selector: str, frequency: int, confidence: float):
        self.pattern_type = pattern_type
        self.selector = selector
        self.frequency = frequency
        self.confidence = confidence
        self.examples: List[str] = []
    
    def add_example(self, example: str) -> None:
        """Add an example of this pattern."""
        if len(self.examples) < 3:  # Keep only top 3 examples
            self.examples.append(example)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": self.pattern_type,
            "selector": self.selector,
            "frequency": self.frequency,
            "confidence": self.confidence,
            "examples": self.examples
        }


class NavigationInfo:
    """Information about website navigation structure."""
    
    def __init__(self):
        self.main_menu_selectors: List[str] = []
        self.pagination_selectors: List[str] = []
        self.breadcrumb_selectors: List[str] = []
        self.search_form_selectors: List[str] = []
        self.has_infinite_scroll: bool = False
        self.has_load_more_button: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "main_menu": self.main_menu_selectors,
            "pagination": self.pagination_selectors,
            "breadcrumbs": self.breadcrumb_selectors,
            "search_forms": self.search_form_selectors,
            "infinite_scroll": self.has_infinite_scroll,
            "load_more": self.has_load_more_button
        }


class WebsiteStructureAnalysis:
    """Complete analysis of website structure and patterns."""
    
    def __init__(self, url: str, title: str = ""):
        self.url = url
        self.title = title
        self.analyzed_at = datetime.utcnow()
        self.content_patterns: List[ContentPattern] = []
        self.navigation_info = NavigationInfo()
        self.metadata: Dict[str, Any] = {}
        self.quality_score: float = 0.0
        self.analysis_confidence: float = 0.0
    
    def add_content_pattern(self, pattern: ContentPattern) -> None:
        """Add a content pattern to the analysis."""
        self.content_patterns.append(pattern)
    
    def get_best_patterns(self, limit: int = 5) -> List[ContentPattern]:
        """Get the best content patterns sorted by confidence."""
        return sorted(
            self.content_patterns, 
            key=lambda p: p.confidence, 
            reverse=True
        )[:limit]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "url": self.url,
            "title": self.title,
            "analyzed_at": self.analyzed_at.isoformat(),
            "content_patterns": [p.to_dict() for p in self.content_patterns],
            "navigation_info": self.navigation_info.to_dict(),
            "metadata": self.metadata,
            "quality_score": self.quality_score,
            "analysis_confidence": self.analysis_confidence
        }


class WebsiteAnalysisContextProvider(SystemPromptContextProviderBase):
    """
    Provides dynamic website analysis context to agents.
    
    This context provider demonstrates advanced patterns for injecting
    real-time website analysis information into agent prompts, enhancing
    their ability to make informed scraping decisions.
    """
    
    def __init__(self, title: str = "Website Analysis Context"):
        super().__init__(title=title)
        self.analysis_results: Optional[WebsiteStructureAnalysis] = None
        self.content_patterns: List[ContentPattern] = []
        self.navigation_info: Optional[NavigationInfo] = None
        self.analysis_cache: Dict[str, WebsiteStructureAnalysis] = {}
        self.cache_ttl_seconds: int = 3600  # 1 hour cache
    
    def set_analysis_results(self, analysis: WebsiteStructureAnalysis) -> None:
        """
        Set the current website analysis results.
        
        Args:
            analysis: Website structure analysis to use for context
        """
        self.analysis_results = analysis
        self.content_patterns = analysis.content_patterns
        self.navigation_info = analysis.navigation_info
        
        # Cache the analysis
        self.analysis_cache[analysis.url] = analysis
    
    def get_cached_analysis(self, url: str) -> Optional[WebsiteStructureAnalysis]:
        """
        Get cached analysis for a URL if available and not expired.
        
        Args:
            url: URL to get cached analysis for
            
        Returns:
            Cached analysis if available and fresh, None otherwise
        """
        if url not in self.analysis_cache:
            return None
        
        analysis = self.analysis_cache[url]
        age_seconds = (datetime.utcnow() - analysis.analyzed_at).total_seconds()
        
        if age_seconds > self.cache_ttl_seconds:
            # Remove expired cache entry
            del self.analysis_cache[url]
            return None
        
        return analysis
    
    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self.analysis_cache.clear()
    
    def get_info(self) -> str:
        """
        Return formatted website analysis information for agent context.
        
        This method demonstrates how to format complex analysis data
        into clear, actionable context for AI agents.
        """
        if not self.analysis_results:
            return self._get_no_analysis_context()
        
        analysis = self.analysis_results
        context_parts = []
        
        # Header with basic info
        context_parts.append(f"## Website Analysis: {analysis.title}")
        context_parts.append(f"**URL:** {analysis.url}")
        context_parts.append(f"**Analyzed:** {analysis.analyzed_at.strftime('%Y-%m-%d %H:%M:%S')}")
        context_parts.append(f"**Quality Score:** {analysis.quality_score:.1f}%")
        context_parts.append(f"**Analysis Confidence:** {analysis.analysis_confidence:.1f}%")
        context_parts.append("")
        
        # Content patterns section
        if analysis.content_patterns:
            context_parts.append("### Identified Content Patterns")
            best_patterns = analysis.get_best_patterns(5)
            
            for i, pattern in enumerate(best_patterns, 1):
                context_parts.append(f"**Pattern {i}: {pattern.pattern_type.title()}**")
                context_parts.append(f"- Selector: `{pattern.selector}`")
                context_parts.append(f"- Frequency: {pattern.frequency} occurrences")
                context_parts.append(f"- Confidence: {pattern.confidence:.1f}%")
                
                if pattern.examples:
                    context_parts.append(f"- Examples: {', '.join(pattern.examples[:2])}")
                context_parts.append("")
        else:
            context_parts.append("### Content Patterns")
            context_parts.append("No specific content patterns identified. Using generic extraction approach.")
            context_parts.append("")
        
        # Navigation structure section
        if self.navigation_info:
            context_parts.append("### Navigation Structure")
            nav = self.navigation_info
            
            if nav.main_menu_selectors:
                context_parts.append(f"- **Main Menu:** {', '.join(nav.main_menu_selectors[:3])}")
            
            if nav.pagination_selectors:
                context_parts.append(f"- **Pagination:** {', '.join(nav.pagination_selectors[:3])}")
            
            if nav.breadcrumb_selectors:
                context_parts.append(f"- **Breadcrumbs:** {', '.join(nav.breadcrumb_selectors[:3])}")
            
            if nav.search_form_selectors:
                context_parts.append(f"- **Search Forms:** {', '.join(nav.search_form_selectors[:3])}")
            
            if nav.has_infinite_scroll:
                context_parts.append("- **Infinite Scroll:** Detected")
            
            if nav.has_load_more_button:
                context_parts.append("- **Load More Button:** Detected")
            
            context_parts.append("")
        
        # Metadata section
        if analysis.metadata:
            context_parts.append("### Additional Metadata")
            for key, value in analysis.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    context_parts.append(f"- **{key.title()}:** {value}")
            context_parts.append("")
        
        # Recommendations section
        context_parts.append("### Scraping Recommendations")
        recommendations = self._generate_scraping_recommendations(analysis)
        for rec in recommendations:
            context_parts.append(f"- {rec}")
        
        return "\n".join(context_parts)
    
    def _get_no_analysis_context(self) -> str:
        """Return context when no analysis is available."""
        return """## Website Analysis Context

**Status:** No website analysis available

### Fallback Approach
- Use common HTML patterns and semantic selectors
- Apply generic content extraction strategies
- Implement robust error handling and fallback mechanisms
- Consider multiple selector strategies for reliability

### Recommended Selectors
- **Headings:** h1, h2, h3, .title, .heading
- **Content:** p, .content, .description, .summary
- **Links:** a[href], .link, .more-info
- **Lists:** ul, ol, .list, .items
- **Articles:** article, .article, .post, .item

### Best Practices
- Start with semantic HTML elements
- Use CSS classes as secondary options
- Implement progressive selector fallbacks
- Apply quality scoring to extracted content"""
    
    def _generate_scraping_recommendations(self, analysis: WebsiteStructureAnalysis) -> List[str]:
        """
        Generate specific scraping recommendations based on analysis.
        
        Args:
            analysis: Website analysis to base recommendations on
            
        Returns:
            List of actionable scraping recommendations
        """
        recommendations = []
        
        # Pattern-based recommendations
        if analysis.content_patterns:
            best_pattern = analysis.get_best_patterns(1)[0]
            recommendations.append(
                f"Primary extraction strategy: Use '{best_pattern.selector}' "
                f"for {best_pattern.pattern_type} content (confidence: {best_pattern.confidence:.1f}%)"
            )
            
            if len(analysis.content_patterns) > 1:
                recommendations.append(
                    f"Implement fallback selectors from {len(analysis.content_patterns)} "
                    "identified patterns for robustness"
                )
        else:
            recommendations.append("Use semantic HTML selectors as primary strategy")
        
        # Navigation-based recommendations
        if self.navigation_info:
            nav = self.navigation_info
            
            if nav.pagination_selectors:
                recommendations.append(
                    f"Pagination detected: Use '{nav.pagination_selectors[0]}' "
                    "for multi-page scraping"
                )
            
            if nav.has_infinite_scroll:
                recommendations.append(
                    "Infinite scroll detected: Consider scroll-based pagination strategy"
                )
            
            if nav.has_load_more_button:
                recommendations.append(
                    "Load more button detected: Implement click-based pagination"
                )
        
        # Quality-based recommendations
        if analysis.quality_score < 70:
            recommendations.append(
                "Low analysis quality score: Use conservative extraction approach "
                "with extensive validation"
            )
        elif analysis.quality_score > 90:
            recommendations.append(
                "High analysis quality score: Can use optimized extraction strategies"
            )
        
        # Confidence-based recommendations
        if analysis.analysis_confidence < 60:
            recommendations.append(
                "Low analysis confidence: Implement multiple extraction strategies "
                "and cross-validation"
            )
        
        # Default recommendation
        if not recommendations:
            recommendations.append(
                "Apply standard web scraping best practices with quality validation"
            )
        
        return recommendations
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """
        Get a summary of content patterns for programmatic use.
        
        Returns:
            Dictionary containing pattern summary information
        """
        if not self.analysis_results:
            return {"patterns": [], "total_patterns": 0, "confidence": 0.0}
        
        patterns_summary = []
        for pattern in self.content_patterns:
            patterns_summary.append({
                "type": pattern.pattern_type,
                "selector": pattern.selector,
                "frequency": pattern.frequency,
                "confidence": pattern.confidence
            })
        
        return {
            "patterns": patterns_summary,
            "total_patterns": len(self.content_patterns),
            "confidence": self.analysis_results.analysis_confidence,
            "quality_score": self.analysis_results.quality_score
        }
    
    def get_navigation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of navigation information for programmatic use.
        
        Returns:
            Dictionary containing navigation summary information
        """
        if not self.navigation_info:
            return {"available": False}
        
        return {
            "available": True,
            "navigation": self.navigation_info.to_dict(),
            "pagination_available": bool(self.navigation_info.pagination_selectors),
            "infinite_scroll": self.navigation_info.has_infinite_scroll,
            "load_more": self.navigation_info.has_load_more_button
        }
    
    def update_analysis(self, url: str, additional_data: Dict[str, Any]) -> None:
        """
        Update existing analysis with additional data.
        
        Args:
            url: URL of the analysis to update
            additional_data: Additional data to merge into the analysis
        """
        if self.analysis_results and self.analysis_results.url == url:
            self.analysis_results.metadata.update(additional_data)
        
        # Also update cache if present
        if url in self.analysis_cache:
            self.analysis_cache[url].metadata.update(additional_data)
    
    def get_analysis_age(self) -> Optional[float]:
        """
        Get the age of the current analysis in seconds.
        
        Returns:
            Age in seconds, or None if no analysis available
        """
        if not self.analysis_results:
            return None
        
        return (datetime.utcnow() - self.analysis_results.analyzed_at).total_seconds()
    
    def is_analysis_fresh(self, max_age_seconds: int = 3600) -> bool:
        """
        Check if the current analysis is fresh enough to use.
        
        Args:
            max_age_seconds: Maximum age in seconds to consider fresh
            
        Returns:
            True if analysis is fresh, False otherwise
        """
        age = self.get_analysis_age()
        return age is not None and age <= max_age_seconds