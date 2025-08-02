"""
Unit tests for WebsiteAnalysisContextProvider.

Tests the website analysis context provider functionality including
dynamic context injection, caching, and performance optimization.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from intelligent_web_scraper.context_providers.website_analysis import (
    WebsiteAnalysisContextProvider,
    WebsiteStructureAnalysis,
    ContentPattern,
    NavigationInfo
)


class TestContentPattern:
    """Test ContentPattern functionality."""
    
    def test_content_pattern_creation(self):
        """Test creating a content pattern."""
        pattern = ContentPattern(
            pattern_type="article",
            selector="article.post",
            frequency=15,
            confidence=85.5
        )
        
        assert pattern.pattern_type == "article"
        assert pattern.selector == "article.post"
        assert pattern.frequency == 15
        assert pattern.confidence == 85.5
        assert pattern.examples == []
    
    def test_add_example(self):
        """Test adding examples to a pattern."""
        pattern = ContentPattern("title", "h1", 10, 90.0)
        
        pattern.add_example("Example Title 1")
        pattern.add_example("Example Title 2")
        pattern.add_example("Example Title 3")
        pattern.add_example("Example Title 4")  # Should not be added (limit 3)
        
        assert len(pattern.examples) == 3
        assert "Example Title 1" in pattern.examples
        assert "Example Title 4" not in pattern.examples
    
    def test_to_dict(self):
        """Test converting pattern to dictionary."""
        pattern = ContentPattern("link", "a.external", 5, 75.0)
        pattern.add_example("https://example.com")
        
        pattern_dict = pattern.to_dict()
        
        assert pattern_dict["type"] == "link"
        assert pattern_dict["selector"] == "a.external"
        assert pattern_dict["frequency"] == 5
        assert pattern_dict["confidence"] == 75.0
        assert pattern_dict["examples"] == ["https://example.com"]


class TestNavigationInfo:
    """Test NavigationInfo functionality."""
    
    def test_navigation_info_creation(self):
        """Test creating navigation info."""
        nav_info = NavigationInfo()
        
        assert nav_info.main_menu_selectors == []
        assert nav_info.pagination_selectors == []
        assert nav_info.breadcrumb_selectors == []
        assert nav_info.search_form_selectors == []
        assert nav_info.has_infinite_scroll is False
        assert nav_info.has_load_more_button is False
    
    def test_navigation_info_population(self):
        """Test populating navigation info."""
        nav_info = NavigationInfo()
        nav_info.main_menu_selectors = ["nav.main", ".navigation"]
        nav_info.pagination_selectors = [".pagination", ".page-nav"]
        nav_info.has_infinite_scroll = True
        
        assert len(nav_info.main_menu_selectors) == 2
        assert len(nav_info.pagination_selectors) == 2
        assert nav_info.has_infinite_scroll is True
    
    def test_to_dict(self):
        """Test converting navigation info to dictionary."""
        nav_info = NavigationInfo()
        nav_info.main_menu_selectors = ["nav"]
        nav_info.pagination_selectors = [".pagination"]
        nav_info.has_infinite_scroll = True
        nav_info.has_load_more_button = False
        
        nav_dict = nav_info.to_dict()
        
        assert nav_dict["main_menu"] == ["nav"]
        assert nav_dict["pagination"] == [".pagination"]
        assert nav_dict["infinite_scroll"] is True
        assert nav_dict["load_more"] is False


class TestWebsiteStructureAnalysis:
    """Test WebsiteStructureAnalysis functionality."""
    
    def test_analysis_creation(self):
        """Test creating website structure analysis."""
        analysis = WebsiteStructureAnalysis(
            url="https://example.com",
            title="Example Site"
        )
        
        assert analysis.url == "https://example.com"
        assert analysis.title == "Example Site"
        assert isinstance(analysis.analyzed_at, datetime)
        assert analysis.content_patterns == []
        assert isinstance(analysis.navigation_info, NavigationInfo)
        assert analysis.metadata == {}
        assert analysis.quality_score == 0.0
        assert analysis.analysis_confidence == 0.0
    
    def test_add_content_pattern(self):
        """Test adding content patterns to analysis."""
        analysis = WebsiteStructureAnalysis("https://example.com")
        
        pattern1 = ContentPattern("title", "h1", 10, 90.0)
        pattern2 = ContentPattern("content", "p", 25, 80.0)
        
        analysis.add_content_pattern(pattern1)
        analysis.add_content_pattern(pattern2)
        
        assert len(analysis.content_patterns) == 2
        assert pattern1 in analysis.content_patterns
        assert pattern2 in analysis.content_patterns
    
    def test_get_best_patterns(self):
        """Test getting best patterns sorted by confidence."""
        analysis = WebsiteStructureAnalysis("https://example.com")
        
        # Add patterns with different confidence scores
        pattern1 = ContentPattern("low", "div", 5, 60.0)
        pattern2 = ContentPattern("high", "article", 10, 95.0)
        pattern3 = ContentPattern("medium", "section", 8, 75.0)
        
        analysis.add_content_pattern(pattern1)
        analysis.add_content_pattern(pattern2)
        analysis.add_content_pattern(pattern3)
        
        best_patterns = analysis.get_best_patterns(2)
        
        assert len(best_patterns) == 2
        assert best_patterns[0].confidence == 95.0  # Highest confidence first
        assert best_patterns[1].confidence == 75.0  # Second highest
    
    def test_to_dict(self):
        """Test converting analysis to dictionary."""
        analysis = WebsiteStructureAnalysis("https://example.com", "Test Site")
        analysis.quality_score = 85.0
        analysis.analysis_confidence = 90.0
        analysis.metadata = {"page_type": "blog"}
        
        pattern = ContentPattern("title", "h1", 5, 80.0)
        analysis.add_content_pattern(pattern)
        
        analysis_dict = analysis.to_dict()
        
        assert analysis_dict["url"] == "https://example.com"
        assert analysis_dict["title"] == "Test Site"
        assert "analyzed_at" in analysis_dict
        assert analysis_dict["quality_score"] == 85.0
        assert analysis_dict["analysis_confidence"] == 90.0
        assert analysis_dict["metadata"]["page_type"] == "blog"
        assert len(analysis_dict["content_patterns"]) == 1


class TestWebsiteAnalysisContextProvider:
    """Test WebsiteAnalysisContextProvider functionality."""
    
    @pytest.fixture
    def context_provider(self):
        """Create a context provider for testing."""
        return WebsiteAnalysisContextProvider()
    
    @pytest.fixture
    def sample_analysis(self):
        """Create a sample website analysis for testing."""
        analysis = WebsiteStructureAnalysis(
            url="https://example.com/blog",
            title="Example Blog"
        )
        analysis.quality_score = 85.0
        analysis.analysis_confidence = 90.0
        analysis.metadata = {"page_type": "blog", "total_posts": 25}
        
        # Add content patterns
        title_pattern = ContentPattern("title", "h2.post-title", 25, 95.0)
        title_pattern.add_example("First Blog Post")
        title_pattern.add_example("Second Blog Post")
        
        content_pattern = ContentPattern("content", ".post-content", 25, 88.0)
        content_pattern.add_example("This is the first paragraph...")
        
        analysis.add_content_pattern(title_pattern)
        analysis.add_content_pattern(content_pattern)
        
        # Add navigation info
        analysis.navigation_info.pagination_selectors = [".pagination", ".page-nav"]
        analysis.navigation_info.main_menu_selectors = ["nav.main"]
        analysis.navigation_info.has_infinite_scroll = False
        analysis.navigation_info.has_load_more_button = True
        
        return analysis
    
    def test_context_provider_initialization(self, context_provider):
        """Test context provider initialization."""
        assert context_provider.title == "Website Analysis Context"
        assert context_provider.analysis_results is None
        assert context_provider.content_patterns == []
        assert context_provider.navigation_info is None
        assert context_provider.analysis_cache == {}
        assert context_provider.cache_ttl_seconds == 3600
    
    def test_set_analysis_results(self, context_provider, sample_analysis):
        """Test setting analysis results."""
        context_provider.set_analysis_results(sample_analysis)
        
        assert context_provider.analysis_results == sample_analysis
        assert len(context_provider.content_patterns) == 2
        assert context_provider.navigation_info == sample_analysis.navigation_info
        assert sample_analysis.url in context_provider.analysis_cache
    
    def test_get_info_with_analysis(self, context_provider, sample_analysis):
        """Test getting context info with analysis data."""
        context_provider.set_analysis_results(sample_analysis)
        
        info = context_provider.get_info()
        
        # Check that key information is included
        assert "Website Analysis: Example Blog" in info
        assert "https://example.com/blog" in info
        assert "Quality Score:** 85.0%" in info
        assert "Analysis Confidence:** 90.0%" in info
        assert "Identified Content Patterns" in info
        assert "h2.post-title" in info
        assert "Navigation Structure" in info
        assert "Pagination:" in info
        assert "Load More Button:** Detected" in info
        assert "Scraping Recommendations" in info
        assert "Page_Type" in info
    
    def test_get_info_without_analysis(self, context_provider):
        """Test getting context info without analysis data."""
        info = context_provider.get_info()
        
        assert "No website analysis available" in info
        assert "Fallback Approach" in info
        assert "Recommended Selectors" in info
        assert "Best Practices" in info
        assert "h1, h2, h3" in info  # Common selectors
    
    def test_caching_functionality(self, context_provider, sample_analysis):
        """Test analysis caching functionality."""
        # Set analysis (should cache it)
        context_provider.set_analysis_results(sample_analysis)
        
        # Get cached analysis
        cached = context_provider.get_cached_analysis(sample_analysis.url)
        assert cached == sample_analysis
        
        # Test cache miss
        cached_miss = context_provider.get_cached_analysis("https://other.com")
        assert cached_miss is None
    
    def test_cache_expiration(self, context_provider):
        """Test cache expiration functionality."""
        # Create an old analysis
        old_analysis = WebsiteStructureAnalysis("https://example.com")
        old_analysis.analyzed_at = datetime.utcnow() - timedelta(hours=2)
        
        # Manually add to cache
        context_provider.analysis_cache[old_analysis.url] = old_analysis
        
        # Should return None due to expiration
        cached = context_provider.get_cached_analysis(old_analysis.url)
        assert cached is None
        
        # Should also remove from cache
        assert old_analysis.url not in context_provider.analysis_cache
    
    def test_clear_cache(self, context_provider, sample_analysis):
        """Test clearing the cache."""
        context_provider.set_analysis_results(sample_analysis)
        assert len(context_provider.analysis_cache) == 1
        
        context_provider.clear_cache()
        assert len(context_provider.analysis_cache) == 0
    
    def test_get_pattern_summary(self, context_provider, sample_analysis):
        """Test getting pattern summary."""
        # Test without analysis
        summary = context_provider.get_pattern_summary()
        assert summary["patterns"] == []
        assert summary["total_patterns"] == 0
        assert summary["confidence"] == 0.0
        
        # Test with analysis
        context_provider.set_analysis_results(sample_analysis)
        summary = context_provider.get_pattern_summary()
        
        assert len(summary["patterns"]) == 2
        assert summary["total_patterns"] == 2
        assert summary["confidence"] == 90.0
        assert summary["patterns"][0]["type"] in ["title", "content"]
        assert "selector" in summary["patterns"][0]
        assert "frequency" in summary["patterns"][0]
        assert "confidence" in summary["patterns"][0]
    
    def test_get_navigation_summary(self, context_provider, sample_analysis):
        """Test getting navigation summary."""
        # Test without navigation info
        summary = context_provider.get_navigation_summary()
        assert summary["available"] is False
        
        # Test with navigation info
        context_provider.set_analysis_results(sample_analysis)
        summary = context_provider.get_navigation_summary()
        
        assert summary["available"] is True
        assert summary["pagination_available"] is True
        assert summary["infinite_scroll"] is False
        assert summary["load_more"] is True
        assert "navigation" in summary
    
    def test_update_analysis(self, context_provider, sample_analysis):
        """Test updating analysis with additional data."""
        context_provider.set_analysis_results(sample_analysis)
        
        additional_data = {"author_count": 5, "category": "tech"}
        context_provider.update_analysis(sample_analysis.url, additional_data)
        
        assert context_provider.analysis_results.metadata["author_count"] == 5
        assert context_provider.analysis_results.metadata["category"] == "tech"
        
        # Check cache is also updated
        cached = context_provider.analysis_cache[sample_analysis.url]
        assert cached.metadata["author_count"] == 5
    
    def test_get_analysis_age(self, context_provider, sample_analysis):
        """Test getting analysis age."""
        # Test without analysis
        age = context_provider.get_analysis_age()
        assert age is None
        
        # Test with analysis
        context_provider.set_analysis_results(sample_analysis)
        age = context_provider.get_analysis_age()
        
        assert isinstance(age, float)
        assert age >= 0
        assert age < 60  # Should be very recent
    
    def test_is_analysis_fresh(self, context_provider, sample_analysis):
        """Test checking if analysis is fresh."""
        # Test without analysis
        assert context_provider.is_analysis_fresh() is False
        
        # Test with fresh analysis
        context_provider.set_analysis_results(sample_analysis)
        assert context_provider.is_analysis_fresh() is True
        assert context_provider.is_analysis_fresh(max_age_seconds=3600) is True
        
        # Test with very strict freshness requirement
        assert context_provider.is_analysis_fresh(max_age_seconds=0) is False
    
    def test_generate_scraping_recommendations(self, context_provider, sample_analysis):
        """Test generating scraping recommendations."""
        context_provider.set_analysis_results(sample_analysis)
        
        # Access the private method for testing
        recommendations = context_provider._generate_scraping_recommendations(sample_analysis)
        
        assert len(recommendations) > 0
        assert any("h2.post-title" in rec for rec in recommendations)
        assert any("confidence: 95.0%" in rec for rec in recommendations)
        assert any("Load more button detected" in rec for rec in recommendations)
    
    def test_recommendations_for_low_quality_analysis(self, context_provider):
        """Test recommendations for low quality analysis."""
        low_quality_analysis = WebsiteStructureAnalysis("https://example.com")
        low_quality_analysis.quality_score = 30.0
        low_quality_analysis.analysis_confidence = 40.0
        
        context_provider.set_analysis_results(low_quality_analysis)
        recommendations = context_provider._generate_scraping_recommendations(low_quality_analysis)
        
        assert any("Low analysis quality score" in rec for rec in recommendations)
        assert any("Low analysis confidence" in rec for rec in recommendations)
    
    def test_recommendations_for_high_quality_analysis(self, context_provider):
        """Test recommendations for high quality analysis."""
        high_quality_analysis = WebsiteStructureAnalysis("https://example.com")
        high_quality_analysis.quality_score = 95.0
        high_quality_analysis.analysis_confidence = 98.0
        
        # Add a high-confidence pattern
        pattern = ContentPattern("article", "article.post", 20, 96.0)
        high_quality_analysis.add_content_pattern(pattern)
        
        context_provider.set_analysis_results(high_quality_analysis)
        recommendations = context_provider._generate_scraping_recommendations(high_quality_analysis)
        
        assert any("High analysis quality score" in rec for rec in recommendations)
        assert any("article.post" in rec for rec in recommendations)
    
    def test_context_provider_inheritance(self, context_provider):
        """Test that context provider properly inherits from base class."""
        from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase
        
        assert isinstance(context_provider, SystemPromptContextProviderBase)
        assert hasattr(context_provider, 'get_info')
        assert hasattr(context_provider, 'title')
    
    def test_custom_title(self):
        """Test creating context provider with custom title."""
        custom_provider = WebsiteAnalysisContextProvider(title="Custom Analysis Context")
        assert custom_provider.title == "Custom Analysis Context"
    
    def test_context_formatting_edge_cases(self, context_provider):
        """Test context formatting with edge cases."""
        # Analysis with no patterns but with navigation
        analysis = WebsiteStructureAnalysis("https://example.com", "Empty Site")
        analysis.navigation_info.main_menu_selectors = ["nav"]
        
        context_provider.set_analysis_results(analysis)
        info = context_provider.get_info()
        
        assert "No specific content patterns identified" in info
        assert "Main Menu:** nav" in info
    
    def test_pattern_examples_truncation(self, context_provider):
        """Test that pattern examples are properly truncated."""
        analysis = WebsiteStructureAnalysis("https://example.com")
        
        pattern = ContentPattern("title", "h1", 10, 90.0)
        pattern.add_example("Example 1")
        pattern.add_example("Example 2")
        pattern.add_example("Example 3")
        
        analysis.add_content_pattern(pattern)
        context_provider.set_analysis_results(analysis)
        
        info = context_provider.get_info()
        
        # Should show only first 2 examples in context
        assert "Example 1, Example 2" in info
        assert "Example 3" not in info or info.count("Example 3") == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])