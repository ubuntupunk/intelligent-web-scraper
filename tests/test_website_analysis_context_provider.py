"""
Unit tests for WebsiteAnalysisContextProvider.

Tests the website analysis context provider functionality including
content pattern analysis, navigation detection, and context formatting.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, Any, List

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
        pattern = ContentPattern("article", "article.post", 15, 85.5)
        
        assert pattern.pattern_type == "article"
        assert pattern.selector == "article.post"
        assert pattern.frequency == 15
        assert pattern.confidence == 85.5
        assert pattern.examples == []
    
    def test_add_example(self):
        """Test adding examples to pattern."""
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
        
        result = pattern.to_dict()
        
        assert result["type"] == "link"
        assert result["selector"] == "a.external"
        assert result["frequency"] == 5
        assert result["confidence"] == 75.0
        assert result["examples"] == ["https://example.com"]


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
        nav_info.breadcrumb_selectors = [".breadcrumb"]
        nav_info.search_form_selectors = ["form.search"]
        nav_info.has_infinite_scroll = True
        nav_info.has_load_more_button = False
        
        assert len(nav_info.main_menu_selectors) == 2
        assert len(nav_info.pagination_selectors) == 2
        assert nav_info.has_infinite_scroll is True
    
    def test_to_dict(self):
        """Test converting navigation info to dictionary."""
        nav_info = NavigationInfo()
        nav_info.main_menu_selectors = ["nav"]
        nav_info.has_infinite_scroll = True
        
        result = nav_info.to_dict()
        
        assert result["main_menu"] == ["nav"]
        assert result["infinite_scroll"] is True
        assert result["load_more"] is False


class TestWebsiteStructureAnalysis:
    """Test WebsiteStructureAnalysis functionality."""
    
    def test_analysis_creation(self):
        """Test creating website structure analysis."""
        analysis = WebsiteStructureAnalysis("https://example.com", "Example Site")
        
        assert analysis.url == "https://example.com"
        assert analysis.title == "Example Site"
        assert isinstance(analysis.analyzed_at, datetime)
        assert analysis.content_patterns == []
        assert isinstance(analysis.navigation_info, NavigationInfo)
        assert analysis.metadata == {}
        assert analysis.quality_score == 0.0
        assert analysis.analysis_confidence == 0.0
    
    def test_add_content_pattern(self):
        """Test adding content patterns."""
        analysis = WebsiteStructureAnalysis("https://example.com")
        pattern1 = ContentPattern("title", "h1", 10, 90.0)
        pattern2 = ContentPattern("content", "p", 25, 75.0)
        
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
        pattern3 = ContentPattern("medium", "section", 8, 80.0)
        
        analysis.add_content_pattern(pattern1)
        analysis.add_content_pattern(pattern2)
        analysis.add_content_pattern(pattern3)
        
        best_patterns = analysis.get_best_patterns(2)
        
        assert len(best_patterns) == 2
        assert best_patterns[0].confidence == 95.0  # Highest confidence first
        assert best_patterns[1].confidence == 80.0  # Second highest
    
    def test_to_dict(self):
        """Test converting analysis to dictionary."""
        analysis = WebsiteStructureAnalysis("https://example.com", "Test Site")
        analysis.quality_score = 85.0
        analysis.analysis_confidence = 90.0
        analysis.metadata = {"page_type": "article"}
        
        pattern = ContentPattern("title", "h1", 5, 95.0)
        analysis.add_content_pattern(pattern)
        
        result = analysis.to_dict()
        
        assert result["url"] == "https://example.com"
        assert result["title"] == "Test Site"
        assert result["quality_score"] == 85.0
        assert result["analysis_confidence"] == 90.0
        assert result["metadata"]["page_type"] == "article"
        assert len(result["content_patterns"]) == 1
        assert "navigation_info" in result
        assert "analyzed_at" in result


class TestWebsiteAnalysisContextProvider:
    """Test WebsiteAnalysisContextProvider functionality."""
    
    @pytest.fixture
    def context_provider(self):
        """Create a context provider for testing."""
        return WebsiteAnalysisContextProvider()
    
    @pytest.fixture
    def sample_analysis(self):
        """Create sample website analysis for testing."""
        analysis = WebsiteStructureAnalysis("https://example.com", "Example Website")
        analysis.quality_score = 85.0
        analysis.analysis_confidence = 90.0
        
        # Add content patterns
        title_pattern = ContentPattern("title", "h1.main-title", 10, 95.0)
        title_pattern.add_example("Main Article Title")
        title_pattern.add_example("Secondary Title")
        
        content_pattern = ContentPattern("content", "div.content", 15, 80.0)
        content_pattern.add_example("Article content here...")
        
        analysis.add_content_pattern(title_pattern)
        analysis.add_content_pattern(content_pattern)
        
        # Add navigation info
        analysis.navigation_info.main_menu_selectors = ["nav.primary"]
        analysis.navigation_info.pagination_selectors = [".pagination"]
        analysis.navigation_info.has_infinite_scroll = False
        analysis.navigation_info.has_load_more_button = True
        
        # Add metadata
        analysis.metadata = {
            "page_type": "article_list",
            "total_pages": 25,
            "items_per_page": 20
        }
        
        return analysis
    
    def test_context_provider_initialization(self, context_provider):
        """Test context provider initialization."""
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
    
    def test_get_cached_analysis(self, context_provider, sample_analysis):
        """Test getting cached analysis."""
        # Set analysis (which caches it)
        context_provider.set_analysis_results(sample_analysis)
        
        # Get cached analysis
        cached = context_provider.get_cached_analysis("https://example.com")
        assert cached == sample_analysis
        
        # Test non-existent URL
        cached = context_provider.get_cached_analysis("https://nonexistent.com")
        assert cached is None
    
    def test_cache_expiration(self, context_provider, sample_analysis):
        """Test cache expiration functionality."""
        # Set short TTL for testing
        context_provider.cache_ttl_seconds = 1
        
        # Set analysis
        context_provider.set_analysis_results(sample_analysis)
        
        # Should be cached immediately
        cached = context_provider.get_cached_analysis("https://example.com")
        assert cached is not None
        
        # Mock time passage
        with patch('intelligent_web_scraper.context_providers.website_analysis.datetime') as mock_datetime:
            # Set current time to be past TTL
            future_time = datetime.utcnow() + timedelta(seconds=2)
            mock_datetime.utcnow.return_value = future_time
            
            # Should be expired and removed
            cached = context_provider.get_cached_analysis("https://example.com")
            assert cached is None
    
    def test_clear_cache(self, context_provider, sample_analysis):
        """Test clearing cache."""
        context_provider.set_analysis_results(sample_analysis)
        assert len(context_provider.analysis_cache) == 1
        
        context_provider.clear_cache()
        assert len(context_provider.analysis_cache) == 0
    
    def test_get_info_no_analysis(self, context_provider):
        """Test getting context info with no analysis."""
        info = context_provider.get_info()
        
        assert "No website analysis available" in info
        assert "Fallback Approach" in info
        assert "Recommended Selectors" in info
        assert "Best Practices" in info
    
    def test_get_info_with_analysis(self, context_provider, sample_analysis):
        """Test getting context info with analysis data."""
        context_provider.set_analysis_results(sample_analysis)
        
        info = context_provider.get_info()
        
        # Check header information
        assert "Website Analysis: Example Website" in info
        assert "https://example.com" in info
        assert "Quality Score: 85.0%" in info
        assert "Analysis Confidence: 90.0%" in info
        
        # Check content patterns section
        assert "Identified Content Patterns" in info
        assert "Pattern 1: Title" in info
        assert "h1.main-title" in info
        assert "Confidence: 95.0%" in info
        
        # Check navigation structure
        assert "Navigation Structure" in info
        assert "Main Menu: nav.primary" in info
        assert "Pagination: .pagination" in info
        assert "Load More Button: Detected" in info
        
        # Check metadata
        assert "Additional Metadata" in info
        assert "Page Type: article_list" in info
        
        # Check recommendations
        assert "Scraping Recommendations" in info
    
    def test_generate_scraping_recommendations(self, context_provider, sample_analysis):
        """Test generating scraping recommendations."""
        context_provider.set_analysis_results(sample_analysis)
        
        recommendations = context_provider._generate_scraping_recommendations(sample_analysis)
        
        assert len(recommendations) > 0
        assert any("h1.main-title" in rec for rec in recommendations)
        assert any("confidence: 95.0%" in rec for rec in recommendations)
        assert any("Load more button detected" in rec for rec in recommendations)
    
    def test_generate_recommendations_low_quality(self, context_provider):
        """Test recommendations for low quality analysis."""
        low_quality_analysis = WebsiteStructureAnalysis("https://example.com")
        low_quality_analysis.quality_score = 45.0  # Low quality
        low_quality_analysis.analysis_confidence = 30.0  # Low confidence
        
        context_provider.set_analysis_results(low_quality_analysis)
        recommendations = context_provider._generate_scraping_recommendations(low_quality_analysis)
        
        assert any("conservative extraction approach" in rec.lower() for rec in recommendations)
        assert any("multiple extraction strategies" in rec.lower() for rec in recommendations)
    
    def test_get_pattern_summary(self, context_provider, sample_analysis):
        """Test getting pattern summary."""
        context_provider.set_analysis_results(sample_analysis)
        
        summary = context_provider.get_pattern_summary()
        
        assert summary["total_patterns"] == 2
        assert summary["confidence"] == 90.0
        assert summary["quality_score"] == 85.0
        assert len(summary["patterns"]) == 2
        
        # Check pattern details
        title_pattern = next(p for p in summary["patterns"] if p["type"] == "title")
        assert title_pattern["selector"] == "h1.main-title"
        assert title_pattern["confidence"] == 95.0
    
    def test_get_pattern_summary_no_analysis(self, context_provider):
        """Test getting pattern summary with no analysis."""
        summary = context_provider.get_pattern_summary()
        
        assert summary["patterns"] == []
        assert summary["total_patterns"] == 0
        assert summary["confidence"] == 0.0
    
    def test_get_navigation_summary(self, context_provider, sample_analysis):
        """Test getting navigation summary."""
        context_provider.set_analysis_results(sample_analysis)
        
        summary = context_provider.get_navigation_summary()
        
        assert summary["available"] is True
        assert summary["pagination_available"] is True
        assert summary["infinite_scroll"] is False
        assert summary["load_more"] is True
        assert "navigation" in summary
    
    def test_get_navigation_summary_no_info(self, context_provider):
        """Test getting navigation summary with no info."""
        summary = context_provider.get_navigation_summary()
        
        assert summary["available"] is False
    
    def test_update_analysis(self, context_provider, sample_analysis):
        """Test updating analysis with additional data."""
        context_provider.set_analysis_results(sample_analysis)
        
        additional_data = {"new_field": "new_value", "updated_field": "updated"}
        context_provider.update_analysis("https://example.com", additional_data)
        
        assert context_provider.analysis_results.metadata["new_field"] == "new_value"
        assert context_provider.analysis_results.metadata["updated_field"] == "updated"
        
        # Check cache is also updated
        cached = context_provider.get_cached_analysis("https://example.com")
        assert cached.metadata["new_field"] == "new_value"
    
    def test_get_analysis_age(self, context_provider, sample_analysis):
        """Test getting analysis age."""
        # No analysis
        age = context_provider.get_analysis_age()
        assert age is None
        
        # With analysis
        context_provider.set_analysis_results(sample_analysis)
        age = context_provider.get_analysis_age()
        assert isinstance(age, float)
        assert age >= 0
    
    def test_is_analysis_fresh(self, context_provider, sample_analysis):
        """Test checking if analysis is fresh."""
        # No analysis
        assert context_provider.is_analysis_fresh() is False
        
        # Fresh analysis
        context_provider.set_analysis_results(sample_analysis)
        assert context_provider.is_analysis_fresh(max_age_seconds=3600) is True
        
        # Old analysis
        assert context_provider.is_analysis_fresh(max_age_seconds=0) is False
    
    def test_multiple_analyses_caching(self, context_provider):
        """Test caching multiple analyses."""
        # Create multiple analyses
        analysis1 = WebsiteStructureAnalysis("https://site1.com", "Site 1")
        analysis2 = WebsiteStructureAnalysis("https://site2.com", "Site 2")
        
        # Set both analyses
        context_provider.set_analysis_results(analysis1)
        context_provider.set_analysis_results(analysis2)  # This should update current but keep cache
        
        # Both should be cached
        cached1 = context_provider.get_cached_analysis("https://site1.com")
        cached2 = context_provider.get_cached_analysis("https://site2.com")
        
        assert cached1 == analysis1
        assert cached2 == analysis2
        assert context_provider.analysis_results == analysis2  # Current should be the last set
    
    def test_recommendations_with_pagination(self, context_provider):
        """Test recommendations with different pagination types."""
        analysis = WebsiteStructureAnalysis("https://example.com")
        
        # Test with infinite scroll
        analysis.navigation_info.has_infinite_scroll = True
        context_provider.set_analysis_results(analysis)
        recommendations = context_provider._generate_scraping_recommendations(analysis)
        assert any("infinite scroll" in rec.lower() for rec in recommendations)
        
        # Test with load more button
        analysis.navigation_info.has_infinite_scroll = False
        analysis.navigation_info.has_load_more_button = True
        context_provider.set_analysis_results(analysis)
        recommendations = context_provider._generate_scraping_recommendations(analysis)
        assert any("load more button" in rec.lower() for rec in recommendations)
        
        # Test with pagination selectors
        analysis.navigation_info.has_load_more_button = False
        analysis.navigation_info.pagination_selectors = [".pagination"]
        context_provider.set_analysis_results(analysis)
        recommendations = context_provider._generate_scraping_recommendations(analysis)
        assert any("pagination detected" in rec.lower() for rec in recommendations)
    
    def test_context_formatting_edge_cases(self, context_provider):
        """Test context formatting with edge cases."""
        # Analysis with no patterns but navigation info
        analysis = WebsiteStructureAnalysis("https://example.com", "Edge Case Site")
        analysis.navigation_info.main_menu_selectors = ["nav"]
        analysis.quality_score = 100.0
        analysis.analysis_confidence = 100.0
        
        context_provider.set_analysis_results(analysis)
        info = context_provider.get_info()
        
        assert "No specific content patterns identified" in info
        assert "Main Menu: nav" in info
        assert "High analysis quality score" in info or "optimized extraction strategies" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])