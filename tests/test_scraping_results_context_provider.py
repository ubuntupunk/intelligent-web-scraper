"""
Unit tests for ScrapingResultsContextProvider.

Tests the scraping results context provider functionality including
result summarization, statistics generation, and quality metrics.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from intelligent_web_scraper.context_providers.scraping_results import (
    ScrapingResultsContextProvider,
    ScrapedItem,
    ExtractionStatistics,
    QualityMetrics,
    ScrapingSession
)


class TestScrapedItem:
    """Test ScrapedItem functionality."""
    
    def test_scraped_item_creation(self):
        """Test creating a scraped item."""
        data = {"title": "Test Title", "description": "Test Description"}
        item = ScrapedItem(data=data, quality_score=75.0, source_url="https://example.com")
        
        assert item.data == data
        assert item.quality_score == 75.0
        assert item.source_url == "https://example.com"
        assert isinstance(item.scraped_at, datetime)
        assert item.field_completeness == {}
        assert item.validation_errors == []
    
    def test_add_validation_error(self):
        """Test adding validation errors."""
        item = ScrapedItem(data={}, quality_score=50.0)
        
        item.add_validation_error("Missing required field")
        item.add_validation_error("Invalid format")
        
        assert len(item.validation_errors) == 2
        assert "Missing required field" in item.validation_errors
        assert "Invalid format" in item.validation_errors
    
    def test_set_field_completeness(self):
        """Test setting field completeness scores."""
        item = ScrapedItem(data={}, quality_score=50.0)
        
        item.set_field_completeness("title", 100.0)
        item.set_field_completeness("description", 75.0)
        
        assert item.field_completeness["title"] == 100.0
        assert item.field_completeness["description"] == 75.0
    
    def test_get_overall_completeness(self):
        """Test calculating overall completeness."""
        item = ScrapedItem(data={}, quality_score=50.0)
        
        # No completeness data
        assert item.get_overall_completeness() == 0.0
        
        # With completeness data
        item.set_field_completeness("title", 100.0)
        item.set_field_completeness("description", 80.0)
        
        expected_completeness = (100.0 + 80.0) / 2
        assert item.get_overall_completeness() == expected_completeness
    
    def test_to_dict(self):
        """Test converting item to dictionary."""
        data = {"title": "Test", "url": "https://example.com"}
        item = ScrapedItem(data=data, quality_score=85.0, source_url="https://source.com")
        item.add_validation_error("Test error")
        item.set_field_completeness("title", 100.0)
        
        result = item.to_dict()
        
        assert result["data"] == data
        assert result["quality_score"] == 85.0
        assert result["source_url"] == "https://source.com"
        assert "scraped_at" in result
        assert result["validation_errors"] == ["Test error"]
        assert result["field_completeness"] == {"title": 100.0}
        assert "overall_completeness" in result


class TestExtractionStatistics:
    """Test ExtractionStatistics functionality."""
    
    def test_statistics_initialization(self):
        """Test statistics initialization."""
        stats = ExtractionStatistics()
        
        assert stats.total_items_found == 0
        assert stats.total_items_extracted == 0
        assert stats.total_processing_time == 0.0
        assert stats.extraction_rate == 0.0
        assert stats.success_rate == 0.0
        assert stats.started_at is None
        assert stats.completed_at is None
    
    def test_start_and_complete_extraction(self):
        """Test starting and completing extraction."""
        stats = ExtractionStatistics()
        
        # Start extraction
        stats.start_extraction()
        assert stats.started_at is not None
        
        # Complete extraction
        stats.total_items_found = 100
        stats.total_items_extracted = 85
        stats.total_processing_time = 10.0
        stats.complete_extraction()
        
        assert stats.completed_at is not None
        assert stats.success_rate == 85.0  # 85/100 * 100
        assert stats.extraction_rate == 8.5  # 85/10
    
    def test_add_processing_time(self):
        """Test adding processing times."""
        stats = ExtractionStatistics()
        
        stats.add_processing_time(2.5)
        stats.add_processing_time(3.0)
        
        assert len(stats.processing_times) == 2
        assert stats.total_processing_time == 5.5
    
    def test_add_error(self):
        """Test adding errors."""
        stats = ExtractionStatistics()
        
        stats.add_error("network_error")
        stats.add_error("parsing_error")
        stats.add_error("network_error")
        
        assert stats.error_counts["network_error"] == 2
        assert stats.error_counts["parsing_error"] == 1
    
    def test_add_quality_score(self):
        """Test adding quality scores to distribution."""
        stats = ExtractionStatistics()
        
        stats.add_quality_score(95.0)  # excellent
        stats.add_quality_score(75.0)  # good
        stats.add_quality_score(55.0)  # fair
        stats.add_quality_score(25.0)  # poor
        
        assert stats.quality_distribution["excellent"] == 1
        assert stats.quality_distribution["good"] == 1
        assert stats.quality_distribution["fair"] == 1
        assert stats.quality_distribution["poor"] == 1
    
    def test_get_duration(self):
        """Test getting extraction duration."""
        stats = ExtractionStatistics()
        
        # No duration when not started
        assert stats.get_duration() is None
        
        # Duration when started but not completed
        stats.start_extraction()
        duration = stats.get_duration()
        assert isinstance(duration, timedelta)
        
        # Duration when completed
        stats.complete_extraction()
        duration = stats.get_duration()
        assert isinstance(duration, timedelta)
    
    def test_to_dict(self):
        """Test converting statistics to dictionary."""
        stats = ExtractionStatistics()
        stats.start_extraction()
        stats.total_items_found = 50
        stats.total_items_extracted = 45
        stats.add_error("test_error")
        stats.complete_extraction()
        
        result = stats.to_dict()
        
        assert result["total_items_found"] == 50
        assert result["total_items_extracted"] == 45
        assert result["success_rate"] == 90.0
        assert "error_counts" in result
        assert "duration_seconds" in result


class TestQualityMetrics:
    """Test QualityMetrics functionality."""
    
    def test_quality_metrics_initialization(self):
        """Test quality metrics initialization."""
        metrics = QualityMetrics()
        
        assert metrics.overall_quality_score == 0.0
        assert metrics.quality_improvements == 0
        assert metrics.quality_degradations == 0
        assert len(metrics.quality_trends) == 0
    
    def test_update_quality_score(self):
        """Test updating quality scores and tracking trends."""
        metrics = QualityMetrics()
        
        # First update
        metrics.update_quality_score(75.0)
        assert metrics.overall_quality_score == 75.0
        assert len(metrics.quality_trends) == 1
        
        # Improvement
        metrics.update_quality_score(80.0)
        assert metrics.quality_improvements == 1
        assert metrics.quality_degradations == 0
        
        # Degradation
        metrics.update_quality_score(70.0)
        assert metrics.quality_improvements == 1
        assert metrics.quality_degradations == 1
    
    def test_update_field_quality(self):
        """Test updating field-specific quality scores."""
        metrics = QualityMetrics()
        
        metrics.update_field_quality("title", 90.0)
        metrics.update_field_quality("description", 75.0)
        
        assert metrics.field_quality_scores["title"] == 90.0
        assert metrics.field_quality_scores["description"] == 75.0
    
    def test_add_threshold_violation(self):
        """Test recording threshold violations."""
        metrics = QualityMetrics()
        
        metrics.add_threshold_violation()
        metrics.add_threshold_violation()
        
        assert metrics.quality_threshold_violations == 2
    
    def test_get_quality_trend(self):
        """Test getting quality trends."""
        metrics = QualityMetrics()
        
        # Insufficient data
        assert metrics.get_quality_trend() == "insufficient_data"
        
        # Add enough data for trend analysis
        for score in [60, 62, 64, 66, 68, 70, 72, 74, 76, 78]:
            metrics.update_quality_score(score)
        
        trend = metrics.get_quality_trend()
        assert trend == "improving"
    
    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = QualityMetrics()
        metrics.update_quality_score(75.0)
        metrics.update_field_quality("title", 85.0)
        metrics.add_threshold_violation()
        
        result = metrics.to_dict()
        
        assert result["overall_quality_score"] == 75.0
        assert result["field_quality_scores"]["title"] == 85.0
        assert result["quality_threshold_violations"] == 1
        assert "quality_trend" in result
        assert "last_quality_check" in result


class TestScrapingSession:
    """Test ScrapingSession functionality."""
    
    def test_session_creation(self):
        """Test creating a scraping session."""
        session = ScrapingSession("test_session", "https://example.com", "list_strategy")
        
        assert session.session_id == "test_session"
        assert session.target_url == "https://example.com"
        assert session.strategy_used == "list_strategy"
        assert session.status == "running"
        assert isinstance(session.started_at, datetime)
        assert session.completed_at is None
        assert len(session.items) == 0
    
    def test_add_item(self):
        """Test adding items to session."""
        session = ScrapingSession("test", "https://example.com", "list")
        item = ScrapedItem({"title": "Test"}, quality_score=80.0)
        
        session.add_item(item)
        
        assert len(session.items) == 1
        assert session.items[0] == item
        assert session.quality_metrics.overall_quality_score == 80.0
    
    def test_add_error_and_warning(self):
        """Test adding errors and warnings."""
        session = ScrapingSession("test", "https://example.com", "list")
        
        session.add_error("Test error", "network")
        session.add_warning("Test warning")
        
        assert len(session.errors) == 1
        assert len(session.warnings) == 1
        assert session.statistics.error_counts["network"] == 1
    
    def test_complete_session(self):
        """Test completing a session."""
        session = ScrapingSession("test", "https://example.com", "list")
        
        session.complete_session("completed")
        
        assert session.status == "completed"
        assert session.completed_at is not None
    
    def test_get_duration(self):
        """Test getting session duration."""
        session = ScrapingSession("test", "https://example.com", "list")
        
        duration = session.get_duration()
        assert isinstance(duration, timedelta)
        
        # Complete session and test duration
        session.complete_session()
        duration = session.get_duration()
        assert isinstance(duration, timedelta)
    
    def test_get_summary(self):
        """Test getting session summary."""
        session = ScrapingSession("test", "https://example.com", "list")
        item = ScrapedItem({"title": "Test"}, quality_score=75.0)
        session.add_item(item)
        session.add_error("Test error")
        
        summary = session.get_summary()
        
        assert summary["session_id"] == "test"
        assert summary["target_url"] == "https://example.com"
        assert summary["items_scraped"] == 1
        assert summary["errors_count"] == 1
        assert summary["average_quality"] == 75.0
    
    def test_to_dict(self):
        """Test converting session to dictionary."""
        session = ScrapingSession("test", "https://example.com", "list")
        item = ScrapedItem({"title": "Test"}, quality_score=75.0)
        session.add_item(item)
        session.complete_session()
        
        result = session.to_dict()
        
        assert result["session_id"] == "test"
        assert result["target_url"] == "https://example.com"
        assert result["status"] == "completed"
        assert len(result["items"]) == 1
        assert "statistics" in result
        assert "quality_metrics" in result
        assert "summary" in result


class TestScrapingResultsContextProvider:
    """Test ScrapingResultsContextProvider functionality."""
    
    @pytest.fixture
    def context_provider(self):
        """Create a context provider for testing."""
        return ScrapingResultsContextProvider()
    
    @pytest.fixture
    def sample_items(self):
        """Create sample scraped items for testing."""
        items = []
        for i in range(5):
            data = {
                "title": f"Title {i}",
                "description": f"Description {i}",
                "url": f"https://example.com/item{i}"
            }
            item = ScrapedItem(data, quality_score=70.0 + i * 5, source_url="https://example.com")
            item.set_field_completeness("title", 100.0)
            item.set_field_completeness("description", 80.0 + i * 4)
            items.append(item)
        return items
    
    def test_context_provider_initialization(self, context_provider):
        """Test context provider initialization."""
        assert context_provider.current_session is None
        assert len(context_provider.session_history) == 0
        assert len(context_provider.results) == 0
        assert isinstance(context_provider.quality_metrics, QualityMetrics)
        assert isinstance(context_provider.extraction_statistics, ExtractionStatistics)
    
    def test_start_new_session(self, context_provider):
        """Test starting a new scraping session."""
        session = context_provider.start_new_session("test_session", "https://example.com", "list")
        
        assert context_provider.current_session == session
        assert session.session_id == "test_session"
        assert session.target_url == "https://example.com"
        assert session.strategy_used == "list"
        assert context_provider.extraction_statistics.started_at is not None
    
    def test_complete_current_session(self, context_provider):
        """Test completing the current session."""
        # Start a session
        session = context_provider.start_new_session("test", "https://example.com", "list")
        
        # Complete it
        context_provider.complete_current_session("completed")
        
        assert session.status == "completed"
        assert session.completed_at is not None
        assert len(context_provider.session_history) == 1
        assert context_provider.session_history[0] == session
    
    def test_add_scraped_item(self, context_provider, sample_items):
        """Test adding scraped items."""
        context_provider.start_new_session("test", "https://example.com", "list")
        
        for item in sample_items:
            context_provider.add_scraped_item(item)
        
        assert len(context_provider.results) == 5
        assert len(context_provider.current_session.items) == 5
        assert context_provider.quality_metrics.overall_quality_score > 0
    
    def test_add_extraction_error(self, context_provider):
        """Test adding extraction errors."""
        context_provider.start_new_session("test", "https://example.com", "list")
        
        context_provider.add_extraction_error("Network timeout", "network")
        context_provider.add_extraction_error("Parse error", "parsing")
        
        assert context_provider.extraction_statistics.error_counts["network"] == 1
        assert context_provider.extraction_statistics.error_counts["parsing"] == 1
        assert len(context_provider.current_session.errors) == 2
    
    def test_update_extraction_statistics(self, context_provider):
        """Test updating extraction statistics."""
        context_provider.update_extraction_statistics(
            total_items_found=100,
            total_items_extracted=85,
            pages_processed=5
        )
        
        stats = context_provider.extraction_statistics
        assert stats.total_items_found == 100
        assert stats.total_items_extracted == 85
        assert stats.pages_processed == 5
    
    def test_get_info_no_data(self, context_provider):
        """Test getting context info with no data."""
        info = context_provider.get_info()
        
        assert "No active scraping session" in info
        assert "Scraping Results Context" in info
    
    def test_get_info_with_data(self, context_provider, sample_items):
        """Test getting context info with data."""
        # Start session and add items
        context_provider.start_new_session("test_session", "https://example.com", "list")
        for item in sample_items:
            context_provider.add_scraped_item(item)
        
        # Update statistics
        context_provider.update_extraction_statistics(
            total_items_found=10,
            total_items_extracted=5,
            pages_processed=2
        )
        
        info = context_provider.get_info()
        
        assert "Current Scraping Session: test_session" in info
        assert "https://example.com" in info
        assert "**Total Items:** 5" in info
        assert "**Items Found:** 10" in info
        assert "**Items Extracted:** 5" in info
        assert "**Pages Processed:** 2" in info
    
    def test_get_results_summary(self, context_provider, sample_items):
        """Test getting results summary."""
        # Add items
        for item in sample_items:
            context_provider.add_scraped_item(item)
        
        summary = context_provider.get_results_summary()
        
        assert summary["total_items"] == 5
        assert summary["average_quality"] > 0
        assert "quality_distribution" in summary
        assert "field_analysis" in summary
        assert "extraction_statistics" in summary
    
    def test_get_results_summary_no_data(self, context_provider):
        """Test getting results summary with no data."""
        summary = context_provider.get_results_summary()
        
        assert summary["total_items"] == 0
        assert summary["average_quality"] == 0.0
        assert summary["status"] == "no_results"
    
    def test_get_quality_report(self, context_provider, sample_items):
        """Test getting quality report."""
        # Add items with validation errors
        for i, item in enumerate(sample_items):
            if i % 2 == 0:
                item.add_validation_error("Missing field")
            context_provider.add_scraped_item(item)
        
        report = context_provider.get_quality_report()
        
        assert "overall_quality" in report
        assert "completeness_analysis" in report
        assert "quality_issues" in report
        assert "recommendations" in report
        assert report["quality_issues"]["validation_errors"] > 0
    
    def test_get_quality_report_no_data(self, context_provider):
        """Test getting quality report with no data."""
        report = context_provider.get_quality_report()
        
        assert report["status"] == "no_data"
    
    def test_quality_distribution(self, context_provider):
        """Test quality distribution calculation."""
        # Add items with different quality scores
        items = [
            ScrapedItem({"title": "Excellent"}, quality_score=95.0),
            ScrapedItem({"title": "Good"}, quality_score=75.0),
            ScrapedItem({"title": "Fair"}, quality_score=55.0),
            ScrapedItem({"title": "Poor"}, quality_score=25.0)
        ]
        
        for item in items:
            context_provider.add_scraped_item(item)
        
        distribution = context_provider._get_quality_distribution()
        
        assert distribution["excellent"] == 1
        assert distribution["good"] == 1
        assert distribution["fair"] == 1
        assert distribution["poor"] == 1
    
    def test_field_analysis(self, context_provider, sample_items):
        """Test field-level analysis."""
        for item in sample_items:
            context_provider.add_scraped_item(item)
        
        analysis = context_provider._get_field_analysis()
        
        assert "title" in analysis
        assert "description" in analysis
        assert "url" in analysis
        
        # All items have title, so extraction rate should be 100%
        assert analysis["title"]["extraction_rate"] == 100.0
        assert analysis["title"]["avg_quality"] > 0
    
    def test_performance_insights(self, context_provider, sample_items):
        """Test performance insights generation."""
        # Add items and set up statistics
        for item in sample_items:
            context_provider.add_scraped_item(item)
        
        context_provider.extraction_statistics.extraction_rate = 3.5
        context_provider.quality_metrics.overall_quality_score = 85.0
        
        insights = context_provider._generate_performance_insights()
        
        assert len(insights) > 0
        assert any("High quality extraction achieved" in insight for insight in insights)
    
    def test_improvement_recommendations(self, context_provider, sample_items):
        """Test improvement recommendations generation."""
        # Add items with some quality issues
        for item in sample_items:
            item.quality_score = 45.0  # Low quality
            context_provider.add_scraped_item(item)
        
        # Add some errors
        context_provider.add_extraction_error("Network error", "network")
        context_provider.add_extraction_error("Network error", "network")
        
        recommendations = context_provider._generate_improvement_recommendations()
        
        assert len(recommendations) > 0
        assert any("refining extraction selectors" in rec.lower() for rec in recommendations)
        assert any("network" in rec.lower() for rec in recommendations)
    
    def test_session_history_management(self, context_provider):
        """Test session history management."""
        # Create multiple sessions
        for i in range(3):
            session = context_provider.start_new_session(f"session_{i}", f"https://example{i}.com", "list")
            item = ScrapedItem({"title": f"Item {i}"}, quality_score=70.0 + i * 10)
            context_provider.add_scraped_item(item)
            context_provider.complete_current_session()
        
        assert len(context_provider.session_history) == 3
        
        # Test session retrieval
        session = context_provider.get_session_by_id("session_1")
        assert session is not None
        assert session.target_url == "https://example1.com"
        
        # Test non-existent session
        session = context_provider.get_session_by_id("nonexistent")
        assert session is None
    
    def test_context_for_session(self, context_provider):
        """Test getting context for specific session."""
        # Create and complete a session
        session = context_provider.start_new_session("test_session", "https://example.com", "list")
        item = ScrapedItem({"title": "Test"}, quality_score=80.0)
        context_provider.add_scraped_item(item)
        context_provider.complete_current_session()
        
        context = context_provider.get_context_for_session("test_session")
        
        assert "Session Context: test_session" in context
        assert "https://example.com" in context
        assert "**Items Scraped:** 1" in context
        assert "**Average Quality:** 80.0%" in context
    
    def test_context_for_nonexistent_session(self, context_provider):
        """Test getting context for non-existent session."""
        context = context_provider.get_context_for_session("nonexistent")
        
        assert "Session nonexistent not found" in context
    
    def test_export_results(self, context_provider, sample_items):
        """Test exporting results in different formats."""
        for item in sample_items:
            context_provider.add_scraped_item(item)
        
        # Test list format
        list_export = context_provider.export_results("list")
        assert isinstance(list_export, list)
        assert len(list_export) == 5
        
        # Test summary format
        summary_export = context_provider.export_results("summary")
        assert isinstance(summary_export, dict)
        assert "total_items" in summary_export
        
        # Test dict format (default)
        dict_export = context_provider.export_results("dict")
        assert isinstance(dict_export, dict)
        assert "results" in dict_export
        assert "statistics" in dict_export
        assert "quality_metrics" in dict_export
    
    def test_clear_results(self, context_provider, sample_items):
        """Test clearing all results."""
        # Add some data
        context_provider.start_new_session("test", "https://example.com", "list")
        for item in sample_items:
            context_provider.add_scraped_item(item)
        
        # Clear results
        context_provider.clear_results()
        
        assert len(context_provider.results) == 0
        assert context_provider.current_session is None
        assert context_provider.quality_metrics.overall_quality_score == 0.0
    
    def test_cache_functionality(self, context_provider, sample_items):
        """Test context caching functionality."""
        # Add data
        for item in sample_items:
            context_provider.add_scraped_item(item)
        
        # First call should populate cache
        info1 = context_provider.get_info()
        assert "main_context" in context_provider.context_cache
        
        # Second call should use cache
        info2 = context_provider.get_info()
        assert info1 == info2
        
        # Adding new item should clear cache
        new_item = ScrapedItem({"title": "New"}, quality_score=90.0)
        context_provider.add_scraped_item(new_item)
        
        # Cache should be cleared
        assert len(context_provider.context_cache) == 0
    
    def test_aggregated_metrics(self, context_provider):
        """Test aggregated metrics calculation."""
        # Create multiple sessions with different strategies
        strategies = ["list", "detail", "search"]
        
        for i, strategy in enumerate(strategies):
            session = context_provider.start_new_session(f"session_{i}", f"https://example{i}.com", strategy)
            
            # Add items with different quality scores
            for j in range(3):
                item = ScrapedItem({"title": f"Item {j}"}, quality_score=60.0 + i * 10 + j * 5)
                context_provider.add_scraped_item(item)
            
            context_provider.complete_current_session()
        
        # Check aggregated metrics
        assert "total_sessions" in context_provider.aggregated_metrics
        assert "total_items_scraped" in context_provider.aggregated_metrics
        assert "most_successful_strategy" in context_provider.aggregated_metrics
        
        assert context_provider.aggregated_metrics["total_sessions"] == 3
        assert context_provider.aggregated_metrics["total_items_scraped"] == 9
    
    def test_most_successful_strategy(self, context_provider):
        """Test determining most successful strategy."""
        # Create sessions with different strategies and quality scores
        strategies_quality = [("list", 90.0), ("detail", 75.0), ("search", 85.0)]
        
        for strategy, quality in strategies_quality:
            session = context_provider.start_new_session(f"session_{strategy}", "https://example.com", strategy)
            item = ScrapedItem({"title": "Test"}, quality_score=quality)
            context_provider.add_scraped_item(item)
            context_provider.complete_current_session()
        
        most_successful = context_provider._get_most_successful_strategy()
        assert most_successful == "list"  # Highest quality score
    
    @patch('intelligent_web_scraper.context_providers.scraping_results.datetime')
    def test_cache_expiration(self, mock_datetime, context_provider):
        """Test cache expiration functionality."""
        # Mock datetime to control cache timing
        base_time = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.utcnow.return_value = base_time
        
        # Add data and get info (populates cache)
        item = ScrapedItem({"title": "Test"}, quality_score=75.0)
        context_provider.add_scraped_item(item)
        info1 = context_provider.get_info()
        
        # Cache should be valid
        assert context_provider._is_cache_valid("main_context")
        
        # Move time forward beyond cache TTL
        mock_datetime.utcnow.return_value = base_time + timedelta(seconds=400)
        
        # Cache should be invalid
        assert not context_provider._is_cache_valid("main_context")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])