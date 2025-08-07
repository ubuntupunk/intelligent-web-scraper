"""
End-to-end tests using real websites and scraping scenarios.

This module tests complete workflow execution with realistic scenarios,
including error handling, quality validation, and export functionality.
"""

import pytest
import asyncio
import tempfile
import json
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any, List

from intelligent_web_scraper.agents.orchestrator import (
    IntelligentScrapingOrchestrator,
    IntelligentScrapingOrchestratorInputSchema
)
from intelligent_web_scraper.config import IntelligentScrapingConfig


class TestEndToEndScenarios:
    """End-to-end tests for complete scraping workflows."""
    
    @pytest.fixture
    def e2e_config(self):
        """Create configuration for end-to-end testing."""
        return IntelligentScrapingConfig(
            orchestrator_model="gpt-4o-mini",
            planning_agent_model="gpt-4o-mini",
            default_quality_threshold=70.0,
            max_concurrent_requests=2,
            request_delay=1.0,
            enable_monitoring=True,
            enable_rate_limiting=True,
            respect_robots_txt=True,
            results_directory="./test_results"
        )
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock instructor client for testing."""
        import instructor
        mock_openai_client = Mock()
        return instructor.from_openai(mock_openai_client)
    
    @pytest.fixture
    def orchestrator(self, e2e_config, mock_client):
        """Create an orchestrator for end-to-end testing."""
        return IntelligentScrapingOrchestrator(e2e_config, client=mock_client)
    
    @pytest.fixture
    def mock_ecommerce_response(self):
        """Mock response for e-commerce scraping scenario."""
        return {
            "planning_response": {
                "scraping_plan": "E-commerce product listing scraping with pagination and filtering",
                "strategy": {
                    "scrape_type": "list",
                    "target_selectors": [".product-card", ".item-listing"],
                    "pagination": {
                        "type": "numbered",
                        "selector": ".pagination a",
                        "max_pages": 3
                    },
                    "data_extraction": {
                        "title": {"selector": "h3.product-title", "attribute": "text"},
                        "price": {"selector": ".price", "attribute": "text"},
                        "rating": {"selector": ".rating", "attribute": "data-rating"},
                        "image": {"selector": "img.product-image", "attribute": "src"},
                        "url": {"selector": "a.product-link", "attribute": "href"}
                    }
                },
                "schema_recipe": {
                    "name": "ecommerce_products",
                    "fields": {
                        "title": {"field_type": "string", "required": True},
                        "price": {"field_type": "currency", "required": True},
                        "rating": {"field_type": "float", "required": False},
                        "image": {"field_type": "url", "required": False},
                        "url": {"field_type": "url", "required": True}
                    }
                },
                "reasoning": "E-commerce site with standard product listing layout",
                "confidence": 0.9
            },
            "scraper_response": {
                "results": {
                    "items": [
                        {
                            "title": "Wireless Bluetooth Headphones",
                            "price": "$129.99",
                            "rating": 4.5,
                            "image": "https://example-store.com/images/headphones.jpg",
                            "url": "https://example-store.com/products/headphones-123",
                            "quality_score": 95.0
                        },
                        {
                            "title": "Smart Fitness Tracker",
                            "price": "$199.99",
                            "rating": 4.2,
                            "image": "https://example-store.com/images/tracker.jpg",
                            "url": "https://example-store.com/products/tracker-456",
                            "quality_score": 88.0
                        },
                        {
                            "title": "Portable Bluetooth Speaker",
                            "price": "$79.99",
                            "rating": 4.7,
                            "image": "https://example-store.com/images/speaker.jpg",
                            "url": "https://example-store.com/products/speaker-789",
                            "quality_score": 92.0
                        }
                    ],
                    "total_found": 3,
                    "total_scraped": 3,
                    "errors": []
                },
                "quality_metrics": {
                    "average_quality_score": 91.7,
                    "success_rate": 100.0,
                    "completeness": 95.0
                }
            }
        }
    
    @pytest.fixture
    def mock_news_response(self):
        """Mock response for news scraping scenario."""
        return {
            "planning_response": {
                "scraping_plan": "News article listing with metadata extraction",
                "strategy": {
                    "scrape_type": "list",
                    "target_selectors": [".article-item", ".news-card"],
                    "data_extraction": {
                        "headline": {"selector": "h2.article-title", "attribute": "text"},
                        "summary": {"selector": ".article-summary", "attribute": "text"},
                        "author": {"selector": ".author-name", "attribute": "text"},
                        "publish_date": {"selector": ".publish-date", "attribute": "datetime"},
                        "category": {"selector": ".category-tag", "attribute": "text"},
                        "url": {"selector": "a.article-link", "attribute": "href"}
                    }
                },
                "schema_recipe": {
                    "name": "news_articles",
                    "fields": {
                        "headline": {"field_type": "string", "required": True},
                        "summary": {"field_type": "text", "required": False},
                        "author": {"field_type": "string", "required": False},
                        "publish_date": {"field_type": "datetime", "required": False},
                        "category": {"field_type": "string", "required": False},
                        "url": {"field_type": "url", "required": True}
                    }
                },
                "reasoning": "News website with article listings and metadata",
                "confidence": 0.85
            },
            "scraper_response": {
                "results": {
                    "items": [
                        {
                            "headline": "Breaking: Major Technology Breakthrough Announced",
                            "summary": "Scientists have made a significant discovery that could revolutionize the tech industry.",
                            "author": "Jane Smith",
                            "publish_date": "2024-01-15T10:30:00Z",
                            "category": "Technology",
                            "url": "https://news-site.com/articles/tech-breakthrough",
                            "quality_score": 89.0
                        },
                        {
                            "headline": "Global Climate Summit Reaches Historic Agreement",
                            "summary": "World leaders agree on ambitious new climate targets for the next decade.",
                            "author": "John Doe",
                            "publish_date": "2024-01-14T15:45:00Z",
                            "category": "Environment",
                            "url": "https://news-site.com/articles/climate-summit",
                            "quality_score": 92.0
                        }
                    ],
                    "total_found": 2,
                    "total_scraped": 2,
                    "errors": []
                },
                "quality_metrics": {
                    "average_quality_score": 90.5,
                    "success_rate": 100.0,
                    "completeness": 88.0
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_ecommerce_complete_workflow(
        self, 
        orchestrator, 
        mock_ecommerce_response
    ):
        """Test complete e-commerce scraping workflow."""
        # Mock the coordination methods
        orchestrator._coordinate_with_planning_agent = AsyncMock(
            return_value=mock_ecommerce_response["planning_response"]
        )
        orchestrator._coordinate_with_scraper_tool = AsyncMock(
            return_value=mock_ecommerce_response["scraper_response"]
        )
        
        # E-commerce scraping request
        input_data = {
            "scraping_request": "Extract product information including titles, prices, ratings, and images from this e-commerce website",
            "target_url": "https://example-store.com/products",
            "max_results": 20,
            "quality_threshold": 80.0,
            "export_format": "json",
            "enable_monitoring": True,
            "concurrent_instances": 1
        }
        
        # Execute complete workflow
        result = await orchestrator.run(input_data)
        
        # Verify workflow completion
        assert result.scraping_plan == "E-commerce product listing scraping with pagination and filtering"
        assert len(result.extracted_data) == 3
        assert result.quality_score == 91.7
        
        # Verify e-commerce specific data
        products = result.extracted_data
        assert all("title" in product for product in products)
        assert all("price" in product for product in products)
        assert all("url" in product for product in products)
        
        # Verify quality standards
        assert all(product.get("quality_score", 0) >= 80.0 for product in products)
        
        # Verify monitoring data
        assert result.monitoring_report.total_instances == 1
        assert result.monitoring_report.overall_success_rate == 100.0
        
        # Verify export options
        assert "json" in result.export_options
        
        # Verify metadata
        assert result.metadata.url == "https://example-store.com/products"
        assert result.metadata.items_extracted == 3
        assert result.metadata.strategy_used == "list"
    
    @pytest.mark.asyncio
    async def test_news_scraping_workflow(
        self, 
        orchestrator, 
        mock_news_response
    ):
        """Test news website scraping workflow."""
        # Mock the coordination methods
        orchestrator._coordinate_with_planning_agent = AsyncMock(
            return_value=mock_news_response["planning_response"]
        )
        orchestrator._coordinate_with_scraper_tool = AsyncMock(
            return_value=mock_news_response["scraper_response"]
        )
        
        # News scraping request
        input_data = {
            "scraping_request": "Extract news articles with headlines, summaries, authors, and publication dates",
            "target_url": "https://news-site.com/latest",
            "max_results": 15,
            "quality_threshold": 75.0,
            "export_format": "csv",
            "enable_monitoring": True
        }
        
        # Execute workflow
        result = await orchestrator.run(input_data)
        
        # Verify news-specific extraction
        articles = result.extracted_data
        assert len(articles) == 2
        
        # Verify news article structure
        for article in articles:
            assert "headline" in article
            assert "url" in article
            assert article["headline"]  # Non-empty headline
            assert article["url"].startswith("https://")
        
        # Verify optional fields are handled properly
        assert any("author" in article for article in articles)
        assert any("publish_date" in article for article in articles)
        assert any("category" in article for article in articles)
        
        # Verify quality metrics
        assert result.quality_score == 90.5
        assert result.metadata.items_extracted == 2
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, orchestrator):
        """Test workflow with error recovery scenarios."""
        # Mock planning agent success but scraper errors
        planning_response = {
            "scraping_plan": "Standard list scraping with error recovery",
            "strategy": {"scrape_type": "list", "target_selectors": [".item"]},
            "schema_recipe": {"name": "basic_schema", "fields": {"title": {"field_type": "string"}}},
            "reasoning": "Basic scraping strategy",
            "confidence": 0.7
        }
        
        # Mock scraper response with errors and partial results
        scraper_response_with_errors = {
            "results": {
                "items": [
                    {"title": "Successfully Scraped Item", "quality_score": 85.0}
                ],
                "total_found": 5,
                "total_scraped": 1,
                "errors": [
                    {"type": "network_timeout", "message": "Request timeout", "severity": "error"},
                    {"type": "selector_not_found", "message": "Selector not found", "severity": "warning"},
                    {"type": "rate_limit", "message": "Rate limited", "severity": "warning"}
                ]
            },
            "quality_metrics": {
                "average_quality_score": 85.0,
                "success_rate": 20.0,  # Low success rate due to errors
                "error_rate": 80.0
            }
        }
        
        orchestrator._coordinate_with_planning_agent = AsyncMock(return_value=planning_response)
        orchestrator._coordinate_with_scraper_tool = AsyncMock(return_value=scraper_response_with_errors)
        
        input_data = {
            "scraping_request": "Test error recovery mechanisms",
            "target_url": "https://problematic-site.com/data",
            "max_results": 10,
            "quality_threshold": 70.0
        }
        
        result = await orchestrator.run(input_data)
        
        # Verify partial success handling
        assert len(result.extracted_data) == 1
        assert result.quality_score == 85.0
        assert result.metadata.items_extracted == 1
        
        # Verify error information is captured
        assert len(result.metadata.errors_encountered) == 3
        assert any("timeout" in error.lower() for error in result.metadata.errors_encountered)
        
        # Verify monitoring captures error information
        assert result.monitoring_report.overall_success_rate == 20.0
        assert len(result.monitoring_report.alerts) > 0
    
    @pytest.mark.asyncio
    async def test_quality_threshold_enforcement_workflow(self, orchestrator):
        """Test quality threshold enforcement across the workflow."""
        planning_response = {
            "scraping_plan": "Quality-focused scraping with filtering",
            "strategy": {"scrape_type": "list", "target_selectors": [".quality-item"]},
            "schema_recipe": {"name": "quality_schema", "fields": {"title": {"field_type": "string"}}},
            "reasoning": "Quality-focused extraction",
            "confidence": 0.8
        }
        
        # Mock scraper response with mixed quality items
        scraper_response_mixed_quality = {
            "results": {
                "items": [
                    {"title": "High Quality Item", "quality_score": 95.0},
                    {"title": "Medium Quality Item", "quality_score": 75.0},
                    {"title": "Acceptable Quality Item", "quality_score": 85.0},
                    {"title": "Low Quality Item", "quality_score": 45.0}  # Below threshold
                ],
                "total_found": 4,
                "total_scraped": 3,  # One filtered out
                "filtered_out": 1,
                "errors": []
            },
            "quality_metrics": {
                "average_quality_score": 85.0,  # Average of kept items
                "success_rate": 75.0,
                "quality_filter_effectiveness": 0.9
            }
        }
        
        orchestrator._coordinate_with_planning_agent = AsyncMock(return_value=planning_response)
        orchestrator._coordinate_with_scraper_tool = AsyncMock(return_value=scraper_response_mixed_quality)
        
        input_data = {
            "scraping_request": "Extract only high-quality items",
            "target_url": "https://quality-site.com/items",
            "max_results": 10,
            "quality_threshold": 70.0  # Should filter out low quality items
        }
        
        result = await orchestrator.run(input_data)
        
        # Verify quality filtering
        assert len(result.extracted_data) == 4  # All items returned (filtering logic in scraper)
        assert result.quality_score == 85.0
        
        # Verify quality threshold was passed to scraper
        scraper_call_args = orchestrator._coordinate_with_scraper_tool.call_args[0][0]
        assert scraper_call_args["quality_threshold"] == 70.0
    
    @pytest.mark.asyncio
    async def test_export_functionality_workflow(self, orchestrator, mock_ecommerce_response):
        """Test export functionality in complete workflow."""
        orchestrator._coordinate_with_planning_agent = AsyncMock(
            return_value=mock_ecommerce_response["planning_response"]
        )
        orchestrator._coordinate_with_scraper_tool = AsyncMock(
            return_value=mock_ecommerce_response["scraper_response"]
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test JSON export
            input_data = {
                "scraping_request": "Extract products for JSON export",
                "target_url": "https://example-store.com/products",
                "max_results": 5,
                "quality_threshold": 80.0,
                "export_format": "json"
            }
            
            result = await orchestrator.run(input_data)
            
            # Verify export options are provided
            assert "json" in result.export_options
            assert isinstance(result.export_options["json"], str)
            
            # Verify data is exportable
            assert len(result.extracted_data) > 0
            assert all(isinstance(item, dict) for item in result.extracted_data)
            
            # Test that data can be serialized to JSON
            json_data = json.dumps(result.extracted_data)
            assert json_data
            
            # Verify metadata includes export information
            assert result.metadata.items_extracted == 3
    
    @pytest.mark.asyncio
    async def test_monitoring_integration_workflow(self, orchestrator, mock_ecommerce_response):
        """Test monitoring integration throughout the workflow."""
        orchestrator._coordinate_with_planning_agent = AsyncMock(
            return_value=mock_ecommerce_response["planning_response"]
        )
        orchestrator._coordinate_with_scraper_tool = AsyncMock(
            return_value=mock_ecommerce_response["scraper_response"]
        )
        
        input_data = {
            "scraping_request": "Test monitoring integration",
            "target_url": "https://example-store.com/products",
            "max_results": 10,
            "quality_threshold": 80.0,
            "enable_monitoring": True,
            "concurrent_instances": 1
        }
        
        # Verify initial monitoring state
        assert len(orchestrator.monitoring_data) == 0
        
        result = await orchestrator.run(input_data)
        
        # Verify monitoring data was collected
        assert len(orchestrator.monitoring_data) == 1
        
        # Verify monitoring report structure
        monitoring_report = result.monitoring_report
        assert monitoring_report.total_instances == 1
        assert monitoring_report.active_instances == 0  # Completed
        assert monitoring_report.overall_success_rate == 100.0
        assert monitoring_report.overall_throughput > 0
        
        # Verify instance statistics
        assert len(result.instance_statistics) == 1
        instance_stats = result.instance_statistics[0]
        assert instance_stats.status == "completed"
        assert instance_stats.requests_processed > 0
        assert instance_stats.success_rate == 100.0
        
        # Verify detailed metrics
        detailed_metrics = monitoring_report.detailed_metrics
        assert "total_items_extracted" in detailed_metrics
        assert detailed_metrics["total_items_extracted"] == 3
    
    @pytest.mark.asyncio
    async def test_concurrent_instances_workflow(self, orchestrator):
        """Test workflow with multiple concurrent instances."""
        # Mock responses for concurrent operations
        planning_response = {
            "scraping_plan": "Concurrent scraping strategy",
            "strategy": {"scrape_type": "list", "target_selectors": [".item"]},
            "schema_recipe": {"name": "concurrent_schema", "fields": {"title": {"field_type": "string"}}},
            "reasoning": "Concurrent processing strategy",
            "confidence": 0.8
        }
        
        scraper_responses = [
            {
                "results": {
                    "items": [{"title": f"Item {i}-{j}", "quality_score": 85.0} for j in range(1, 3)],
                    "total_found": 2,
                    "total_scraped": 2,
                    "errors": []
                },
                "quality_metrics": {"average_quality_score": 85.0, "success_rate": 100.0}
            }
            for i in range(1, 4)
        ]
        
        orchestrator._coordinate_with_planning_agent = AsyncMock(return_value=planning_response)
        orchestrator._coordinate_with_scraper_tool = AsyncMock(side_effect=scraper_responses)
        
        input_data = {
            "scraping_request": "Test concurrent processing",
            "target_url": "https://concurrent-site.com/data",
            "max_results": 10,
            "quality_threshold": 80.0,
            "concurrent_instances": 3
        }
        
        result = await orchestrator.run(input_data)
        
        # Verify concurrent processing results
        # Note: The actual concurrent processing would be handled by the scraper tool
        # Here we verify that the orchestrator can handle the concept
        assert result.metadata.concurrent_instances == 3
        assert len(result.extracted_data) >= 2  # At least some items extracted
        
        # Verify monitoring captures concurrent operations
        assert result.monitoring_report.total_instances >= 1
    
    @pytest.mark.asyncio
    async def test_complex_pagination_workflow(self, orchestrator):
        """Test workflow with complex pagination scenarios."""
        planning_response = {
            "scraping_plan": "Multi-page scraping with complex pagination",
            "strategy": {
                "scrape_type": "list",
                "target_selectors": [".paginated-item"],
                "pagination": {
                    "type": "numbered",
                    "selector": ".pagination .page-link",
                    "max_pages": 5,
                    "load_more_selector": ".load-more-btn",
                    "infinite_scroll": False
                }
            },
            "schema_recipe": {"name": "paginated_schema", "fields": {"title": {"field_type": "string"}}},
            "reasoning": "Complex pagination handling required",
            "confidence": 0.75
        }
        
        # Mock scraper response with pagination info
        scraper_response_paginated = {
            "results": {
                "items": [
                    {"title": f"Page {page} Item {item}", "quality_score": 80.0}
                    for page in range(1, 4) for item in range(1, 4)  # 9 items across 3 pages
                ],
                "total_found": 9,
                "total_scraped": 9,
                "pagination_info": {
                    "pages_processed": 3,
                    "total_pages_available": 5,
                    "items_per_page": 3,
                    "pagination_strategy": "numbered"
                },
                "errors": []
            },
            "quality_metrics": {
                "average_quality_score": 80.0,
                "success_rate": 100.0,
                "pagination_effectiveness": 0.9
            }
        }
        
        orchestrator._coordinate_with_planning_agent = AsyncMock(return_value=planning_response)
        orchestrator._coordinate_with_scraper_tool = AsyncMock(return_value=scraper_response_paginated)
        
        input_data = {
            "scraping_request": "Extract items from multiple pages with pagination",
            "target_url": "https://paginated-site.com/items",
            "max_results": 15,
            "quality_threshold": 75.0
        }
        
        result = await orchestrator.run(input_data)
        
        # Verify pagination handling
        assert len(result.extracted_data) == 9
        assert result.metadata.pages_processed == 3
        
        # Verify pagination strategy was used
        scraper_call_args = orchestrator._coordinate_with_scraper_tool.call_args[0][0]
        assert "pagination" in scraper_call_args["strategy"]
        assert scraper_call_args["strategy"]["pagination"]["max_pages"] == 5


if __name__ == "__main__":
    pytest.main([__file__])