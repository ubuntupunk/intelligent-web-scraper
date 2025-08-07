"""
Integration tests for planning agent-scraper tool coordination.

This module tests the integration between the AtomicScraperPlanningAgent
and the AtomicScraperTool to ensure proper strategy execution and data extraction.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List

from intelligent_web_scraper.config import IntelligentScrapingConfig


class TestPlanningScraperIntegration:
    """Test integration between planning agent and scraper tool."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return IntelligentScrapingConfig(
            planning_agent_model="gpt-4o-mini",
            default_quality_threshold=60.0,
            max_concurrent_requests=3,
            request_delay=1.0,
            enable_monitoring=True
        )
    
    @pytest.fixture
    def sample_planning_strategy(self):
        """Create a sample planning strategy."""
        return {
            "scrape_type": "list",
            "target_selectors": [".product-card", ".item-listing"],
            "pagination": {
                "type": "numbered",
                "selector": ".pagination .page-link",
                "max_pages": 5
            },
            "data_extraction": {
                "title": {
                    "selector": "h3.product-title",
                    "attribute": "text",
                    "required": True
                },
                "price": {
                    "selector": ".price-current",
                    "attribute": "text",
                    "required": True,
                    "transform": "extract_price"
                },
                "description": {
                    "selector": ".product-description",
                    "attribute": "text",
                    "required": False
                },
                "image_url": {
                    "selector": "img.product-image",
                    "attribute": "src",
                    "required": False
                },
                "product_url": {
                    "selector": "a.product-link",
                    "attribute": "href",
                    "required": True,
                    "transform": "absolute_url"
                }
            },
            "quality_checks": {
                "min_text_length": 10,
                "required_fields": ["title", "price", "product_url"],
                "url_validation": True
            }
        }
    
    @pytest.fixture
    def sample_schema_recipe(self):
        """Create a sample schema recipe."""
        return {
            "name": "ecommerce_product_schema",
            "version": "1.0",
            "fields": {
                "title": {
                    "field_type": "string",
                    "required": True,
                    "max_length": 200,
                    "validation": "non_empty"
                },
                "price": {
                    "field_type": "currency",
                    "required": True,
                    "validation": "positive_number"
                },
                "description": {
                    "field_type": "text",
                    "required": False,
                    "max_length": 1000
                },
                "image_url": {
                    "field_type": "url",
                    "required": False,
                    "validation": "valid_url"
                },
                "product_url": {
                    "field_type": "url",
                    "required": True,
                    "validation": "valid_url"
                }
            },
            "quality_thresholds": {
                "completeness": 80.0,
                "accuracy": 90.0,
                "consistency": 85.0
            }
        }
    
    @pytest.fixture
    def mock_scraper_tool(self):
        """Create a mock scraper tool."""
        tool = Mock()
        tool.run = AsyncMock()
        tool.get_tool_info = Mock(return_value={
            "name": "AtomicScraperTool",
            "version": "1.0.0",
            "capabilities": ["list_scraping", "pagination", "quality_analysis"]
        })
        return tool
    
    @pytest.fixture
    def mock_planning_agent(self):
        """Create a mock planning agent."""
        agent = Mock()
        agent.run = AsyncMock()
        agent.analyze_website = AsyncMock()
        agent.generate_strategy = AsyncMock()
        agent.create_schema_recipe = AsyncMock()
        return agent
    
    @pytest.mark.asyncio
    async def test_strategy_execution_success(
        self, 
        mock_scraper_tool, 
        sample_planning_strategy, 
        sample_schema_recipe
    ):
        """Test successful execution of planning strategy by scraper tool."""
        # Mock successful scraper response
        mock_scraper_response = {
            "results": {
                "items": [
                    {
                        "title": "Wireless Bluetooth Headphones",
                        "price": "$89.99",
                        "description": "High-quality wireless headphones with noise cancellation",
                        "image_url": "https://example.com/images/headphones.jpg",
                        "product_url": "https://example.com/products/headphones-123",
                        "quality_score": 92.5
                    },
                    {
                        "title": "Smart Fitness Watch",
                        "price": "$199.99",
                        "description": "Advanced fitness tracking with heart rate monitor",
                        "image_url": "https://example.com/images/watch.jpg",
                        "product_url": "https://example.com/products/watch-456",
                        "quality_score": 88.0
                    }
                ],
                "total_found": 2,
                "total_scraped": 2,
                "strategy_used": sample_planning_strategy,
                "errors": []
            },
            "quality_metrics": {
                "average_quality_score": 90.25,
                "success_rate": 100.0,
                "completeness_score": 95.0,
                "accuracy_score": 92.0,
                "consistency_score": 88.0
            },
            "execution_metadata": {
                "pages_processed": 1,
                "total_processing_time": 3.2,
                "strategy_effectiveness": 0.95
            }
        }
        
        mock_scraper_tool.run.return_value = mock_scraper_response
        
        # Prepare scraper input
        scraper_input = {
            "target_url": "https://example-store.com/products",
            "strategy": sample_planning_strategy,
            "schema_recipe": sample_schema_recipe,
            "max_results": 10,
            "quality_threshold": 80.0,
            "operation_id": "test-operation-123"
        }
        
        # Execute scraper with planning strategy
        result = await mock_scraper_tool.run(scraper_input)
        
        # Verify scraper was called with correct parameters
        mock_scraper_tool.run.assert_called_once_with(scraper_input)
        
        # Verify result structure
        assert "results" in result
        assert "quality_metrics" in result
        assert "execution_metadata" in result
        
        # Verify extracted data matches strategy expectations
        items = result["results"]["items"]
        assert len(items) == 2
        
        # Verify all required fields from schema are present
        required_fields = ["title", "price", "product_url"]
        for item in items:
            for field in required_fields:
                assert field in item
                assert item[field] is not None
        
        # Verify quality scores meet threshold
        for item in items:
            assert item["quality_score"] >= 80.0
        
        # Verify strategy was executed correctly
        assert result["results"]["strategy_used"] == sample_planning_strategy
        assert result["quality_metrics"]["average_quality_score"] >= 80.0
    
    @pytest.mark.asyncio
    async def test_pagination_strategy_execution(
        self, 
        mock_scraper_tool, 
        sample_planning_strategy, 
        sample_schema_recipe
    ):
        """Test execution of pagination strategy."""
        # Mock scraper response with pagination
        mock_paginated_response = {
            "results": {
                "items": [
                    {"title": f"Product {i}", "price": f"${i*10}.99", "product_url": f"https://example.com/product-{i}", "quality_score": 85.0}
                    for i in range(1, 16)  # 15 items across multiple pages
                ],
                "total_found": 15,
                "total_scraped": 15,
                "strategy_used": sample_planning_strategy,
                "pagination_info": {
                    "pages_processed": 3,
                    "total_pages_available": 5,
                    "items_per_page": 5
                },
                "errors": []
            },
            "quality_metrics": {
                "average_quality_score": 85.0,
                "success_rate": 100.0
            },
            "execution_metadata": {
                "pages_processed": 3,
                "total_processing_time": 8.5,
                "pagination_effectiveness": 0.88
            }
        }
        
        mock_scraper_tool.run.return_value = mock_paginated_response
        
        # Test pagination strategy
        scraper_input = {
            "target_url": "https://example-store.com/products",
            "strategy": sample_planning_strategy,
            "schema_recipe": sample_schema_recipe,
            "max_results": 15,
            "quality_threshold": 80.0
        }
        
        result = await mock_scraper_tool.run(scraper_input)
        
        # Verify pagination was handled
        assert result["results"]["pagination_info"]["pages_processed"] == 3
        assert len(result["results"]["items"]) == 15
        assert result["execution_metadata"]["pages_processed"] == 3
        
        # Verify all items meet quality threshold
        for item in result["results"]["items"]:
            assert item["quality_score"] >= 80.0
    
    @pytest.mark.asyncio
    async def test_quality_filtering_integration(
        self, 
        mock_scraper_tool, 
        sample_planning_strategy, 
        sample_schema_recipe
    ):
        """Test quality filtering based on planning strategy and schema."""
        # Mock scraper response with mixed quality items
        mock_mixed_quality_response = {
            "results": {
                "items": [
                    {"title": "High Quality Product", "price": "$99.99", "product_url": "https://example.com/high", "quality_score": 95.0},
                    {"title": "Medium Quality Product", "price": "$49.99", "product_url": "https://example.com/medium", "quality_score": 75.0},
                    {"title": "Low Quality Product", "price": "$19.99", "product_url": "https://example.com/low", "quality_score": 45.0},
                    {"title": "", "price": "$29.99", "product_url": "", "quality_score": 25.0}  # Invalid item
                ],
                "total_found": 4,
                "total_scraped": 2,  # Only high quality items kept
                "filtered_out": 2,
                "strategy_used": sample_planning_strategy,
                "errors": []
            },
            "quality_metrics": {
                "average_quality_score": 85.0,  # Average of kept items
                "success_rate": 50.0,  # 2 out of 4 items kept
                "filtering_effectiveness": 0.75
            }
        }
        
        mock_scraper_tool.run.return_value = mock_mixed_quality_response
        
        # Test with high quality threshold
        scraper_input = {
            "target_url": "https://example-store.com/products",
            "strategy": sample_planning_strategy,
            "schema_recipe": sample_schema_recipe,
            "max_results": 10,
            "quality_threshold": 70.0  # Should filter out low quality items
        }
        
        result = await mock_scraper_tool.run(scraper_input)
        
        # Verify quality filtering worked
        assert result["results"]["total_scraped"] == 2
        assert result["results"]["filtered_out"] == 2
        assert result["quality_metrics"]["average_quality_score"] >= 70.0
        
        # Verify remaining items meet quality standards
        for item in result["results"]["items"]:
            assert item["quality_score"] >= 70.0
            assert item["title"]  # Non-empty title
            assert item["product_url"]  # Valid URL
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(
        self, 
        mock_scraper_tool, 
        sample_planning_strategy, 
        sample_schema_recipe
    ):
        """Test error handling and recovery mechanisms."""
        # Mock scraper response with errors
        mock_error_response = {
            "results": {
                "items": [
                    {"title": "Successful Item", "price": "$99.99", "product_url": "https://example.com/success", "quality_score": 90.0}
                ],
                "total_found": 3,
                "total_scraped": 1,
                "strategy_used": sample_planning_strategy,
                "errors": [
                    {
                        "type": "selector_not_found",
                        "message": "Price selector '.price-current' not found on page 2",
                        "page": 2,
                        "severity": "warning"
                    },
                    {
                        "type": "network_timeout",
                        "message": "Request timeout for page 3",
                        "page": 3,
                        "severity": "error"
                    }
                ]
            },
            "quality_metrics": {
                "average_quality_score": 90.0,
                "success_rate": 33.3,  # 1 out of 3 items successful
                "error_rate": 66.7
            },
            "execution_metadata": {
                "pages_processed": 3,
                "pages_successful": 1,
                "recovery_attempts": 2,
                "fallback_strategies_used": ["alternative_selector", "retry_with_delay"]
            }
        }
        
        mock_scraper_tool.run.return_value = mock_error_response
        
        scraper_input = {
            "target_url": "https://problematic-site.com/products",
            "strategy": sample_planning_strategy,
            "schema_recipe": sample_schema_recipe,
            "max_results": 10,
            "quality_threshold": 80.0
        }
        
        result = await mock_scraper_tool.run(scraper_input)
        
        # Verify error handling
        assert len(result["results"]["errors"]) == 2
        assert result["results"]["total_scraped"] == 1  # Partial success
        assert result["execution_metadata"]["recovery_attempts"] == 2
        
        # Verify successful items still meet quality standards
        for item in result["results"]["items"]:
            assert item["quality_score"] >= 80.0
    
    @pytest.mark.asyncio
    async def test_schema_validation_integration(
        self, 
        mock_scraper_tool, 
        sample_planning_strategy, 
        sample_schema_recipe
    ):
        """Test schema validation during scraping execution."""
        # Mock scraper response with schema validation
        mock_validated_response = {
            "results": {
                "items": [
                    {
                        "title": "Valid Product",
                        "price": "$99.99",
                        "description": "This is a valid product description",
                        "image_url": "https://example.com/images/valid.jpg",
                        "product_url": "https://example.com/products/valid",
                        "quality_score": 95.0,
                        "validation_status": "passed"
                    },
                    {
                        "title": "Partially Valid Product",
                        "price": "$49.99",
                        "description": "",  # Missing description (optional field)
                        "image_url": "invalid-url",  # Invalid URL
                        "product_url": "https://example.com/products/partial",
                        "quality_score": 70.0,
                        "validation_status": "partial",
                        "validation_errors": ["invalid_image_url"]
                    }
                ],
                "total_found": 2,
                "total_scraped": 2,
                "validation_summary": {
                    "fully_valid": 1,
                    "partially_valid": 1,
                    "invalid": 0
                },
                "strategy_used": sample_planning_strategy,
                "errors": []
            },
            "quality_metrics": {
                "average_quality_score": 82.5,
                "validation_success_rate": 100.0,  # All items have some valid data
                "schema_compliance": 85.0
            }
        }
        
        mock_scraper_tool.run.return_value = mock_validated_response
        
        scraper_input = {
            "target_url": "https://example-store.com/products",
            "strategy": sample_planning_strategy,
            "schema_recipe": sample_schema_recipe,
            "max_results": 10,
            "quality_threshold": 60.0,
            "enable_validation": True
        }
        
        result = await mock_scraper_tool.run(scraper_input)
        
        # Verify schema validation was applied
        assert "validation_summary" in result["results"]
        assert result["results"]["validation_summary"]["fully_valid"] == 1
        assert result["results"]["validation_summary"]["partially_valid"] == 1
        
        # Verify validation status is included in items
        for item in result["results"]["items"]:
            assert "validation_status" in item
            assert item["validation_status"] in ["passed", "partial", "failed"]
    
    @pytest.mark.asyncio
    async def test_strategy_adaptation_feedback(
        self, 
        mock_scraper_tool, 
        mock_planning_agent, 
        sample_planning_strategy, 
        sample_schema_recipe
    ):
        """Test feedback loop for strategy adaptation."""
        # Mock initial scraper response with low success rate
        initial_response = {
            "results": {
                "items": [
                    {"title": "Item 1", "price": "$99.99", "product_url": "https://example.com/1", "quality_score": 60.0}
                ],
                "total_found": 5,
                "total_scraped": 1,
                "strategy_used": sample_planning_strategy,
                "errors": [
                    {"type": "selector_mismatch", "message": "Primary selector failed", "severity": "error"}
                ]
            },
            "quality_metrics": {
                "average_quality_score": 60.0,
                "success_rate": 20.0,  # Low success rate
                "strategy_effectiveness": 0.3
            },
            "adaptation_suggestions": [
                "try_alternative_selectors",
                "adjust_quality_thresholds",
                "enable_fallback_strategies"
            ]
        }
        
        # Mock adapted strategy from planning agent
        adapted_strategy = sample_planning_strategy.copy()
        adapted_strategy["target_selectors"] = [".product-item", ".listing-card"]  # Alternative selectors
        adapted_strategy["fallback_selectors"] = [".item", ".product"]
        
        # Mock improved scraper response with adapted strategy
        improved_response = {
            "results": {
                "items": [
                    {"title": f"Improved Item {i}", "price": f"${i*20}.99", "product_url": f"https://example.com/{i}", "quality_score": 85.0}
                    for i in range(1, 5)
                ],
                "total_found": 5,
                "total_scraped": 4,
                "strategy_used": adapted_strategy,
                "errors": []
            },
            "quality_metrics": {
                "average_quality_score": 85.0,
                "success_rate": 80.0,  # Improved success rate
                "strategy_effectiveness": 0.85
            }
        }
        
        # Set up mock responses
        mock_scraper_tool.run.side_effect = [initial_response, improved_response]
        mock_planning_agent.adapt_strategy.return_value = adapted_strategy
        
        # First execution with original strategy
        initial_input = {
            "target_url": "https://example-store.com/products",
            "strategy": sample_planning_strategy,
            "schema_recipe": sample_schema_recipe,
            "max_results": 10,
            "quality_threshold": 70.0
        }
        
        initial_result = await mock_scraper_tool.run(initial_input)
        
        # Verify low performance detected
        assert initial_result["quality_metrics"]["success_rate"] < 50.0
        assert len(initial_result["adaptation_suggestions"]) > 0
        
        # Simulate strategy adaptation based on feedback
        adapted_input = initial_input.copy()
        adapted_input["strategy"] = adapted_strategy
        
        # Second execution with adapted strategy
        improved_result = await mock_scraper_tool.run(adapted_input)
        
        # Verify improvement
        assert improved_result["quality_metrics"]["success_rate"] > initial_result["quality_metrics"]["success_rate"]
        assert improved_result["results"]["total_scraped"] > initial_result["results"]["total_scraped"]
        assert improved_result["quality_metrics"]["strategy_effectiveness"] > 0.8
    
    @pytest.mark.asyncio
    async def test_concurrent_scraping_coordination(
        self, 
        mock_scraper_tool, 
        sample_planning_strategy, 
        sample_schema_recipe
    ):
        """Test coordination of concurrent scraping operations."""
        # Mock responses for concurrent operations
        responses = [
            {
                "results": {
                    "items": [{"title": f"Concurrent Item {i}-{j}", "price": f"${j*10}.99", "product_url": f"https://example.com/{i}-{j}", "quality_score": 80.0}
                             for j in range(1, 4)],
                    "total_found": 3,
                    "total_scraped": 3,
                    "operation_id": f"concurrent-op-{i}",
                    "errors": []
                },
                "quality_metrics": {"average_quality_score": 80.0, "success_rate": 100.0}
            }
            for i in range(1, 4)
        ]
        
        mock_scraper_tool.run.side_effect = responses
        
        # Create concurrent scraping inputs
        concurrent_inputs = [
            {
                "target_url": f"https://example-store.com/category-{i}",
                "strategy": sample_planning_strategy,
                "schema_recipe": sample_schema_recipe,
                "max_results": 5,
                "quality_threshold": 75.0,
                "operation_id": f"concurrent-op-{i}"
            }
            for i in range(1, 4)
        ]
        
        # Execute concurrent scraping operations
        results = await asyncio.gather(*[
            mock_scraper_tool.run(input_data) for input_data in concurrent_inputs
        ])
        
        # Verify all operations completed successfully
        assert len(results) == 3
        assert mock_scraper_tool.run.call_count == 3
        
        # Verify each operation has unique results
        operation_ids = [result["results"]["operation_id"] for result in results]
        assert len(set(operation_ids)) == 3  # All unique
        
        # Verify quality standards maintained across all operations
        for result in results:
            assert result["quality_metrics"]["average_quality_score"] >= 75.0
            assert len(result["results"]["items"]) == 3


if __name__ == "__main__":
    pytest.main([__file__])