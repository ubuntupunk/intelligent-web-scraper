"""
Integration tests for orchestrator-planning agent coordination.

This module tests the integration between the IntelligentScrapingOrchestrator
and the AtomicScraperPlanningAgent to ensure proper coordination and data flow.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any

from intelligent_web_scraper.agents.orchestrator import (
    IntelligentScrapingOrchestrator,
    IntelligentScrapingOrchestratorInputSchema,
    IntelligentScrapingOrchestratorOutputSchema
)
from intelligent_web_scraper.config import IntelligentScrapingConfig


class TestOrchestratorPlanningIntegration:
    """Test integration between orchestrator and planning agent."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return IntelligentScrapingConfig(
            orchestrator_model="gpt-4o-mini",
            planning_agent_model="gpt-4o-mini",
            default_quality_threshold=70.0,
            max_concurrent_requests=3,
            enable_monitoring=True
        )
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock instructor client."""
        import instructor
        mock_openai_client = Mock()
        return instructor.from_openai(mock_openai_client)
    
    @pytest.fixture
    def orchestrator(self, mock_config, mock_client):
        """Create an orchestrator for testing."""
        return IntelligentScrapingOrchestrator(mock_config, client=mock_client)
    
    @pytest.fixture
    def sample_planning_response(self):
        """Create a sample planning agent response."""
        return {
            "scraping_plan": "Comprehensive scraping plan for e-commerce product data",
            "strategy": {
                "scrape_type": "list",
                "target_selectors": [".product-item", ".item-card"],
                "pagination": {
                    "type": "numbered",
                    "selector": ".pagination a",
                    "max_pages": 10
                },
                "data_extraction": {
                    "title": {"selector": "h2.product-title", "attribute": "text"},
                    "price": {"selector": ".price", "attribute": "text"},
                    "url": {"selector": "a.product-link", "attribute": "href"},
                    "image": {"selector": "img.product-image", "attribute": "src"}
                }
            },
            "schema_recipe": {
                "name": "product_schema",
                "fields": {
                    "title": {"field_type": "string", "required": True},
                    "price": {"field_type": "string", "required": True},
                    "url": {"field_type": "url", "required": True},
                    "image": {"field_type": "url", "required": False}
                }
            },
            "reasoning": "The website appears to be an e-commerce platform with product listings. Using list scraping strategy with pagination support.",
            "confidence": 0.85,
            "estimated_items": 150,
            "complexity_score": 3.2
        }
    
    @pytest.fixture
    def sample_scraper_response(self):
        """Create a sample scraper tool response."""
        return {
            "results": {
                "items": [
                    {
                        "title": "Premium Wireless Headphones",
                        "price": "$199.99",
                        "url": "https://example.com/products/headphones-1",
                        "image": "https://example.com/images/headphones-1.jpg",
                        "quality_score": 92.5
                    },
                    {
                        "title": "Smart Fitness Tracker",
                        "price": "$149.99",
                        "url": "https://example.com/products/tracker-1",
                        "image": "https://example.com/images/tracker-1.jpg",
                        "quality_score": 88.0
                    },
                    {
                        "title": "Bluetooth Speaker",
                        "price": "$79.99",
                        "url": "https://example.com/products/speaker-1",
                        "image": "https://example.com/images/speaker-1.jpg",
                        "quality_score": 85.5
                    }
                ],
                "total_found": 3,
                "total_scraped": 3,
                "strategy_used": {
                    "scrape_type": "list",
                    "selectors_used": [".product-item"],
                    "pagination_used": False
                },
                "errors": []
            },
            "summary": "Successfully scraped 3 product items from the target website",
            "quality_metrics": {
                "average_quality_score": 88.7,
                "success_rate": 100.0,
                "total_items_found": 3.0,
                "total_items_scraped": 3.0,
                "execution_time": 4.2,
                "data_completeness": 95.0
            },
            "metadata": {
                "scraping_timestamp": datetime.utcnow().isoformat(),
                "strategy_effectiveness": 0.92,
                "website_complexity": 2.8
            }
        }
    
    @pytest.mark.asyncio
    async def test_orchestrator_planning_coordination_success(
        self, 
        orchestrator, 
        sample_planning_response, 
        sample_scraper_response
    ):
        """Test successful coordination between orchestrator and planning agent."""
        # Mock the coordination methods
        orchestrator._coordinate_with_planning_agent = AsyncMock(return_value=sample_planning_response)
        orchestrator._coordinate_with_scraper_tool = AsyncMock(return_value=sample_scraper_response)
        
        # Test input
        input_data = {
            "scraping_request": "Extract product information from this e-commerce website",
            "target_url": "https://example-store.com/products",
            "max_results": 10,
            "quality_threshold": 80.0,
            "export_format": "json",
            "enable_monitoring": True,
            "concurrent_instances": 1
        }
        
        # Execute orchestration
        result = await orchestrator.run(input_data)
        
        # Verify coordination was called
        orchestrator._coordinate_with_planning_agent.assert_called_once()
        orchestrator._coordinate_with_scraper_tool.assert_called_once()
        
        # Verify result structure
        assert isinstance(result, IntelligentScrapingOrchestratorOutputSchema)
        assert result.scraping_plan == sample_planning_response["scraping_plan"]
        assert len(result.extracted_data) == 3
        assert result.quality_score == 88.7
        assert result.reasoning == sample_planning_response["reasoning"]
        
        # Verify extracted data
        assert result.extracted_data[0]["title"] == "Premium Wireless Headphones"
        assert result.extracted_data[0]["price"] == "$199.99"
        assert result.extracted_data[1]["title"] == "Smart Fitness Tracker"
        assert result.extracted_data[2]["title"] == "Bluetooth Speaker"
        
        # Verify metadata
        assert result.metadata.url == "https://example-store.com/products"
        assert result.metadata.items_extracted == 3
        assert result.metadata.strategy_used == "list"
        assert result.metadata.monitoring_enabled is True
    
    @pytest.mark.asyncio
    async def test_planning_agent_input_transformation(
        self, 
        orchestrator, 
        sample_planning_response
    ):
        """Test that orchestrator properly transforms input for planning agent."""
        orchestrator._coordinate_with_planning_agent = AsyncMock(return_value=sample_planning_response)
        orchestrator._coordinate_with_scraper_tool = AsyncMock(return_value={
            "results": {"items": [], "total_found": 0, "total_scraped": 0, "errors": []},
            "quality_metrics": {"average_quality_score": 0.0}
        })
        
        input_data = {
            "scraping_request": "Find all product listings with prices and descriptions",
            "target_url": "https://shop.example.com/catalog",
            "max_results": 25,
            "quality_threshold": 75.0
        }
        
        await orchestrator.run(input_data)
        
        # Verify planning agent was called with correct parameters
        call_args = orchestrator._coordinate_with_planning_agent.call_args[0][0]
        
        assert call_args["scraping_request"] == "Find all product listings with prices and descriptions"
        assert call_args["target_url"] == "https://shop.example.com/catalog"
        assert call_args["max_results"] == 25
        assert call_args["quality_threshold"] == 75.0
        assert "operation_id" in call_args
        assert "timestamp" in call_args
    
    @pytest.mark.asyncio
    async def test_planning_to_scraper_transformation(
        self, 
        orchestrator, 
        sample_planning_response, 
        sample_scraper_response
    ):
        """Test transformation of planning agent output to scraper tool input."""
        orchestrator._coordinate_with_planning_agent = AsyncMock(return_value=sample_planning_response)
        orchestrator._coordinate_with_scraper_tool = AsyncMock(return_value=sample_scraper_response)
        
        input_data = {
            "scraping_request": "Extract product data",
            "target_url": "https://example.com/products",
            "max_results": 15
        }
        
        await orchestrator.run(input_data)
        
        # Verify scraper tool was called with transformed planning output
        scraper_call_args = orchestrator._coordinate_with_scraper_tool.call_args[0][0]
        
        assert scraper_call_args["target_url"] == "https://example.com/products"
        assert scraper_call_args["strategy"] == sample_planning_response["strategy"]
        assert scraper_call_args["schema_recipe"] == sample_planning_response["schema_recipe"]
        assert scraper_call_args["max_results"] == 15
        assert scraper_call_args["quality_threshold"] == 50.0  # Default from config
        assert "operation_id" in scraper_call_args
    
    @pytest.mark.asyncio
    async def test_planning_agent_error_handling(
        self, 
        orchestrator, 
        sample_scraper_response
    ):
        """Test error handling when planning agent fails."""
        # Mock planning agent to raise an error
        orchestrator._coordinate_with_planning_agent = AsyncMock(
            side_effect=Exception("Planning agent connection timeout")
        )
        orchestrator._coordinate_with_scraper_tool = AsyncMock(return_value=sample_scraper_response)
        
        input_data = {
            "scraping_request": "Extract data from website",
            "target_url": "https://example.com/data"
        }
        
        result = await orchestrator.run(input_data)
        
        # Verify graceful error handling
        assert isinstance(result, IntelligentScrapingOrchestratorOutputSchema)
        assert "Operation failed during coordination" in result.scraping_plan
        assert len(result.extracted_data) == 0
        assert result.quality_score == 0.0
        assert "Orchestration failed" in result.reasoning
        
        # Verify fallback strategy was used
        orchestrator._coordinate_with_scraper_tool.assert_called_once()
        scraper_call_args = orchestrator._coordinate_with_scraper_tool.call_args[0][0]
        assert scraper_call_args["strategy"]["scrape_type"] == "list"  # Fallback strategy
    
    @pytest.mark.asyncio
    async def test_context_provider_integration(
        self, 
        orchestrator, 
        sample_planning_response, 
        sample_scraper_response
    ):
        """Test that context providers are properly updated during coordination."""
        orchestrator._coordinate_with_planning_agent = AsyncMock(return_value=sample_planning_response)
        orchestrator._coordinate_with_scraper_tool = AsyncMock(return_value=sample_scraper_response)
        
        input_data = {
            "scraping_request": "Extract product information",
            "target_url": "https://example.com/products",
            "enable_monitoring": True
        }
        
        # Verify initial context state
        assert len(orchestrator.scraping_results_provider.results) == 0
        assert orchestrator.scraping_results_provider.operation_metadata == {}
        
        await orchestrator.run(input_data)
        
        # Verify context providers were updated
        assert len(orchestrator.scraping_results_provider.results) == 3
        assert orchestrator.scraping_results_provider.operation_metadata["target_url"] == "https://example.com/products"
        assert orchestrator.scraping_results_provider.operation_metadata["scraping_request"] == "Extract product information"
        
        # Verify website analysis context was attempted to be updated
        # (This would be called during planning agent coordination)
        orchestrator._coordinate_with_planning_agent.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_monitoring_data_collection(
        self, 
        orchestrator, 
        sample_planning_response, 
        sample_scraper_response
    ):
        """Test that monitoring data is properly collected during coordination."""
        orchestrator._coordinate_with_planning_agent = AsyncMock(return_value=sample_planning_response)
        orchestrator._coordinate_with_scraper_tool = AsyncMock(return_value=sample_scraper_response)
        
        input_data = {
            "scraping_request": "Monitor this scraping operation",
            "target_url": "https://example.com/monitor",
            "enable_monitoring": True
        }
        
        # Verify initial monitoring state
        assert len(orchestrator.monitoring_data) == 0
        
        result = await orchestrator.run(input_data)
        
        # Verify monitoring data was collected
        assert len(orchestrator.monitoring_data) == 1
        
        operation_id = list(orchestrator.monitoring_data.keys())[0]
        monitoring_entry = orchestrator.monitoring_data[operation_id]
        
        assert monitoring_entry["status"] == "completed"
        assert monitoring_entry["result_summary"]["items_extracted"] == 3
        assert monitoring_entry["result_summary"]["quality_score"] == 88.7
        assert isinstance(monitoring_entry["timestamp"], datetime)
        
        # Verify monitoring report includes this data
        assert result.monitoring_report.total_instances == 1
        assert result.monitoring_report.overall_success_rate == 100.0
        assert result.monitoring_report.detailed_metrics["total_items_extracted"] == 3
    
    @pytest.mark.asyncio
    async def test_schema_alignment_validation(
        self, 
        orchestrator, 
        sample_planning_response, 
        sample_scraper_response
    ):
        """Test that schemas are properly aligned between planning and scraping."""
        orchestrator._coordinate_with_planning_agent = AsyncMock(return_value=sample_planning_response)
        orchestrator._coordinate_with_scraper_tool = AsyncMock(return_value=sample_scraper_response)
        
        input_data = {
            "scraping_request": "Test schema alignment",
            "target_url": "https://example.com/test"
        }
        
        await orchestrator.run(input_data)
        
        # Verify planning agent output schema
        planning_result = await orchestrator._coordinate_with_planning_agent.return_value
        assert "strategy" in planning_result
        assert "schema_recipe" in planning_result
        assert "reasoning" in planning_result
        assert "confidence" in planning_result
        
        # Verify scraper tool input includes planning output
        scraper_call_args = orchestrator._coordinate_with_scraper_tool.call_args[0][0]
        assert scraper_call_args["strategy"] == planning_result["strategy"]
        assert scraper_call_args["schema_recipe"] == planning_result["schema_recipe"]
        
        # Verify scraper output matches expected schema
        scraper_result = await orchestrator._coordinate_with_scraper_tool.return_value
        assert "results" in scraper_result
        assert "quality_metrics" in scraper_result
        assert "items" in scraper_result["results"]
        assert "total_found" in scraper_result["results"]
    
    @pytest.mark.asyncio
    async def test_concurrent_coordination_handling(
        self, 
        orchestrator, 
        sample_planning_response, 
        sample_scraper_response
    ):
        """Test handling of concurrent coordination requests."""
        orchestrator._coordinate_with_planning_agent = AsyncMock(return_value=sample_planning_response)
        orchestrator._coordinate_with_scraper_tool = AsyncMock(return_value=sample_scraper_response)
        
        # Create multiple concurrent requests
        input_data_1 = {
            "scraping_request": "Concurrent request 1",
            "target_url": "https://example.com/concurrent1"
        }
        
        input_data_2 = {
            "scraping_request": "Concurrent request 2", 
            "target_url": "https://example.com/concurrent2"
        }
        
        # Execute concurrently
        results = await asyncio.gather(
            orchestrator.run(input_data_1),
            orchestrator.run(input_data_2)
        )
        
        # Verify both requests completed successfully
        assert len(results) == 2
        assert all(isinstance(result, IntelligentScrapingOrchestratorOutputSchema) for result in results)
        
        # Verify both planning and scraper coordination were called for each request
        assert orchestrator._coordinate_with_planning_agent.call_count == 2
        assert orchestrator._coordinate_with_scraper_tool.call_count == 2
        
        # Verify monitoring data was collected for both operations
        assert len(orchestrator.monitoring_data) == 2
    
    @pytest.mark.asyncio
    async def test_quality_threshold_enforcement(
        self, 
        orchestrator, 
        sample_planning_response
    ):
        """Test that quality thresholds are properly enforced in coordination."""
        # Create scraper response with mixed quality scores
        mixed_quality_response = {
            "results": {
                "items": [
                    {"title": "High Quality Item", "quality_score": 95.0},
                    {"title": "Medium Quality Item", "quality_score": 65.0},
                    {"title": "Low Quality Item", "quality_score": 35.0}
                ],
                "total_found": 3,
                "total_scraped": 3,
                "errors": []
            },
            "quality_metrics": {
                "average_quality_score": 65.0,
                "success_rate": 100.0
            }
        }
        
        orchestrator._coordinate_with_planning_agent = AsyncMock(return_value=sample_planning_response)
        orchestrator._coordinate_with_scraper_tool = AsyncMock(return_value=mixed_quality_response)
        
        input_data = {
            "scraping_request": "Test quality filtering",
            "target_url": "https://example.com/quality",
            "quality_threshold": 70.0  # Should filter out low quality items
        }
        
        result = await orchestrator.run(input_data)
        
        # Verify quality threshold was passed to scraper
        scraper_call_args = orchestrator._coordinate_with_scraper_tool.call_args[0][0]
        assert scraper_call_args["quality_threshold"] == 70.0
        
        # Verify result reflects quality considerations
        assert result.quality_score == 65.0
        assert len(result.extracted_data) == 3  # All items returned (filtering happens in scraper)


if __name__ == "__main__":
    pytest.main([__file__])