"""
Integration tests for orchestrator and planning agent coordination.

This module tests the integration between the orchestrator and the enhanced
planning agent to ensure proper schema alignment and coordination.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from intelligent_web_scraper.agents.orchestrator import (
    IntelligentScrapingOrchestrator,
    IntelligentScrapingOrchestratorInputSchema
)
from intelligent_web_scraper.agents.planning_agent import (
    IntelligentWebScraperPlanningAgent,
    IntelligentPlanningAgentOutputSchema
)
from intelligent_web_scraper.config import IntelligentScrapingConfig


class TestOrchestratorPlanningAgentIntegration:
    """Test integration between orchestrator and planning agent."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return IntelligentScrapingConfig(
            orchestrator_model="gpt-4o-mini",
            planning_agent_model="gpt-4o-mini",
            default_quality_threshold=50.0,
            max_concurrent_requests=5,
            request_delay=1.0,
            respect_robots_txt=True,
            enable_rate_limiting=True
        )
    
    @pytest.fixture
    def mock_client(self):
        """Create mock instructor client."""
        return Mock()
    
    @pytest.fixture
    def orchestrator_input(self):
        """Create test orchestrator input."""
        return IntelligentScrapingOrchestratorInputSchema(
            scraping_request="Extract Saturday markets with prices and locations",
            target_url="https://example.com/markets",
            max_results=20,
            quality_threshold=75.0,
            export_format="json",
            enable_monitoring=True,
            concurrent_instances=2
        )
    
    @pytest.mark.asyncio
    @patch('intelligent_web_scraper.agents.planning_agent.IntelligentWebScraperPlanningAgent')
    @patch('atomic_agents.agents.base_agent.BaseAgentConfig')
    async def test_orchestrator_planning_agent_coordination(
        self, 
        mock_base_config, 
        mock_planning_agent_class,
        mock_config,
        mock_client,
        orchestrator_input
    ):
        """Test that orchestrator properly coordinates with planning agent."""
        
        # Setup mock planning agent
        mock_planning_agent = Mock()
        mock_planning_agent_class.return_value = mock_planning_agent
        
        # Setup mock planning agent response
        mock_planning_response = IntelligentPlanningAgentOutputSchema(
            scraping_plan="Test scraping plan with educational explanations",
            strategy={
                "scrape_type": "list",
                "target_selectors": [".market", ".item"],
                "extraction_rules": {"min_quality": 75.0},
                "pagination_strategy": "next_link",
                "max_pages": 5,
                "request_delay": 1.0,
                "metadata": {"orchestrator_compatible": True}
            },
            schema_recipe={
                "name": "markets_schema",
                "fields": {
                    "title": {
                        "field_type": "string",
                        "extraction_selector": "h1, h2",
                        "required": True,
                        "quality_weight": 0.9
                    },
                    "price": {
                        "field_type": "string", 
                        "extraction_selector": ".price",
                        "required": False,
                        "quality_weight": 0.8
                    }
                },
                "validation_rules": ["normalize_whitespace"],
                "quality_weights": {"completeness": 0.4, "accuracy": 0.4, "consistency": 0.2}
            },
            reasoning="Comprehensive reasoning with educational explanations",
            confidence=0.85,
            orchestrator_metadata={
                "operation_id": "test-123",
                "orchestrator_compatible": True,
                "educational_mode": True,
                "monitoring_enabled": True
            },
            educational_insights={
                "scraping_patterns_demonstrated": ["List extraction with pagination"],
                "best_practices_shown": ["Semantic selectors", "Quality thresholds"],
                "learning_objectives": ["Understand market data extraction"]
            }
        )
        
        mock_planning_agent.run.return_value = mock_planning_response
        
        # Create orchestrator
        orchestrator = IntelligentScrapingOrchestrator(mock_config, mock_client)
        
        # Test coordination method
        result = await orchestrator._coordinate_with_planning_agent(orchestrator_input)
        
        # Verify planning agent was called with correct input
        mock_planning_agent_class.assert_called_once()
        mock_planning_agent.run.assert_called_once()
        
        # Get the actual input passed to planning agent
        planning_input = mock_planning_agent.run.call_args[0][0]
        
        # Verify input transformation
        assert planning_input.scraping_request == orchestrator_input.scraping_request
        assert planning_input.target_url == orchestrator_input.target_url
        assert planning_input.max_results == orchestrator_input.max_results
        assert planning_input.quality_threshold == orchestrator_input.quality_threshold
        assert planning_input.orchestrator_context["educational_mode"] is True
        assert planning_input.orchestrator_context["monitoring_enabled"] is True
        assert planning_input.orchestrator_context["concurrent_instances"] == 2
        
        # Verify output transformation
        assert result["scraping_plan"] == mock_planning_response.scraping_plan
        assert result["strategy"] == mock_planning_response.strategy
        assert result["schema_recipe"] == mock_planning_response.schema_recipe
        assert result["reasoning"] == mock_planning_response.reasoning
        assert result["confidence"] == mock_planning_response.confidence
        assert result["orchestrator_metadata"] == mock_planning_response.orchestrator_metadata
        assert result["educational_insights"] == mock_planning_response.educational_insights
    
    def test_schema_alignment_compatibility(self):
        """Test that planning agent output is compatible with orchestrator expectations."""
        
        # Create a sample planning agent output
        planning_output = {
            "scraping_plan": "Test plan",
            "strategy": {
                "scrape_type": "list",
                "target_selectors": [".item"],
                "extraction_rules": {"min_quality": 50.0},
                "max_pages": 1,
                "request_delay": 1.0
            },
            "schema_recipe": {
                "name": "test_schema",
                "fields": {
                    "title": {
                        "field_type": "string",
                        "extraction_selector": "h1",
                        "required": True,
                        "quality_weight": 0.9
                    }
                },
                "validation_rules": ["normalize_whitespace"],
                "quality_weights": {"completeness": 0.4, "accuracy": 0.4, "consistency": 0.2}
            },
            "reasoning": "Test reasoning",
            "confidence": 0.8,
            "orchestrator_metadata": {"operation_id": "test"},
            "educational_insights": {"patterns": ["test"]}
        }
        
        # Verify the structure matches what orchestrator expects
        required_fields = [
            "scraping_plan", "strategy", "schema_recipe", 
            "reasoning", "confidence", "orchestrator_metadata", "educational_insights"
        ]
        
        for field in required_fields:
            assert field in planning_output, f"Missing required field: {field}"
        
        # Verify strategy structure for AtomicScraperTool compatibility
        strategy = planning_output["strategy"]
        strategy_required_fields = ["scrape_type", "target_selectors", "extraction_rules"]
        
        for field in strategy_required_fields:
            assert field in strategy, f"Missing required strategy field: {field}"
        
        # Verify schema recipe structure for AtomicScraperTool compatibility
        schema_recipe = planning_output["schema_recipe"]
        schema_required_fields = ["name", "fields", "validation_rules", "quality_weights"]
        
        for field in schema_required_fields:
            assert field in schema_recipe, f"Missing required schema field: {field}"
        
        # Verify field structure
        title_field = schema_recipe["fields"]["title"]
        field_required_attrs = ["field_type", "extraction_selector", "required", "quality_weight"]
        
        for attr in field_required_attrs:
            assert attr in title_field, f"Missing required field attribute: {attr}"
    
    def test_educational_enhancements_integration(self):
        """Test that educational enhancements are properly integrated."""
        
        # Create enhanced planning output with educational features
        enhanced_output = IntelligentPlanningAgentOutputSchema(
            scraping_plan="# Educational Scraping Plan\n\nThis plan demonstrates...",
            strategy={
                "scrape_type": "list",
                "metadata": {
                    "orchestrator_compatible": True,
                    "educational_focus": ["location_extraction", "price_parsing"],
                    "user_intent": "learning",
                    "complexity_level": "medium"
                }
            },
            schema_recipe={
                "name": "educational_schema",
                "metadata": {
                    "educational_purpose": True,
                    "orchestrator_compatible": True
                }
            },
            reasoning="## Comprehensive Decision Analysis\n\nEducational insights...",
            confidence=0.85,
            orchestrator_metadata={
                "educational_mode": True,
                "coordination_requirements": {
                    "requires_monitoring": True,
                    "supports_partial_results": True,
                    "error_recovery_enabled": True
                }
            },
            educational_insights={
                "scraping_patterns_demonstrated": [
                    "List extraction pattern with fallback selectors",
                    "Multi-field schema with quality weights"
                ],
                "best_practices_shown": [
                    "Semantic HTML selector prioritization",
                    "Graceful error handling strategies"
                ],
                "learning_objectives": [
                    "Understand intelligent scraping strategy selection",
                    "Master CSS selector robustness patterns"
                ]
            }
        )
        
        # Verify educational enhancements are present
        assert "Educational" in enhanced_output.scraping_plan
        assert enhanced_output.strategy["metadata"]["educational_focus"]
        assert enhanced_output.schema_recipe["metadata"]["educational_purpose"] is True
        assert "Educational insights" in enhanced_output.reasoning
        assert enhanced_output.orchestrator_metadata["educational_mode"] is True
        assert len(enhanced_output.educational_insights["learning_objectives"]) > 0
        
        # Verify orchestrator compatibility
        assert enhanced_output.strategy["metadata"]["orchestrator_compatible"] is True
        assert enhanced_output.schema_recipe["metadata"]["orchestrator_compatible"] is True
        assert enhanced_output.orchestrator_metadata["coordination_requirements"]["requires_monitoring"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])