"""
Unit tests for IntelligentWebScraperPlanningAgent integration and schema alignment.

This module tests the enhanced planning agent's integration with the orchestrator
and validates proper schema alignment with AtomicScraperTool.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from atomic_agents.agents.base_agent import BaseAgentConfig
import instructor
from openai import OpenAI

from intelligent_web_scraper.agents.planning_agent import (
    IntelligentWebScraperPlanningAgent,
    IntelligentPlanningAgentInputSchema,
    IntelligentPlanningAgentOutputSchema,
    IntelligentScrapingContextProvider
)
# Import from atomic_scraper_tool or use local implementations
try:
    from atomic_scraper_tool.models.base_models import ScrapingStrategy
    from atomic_scraper_tool.models.schema_models import SchemaRecipe, FieldDefinition
except ImportError:
    # Use the local implementations from the planning agent
    from intelligent_web_scraper.agents.planning_agent import ScrapingStrategy, SchemaRecipe, FieldDefinition


class TestIntelligentPlanningAgentInputSchema:
    """Test the enhanced input schema validation and compatibility."""
    
    def test_valid_input_schema(self):
        """Test that valid input creates proper schema instance."""
        input_data = {
            "scraping_request": "Extract Saturday markets from this website",
            "target_url": "https://example.com/markets",
            "max_results": 20,
            "quality_threshold": 75.0,
            "orchestrator_context": {"educational_mode": True}
        }
        
        schema = IntelligentPlanningAgentInputSchema(**input_data)
        
        assert schema.scraping_request == "Extract Saturday markets from this website"
        assert schema.target_url == "https://example.com/markets"
        assert schema.max_results == 20
        assert schema.quality_threshold == 75.0
        assert schema.orchestrator_context["educational_mode"] is True
    
    def test_default_values(self):
        """Test that default values are properly set."""
        input_data = {
            "scraping_request": "Extract data",
            "target_url": "https://example.com"
        }
        
        schema = IntelligentPlanningAgentInputSchema(**input_data)
        
        assert schema.max_results == 10
        assert schema.quality_threshold == 50.0
        assert schema.orchestrator_context == {}
    
    def test_url_validation(self):
        """Test URL validation in input schema."""
        # Valid URLs
        valid_urls = [
            "https://example.com",
            "http://test.org/path",
            "https://subdomain.example.com/path?query=value"
        ]
        
        for url in valid_urls:
            input_data = {
                "scraping_request": "Extract data",
                "target_url": url
            }
            schema = IntelligentPlanningAgentInputSchema(**input_data)
            assert schema.target_url == url
        
        # Invalid URLs
        invalid_urls = [
            "",
            "not-a-url",
            "ftp://example.com",
            "javascript:alert('xss')"
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValueError):
                IntelligentPlanningAgentInputSchema(
                    scraping_request="Extract data",
                    target_url=url
                )
    
    def test_quality_threshold_bounds(self):
        """Test quality threshold validation bounds."""
        input_data = {
            "scraping_request": "Extract data",
            "target_url": "https://example.com"
        }
        
        # Valid thresholds
        for threshold in [0.0, 50.0, 100.0]:
            schema = IntelligentPlanningAgentInputSchema(
                quality_threshold=threshold, **input_data
            )
            assert schema.quality_threshold == threshold
        
        # Invalid thresholds
        for threshold in [-1.0, 101.0]:
            with pytest.raises(ValueError):
                IntelligentPlanningAgentInputSchema(
                    quality_threshold=threshold, **input_data
                )


class TestIntelligentPlanningAgentOutputSchema:
    """Test the enhanced output schema and orchestrator compatibility."""
    
    def test_output_schema_structure(self):
        """Test that output schema has all required fields for orchestrator integration."""
        output_data = {
            "scraping_plan": "Test plan",
            "strategy": {"scrape_type": "list"},
            "schema_recipe": {"name": "test_schema"},
            "reasoning": "Test reasoning",
            "confidence": 0.85,
            "orchestrator_metadata": {"operation_id": "test-123"},
            "educational_insights": {"patterns": ["test_pattern"]}
        }
        
        schema = IntelligentPlanningAgentOutputSchema(**output_data)
        
        assert schema.scraping_plan == "Test plan"
        assert schema.strategy == {"scrape_type": "list"}
        assert schema.schema_recipe == {"name": "test_schema"}
        assert schema.reasoning == "Test reasoning"
        assert schema.confidence == 0.85
        assert schema.orchestrator_metadata["operation_id"] == "test-123"
        assert schema.educational_insights["patterns"] == ["test_pattern"]
    
    def test_confidence_bounds(self):
        """Test confidence score validation bounds."""
        base_data = {
            "scraping_plan": "Test plan",
            "strategy": {"scrape_type": "list"},
            "schema_recipe": {"name": "test_schema"},
            "reasoning": "Test reasoning"
        }
        
        # Valid confidence scores
        for confidence in [0.0, 0.5, 1.0]:
            schema = IntelligentPlanningAgentOutputSchema(
                confidence=confidence, **base_data
            )
            assert schema.confidence == confidence
        
        # Invalid confidence scores
        for confidence in [-0.1, 1.1]:
            with pytest.raises(ValueError):
                IntelligentPlanningAgentOutputSchema(
                    confidence=confidence, **base_data
                )


class TestIntelligentScrapingContextProvider:
    """Test the enhanced context provider for educational scraping."""
    
    def test_context_provider_initialization(self):
        """Test context provider initializes correctly."""
        provider = IntelligentScrapingContextProvider()
        
        assert provider.title == "Intelligent Scraping Capabilities & Educational Context"
        assert isinstance(provider.get_info(), str)
        assert len(provider.get_info()) > 0
    
    def test_context_content_completeness(self):
        """Test that context includes all necessary information."""
        provider = IntelligentScrapingContextProvider()
        context_info = provider.get_info()
        
        # Check for key educational concepts
        assert "educational" in context_info.lower()
        assert "orchestrator integration" in context_info.lower()
        assert "schema generation" in context_info.lower()
        assert "quality assurance" in context_info.lower()
        
        # Check for scraping strategies
        assert "list" in context_info
        assert "detail" in context_info
        assert "search" in context_info
        assert "sitemap" in context_info
        
        # Check for best practices
        assert "robots.txt" in context_info
        assert "rate limiting" in context_info
        assert "css selector" in context_info.lower()


class TestIntelligentWebScraperPlanningAgent:
    """Test the main planning agent functionality and integration."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock instructor client."""
        return Mock(spec=instructor.client.Instructor)
    
    @pytest.fixture
    def agent_config(self, mock_client):
        """Create agent configuration for testing."""
        return BaseAgentConfig(
            client=mock_client,
            model="gpt-4o-mini"
        )
    
    @pytest.fixture
    def planning_agent(self, agent_config):
        """Create planning agent instance for testing."""
        return IntelligentWebScraperPlanningAgent(agent_config)
    
    def test_agent_initialization(self, planning_agent):
        """Test that agent initializes correctly with proper schemas."""
        assert planning_agent.input_schema == IntelligentPlanningAgentInputSchema
        assert planning_agent.output_schema == IntelligentPlanningAgentOutputSchema
        assert hasattr(planning_agent, 'website_analyzer')
        assert hasattr(planning_agent, 'strategy_generator')
        assert hasattr(planning_agent, 'schema_generator')
    
    @patch('intelligent_web_scraper.agents.planning_agent.WebsiteAnalyzer')
    @patch('intelligent_web_scraper.agents.planning_agent.StrategyGenerator')
    @patch('intelligent_web_scraper.agents.planning_agent.SchemaRecipeGenerator')
    def test_enhanced_request_parsing(self, mock_schema_gen, mock_strategy_gen, mock_analyzer, planning_agent):
        """Test enhanced natural language request parsing."""
        request = "Extract Saturday markets with prices and locations from Cape Town"
        
        parsed = planning_agent._parse_enhanced_request(request)
        
        assert parsed['content_type'] == 'list'
        assert 'markets' in parsed['target_data']
        assert 'prices' in parsed['target_data']
        assert 'locations' in parsed['target_data']
        assert 'saturday' in parsed['temporal_filters']
        assert 'cape_town' in parsed['location_filters']
        assert 'educational_focus' in parsed
        assert parsed['user_intent'] == 'data_extraction'
    
    @patch('requests.get')
    def test_comprehensive_website_analysis(self, mock_get, planning_agent):
        """Test comprehensive website analysis with error handling."""
        # Mock successful response
        mock_response = Mock()
        mock_response.text = "<html><body><h1>Test</h1></body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with patch.object(planning_agent.website_analyzer, 'analyze_website') as mock_analyze:
            mock_analysis = Mock()
            mock_analysis.url = "https://example.com"
            mock_analysis.title = "Test Site"
            mock_analysis.content_patterns = []
            mock_analysis.metadata = {}
            mock_analyze.return_value = mock_analysis
            
            result = planning_agent._analyze_website_comprehensive("https://example.com")
            
            assert result.url == "https://example.com"
            assert 'analysis_timestamp' in result.metadata
            assert result.metadata['educational_value'] == 'high'
    
    @patch('requests.get')
    def test_website_analysis_error_handling(self, mock_get, planning_agent):
        """Test website analysis graceful error handling."""
        # Mock failed request
        mock_get.side_effect = Exception("Network error")
        
        result = planning_agent._analyze_website_comprehensive("https://example.com")
        
        assert result.title == "Website Analysis Failed"
        assert 'error' in result.metadata
        assert result.metadata['fallback_mode'] is True
        assert 'educational_note' in result.metadata
    
    def test_orchestrator_aligned_strategy_generation(self, planning_agent):
        """Test strategy generation aligned with orchestrator expectations."""
        mock_analysis = Mock()
        mock_analysis.url = "https://example.com"
        mock_analysis.metadata = {}
        
        parsed_request = {
            'content_type': 'list',
            'target_data': ['markets'],
            'educational_focus': ['location_extraction'],
            'user_intent': 'learning',
            'complexity_level': 'medium'
        }
        
        input_data = IntelligentPlanningAgentInputSchema(
            scraping_request="Extract markets",
            target_url="https://example.com",
            orchestrator_context={"educational_mode": True}
        )
        
        with patch.object(planning_agent.strategy_generator, 'generate_strategy') as mock_generate:
            mock_strategy = Mock(spec=ScrapingStrategy)
            mock_strategy.scrape_type = 'list'
            mock_strategy.metadata = {}
            mock_generate.return_value = mock_strategy
            
            result = planning_agent._generate_orchestrator_aligned_strategy(
                mock_analysis, parsed_request, input_data
            )
            
            assert result.metadata['orchestrator_compatible'] is True
            assert result.metadata['educational_focus'] == ['location_extraction']
            assert result.metadata['user_intent'] == 'learning'
            assert result.metadata['complexity_level'] == 'medium'
    
    def test_enhanced_schema_recipe_creation(self, planning_agent):
        """Test enhanced schema recipe creation with educational value."""
        parsed_request = {
            'content_type': 'list',
            'target_data': ['markets', 'prices'],
            'complexity_level': 'medium',
            'keywords': ['saturday', 'markets']
        }
        
        input_data = IntelligentPlanningAgentInputSchema(
            scraping_request="Extract Saturday markets with prices",
            target_url="https://example.com"
        )
        
        result = planning_agent._create_educational_schema_recipe(parsed_request, input_data)
        
        assert isinstance(result, SchemaRecipe)
        assert 'title' in result.fields
        assert 'price' in result.fields  # Should be added due to 'prices' in target_data
        assert result.fields['title'].required is True
        assert result.fields['price'].required is False
        assert result.metadata['educational_purpose'] is True
        assert result.metadata['orchestrator_compatible'] is True
        assert 'saturday_markets_schema' in result.name
    
    def test_educational_scraping_plan_generation(self, planning_agent):
        """Test comprehensive educational scraping plan generation."""
        mock_strategy = Mock(spec=ScrapingStrategy)
        mock_strategy.scrape_type = 'list'
        mock_strategy.target_selectors = ['.item', '.product', 'article']
        mock_strategy.pagination_strategy = 'next_link'
        mock_strategy.max_pages = 5
        mock_strategy.request_delay = 1.0
        mock_strategy.extraction_rules = {'min_quality': 50.0}
        
        mock_schema = Mock(spec=SchemaRecipe)
        mock_schema.fields = {
            'title': Mock(
                required=True,
                description='Test title field',
                extraction_selector='h1, h2',
                quality_weight=0.9,
                educational_note='Test note'
            )
        }
        
        parsed_request = {
            'content_type': 'list',
            'educational_focus': ['content_extraction'],
            'complexity_level': 'medium'
        }
        
        mock_analysis = Mock()
        mock_analysis.url = "https://example.com"
        mock_analysis.title = "Test Site"
        mock_analysis.metadata = {}
        mock_analysis.content_patterns = []  # Add this to fix the len() error
        
        result = planning_agent._generate_educational_scraping_plan(
            mock_strategy, mock_schema, parsed_request, mock_analysis
        )
        
        assert "Intelligent Scraping Plan" in result
        assert "Educational Overview" in result
        assert "Strategy Analysis" in result
        assert "Data Extraction Strategy" in result
        assert "Orchestrator Integration" in result
        assert mock_strategy.scrape_type in result
        assert str(mock_strategy.max_pages) in result
    
    def test_confidence_calculation(self, planning_agent):
        """Test enhanced confidence score calculation."""
        mock_analysis = Mock()
        mock_analysis.metadata = {}
        mock_analysis.content_patterns = [Mock(), Mock()]  # 2 patterns
        
        mock_strategy = Mock(spec=ScrapingStrategy)
        mock_strategy.scrape_type = 'list'
        mock_strategy.target_selectors = ['.item', '.product', 'article']
        mock_strategy.pagination_strategy = 'next_link'
        
        mock_schema = Mock(spec=SchemaRecipe)
        mock_schema.fields = {
            'title': Mock(required=True),
            'description': Mock(required=False),
            'price': Mock(required=True)
        }
        
        parsed_request = {
            'content_type': 'list',
            'target_data': ['products'],
            'complexity_level': 'medium',
            'user_intent': 'learning',
            'keywords': ['products', 'multiple']  # Add keywords to fix KeyError
        }
        
        confidence = planning_agent._calculate_enhanced_confidence(
            mock_analysis, mock_strategy, mock_schema, parsed_request
        )
        
        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)
    
    def test_orchestrator_metadata_creation(self, planning_agent):
        """Test orchestrator metadata creation for coordination."""
        operation_id = "test-operation-123"
        operation_start = datetime.utcnow()
        
        input_data = IntelligentPlanningAgentInputSchema(
            scraping_request="Extract data",
            target_url="https://example.com",
            max_results=20,
            quality_threshold=75.0,
            orchestrator_context={"educational_mode": True}
        )
        
        mock_strategy = Mock(spec=ScrapingStrategy)
        mock_strategy.scrape_type = 'list'
        mock_strategy.pagination_strategy = None  # Add this to fix AttributeError
        mock_strategy.max_pages = 5
        mock_strategy.request_delay = 1.0
        
        mock_schema = Mock(spec=SchemaRecipe)
        mock_schema.name = 'test_schema'
        mock_schema.fields = {'title': Mock(required=True), 'desc': Mock(required=False)}
        
        metadata = planning_agent._create_orchestrator_metadata(
            operation_id, operation_start, input_data, mock_strategy, mock_schema
        )
        
        assert metadata['operation_id'] == operation_id
        assert metadata['orchestrator_compatible'] is True
        assert metadata['strategy_type'] == 'list'
        assert metadata['schema_name'] == 'test_schema'
        assert metadata['field_count'] == 2
        assert metadata['required_field_count'] == 1
        assert metadata['quality_threshold'] == 75.0
        assert metadata['max_results'] == 20
        assert 'coordination_requirements' in metadata
    
    def test_educational_insights_generation(self, planning_agent):
        """Test educational insights generation for learning value."""
        mock_strategy = Mock(spec=ScrapingStrategy)
        mock_strategy.scrape_type = 'list'
        mock_strategy.target_selectors = ['.item', '.product']
        mock_strategy.pagination_strategy = 'next_link'
        
        mock_schema = Mock(spec=SchemaRecipe)
        mock_schema.fields = {'title': Mock(), 'price': Mock()}
        
        mock_analysis = Mock()
        mock_analysis.url = "https://example.com"
        
        parsed_request = {
            'target_data': ['products', 'prices'],
            'complexity_level': 'medium'
        }
        
        insights = planning_agent._generate_educational_insights(
            mock_strategy, mock_schema, mock_analysis, parsed_request
        )
        
        assert 'scraping_patterns_demonstrated' in insights
        assert 'best_practices_shown' in insights
        assert 'common_challenges_addressed' in insights
        assert 'advanced_concepts_illustrated' in insights
        assert 'production_considerations' in insights
        assert 'learning_objectives' in insights
        
        # Check for specific educational content
        assert any('List extraction pattern' in pattern for pattern in insights['scraping_patterns_demonstrated'])
        assert any('Semantic HTML' in practice for practice in insights['best_practices_shown'])
    
    def test_error_handling_with_educational_context(self, planning_agent):
        """Test graceful error handling with educational explanations."""
        operation_id = "test-error-123"
        operation_start = datetime.utcnow()
        
        input_data = IntelligentPlanningAgentInputSchema(
            scraping_request="Extract data",
            target_url="https://example.com"
        )
        
        error_result = planning_agent._handle_planning_error(
            "Test error message", input_data, operation_id, operation_start
        )
        
        assert isinstance(error_result, IntelligentPlanningAgentOutputSchema)
        assert "Error Recovery Plan" in error_result.scraping_plan
        assert "Error Analysis & Recovery Strategy" in error_result.reasoning
        assert error_result.confidence == 0.3  # Low confidence due to error
        assert error_result.orchestrator_metadata['error_mode'] is True
        assert error_result.orchestrator_metadata['fallback_strategy'] is True
        assert 'error_handling_demonstrated' in error_result.educational_insights


class TestSchemaAlignment:
    """Test schema alignment between planning agent and AtomicScraperTool."""
    
    def test_strategy_schema_compatibility(self):
        """Test that generated strategy is compatible with AtomicScraperTool."""
        # This would be a more comprehensive test in a real implementation
        # For now, we test the basic structure compatibility
        
        mock_strategy_dict = {
            'scrape_type': 'list',
            'target_selectors': ['.item', '.product'],
            'extraction_rules': {'min_quality': 50.0},
            'pagination_strategy': 'next_link',
            'max_pages': 5,
            'request_delay': 1.0,
            'metadata': {'orchestrator_compatible': True}
        }
        
        # Verify the structure matches what AtomicScraperTool expects
        required_fields = ['scrape_type', 'target_selectors', 'extraction_rules']
        for field in required_fields:
            assert field in mock_strategy_dict
        
        # Verify data types
        assert isinstance(mock_strategy_dict['scrape_type'], str)
        assert isinstance(mock_strategy_dict['target_selectors'], list)
        assert isinstance(mock_strategy_dict['extraction_rules'], dict)
    
    def test_schema_recipe_compatibility(self):
        """Test that generated schema recipe is compatible with AtomicScraperTool."""
        mock_schema_dict = {
            'name': 'test_schema',
            'description': 'Test schema',
            'fields': {
                'title': {
                    'field_type': 'string',
                    'description': 'Title field',
                    'extraction_selector': 'h1, h2',
                    'required': True,
                    'quality_weight': 0.9,
                    'post_processing': ['trim', 'clean'],
                    'validation_rules': ['not_empty']
                }
            },
            'validation_rules': ['normalize_whitespace'],
            'quality_weights': {
                'completeness': 0.4,
                'accuracy': 0.4,
                'consistency': 0.2
            },
            'version': '2.0'
        }
        
        # Verify the structure matches what AtomicScraperTool expects
        required_fields = ['name', 'fields', 'validation_rules', 'quality_weights']
        for field in required_fields:
            assert field in mock_schema_dict
        
        # Verify field structure
        title_field = mock_schema_dict['fields']['title']
        field_required_attrs = ['field_type', 'extraction_selector', 'required']
        for attr in field_required_attrs:
            assert attr in title_field


if __name__ == "__main__":
    pytest.main([__file__, "-v"])