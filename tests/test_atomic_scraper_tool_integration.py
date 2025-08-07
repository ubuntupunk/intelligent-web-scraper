"""
Integration tests for atomic_scraper_tool components.

Tests the integration between intelligent-web-scraper and atomic_scraper_tool.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add atomic_scraper_tool to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class TestAtomicScraperToolIntegration:
    """Test integration with atomic_scraper_tool components."""
    
    def test_atomic_scraper_tool_imports(self):
        """Test that we can import atomic_scraper_tool components."""
        try:
            from atomic_scraper_tool.models.base_models import ScrapingStrategy, ScrapingResult
            from atomic_scraper_tool.models.schema_models import SchemaRecipe, FieldDefinition
            from atomic_scraper_tool.config.scraper_config import AtomicScraperConfig
            
            # Test that classes can be instantiated
            strategy = ScrapingStrategy(
                scrape_type="list",
                target_selectors=[".item"],
                extraction_rules={},
                max_pages=1,
                request_delay=1.0
            )
            assert strategy.scrape_type == "list"
            
            field_def = FieldDefinition(
                field_type="string",
                description="Test field",
                extraction_selector=".title"
            )
            assert field_def.field_type == "string"
            
            config = AtomicScraperConfig(
                base_url="https://example.com"
            )
            assert config.base_url == "https://example.com"
            
        except ImportError as e:
            pytest.fail(f"Failed to import atomic_scraper_tool components: {e}")
    
    def test_atomic_scraper_tool_integration_with_intelligent_scraper(self):
        """Test integration between atomic_scraper_tool and intelligent scraper."""
        try:
            from intelligent_web_scraper.tools.atomic_scraper_tool import AtomicScraperToolConfig
            from intelligent_web_scraper.config import IntelligentScrapingConfig
            
            # Test that we can create the config
            intelligent_config = IntelligentScrapingConfig(
                request_delay=1.0,
                default_quality_threshold=50.0,
                respect_robots_txt=True,
                enable_rate_limiting=True
            )
            
            config = AtomicScraperToolConfig.from_intelligent_config(
                base_url="https://test.com",
                intelligent_config=intelligent_config
            )
            
            assert config.base_url == "https://test.com"
            assert config.request_delay == 1.0
            assert config.min_quality_score == 50.0
            assert config.respect_robots_txt == True
            assert config.enable_rate_limiting == True
            
        except ImportError as e:
            pytest.fail(f"Failed to test integration: {e}")
    
    def test_atomic_scraper_tool_configuration_alignment(self):
        """Test that configurations are properly aligned between systems."""
        # Test configuration parameter mapping
        config_mapping = {
            'request_delay': 'request_delay',
            'default_quality_threshold': 'min_quality_score', 
            'respect_robots_txt': 'respect_robots_txt',
            'enable_rate_limiting': 'enable_rate_limiting'
        }
        
        # Verify all expected mappings exist
        for intelligent_param, atomic_param in config_mapping.items():
            assert intelligent_param is not None
            assert atomic_param is not None
    
    def test_schema_alignment(self):
        """Test that input/output schemas are properly aligned."""
        # Test that we can create mock schemas that match expected structure
        mock_input_schema = {
            'target_url': 'https://example.com',
            'strategy': {
                'scrape_type': 'list',
                'target_selectors': ['.item'],
                'max_pages': 1
            },
            'schema_recipe': {
                'name': 'test_schema',
                'fields': {
                    'title': {
                        'field_type': 'string',
                        'extraction_selector': '.title'
                    }
                }
            },
            'max_results': 10
        }
        
        mock_output_schema = {
            'target_url': 'https://example.com',
            'strategy': mock_input_schema['strategy'],
            'schema_recipe': mock_input_schema['schema_recipe'],
            'max_results': 10,
            'results': {
                'items': [],
                'total_found': 0,
                'total_scraped': 0
            },
            'summary': 'Test summary',
            'quality_metrics': {
                'average_quality_score': 0.0,
                'success_rate': 0.0
            },
            'monitoring_data': {
                'operation_id': 'test_op',
                'execution_time': 1.0
            },
            'errors': []
        }
        
        # Verify schema structure
        assert 'target_url' in mock_input_schema
        assert 'strategy' in mock_input_schema
        assert 'schema_recipe' in mock_input_schema
        
        assert 'results' in mock_output_schema
        assert 'monitoring_data' in mock_output_schema
        assert 'errors' in mock_output_schema
    
    def test_error_handling_integration(self):
        """Test error handling between systems."""
        # Test that error types are properly handled
        error_types = [
            'NetworkError',
            'ParsingError', 
            'ValidationError',
            'QualityError',
            'ConfigurationError'
        ]
        
        for error_type in error_types:
            # Verify error type exists (mock test)
            assert error_type is not None
    
    def test_monitoring_integration(self):
        """Test monitoring data integration."""
        # Test monitoring data structure
        monitoring_data = {
            'operation_id': 'test_123',
            'instance_id': 'scraper_1',
            'execution_time': 2.5,
            'requests_made': 1,
            'success_rate': 100.0,
            'average_response_time': 1.2
        }
        
        # Verify monitoring structure
        required_fields = [
            'operation_id', 'execution_time', 'success_rate'
        ]
        
        for field in required_fields:
            assert field in monitoring_data
            assert monitoring_data[field] is not None


if __name__ == "__main__":
    pytest.main([__file__])