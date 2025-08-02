#!/usr/bin/env python3
"""
Example demonstrating AtomicScraperTool integration with atomic-agents patterns.

This example shows how to use the integrated AtomicScraperTool with proper
configuration management, error handling, and monitoring.
"""

import asyncio
import logging
from typing import Dict, Any

from intelligent_web_scraper.config import IntelligentScrapingConfig
from intelligent_web_scraper.tools import (
    AtomicScraperTool,
    AtomicScraperToolConfig,
    AtomicScraperInputSchema,
    AtomicScraperToolFactory,
    ConfigurationManager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_basic_tool_usage():
    """Demonstrate basic AtomicScraperTool usage."""
    print("\n=== Basic AtomicScraperTool Usage ===")
    
    # Create intelligent scraping configuration
    intelligent_config = IntelligentScrapingConfig(
        request_delay=0.5,
        default_quality_threshold=60.0,
        enable_monitoring=True,
        max_concurrent_requests=3
    )
    
    # Create tool configuration
    tool_config = AtomicScraperToolConfig.from_intelligent_config(
        base_url="https://httpbin.org",
        intelligent_config=intelligent_config,
        instance_id="demo_scraper",
        timeout=30
    )
    
    # Create tool instance
    scraper_tool = AtomicScraperTool(
        config=tool_config,
        intelligent_config=intelligent_config
    )
    
    # Create scraping input
    input_data = AtomicScraperInputSchema(
        target_url="https://httpbin.org/html",
        strategy={
            "scrape_type": "detail",
            "target_selectors": {
                "container": "body",
                "item": "body"
            },
            "max_pages": 1
        },
        schema_recipe={
            "fields": {
                "title": {
                    "field_type": "string",
                    "extraction_selector": "h1",
                    "quality_weight": 2.0
                },
                "content": {
                    "field_type": "string", 
                    "extraction_selector": "p",
                    "quality_weight": 1.0
                }
            }
        },
        max_results=5
    )
    
    try:
        # Execute scraping
        result = scraper_tool.run(input_data)
        
        # Display results
        print(f"Summary: {result.summary}")
        print(f"Items scraped: {result.results['total_scraped']}")
        print(f"Quality metrics: {result.quality_metrics}")
        
        if result.monitoring_data:
            print(f"Monitoring data: {result.monitoring_data}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        
        # Get tool information
        tool_info = scraper_tool.get_tool_info()
        print(f"Tool info: {tool_info['name']} v{tool_info['version']}")
        
    except Exception as e:
        logger.error(f"Scraping failed: {e}")


def demonstrate_factory_usage():
    """Demonstrate AtomicScraperToolFactory usage."""
    print("\n=== AtomicScraperToolFactory Usage ===")
    
    # Create intelligent configuration
    intelligent_config = IntelligentScrapingConfig(
        request_delay=0.3,
        default_quality_threshold=50.0,
        enable_monitoring=True
    )
    
    # Create tool factory
    factory = AtomicScraperToolFactory(intelligent_config=intelligent_config)
    
    # Create multiple tool instances
    tool1 = factory.create_tool(
        base_url="https://httpbin.org",
        instance_id="scraper_1",
        config_overrides={"timeout": 20}
    )
    
    tool2 = factory.create_tool(
        base_url="https://httpbin.org",
        instance_id="scraper_2",
        config_overrides={"min_quality_score": 70.0}
    )
    
    # List instances
    instances = factory.list_tool_instances()
    print(f"Created {len(instances)} tool instances:")
    for instance_id, info in instances.items():
        print(f"  - {instance_id}: {info['base_url']}")
    
    # Get factory statistics
    stats = factory.get_factory_stats()
    print(f"Factory stats: {stats['total_instances']} instances, {stats['cached_configs']} cached configs")
    
    # Demonstrate tool retrieval
    retrieved_tool = factory.get_tool("scraper_1")
    if retrieved_tool:
        print(f"Retrieved tool: {retrieved_tool.config.instance_id}")
    
    # Clean up
    factory.clear_all_instances()
    print("Cleared all instances")


def demonstrate_configuration_management():
    """Demonstrate configuration management patterns."""
    print("\n=== Configuration Management ===")
    
    # Create configuration manager
    config_manager = ConfigurationManager()
    
    # Validate environment
    validation_report = config_manager.validate_environment()
    print(f"Environment validation: {'Valid' if validation_report['valid'] else 'Invalid'}")
    
    if validation_report['errors']:
        print("Validation errors:")
        for error in validation_report['errors']:
            print(f"  - {error}")
    
    if validation_report['warnings']:
        print("Validation warnings:")
        for warning in validation_report['warnings']:
            print(f"  - {warning}")
    
    # Demonstrate environment variable handling
    try:
        quality_threshold = config_manager.get_env_value(
            "QUALITY_THRESHOLD",
            default=50.0,
            value_type=float
        )
        print(f"Quality threshold: {quality_threshold}")
        
        max_requests = config_manager.get_env_value(
            "MAX_CONCURRENT_REQUESTS",
            default=5,
            value_type=int
        )
        print(f"Max concurrent requests: {max_requests}")
        
    except Exception as e:
        logger.error(f"Configuration error: {e}")


def demonstrate_error_handling():
    """Demonstrate error handling and monitoring."""
    print("\n=== Error Handling and Monitoring ===")
    
    # Create tool with monitoring enabled
    intelligent_config = IntelligentScrapingConfig(enable_monitoring=True)
    tool_config = AtomicScraperToolConfig.from_intelligent_config(
        base_url="https://invalid-url-that-does-not-exist.com",
        intelligent_config=intelligent_config,
        instance_id="error_demo",
        max_retries=2,
        retry_delay=0.1
    )
    
    scraper_tool = AtomicScraperTool(
        config=tool_config,
        intelligent_config=intelligent_config
    )
    
    # Create input that will likely fail
    input_data = AtomicScraperInputSchema(
        target_url="https://invalid-url-that-does-not-exist.com/test",
        strategy={
            "scrape_type": "list",
            "target_selectors": {"item": "div"}
        },
        schema_recipe={
            "fields": {
                "title": {"field_type": "string", "extraction_selector": "h1"}
            }
        },
        max_results=10
    )
    
    # Execute and handle errors gracefully
    result = scraper_tool.run(input_data)
    
    print(f"Error handling result:")
    print(f"  - Success: {result.results['total_scraped'] > 0}")
    print(f"  - Errors: {len(result.errors)}")
    print(f"  - Summary: {result.summary}")
    
    if result.monitoring_data:
        print(f"  - Monitoring: {result.monitoring_data}")
    
    # Get monitoring data
    monitoring_data = scraper_tool.get_monitoring_data()
    print(f"Tool monitoring data:")
    print(f"  - Requests made: {monitoring_data['requests_made']}")
    print(f"  - Success rate: {monitoring_data['success_rate']:.1f}%")
    print(f"  - Error count: {monitoring_data['error_count']}")


def demonstrate_integration_patterns():
    """Demonstrate atomic-agents integration patterns."""
    print("\n=== Atomic-Agents Integration Patterns ===")
    
    # Show proper inheritance and schema usage
    from atomic_agents.lib.base.base_tool import BaseTool
    from atomic_agents.lib.base.base_io_schema import BaseIOSchema
    
    # Verify inheritance
    tool = AtomicScraperTool()
    print(f"Tool inherits from BaseTool: {isinstance(tool, BaseTool)}")
    
    # Verify schema compliance
    input_schema = AtomicScraperInputSchema(
        target_url="https://example.com",
        strategy={"scrape_type": "list", "target_selectors": {}},
        schema_recipe={"fields": {}},
        max_results=10
    )
    print(f"Input schema is BaseIOSchema: {isinstance(input_schema, BaseIOSchema)}")
    
    # Show configuration integration
    intelligent_config = IntelligentScrapingConfig.from_env()
    tool_config = AtomicScraperToolConfig.from_intelligent_config(
        base_url="https://example.com",
        intelligent_config=intelligent_config
    )
    
    print(f"Configuration integration:")
    print(f"  - Request delay: {tool_config.request_delay}")
    print(f"  - Quality threshold: {tool_config.min_quality_score}")
    print(f"  - Monitoring enabled: {tool_config.enable_monitoring}")


def main():
    """Run all demonstrations."""
    print("AtomicScraperTool Integration Examples")
    print("=" * 50)
    
    try:
        demonstrate_basic_tool_usage()
        demonstrate_factory_usage()
        demonstrate_configuration_management()
        demonstrate_error_handling()
        demonstrate_integration_patterns()
        
        print("\n=== All demonstrations completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()