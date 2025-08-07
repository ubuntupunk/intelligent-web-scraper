#!/usr/bin/env python3
"""
Basic Scraping Example

This example demonstrates the fundamental usage of the Intelligent Web Scraper
with simple, straightforward scraping operations. Perfect for getting started
and understanding the core concepts.
"""

import asyncio
import logging
from typing import Dict, Any

from intelligent_web_scraper import (
    IntelligentScrapingOrchestrator,
    IntelligentScrapingConfig
)

# Configure logging for the example
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def basic_product_scraping():
    """Demonstrate basic product information scraping."""
    print("\n🛍️  Basic Product Scraping Example")
    print("=" * 50)
    
    # Create configuration
    config = IntelligentScrapingConfig(
        orchestrator_model="gpt-4o-mini",
        planning_agent_model="gpt-4o-mini",
        default_quality_threshold=60.0,
        max_concurrent_requests=3,
        enable_monitoring=True
    )
    
    # Initialize orchestrator
    orchestrator = IntelligentScrapingOrchestrator(config=config)
    
    # Define scraping request
    request = {
        "scraping_request": "Extract product names, prices, and ratings from this e-commerce page",
        "target_url": "https://books.toscrape.com/",
        "max_results": 10,
        "quality_threshold": 60.0,
        "export_format": "json"
    }
    
    try:
        print(f"🎯 Target: {request['target_url']}")
        print(f"📋 Request: {request['scraping_request']}")
        print("⏳ Processing...")
        
        # Execute scraping
        result = await orchestrator.run(request)
        
        # Display results
        print(f"\n✅ Scraping completed successfully!")
        print(f"📊 Items extracted: {len(result.extracted_data)}")
        print(f"🎯 Quality score: {result.quality_score:.1f}")
        print(f"⏱️  Processing time: {result.metadata.processing_time:.2f}s")
        
        # Show sample results
        if result.extracted_data:
            print(f"\n📖 Sample results:")
            for i, item in enumerate(result.extracted_data[:3]):
                print(f"  {i+1}. {item}")
        
        # Show reasoning
        print(f"\n🧠 AI Reasoning:")
        print(f"   {result.reasoning}")
        
        # Show export information
        if result.export_options:
            print(f"\n💾 Export options:")
            for format_type, path in result.export_options.items():
                print(f"   {format_type}: {path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        return None


async def basic_news_scraping():
    """Demonstrate basic news article scraping."""
    print("\n📰 Basic News Article Scraping Example")
    print("=" * 50)
    
    # Create configuration with different settings
    config = IntelligentScrapingConfig(
        default_quality_threshold=70.0,
        max_concurrent_requests=2,
        request_delay=1.0,
        enable_monitoring=False  # Disable monitoring for simpler output
    )
    
    orchestrator = IntelligentScrapingOrchestrator(config=config)
    
    request = {
        "scraping_request": "Extract article titles, summaries, and publication dates",
        "target_url": "https://httpbin.org/html",  # Using httpbin for reliable demo
        "max_results": 5,
        "quality_threshold": 50.0,
        "export_format": "markdown"
    }
    
    try:
        print(f"🎯 Target: {request['target_url']}")
        print(f"📋 Request: {request['scraping_request']}")
        print("⏳ Processing...")
        
        result = await orchestrator.run(request)
        
        print(f"\n✅ Scraping completed!")
        print(f"📊 Items found: {len(result.extracted_data)}")
        print(f"🎯 Quality score: {result.quality_score:.1f}")
        
        # Show metadata
        print(f"\n📋 Metadata:")
        print(f"   Strategy: {result.metadata.strategy_used}")
        print(f"   Pages processed: {result.metadata.pages_processed}")
        print(f"   Processing time: {result.metadata.processing_time:.2f}s")
        
        if result.metadata.errors_encountered:
            print(f"   Errors: {len(result.metadata.errors_encountered)}")
        
        return result
        
    except Exception as e:
        logger.error(f"News scraping failed: {e}")
        return None


async def basic_directory_scraping():
    """Demonstrate basic directory/listing scraping."""
    print("\n📂 Basic Directory Scraping Example")
    print("=" * 50)
    
    config = IntelligentScrapingConfig(
        default_quality_threshold=50.0,
        max_concurrent_requests=1,  # Conservative for demo
        enable_monitoring=True
    )
    
    orchestrator = IntelligentScrapingOrchestrator(config=config)
    
    request = {
        "scraping_request": "Extract business names, addresses, and contact information from this directory",
        "target_url": "https://httpbin.org/json",  # JSON endpoint for demo
        "max_results": 20,
        "quality_threshold": 40.0,
        "export_format": "csv"
    }
    
    try:
        print(f"🎯 Target: {request['target_url']}")
        print(f"📋 Request: {request['scraping_request']}")
        print("⏳ Processing...")
        
        result = await orchestrator.run(request)
        
        print(f"\n✅ Directory scraping completed!")
        print(f"📊 Entries found: {len(result.extracted_data)}")
        print(f"🎯 Quality score: {result.quality_score:.1f}")
        
        # Show monitoring report if available
        if result.monitoring_report:
            print(f"\n📊 Monitoring Report:")
            print(f"   Active instances: {result.monitoring_report.active_instances}")
            print(f"   Overall throughput: {result.monitoring_report.overall_throughput:.2f} req/sec")
            print(f"   Success rate: {result.monitoring_report.overall_success_rate:.1%}")
        
        return result
        
    except Exception as e:
        logger.error(f"Directory scraping failed: {e}")
        return None


async def demonstrate_error_handling():
    """Demonstrate error handling with invalid URLs."""
    print("\n⚠️  Error Handling Demonstration")
    print("=" * 50)
    
    config = IntelligentScrapingConfig(
        default_quality_threshold=60.0,
        max_concurrent_requests=1
    )
    
    orchestrator = IntelligentScrapingOrchestrator(config=config)
    
    # Try scraping an invalid URL
    request = {
        "scraping_request": "Extract any available information",
        "target_url": "https://this-url-does-not-exist-12345.com",
        "max_results": 5,
        "quality_threshold": 30.0,
        "export_format": "json"
    }
    
    try:
        print(f"🎯 Target: {request['target_url']} (invalid URL)")
        print(f"📋 Request: {request['scraping_request']}")
        print("⏳ Processing (expecting graceful error handling)...")
        
        result = await orchestrator.run(request)
        
        print(f"\n🛡️  Error handled gracefully!")
        print(f"📊 Items extracted: {len(result.extracted_data)}")
        print(f"🎯 Quality score: {result.quality_score:.1f}")
        
        # Show error information
        if result.metadata.errors_encountered:
            print(f"\n❌ Errors encountered:")
            for error in result.metadata.errors_encountered:
                print(f"   - {error}")
        
        print(f"\n🧠 AI Reasoning about errors:")
        print(f"   {result.reasoning}")
        
        return result
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


async def demonstrate_configuration_options():
    """Demonstrate different configuration options."""
    print("\n⚙️  Configuration Options Demonstration")
    print("=" * 50)
    
    # Show different configuration approaches
    print("1. Environment-based configuration:")
    try:
        env_config = IntelligentScrapingConfig.from_env()
        print(f"   ✅ Loaded from environment")
        print(f"   Model: {env_config.orchestrator_model}")
        print(f"   Quality threshold: {env_config.default_quality_threshold}")
    except Exception as e:
        print(f"   ❌ Environment config failed: {e}")
    
    print("\n2. Custom configuration:")
    custom_config = IntelligentScrapingConfig(
        orchestrator_model="gpt-4o-mini",
        planning_agent_model="gpt-4o-mini",
        default_quality_threshold=75.0,
        max_concurrent_requests=2,
        request_delay=0.5,
        enable_monitoring=True,
        results_directory="./custom_results"
    )
    print(f"   ✅ Custom configuration created")
    print(f"   Quality threshold: {custom_config.default_quality_threshold}")
    print(f"   Max concurrent: {custom_config.max_concurrent_requests}")
    print(f"   Results directory: {custom_config.results_directory}")
    
    print("\n3. Performance-optimized configuration:")
    performance_config = IntelligentScrapingConfig(
        max_concurrent_requests=8,
        request_delay=0.1,
        default_quality_threshold=50.0,
        enable_monitoring=True
    )
    print(f"   ✅ Performance configuration created")
    print(f"   Max concurrent: {performance_config.max_concurrent_requests}")
    print(f"   Request delay: {performance_config.request_delay}s")


def main():
    """Main function to run all basic examples."""
    print("🚀 Intelligent Web Scraper - Basic Examples")
    print("=" * 60)
    print("This example demonstrates fundamental scraping operations:")
    print("- Product information extraction")
    print("- News article scraping")
    print("- Directory/listing scraping")
    print("- Error handling")
    print("- Configuration options")
    print()
    
    async def run_examples():
        """Run all examples asynchronously."""
        try:
            # Run basic examples
            await basic_product_scraping()
            await basic_news_scraping()
            await basic_directory_scraping()
            
            # Demonstrate error handling
            await demonstrate_error_handling()
            
            # Show configuration options
            await demonstrate_configuration_options()
            
            print("\n🎉 All basic examples completed successfully!")
            print("\nNext steps:")
            print("- Try the advanced_orchestration_example.py for complex workflows")
            print("- Explore monitoring_dashboard_demo.py for real-time monitoring")
            print("- Check export_example.py for different output formats")
            
        except Exception as e:
            logger.error(f"Example execution failed: {e}", exc_info=True)
    
    # Run the async examples
    asyncio.run(run_examples())


if __name__ == "__main__":
    main()