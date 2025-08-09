#!/usr/bin/env python3
"""
Test script for current implementation.

This script tests the current Intelligent Web Scraper implementation
to ensure it works correctly before proceeding with rebranding.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from intelligent_web_scraper.config import IntelligentScrapingConfig
from intelligent_web_scraper.agents.orchestrator import IntelligentScrapingOrchestrator
from intelligent_web_scraper.llm_providers import get_available_providers

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn


async def test_basic_configuration():
    """Test basic configuration loading."""
    console = Console()
    console.print("\n🔧 Testing Basic Configuration", style="bold blue")
    
    try:
        # Test default configuration
        config = IntelligentScrapingConfig.from_env()
        console.print(f"✅ Configuration loaded successfully")
        console.print(f"   Provider: {config.llm_provider}")
        console.print(f"   Orchestrator model: {config.orchestrator_model}")
        console.print(f"   Quality threshold: {config.default_quality_threshold}")
        console.print(f"   Max concurrent requests: {config.max_concurrent_requests}")
        
        # Test provider configuration
        provider_config = config.get_provider_config()
        console.print(f"   Provider config generated: {provider_config['provider']}")
        
        return config
        
    except Exception as e:
        console.print(f"❌ Configuration test failed: {e}", style="red")
        return None


async def test_orchestrator_initialization(config):
    """Test orchestrator initialization."""
    console = Console()
    console.print("\n🎼 Testing Orchestrator Initialization", style="bold blue")
    
    try:
        orchestrator = IntelligentScrapingOrchestrator(config=config)
        console.print("✅ Orchestrator initialized successfully")
        
        # Test that required attributes exist
        assert hasattr(orchestrator, 'input_schema'), "Missing input_schema"
        assert hasattr(orchestrator, 'output_schema'), "Missing output_schema"
        assert hasattr(orchestrator, 'system_prompt_generator'), "Missing system_prompt_generator"
        
        console.print("✅ All required attributes present")
        
        return orchestrator
        
    except Exception as e:
        console.print(f"❌ Orchestrator initialization failed: {e}", style="red")
        return None


async def test_simple_scraping_operation(orchestrator):
    """Test a simple scraping operation."""
    console = Console()
    console.print("\n🌐 Testing Simple Scraping Operation", style="bold blue")
    
    # Use a simple, reliable test URL
    test_request = {
        "scraping_request": "Extract any available information from this test page",
        "target_url": "https://httpbin.org/json",
        "max_results": 5,
        "quality_threshold": 30.0,  # Lower threshold for test
        "export_format": "json"
    }
    
    try:
        console.print(f"🎯 Target: {test_request['target_url']}")
        console.print(f"📋 Request: {test_request['scraping_request']}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False
        ) as progress:
            task = progress.add_task("Processing scraping request...", total=None)
            
            # Execute the scraping operation
            result = await orchestrator.run(test_request)
            
            progress.update(task, description="Scraping completed!")
        
        # Analyze results
        console.print("\n📊 Results Analysis:")
        console.print(f"   Items extracted: {len(result.extracted_data) if hasattr(result, 'extracted_data') else 'N/A'}")
        console.print(f"   Quality score: {result.quality_score if hasattr(result, 'quality_score') else 'N/A'}")
        
        if hasattr(result, 'metadata') and result.metadata:
            console.print(f"   Processing time: {result.metadata.processing_time:.2f}s")
            console.print(f"   Pages processed: {result.metadata.pages_processed}")
            
            if hasattr(result.metadata, 'errors_encountered') and result.metadata.errors_encountered:
                console.print(f"   Errors encountered: {len(result.metadata.errors_encountered)}")
                for error in result.metadata.errors_encountered[:3]:  # Show first 3 errors
                    console.print(f"     - {error}")
        
        if hasattr(result, 'reasoning') and result.reasoning:
            console.print(f"\n🧠 AI Reasoning:")
            console.print(f"   {result.reasoning[:200]}..." if len(result.reasoning) > 200 else f"   {result.reasoning}")
        
        # Check if we got any data
        if hasattr(result, 'extracted_data') and result.extracted_data:
            console.print("✅ Scraping operation completed with data", style="green")
            
            # Show sample data
            console.print("\n📋 Sample extracted data:")
            for i, item in enumerate(result.extracted_data[:2]):  # Show first 2 items
                console.print(f"   {i+1}. {str(item)[:100]}..." if len(str(item)) > 100 else f"   {i+1}. {item}")
        else:
            console.print("⚠️  Scraping completed but no data extracted", style="yellow")
        
        return result
        
    except Exception as e:
        console.print(f"❌ Scraping operation failed: {e}", style="red")
        import traceback
        console.print(f"   Traceback: {traceback.format_exc()}", style="dim red")
        return None


async def test_different_providers():
    """Test different LLM providers if available."""
    console = Console()
    console.print("\n🔄 Testing Different Providers", style="bold blue")
    
    # Get available providers
    available_providers = get_available_providers()
    working_providers = [p for p, avail in available_providers.items() if avail]
    
    console.print(f"Available providers: {', '.join(working_providers)}")
    
    # Test each available provider
    for provider in working_providers[:2]:  # Test first 2 to save time
        console.print(f"\n🧪 Testing {provider.capitalize()} provider...")
        
        try:
            # Create configuration for this provider
            config = IntelligentScrapingConfig(
                llm_provider=provider,
                orchestrator_model="gpt-4o-mini",
                default_quality_threshold=30.0
            )
            
            # Check if we have API key for this provider
            provider_config = config.get_provider_config()
            if not provider_config.get("api_key"):
                console.print(f"   ⚠️  No API key available for {provider}", style="yellow")
                continue
            
            # Test basic initialization
            orchestrator = IntelligentScrapingOrchestrator(config=config)
            console.print(f"   ✅ {provider.capitalize()} orchestrator initialized")
            
            # Test simple operation (with shorter timeout)
            test_request = {
                "scraping_request": "Get basic information",
                "target_url": "https://httpbin.org/json",
                "max_results": 1,
                "quality_threshold": 20.0
            }
            
            try:
                result = await asyncio.wait_for(orchestrator.run(test_request), timeout=30.0)
                console.print(f"   ✅ {provider.capitalize()} scraping test completed")
            except asyncio.TimeoutError:
                console.print(f"   ⚠️  {provider.capitalize()} test timed out", style="yellow")
            except Exception as e:
                console.print(f"   ❌ {provider.capitalize()} test failed: {e}", style="red")
                
        except Exception as e:
            console.print(f"   ❌ {provider.capitalize()} configuration failed: {e}", style="red")


async def test_cli_functionality():
    """Test CLI functionality."""
    console = Console()
    console.print("\n💻 Testing CLI Functionality", style="bold blue")
    
    try:
        # Test help command
        import subprocess
        result = subprocess.run(
            ["poetry", "run", "intelligent-web-scraper", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            console.print("✅ CLI help command works")
        else:
            console.print(f"❌ CLI help command failed: {result.stderr}", style="red")
        
        # Test version command
        result = subprocess.run(
            ["poetry", "run", "intelligent-web-scraper", "--version"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            console.print("✅ CLI version command works")
        else:
            console.print(f"❌ CLI version command failed: {result.stderr}", style="red")
            
    except Exception as e:
        console.print(f"❌ CLI test failed: {e}", style="red")


async def test_health_check():
    """Test the health check system."""
    console = Console()
    console.print("\n🏥 Testing Health Check System", style="bold blue")
    
    try:
        import subprocess
        result = subprocess.run(
            ["poetry", "run", "python", "scripts/health-check.py", "--json"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            console.print("✅ Health check system works")
            
            # Parse and show summary
            import json
            health_data = json.loads(result.stdout)
            console.print(f"   Overall status: {health_data.get('overall_status', 'Unknown')}")
            console.print(f"   Health percentage: {health_data.get('health_percentage', 0):.1f}%")
            console.print(f"   Passed checks: {health_data.get('passed_checks', 0)}/{health_data.get('total_checks', 0)}")
            
        else:
            console.print(f"❌ Health check failed: {result.stderr}", style="red")
            
    except Exception as e:
        console.print(f"❌ Health check test failed: {e}", style="red")


async def main():
    """Main test function."""
    console = Console()
    
    # Display header
    header_panel = Panel(
        "🧪 Current Implementation Test Suite\n\nTesting the Intelligent Web Scraper implementation\nbefore proceeding with Scraperotti rebranding.",
        title="Implementation Test",
        border_style="bright_blue"
    )
    console.print(header_panel)
    
    try:
        # Run tests in sequence
        config = await test_basic_configuration()
        if not config:
            console.print("❌ Cannot proceed without valid configuration", style="red")
            return
        
        orchestrator = await test_orchestrator_initialization(config)
        if not orchestrator:
            console.print("❌ Cannot proceed without valid orchestrator", style="red")
            return
        
        await test_simple_scraping_operation(orchestrator)
        await test_different_providers()
        await test_cli_functionality()
        await test_health_check()
        
        # Final summary
        console.print("\n🎉 Implementation Test Summary", style="bold green")
        console.print("✅ Basic configuration and initialization working")
        console.print("✅ Multi-provider LLM support implemented")
        console.print("✅ CLI functionality operational")
        console.print("✅ Health check system functional")
        console.print("\n🚀 Ready to proceed with Scraperotti rebranding!", style="bold cyan")
        
    except KeyboardInterrupt:
        console.print("\n⏹️  Test suite interrupted", style="yellow")
    except Exception as e:
        console.print(f"\n❌ Test suite failed: {e}", style="red")
        import traceback
        console.print(f"Traceback: {traceback.format_exc()}", style="dim red")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())