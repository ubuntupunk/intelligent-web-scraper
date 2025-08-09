#!/usr/bin/env python3
"""
Test script for LLM provider functionality.

This script tests the configuration and connectivity of different LLM providers
to ensure they work correctly with the Intelligent Web Scraper.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from intelligent_web_scraper.config import IntelligentScrapingConfig
from intelligent_web_scraper.llm_providers import (
    get_available_providers,
    test_provider_connection,
    validate_llm_config,
    LLMProviderError
)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


def test_configuration_loading():
    """Test configuration loading with different providers."""
    console = Console()
    console.print("\n🔧 Testing Configuration Loading", style="bold blue")
    
    # Test default configuration
    try:
        config = IntelligentScrapingConfig.from_env()
        console.print(f"✅ Default config loaded: {config.llm_provider} provider")
        
        provider_config = config.get_provider_config()
        console.print(f"   Provider config: {provider_config['provider']}")
        console.print(f"   Orchestrator model: {provider_config['orchestrator_model']}")
        console.print(f"   Planning model: {provider_config['planning_agent_model']}")
        
    except Exception as e:
        console.print(f"❌ Failed to load default config: {e}", style="red")
    
    # Test different provider configurations
    providers_to_test = ["openai", "gemini", "deepseek", "openrouter", "anthropic"]
    
    for provider in providers_to_test:
        try:
            config = IntelligentScrapingConfig(
                llm_provider=provider,
                orchestrator_model="gpt-4o-mini",
                planning_agent_model="gpt-4o-mini"
            )
            
            provider_config = config.get_provider_config()
            console.print(f"✅ {provider.capitalize()} config created")
            console.print(f"   Mapped orchestrator model: {provider_config['orchestrator_model']}")
            console.print(f"   Mapped planning model: {provider_config['planning_agent_model']}")
            
        except Exception as e:
            console.print(f"❌ Failed to create {provider} config: {e}", style="red")


def test_provider_availability():
    """Test which providers are available."""
    console = Console()
    console.print("\n📦 Testing Provider Availability", style="bold blue")
    
    try:
        availability = get_available_providers()
        
        table = Table(title="LLM Provider Availability")
        table.add_column("Provider", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Notes", style="dim")
        
        for provider, available in availability.items():
            if available:
                status = "✅ Available"
                notes = "Dependencies installed"
            else:
                status = "❌ Not Available"
                notes = "Missing dependencies"
            
            table.add_row(provider.capitalize(), status, notes)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"❌ Failed to check provider availability: {e}", style="red")


def test_provider_connections():
    """Test connections to available providers."""
    console = Console()
    console.print("\n🔗 Testing Provider Connections", style="bold blue")
    
    # Get available providers
    try:
        availability = get_available_providers()
        available_providers = [p for p, avail in availability.items() if avail]
        
        if not available_providers:
            console.print("❌ No providers available for testing", style="red")
            return
        
        console.print(f"Testing connections for: {', '.join(available_providers)}")
        
        for provider in available_providers:
            console.print(f"\n🧪 Testing {provider.capitalize()}...")
            
            # Get API key from environment
            api_key_env_vars = {
                "openai": "OPENAI_API_KEY",
                "gemini": "GEMINI_API_KEY", 
                "deepseek": "DEEPSEEK_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY"
            }
            
            api_key = os.getenv(api_key_env_vars.get(provider))
            
            if not api_key:
                console.print(f"   ⚠️  No API key found for {provider} (set {api_key_env_vars.get(provider)})", style="yellow")
                continue
            
            # Create test configuration
            config = IntelligentScrapingConfig(llm_provider=provider)
            provider_config = config.get_provider_config()
            
            # Test connection
            try:
                success = test_provider_connection(provider, provider_config)
                if success:
                    console.print(f"   ✅ Connection successful", style="green")
                else:
                    console.print(f"   ❌ Connection failed", style="red")
                    
            except Exception as e:
                console.print(f"   ❌ Connection error: {e}", style="red")
    
    except Exception as e:
        console.print(f"❌ Failed to test provider connections: {e}", style="red")


def test_model_mapping():
    """Test model name mapping for different providers."""
    console = Console()
    console.print("\n🎯 Testing Model Mapping", style="bold blue")
    
    generic_models = ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"]
    providers = ["openai", "gemini", "deepseek", "openrouter", "anthropic"]
    
    table = Table(title="Model Name Mapping")
    table.add_column("Generic Model", style="cyan")
    
    for provider in providers:
        table.add_column(provider.capitalize(), style="green")
    
    for generic_model in generic_models:
        row = [generic_model]
        
        for provider in providers:
            try:
                config = IntelligentScrapingConfig(
                    llm_provider=provider,
                    orchestrator_model=generic_model
                )
                mapped_model = config.get_provider_model_name(generic_model)
                row.append(mapped_model)
                
            except Exception as e:
                row.append(f"Error: {e}")
        
        table.add_row(*row)
    
    console.print(table)


def test_environment_variables():
    """Test environment variable configuration."""
    console = Console()
    console.print("\n🌍 Testing Environment Variables", style="bold blue")
    
    # Show current environment variables
    env_vars = [
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY", 
        "DEEPSEEK_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "ORCHESTRATOR_MODEL",
        "PLANNING_AGENT_MODEL"
    ]
    
    table = Table(title="Environment Variables")
    table.add_column("Variable", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Status", style="yellow")
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Mask API keys for security
            if "API_KEY" in var:
                display_value = f"{value[:8]}..." if len(value) > 8 else "***"
            else:
                display_value = value
            status = "✅ Set"
        else:
            display_value = "Not set"
            status = "❌ Missing"
        
        table.add_row(var, display_value, status)
    
    console.print(table)


def show_usage_examples():
    """Show usage examples for different providers."""
    console = Console()
    console.print("\n📚 Usage Examples", style="bold blue")
    
    examples = [
        {
            "title": "OpenAI Configuration",
            "code": """
# Environment variables
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_openai_key

# Python code
config = IntelligentScrapingConfig.from_env()
"""
        },
        {
            "title": "Gemini Configuration", 
            "code": """
# Environment variables
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=your_gemini_key

# Python code
config = IntelligentScrapingConfig(
    llm_provider="gemini",
    orchestrator_model="gpt-4o-mini"  # Maps to gemini-1.5-flash
)
"""
        },
        {
            "title": "DeepSeek Configuration",
            "code": """
# Environment variables
export LLM_PROVIDER=deepseek
export DEEPSEEK_API_KEY=your_deepseek_key

# Python code
config = IntelligentScrapingConfig(
    llm_provider="deepseek",
    orchestrator_model="gpt-4"  # Maps to deepseek-chat
)
"""
        }
    ]
    
    for example in examples:
        panel = Panel(
            example["code"].strip(),
            title=example["title"],
            border_style="green"
        )
        console.print(panel)


def main():
    """Main test function."""
    console = Console()
    
    # Display header
    header_text = Text()
    header_text.append("🧪 LLM Provider Test Suite\n", style="bold bright_blue")
    header_text.append("Testing multi-provider support for Intelligent Web Scraper", style="italic")
    
    header_panel = Panel(
        header_text,
        title="Test Suite",
        border_style="bright_blue"
    )
    console.print(header_panel)
    
    # Run tests
    try:
        test_configuration_loading()
        test_provider_availability()
        test_model_mapping()
        test_environment_variables()
        test_provider_connections()
        show_usage_examples()
        
        console.print("\n🎉 Test suite completed!", style="bold green")
        
    except KeyboardInterrupt:
        console.print("\n⏹️  Test suite interrupted", style="yellow")
    except Exception as e:
        console.print(f"\n❌ Test suite failed: {e}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()