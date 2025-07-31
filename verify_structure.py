#!/usr/bin/env python3
"""
Verification script to test the intelligent web scraper project structure.
"""

import asyncio
import sys
from pathlib import Path

# Add the project to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🔍 Testing imports...")
    
    try:
        from intelligent_web_scraper import (
            IntelligentScrapingConfig,
            IntelligentScrapingOrchestrator,
            WebsiteAnalysisContextProvider,
            ScrapingResultsContextProvider,
            ConfigurationContextProvider,
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration creation and environment loading."""
    print("\n🔧 Testing configuration...")
    
    try:
        from intelligent_web_scraper import IntelligentScrapingConfig
        
        # Test default configuration
        config = IntelligentScrapingConfig()
        assert config.orchestrator_model == "gpt-4o-mini"
        assert config.default_quality_threshold == 50.0
        assert config.enable_monitoring is True
        print("✅ Default configuration works")
        
        # Test environment configuration
        config_env = IntelligentScrapingConfig.from_env()
        assert config_env.orchestrator_model is not None
        print("✅ Environment configuration works")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_orchestrator():
    """Test orchestrator creation and basic functionality."""
    print("\n🤖 Testing orchestrator...")
    
    try:
        from intelligent_web_scraper import IntelligentScrapingConfig, IntelligentScrapingOrchestrator
        
        config = IntelligentScrapingConfig()
        orchestrator = IntelligentScrapingOrchestrator(config=config)
        
        # Check basic properties
        assert orchestrator.model == "gpt-4o-mini"
        assert orchestrator.input_schema is not None
        assert orchestrator.output_schema is not None
        assert orchestrator.is_running is False
        
        print("✅ Orchestrator creation successful")
        return True
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        return False

def test_context_providers():
    """Test context provider creation and functionality."""
    print("\n📋 Testing context providers...")
    
    try:
        from intelligent_web_scraper import IntelligentScrapingConfig
        from intelligent_web_scraper.context_providers import (
            WebsiteAnalysisContextProvider,
            ScrapingResultsContextProvider,
            ConfigurationContextProvider,
        )
        
        config = IntelligentScrapingConfig()
        
        # Test website analysis context provider
        website_context = WebsiteAnalysisContextProvider()
        assert website_context.analysis_results is None
        info = website_context.get_info()
        assert isinstance(info, str)
        print("✅ WebsiteAnalysisContextProvider works")
        
        # Test scraping results context provider
        results_context = ScrapingResultsContextProvider()
        assert len(results_context.results) == 0
        info = results_context.get_info()
        assert isinstance(info, str)
        print("✅ ScrapingResultsContextProvider works")
        
        # Test configuration context provider
        config_context = ConfigurationContextProvider(config)
        assert config_context.config == config
        info = config_context.get_info()
        assert isinstance(info, str)
        assert len(info) > 0
        print("✅ ConfigurationContextProvider works")
        
        return True
    except Exception as e:
        print(f"❌ Context providers test failed: {e}")
        return False

async def test_orchestrator_run():
    """Test orchestrator run method with mock data."""
    print("\n🚀 Testing orchestrator run method...")
    
    try:
        from intelligent_web_scraper import IntelligentScrapingConfig, IntelligentScrapingOrchestrator
        
        config = IntelligentScrapingConfig()
        orchestrator = IntelligentScrapingOrchestrator(config=config)
        
        # Test input data
        input_data = {
            "scraping_request": "Extract product information from this e-commerce page",
            "target_url": "https://example.com/products",
            "max_results": 10,
            "quality_threshold": 70.0,
            "export_format": "json",
            "enable_monitoring": True
        }
        
        # Run the orchestrator (should return mock data for now)
        result = await orchestrator.run(input_data)
        
        # Verify the result structure
        assert result.scraping_plan is not None
        assert isinstance(result.extracted_data, list)
        assert result.metadata is not None
        assert result.quality_score >= 0.0
        assert result.reasoning is not None
        assert isinstance(result.export_options, dict)
        assert result.monitoring_report is not None
        assert isinstance(result.instance_statistics, list)
        
        print("✅ Orchestrator run method works")
        print(f"   📊 Plan: {result.scraping_plan[:50]}...")
        print(f"   📈 Quality Score: {result.quality_score}")
        print(f"   📁 Export Options: {list(result.export_options.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Orchestrator run test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_project_structure():
    """Test that all required files and directories exist."""
    print("\n📁 Testing project structure...")
    
    required_files = [
        "pyproject.toml",
        "README.md",
        ".env.example",
        "intelligent_web_scraper/__init__.py",
        "intelligent_web_scraper/config.py",
        "intelligent_web_scraper/main.py",
        "intelligent_web_scraper/agents/__init__.py",
        "intelligent_web_scraper/agents/orchestrator.py",
        "intelligent_web_scraper/context_providers/__init__.py",
        "intelligent_web_scraper/context_providers/website_analysis.py",
        "intelligent_web_scraper/context_providers/scraping_results.py",
        "intelligent_web_scraper/context_providers/configuration.py",
        "intelligent_web_scraper/tools/__init__.py",
        "tests/__init__.py",
        "tests/test_basic_setup.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist")
        return True

async def main():
    """Run all verification tests."""
    print("🧪 Intelligent Web Scraper - Project Structure Verification")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Orchestrator", test_orchestrator),
        ("Context Providers", test_context_providers),
        ("Orchestrator Run", test_orchestrator_run),
    ]
    
    results = []
    for test_name, test_func in tests:
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! The project structure is set up correctly.")
        print("🚀 Ready to implement the remaining tasks!")
    else:
        print(f"\n⚠️  {len(tests) - passed} test(s) failed. Please review the issues above.")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)