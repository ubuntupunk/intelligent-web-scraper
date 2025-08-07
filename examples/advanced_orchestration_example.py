#!/usr/bin/env python3
"""
Advanced Orchestration Example

This example demonstrates sophisticated orchestration patterns including:
- Multi-instance concurrent scraping
- Context provider customization
- Advanced monitoring and alerting
- Complex workflow coordination
- Custom tool integration
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from intelligent_web_scraper import (
    IntelligentScrapingOrchestrator,
    IntelligentScrapingConfig
)
from intelligent_web_scraper.context_providers import (
    WebsiteAnalysisContextProvider,
    ScrapingResultsContextProvider,
    ConfigurationContextProvider
)
from intelligent_web_scraper.agents import ScraperInstanceManager
from intelligent_web_scraper.monitoring import MonitoringDashboard, AlertManager, AlertLevel

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedScrapingWorkflow:
    """Advanced scraping workflow demonstrating orchestration patterns."""
    
    def __init__(self, config: IntelligentScrapingConfig):
        self.config = config
        self.orchestrator = IntelligentScrapingOrchestrator(config=config)
        self.instance_manager = ScraperInstanceManager(max_instances=5)
        self.monitoring_dashboard = MonitoringDashboard(
            refresh_rate=1.0,
            enable_sound_alerts=False
        )
        self.results_history: List[Dict[str, Any]] = []
        
    async def setup_advanced_context_providers(self):
        """Set up advanced context providers with custom data."""
        print("🔧 Setting up advanced context providers...")
        
        # Website analysis context with historical data
        website_context = WebsiteAnalysisContextProvider(title="Advanced Website Analysis")
        website_context.set_analysis_results({
            "site_type": "e-commerce",
            "complexity_score": 7.5,
            "content_patterns": [
                {"pattern": "product_grid", "confidence": 0.9},
                {"pattern": "pagination", "confidence": 0.8},
                {"pattern": "dynamic_loading", "confidence": 0.6}
            ],
            "recommended_strategy": "multi_page_extraction",
            "estimated_processing_time": 45.0
        })
        
        # Scraping results context with performance history
        results_context = ScrapingResultsContextProvider(title="Performance History")
        results_context.set_results_history([
            {
                "timestamp": datetime.now() - timedelta(hours=1),
                "quality_score": 92.5,
                "items_extracted": 150,
                "processing_time": 38.2,
                "strategy_used": "intelligent_extraction"
            },
            {
                "timestamp": datetime.now() - timedelta(hours=2),
                "quality_score": 88.1,
                "items_extracted": 120,
                "processing_time": 42.1,
                "strategy_used": "adaptive_extraction"
            }
        ])
        
        # Configuration context with optimization recommendations
        config_context = ConfigurationContextProvider(title="Optimization Context")
        config_context.set_optimization_recommendations([
            "Increase concurrent requests for better throughput",
            "Use caching for repeated website analysis",
            "Enable monitoring for production workloads"
        ])
        
        # Register context providers
        self.orchestrator.add_context_provider(website_context)
        self.orchestrator.add_context_provider(results_context)
        self.orchestrator.add_context_provider(config_context)
        
        print("✅ Advanced context providers configured")
    
    async def demonstrate_concurrent_multi_site_scraping(self):
        """Demonstrate concurrent scraping of multiple websites."""
        print("\n🌐 Concurrent Multi-Site Scraping")
        print("=" * 50)
        
        # Define multiple scraping targets
        scraping_targets = [
            {
                "name": "Books Site",
                "request": {
                    "scraping_request": "Extract book titles, authors, prices, and ratings",
                    "target_url": "https://books.toscrape.com/",
                    "max_results": 20,
                    "quality_threshold": 70.0,
                    "export_format": "json",
                    "concurrent_instances": 2
                }
            },
            {
                "name": "Test HTML",
                "request": {
                    "scraping_request": "Extract all text content and structure information",
                    "target_url": "https://httpbin.org/html",
                    "max_results": 10,
                    "quality_threshold": 60.0,
                    "export_format": "markdown",
                    "concurrent_instances": 1
                }
            },
            {
                "name": "JSON API",
                "request": {
                    "scraping_request": "Extract structured data from JSON response",
                    "target_url": "https://httpbin.org/json",
                    "max_results": 5,
                    "quality_threshold": 80.0,
                    "export_format": "csv",
                    "concurrent_instances": 1
                }
            }
        ]
        
        print(f"🎯 Scraping {len(scraping_targets)} sites concurrently...")
        
        # Create concurrent tasks
        tasks = []
        for target in scraping_targets:
            task = asyncio.create_task(
                self._execute_scraping_with_monitoring(
                    target["name"], 
                    target["request"]
                )
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        start_time = datetime.now()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = datetime.now()
        
        # Process results
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(results):
            target_name = scraping_targets[i]["name"]
            if isinstance(result, Exception):
                failed_results.append({"target": target_name, "error": str(result)})
                logger.error(f"Failed to scrape {target_name}: {result}")
            else:
                successful_results.append({"target": target_name, "result": result})
                self.results_history.append({
                    "target": target_name,
                    "timestamp": datetime.now(),
                    "quality_score": result.quality_score if result else 0,
                    "items_extracted": len(result.extracted_data) if result else 0
                })
        
        # Display summary
        total_time = (end_time - start_time).total_seconds()
        print(f"\n📊 Concurrent Scraping Summary:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Successful: {len(successful_results)}")
        print(f"   Failed: {len(failed_results)}")
        
        for success in successful_results:
            result = success["result"]
            print(f"   ✅ {success['target']}: {len(result.extracted_data)} items, "
                  f"quality {result.quality_score:.1f}")
        
        for failure in failed_results:
            print(f"   ❌ {failure['target']}: {failure['error']}")
        
        return successful_results
    
    async def _execute_scraping_with_monitoring(self, target_name: str, request: Dict[str, Any]):
        """Execute scraping with detailed monitoring."""
        print(f"🚀 Starting {target_name} scraping...")
        
        try:
            # Execute scraping
            result = await self.orchestrator.run(request)
            
            # Log monitoring data
            if result.monitoring_report:
                print(f"📊 {target_name} monitoring:")
                print(f"   Instances: {result.monitoring_report.active_instances}")
                print(f"   Throughput: {result.monitoring_report.overall_throughput:.2f} req/sec")
                print(f"   Success rate: {result.monitoring_report.overall_success_rate:.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error scraping {target_name}: {e}")
            raise
    
    async def demonstrate_adaptive_quality_control(self):
        """Demonstrate adaptive quality control based on results."""
        print("\n🎯 Adaptive Quality Control")
        print("=" * 50)
        
        # Start with conservative quality threshold
        initial_threshold = 80.0
        print(f"🎚️  Starting with quality threshold: {initial_threshold}")
        
        request = {
            "scraping_request": "Extract high-quality product information with detailed analysis",
            "target_url": "https://books.toscrape.com/",
            "max_results": 15,
            "quality_threshold": initial_threshold,
            "export_format": "json"
        }
        
        # First attempt with high threshold
        print("🔍 First attempt with high quality threshold...")
        result1 = await self.orchestrator.run(request)
        
        print(f"📊 First attempt results:")
        print(f"   Items extracted: {len(result1.extracted_data)}")
        print(f"   Quality score: {result1.quality_score:.1f}")
        print(f"   Processing time: {result1.metadata.processing_time:.2f}s")
        
        # Adaptive adjustment based on results
        if result1.quality_score < initial_threshold or len(result1.extracted_data) < 10:
            adjusted_threshold = max(50.0, result1.quality_score - 10.0)
            print(f"\n🔄 Adapting quality threshold to: {adjusted_threshold}")
            
            request["quality_threshold"] = adjusted_threshold
            request["max_results"] = 25  # Increase max results
            
            print("🔍 Second attempt with adjusted threshold...")
            result2 = await self.orchestrator.run(request)
            
            print(f"📊 Second attempt results:")
            print(f"   Items extracted: {len(result2.extracted_data)}")
            print(f"   Quality score: {result2.quality_score:.1f}")
            print(f"   Processing time: {result2.metadata.processing_time:.2f}s")
            
            # Compare results
            improvement = len(result2.extracted_data) - len(result1.extracted_data)
            print(f"\n📈 Adaptation results:")
            print(f"   Additional items: {improvement}")
            print(f"   Quality change: {result2.quality_score - result1.quality_score:.1f}")
            
            return result2
        else:
            print("✅ Initial quality threshold was appropriate")
            return result1
    
    async def demonstrate_intelligent_retry_logic(self):
        """Demonstrate intelligent retry logic with exponential backoff."""
        print("\n🔄 Intelligent Retry Logic")
        print("=" * 50)
        
        # Create a request that might fail
        request = {
            "scraping_request": "Extract data with retry logic demonstration",
            "target_url": "https://httpbin.org/status/500",  # Will return 500 error
            "max_results": 5,
            "quality_threshold": 50.0,
            "export_format": "json"
        }
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                print(f"🔄 Attempt {attempt + 1}/{max_retries + 1}")
                
                if attempt > 0:
                    # Exponential backoff
                    delay = base_delay * (2 ** (attempt - 1))
                    print(f"⏳ Waiting {delay:.1f}s before retry...")
                    await asyncio.sleep(delay)
                    
                    # Adjust strategy for retry
                    request["target_url"] = "https://httpbin.org/html"  # Switch to working URL
                    request["scraping_request"] = "Extract content after retry"
                
                result = await self.orchestrator.run(request)
                
                print(f"✅ Success on attempt {attempt + 1}!")
                print(f"   Items extracted: {len(result.extracted_data)}")
                print(f"   Quality score: {result.quality_score:.1f}")
                
                return result
                
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries:
                    print("🚫 All retry attempts exhausted")
                    raise
                else:
                    print("🔄 Will retry with adjusted parameters...")
    
    async def demonstrate_performance_optimization(self):
        """Demonstrate performance optimization techniques."""
        print("\n⚡ Performance Optimization")
        print("=" * 50)
        
        # Baseline performance test
        print("📊 Baseline performance test...")
        baseline_config = IntelligentScrapingConfig(
            max_concurrent_requests=1,
            request_delay=1.0,
            enable_monitoring=True
        )
        
        baseline_orchestrator = IntelligentScrapingOrchestrator(config=baseline_config)
        
        request = {
            "scraping_request": "Extract content for performance comparison",
            "target_url": "https://httpbin.org/html",
            "max_results": 10,
            "quality_threshold": 60.0,
            "export_format": "json"
        }
        
        start_time = datetime.now()
        baseline_result = await baseline_orchestrator.run(request)
        baseline_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   Baseline time: {baseline_time:.2f}s")
        print(f"   Items extracted: {len(baseline_result.extracted_data)}")
        
        # Optimized performance test
        print("\n🚀 Optimized performance test...")
        optimized_config = IntelligentScrapingConfig(
            max_concurrent_requests=5,
            request_delay=0.2,
            enable_monitoring=True
        )
        
        optimized_orchestrator = IntelligentScrapingOrchestrator(config=optimized_config)
        
        start_time = datetime.now()
        optimized_result = await optimized_orchestrator.run(request)
        optimized_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   Optimized time: {optimized_time:.2f}s")
        print(f"   Items extracted: {len(optimized_result.extracted_data)}")
        
        # Performance comparison
        speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
        print(f"\n📈 Performance improvement:")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Time saved: {baseline_time - optimized_time:.2f}s")
        
        return {
            "baseline_time": baseline_time,
            "optimized_time": optimized_time,
            "speedup": speedup
        }
    
    async def generate_comprehensive_report(self):
        """Generate a comprehensive workflow report."""
        print("\n📋 Comprehensive Workflow Report")
        print("=" * 50)
        
        report = {
            "workflow_summary": {
                "total_executions": len(self.results_history),
                "successful_extractions": sum(1 for r in self.results_history if r.get("items_extracted", 0) > 0),
                "average_quality_score": sum(r.get("quality_score", 0) for r in self.results_history) / len(self.results_history) if self.results_history else 0,
                "total_items_extracted": sum(r.get("items_extracted", 0) for r in self.results_history)
            },
            "performance_metrics": {
                "orchestrator_model": self.config.orchestrator_model,
                "planning_agent_model": self.config.planning_agent_model,
                "max_concurrent_requests": self.config.max_concurrent_requests,
                "default_quality_threshold": self.config.default_quality_threshold
            },
            "recommendations": [
                "Consider increasing concurrent requests for better throughput",
                "Monitor quality scores and adjust thresholds based on requirements",
                "Use caching for repeated website analysis",
                "Implement custom context providers for domain-specific knowledge"
            ]
        }
        
        # Save report
        report_filename = f"workflow_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Workflow Summary:")
        print(f"   Total executions: {report['workflow_summary']['total_executions']}")
        print(f"   Successful extractions: {report['workflow_summary']['successful_extractions']}")
        print(f"   Average quality: {report['workflow_summary']['average_quality_score']:.1f}")
        print(f"   Total items: {report['workflow_summary']['total_items_extracted']}")
        print(f"   Report saved: {report_filename}")
        
        return report


async def main():
    """Main function demonstrating advanced orchestration patterns."""
    print("🎭 Intelligent Web Scraper - Advanced Orchestration")
    print("=" * 60)
    print("This example demonstrates sophisticated orchestration patterns:")
    print("- Multi-instance concurrent scraping")
    print("- Advanced context provider customization")
    print("- Adaptive quality control")
    print("- Intelligent retry logic")
    print("- Performance optimization")
    print("- Comprehensive reporting")
    print()
    
    # Create advanced configuration
    config = IntelligentScrapingConfig(
        orchestrator_model="gpt-4o-mini",
        planning_agent_model="gpt-4o-mini",
        max_concurrent_requests=3,
        request_delay=0.5,
        default_quality_threshold=70.0,
        enable_monitoring=True,
        results_directory="./advanced_results"
    )
    
    # Initialize advanced workflow
    workflow = AdvancedScrapingWorkflow(config)
    
    try:
        # Set up advanced context providers
        await workflow.setup_advanced_context_providers()
        
        # Demonstrate concurrent multi-site scraping
        await workflow.demonstrate_concurrent_multi_site_scraping()
        
        # Demonstrate adaptive quality control
        await workflow.demonstrate_adaptive_quality_control()
        
        # Demonstrate intelligent retry logic
        await workflow.demonstrate_intelligent_retry_logic()
        
        # Demonstrate performance optimization
        performance_results = await workflow.demonstrate_performance_optimization()
        
        # Generate comprehensive report
        final_report = await workflow.generate_comprehensive_report()
        
        print("\n🎉 Advanced orchestration demonstration completed!")
        print(f"📊 Performance improvement: {performance_results['speedup']:.2f}x speedup achieved")
        print(f"📋 Comprehensive report generated with {final_report['workflow_summary']['total_executions']} executions")
        
        print("\n🚀 Next steps:")
        print("- Explore custom context provider development")
        print("- Implement domain-specific orchestration patterns")
        print("- Scale to production workloads with monitoring")
        print("- Integrate with other atomic-agents tools")
        
    except Exception as e:
        logger.error(f"Advanced orchestration failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())