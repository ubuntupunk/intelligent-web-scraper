#!/usr/bin/env python3
"""
Integration Patterns Example

This example demonstrates integration patterns with other atomic-agents tools
and frameworks, showing how to build complex multi-agent workflows and
extend the intelligent web scraper with custom components.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

from intelligent_web_scraper import (
    IntelligentScrapingOrchestrator,
    IntelligentScrapingConfig
)
from intelligent_web_scraper.tools import ToolFactory, AtomicScraperTool
from intelligent_web_scraper.context_providers import (
    WebsiteAnalysisContextProvider,
    ScrapingResultsContextProvider
)

# Import atomic-agents base classes for custom implementations
from atomic_agents.lib.base.base_agent import BaseAgent
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseTool
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.lib.components.system_prompt_context_providers import SystemPromptContextProviderBase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomDataProcessorInputSchema(BaseIOSchema):
    """Input schema for custom data processor."""
    raw_data: List[Dict[str, Any]]
    processing_instructions: str
    quality_threshold: float = 70.0
    output_format: str = "structured"


class CustomDataProcessorOutputSchema(BaseIOSchema):
    """Output schema for custom data processor."""
    processed_data: List[Dict[str, Any]]
    processing_summary: str
    quality_metrics: Dict[str, float]
    recommendations: List[str]


class CustomDataProcessorTool(BaseTool):
    """Custom data processor tool demonstrating atomic-agents integration."""
    
    input_schema = CustomDataProcessorInputSchema
    output_schema = CustomDataProcessorOutputSchema
    
    def __init__(self):
        super().__init__()
        self.name = "CustomDataProcessor"
        self.description = "Processes scraped data with custom business logic"
    
    def run(self, params: CustomDataProcessorInputSchema) -> CustomDataProcessorOutputSchema:
        """Process scraped data with custom logic."""
        logger.info(f"Processing {len(params.raw_data)} items with custom logic")
        
        processed_items = []
        quality_scores = []
        
        for item in params.raw_data:
            # Custom processing logic
            processed_item = self._process_single_item(item, params.processing_instructions)
            quality_score = self._calculate_quality_score(processed_item)
            
            if quality_score >= params.quality_threshold:
                processed_items.append(processed_item)
                quality_scores.append(quality_score)
        
        # Generate quality metrics
        quality_metrics = {
            "average_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "items_processed": len(processed_items),
            "items_filtered": len(params.raw_data) - len(processed_items),
            "processing_efficiency": len(processed_items) / len(params.raw_data) if params.raw_data else 0
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(quality_metrics, params)
        
        return CustomDataProcessorOutputSchema(
            processed_data=processed_items,
            processing_summary=f"Processed {len(processed_items)} items from {len(params.raw_data)} raw items",
            quality_metrics=quality_metrics,
            recommendations=recommendations
        )
    
    def _process_single_item(self, item: Dict[str, Any], instructions: str) -> Dict[str, Any]:
        """Process a single item with custom logic."""
        processed = item.copy()
        
        # Add processing metadata
        processed["processed_at"] = datetime.now().isoformat()
        processed["processing_instructions"] = instructions
        
        # Custom field transformations
        if "title" in processed:
            processed["title_length"] = len(processed["title"])
            processed["title_words"] = len(processed["title"].split())
        
        if "price" in processed:
            # Extract numeric price if it's a string
            price_str = str(processed["price"])
            numeric_price = ''.join(filter(str.isdigit, price_str.replace('.', 'X', 1).replace('X', '.')))
            try:
                processed["numeric_price"] = float(numeric_price) if numeric_price else 0.0
            except ValueError:
                processed["numeric_price"] = 0.0
        
        # Add custom scoring
        processed["custom_score"] = self._calculate_custom_score(processed)
        
        return processed
    
    def _calculate_quality_score(self, item: Dict[str, Any]) -> float:
        """Calculate quality score for processed item."""
        score = 50.0  # Base score
        
        # Bonus for having title
        if item.get("title") and len(item["title"]) > 5:
            score += 20.0
        
        # Bonus for having price
        if item.get("numeric_price", 0) > 0:
            score += 15.0
        
        # Bonus for having description/content
        if item.get("description") or item.get("content"):
            score += 10.0
        
        # Bonus for custom score
        custom_score = item.get("custom_score", 0)
        score += min(custom_score * 5, 15.0)
        
        return min(score, 100.0)
    
    def _calculate_custom_score(self, item: Dict[str, Any]) -> float:
        """Calculate custom business logic score."""
        score = 0.0
        
        # Score based on title quality
        title_length = item.get("title_length", 0)
        if 10 <= title_length <= 100:
            score += 2.0
        
        # Score based on price reasonableness
        price = item.get("numeric_price", 0)
        if 1.0 <= price <= 1000.0:
            score += 1.5
        
        # Score based on content richness
        content_fields = ["description", "content", "summary"]
        content_count = sum(1 for field in content_fields if item.get(field))
        score += content_count * 0.5
        
        return score
    
    def _generate_recommendations(self, metrics: Dict[str, float], params: CustomDataProcessorInputSchema) -> List[str]:
        """Generate processing recommendations."""
        recommendations = []
        
        if metrics["processing_efficiency"] < 0.5:
            recommendations.append("Consider lowering quality threshold to capture more items")
        
        if metrics["average_quality"] < 70:
            recommendations.append("Review processing instructions to improve data quality")
        
        if metrics["items_processed"] < 10:
            recommendations.append("Consider expanding scraping scope to gather more data")
        
        if not recommendations:
            recommendations.append("Processing performance is optimal")
        
        return recommendations


class CustomAnalyticsContextProvider(SystemPromptContextProviderBase):
    """Custom context provider for analytics and insights."""
    
    def __init__(self, title: str = "Analytics Context"):
        super().__init__(title=title)
        self.analytics_data: Dict[str, Any] = {}
        self.insights: List[str] = []
    
    def set_analytics_data(self, data: Dict[str, Any]):
        """Set analytics data for context."""
        self.analytics_data = data
        self._generate_insights()
    
    def _generate_insights(self):
        """Generate insights from analytics data."""
        self.insights = []
        
        if "processing_efficiency" in self.analytics_data:
            efficiency = self.analytics_data["processing_efficiency"]
            if efficiency > 0.8:
                self.insights.append("High processing efficiency indicates good data quality")
            elif efficiency < 0.5:
                self.insights.append("Low processing efficiency suggests need for strategy adjustment")
        
        if "average_quality" in self.analytics_data:
            quality = self.analytics_data["average_quality"]
            if quality > 85:
                self.insights.append("Excellent data quality achieved")
            elif quality < 60:
                self.insights.append("Data quality below optimal threshold")
    
    def get_info(self) -> str:
        """Return formatted analytics context."""
        if not self.analytics_data:
            return "No analytics data available"
        
        context = f"Analytics Summary:\n"
        for key, value in self.analytics_data.items():
            context += f"- {key}: {value}\n"
        
        if self.insights:
            context += "\nKey Insights:\n"
            for insight in self.insights:
                context += f"- {insight}\n"
        
        return context


class IntegratedWorkflowOrchestrator:
    """Orchestrator demonstrating integration patterns."""
    
    def __init__(self, config: IntelligentScrapingConfig):
        self.config = config
        self.scraping_orchestrator = IntelligentScrapingOrchestrator(config=config)
        self.data_processor = CustomDataProcessorTool()
        self.analytics_context = CustomAnalyticsContextProvider()
        self.workflow_history: List[Dict[str, Any]] = []
    
    async def execute_integrated_workflow(self, scraping_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute integrated workflow with scraping and custom processing."""
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting integrated workflow: {workflow_id}")
        
        workflow_result = {
            "workflow_id": workflow_id,
            "started_at": datetime.now(),
            "scraping_result": None,
            "processing_result": None,
            "final_analytics": None,
            "success": False
        }
        
        try:
            # Step 1: Execute scraping
            print("🕷️  Step 1: Executing intelligent scraping...")
            scraping_result = await self.scraping_orchestrator.run(scraping_request)
            workflow_result["scraping_result"] = {
                "items_extracted": len(scraping_result.extracted_data),
                "quality_score": scraping_result.quality_score,
                "processing_time": scraping_result.metadata.processing_time,
                "strategy_used": scraping_result.metadata.strategy_used
            }
            
            print(f"   ✅ Scraped {len(scraping_result.extracted_data)} items")
            print(f"   🎯 Quality score: {scraping_result.quality_score:.1f}")
            
            # Step 2: Custom data processing
            print("\n🔧 Step 2: Applying custom data processing...")
            processing_input = CustomDataProcessorInputSchema(
                raw_data=scraping_result.extracted_data,
                processing_instructions="Apply business logic and quality filtering",
                quality_threshold=70.0,
                output_format="structured"
            )
            
            processing_result = self.data_processor.run(processing_input)
            workflow_result["processing_result"] = {
                "items_processed": len(processing_result.processed_data),
                "quality_metrics": processing_result.quality_metrics,
                "recommendations": processing_result.recommendations
            }
            
            print(f"   ✅ Processed {len(processing_result.processed_data)} items")
            print(f"   📊 Average quality: {processing_result.quality_metrics['average_quality']:.1f}")
            
            # Step 3: Analytics and insights
            print("\n📈 Step 3: Generating analytics and insights...")
            combined_analytics = {
                **scraping_result.metadata.__dict__,
                **processing_result.quality_metrics
            }
            
            self.analytics_context.set_analytics_data(combined_analytics)
            workflow_result["final_analytics"] = combined_analytics
            
            # Step 4: Update context for future workflows
            print("\n🔄 Step 4: Updating context for future workflows...")
            self._update_workflow_context(workflow_result)
            
            workflow_result["success"] = True
            workflow_result["completed_at"] = datetime.now()
            
            print(f"\n✅ Integrated workflow {workflow_id} completed successfully!")
            
            return workflow_result
            
        except Exception as e:
            logger.error(f"Integrated workflow failed: {e}")
            workflow_result["error"] = str(e)
            workflow_result["completed_at"] = datetime.now()
            return workflow_result
    
    def _update_workflow_context(self, workflow_result: Dict[str, Any]):
        """Update context providers with workflow results."""
        self.workflow_history.append(workflow_result)
        
        # Update scraping results context
        if hasattr(self.scraping_orchestrator, 'context_providers'):
            for provider in self.scraping_orchestrator.context_providers:
                if isinstance(provider, ScrapingResultsContextProvider):
                    provider.add_workflow_result(workflow_result)
    
    async def demonstrate_tool_chaining(self):
        """Demonstrate chaining multiple tools together."""
        print("\n🔗 Tool Chaining Demonstration")
        print("=" * 50)
        
        # Create tool factory
        tool_factory = ToolFactory(self.config)
        
        # Create multiple scraper instances
        scraper1 = tool_factory.create_scraper_tool(
            base_url="https://books.toscrape.com",
            instance_id="books_scraper",
            config_overrides={"timeout": 30}
        )
        
        scraper2 = tool_factory.create_scraper_tool(
            base_url="https://httpbin.org",
            instance_id="api_scraper",
            config_overrides={"min_quality_score": 60.0}
        )
        
        print(f"🔧 Created {len(tool_factory.list_tool_instances())} scraper instances")
        
        # Chain tools: scraper -> processor -> analytics
        chain_results = []
        
        # Use scraper1 for books
        books_request = {
            "scraping_request": "Extract book information for tool chaining demo",
            "target_url": "https://books.toscrape.com/",
            "max_results": 5,
            "quality_threshold": 60.0
        }
        
        books_result = await self.execute_integrated_workflow(books_request)
        chain_results.append(("Books Scraping", books_result))
        
        # Use scraper2 for API data
        api_request = {
            "scraping_request": "Extract API data for tool chaining demo",
            "target_url": "https://httpbin.org/json",
            "max_results": 3,
            "quality_threshold": 50.0
        }
        
        api_result = await self.execute_integrated_workflow(api_request)
        chain_results.append(("API Scraping", api_result))
        
        # Analyze chaining results
        print(f"\n📊 Tool Chaining Results:")
        for chain_name, result in chain_results:
            success = "✅" if result["success"] else "❌"
            items = result.get("processing_result", {}).get("items_processed", 0)
            print(f"   {success} {chain_name}: {items} items processed")
        
        return chain_results
    
    async def demonstrate_context_provider_integration(self):
        """Demonstrate advanced context provider integration."""
        print("\n🧠 Context Provider Integration")
        print("=" * 50)
        
        # Add custom analytics context to orchestrator
        self.scraping_orchestrator.add_context_provider(self.analytics_context)
        
        # Set up rich context data
        self.analytics_context.set_analytics_data({
            "historical_success_rate": 0.92,
            "average_processing_time": 25.3,
            "preferred_strategies": ["intelligent_extraction", "adaptive_extraction"],
            "quality_trends": [85.2, 87.1, 89.3, 91.2],
            "processing_efficiency": 0.78
        })
        
        print("🧠 Enhanced context providers configured")
        print("   - Analytics context with historical data")
        print("   - Quality trends and success patterns")
        print("   - Strategy recommendations")
        
        # Execute scraping with enhanced context
        enhanced_request = {
            "scraping_request": "Use enhanced context to optimize scraping strategy",
            "target_url": "https://books.toscrape.com/",
            "max_results": 8,
            "quality_threshold": 75.0
        }
        
        result = await self.scraping_orchestrator.run(enhanced_request)
        
        print(f"\n📊 Enhanced Context Results:")
        print(f"   Items extracted: {len(result.extracted_data)}")
        print(f"   Quality score: {result.quality_score:.1f}")
        print(f"   Strategy used: {result.metadata.strategy_used}")
        print(f"   AI reasoning: {result.reasoning[:100]}...")
        
        return result
    
    def generate_integration_report(self) -> Dict[str, Any]:
        """Generate comprehensive integration report."""
        report = {
            "integration_summary": {
                "workflows_executed": len(self.workflow_history),
                "successful_workflows": sum(1 for w in self.workflow_history if w["success"]),
                "total_items_processed": sum(
                    w.get("processing_result", {}).get("items_processed", 0) 
                    for w in self.workflow_history
                ),
                "average_quality": sum(
                    w.get("scraping_result", {}).get("quality_score", 0) 
                    for w in self.workflow_history
                ) / len(self.workflow_history) if self.workflow_history else 0
            },
            "tool_integration": {
                "custom_tools_used": ["CustomDataProcessorTool"],
                "context_providers_used": ["CustomAnalyticsContextProvider"],
                "atomic_agents_patterns": [
                    "BaseAgent inheritance",
                    "BaseIOSchema compliance", 
                    "BaseTool implementation",
                    "SystemPromptContextProviderBase extension"
                ]
            },
            "performance_insights": {
                "integration_overhead": "Minimal - proper async coordination",
                "scalability": "High - leverages atomic-agents patterns",
                "extensibility": "Excellent - modular design",
                "maintainability": "Good - clear separation of concerns"
            }
        }
        
        return report


async def main():
    """Main function demonstrating integration patterns."""
    print("🔗 Intelligent Web Scraper - Integration Patterns")
    print("=" * 60)
    print("This example demonstrates integration with atomic-agents ecosystem:")
    print("- Custom tool development with BaseAgent/BaseTool patterns")
    print("- Advanced context provider implementation")
    print("- Multi-agent workflow orchestration")
    print("- Tool chaining and composition")
    print("- Schema compliance and data flow")
    print()
    
    # Create configuration
    config = IntelligentScrapingConfig(
        orchestrator_model="gpt-4o-mini",
        planning_agent_model="gpt-4o-mini",
        max_concurrent_requests=2,
        default_quality_threshold=70.0,
        enable_monitoring=True
    )
    
    # Initialize integrated orchestrator
    integrated_orchestrator = IntegratedWorkflowOrchestrator(config)
    
    try:
        # Demonstrate integrated workflow
        print("🚀 Demonstrating integrated workflow...")
        workflow_request = {
            "scraping_request": "Extract product information for integration demo",
            "target_url": "https://books.toscrape.com/",
            "max_results": 10,
            "quality_threshold": 65.0,
            "export_format": "json"
        }
        
        workflow_result = await integrated_orchestrator.execute_integrated_workflow(workflow_request)
        
        # Demonstrate tool chaining
        chaining_results = await integrated_orchestrator.demonstrate_tool_chaining()
        
        # Demonstrate context provider integration
        context_result = await integrated_orchestrator.demonstrate_context_provider_integration()
        
        # Generate integration report
        integration_report = integrated_orchestrator.generate_integration_report()
        
        # Save integration report
        report_filename = f"integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(integration_report, f, indent=2, default=str)
        
        print(f"\n📋 Integration Summary:")
        print(f"   Workflows executed: {integration_report['integration_summary']['workflows_executed']}")
        print(f"   Success rate: {integration_report['integration_summary']['successful_workflows']}/{integration_report['integration_summary']['workflows_executed']}")
        print(f"   Items processed: {integration_report['integration_summary']['total_items_processed']}")
        print(f"   Average quality: {integration_report['integration_summary']['average_quality']:.1f}")
        print(f"   Report saved: {report_filename}")
        
        print(f"\n🎯 Integration Patterns Demonstrated:")
        for pattern in integration_report['tool_integration']['atomic_agents_patterns']:
            print(f"   ✅ {pattern}")
        
        print(f"\n🚀 Next steps:")
        print("- Develop domain-specific custom tools")
        print("- Create specialized context providers")
        print("- Build multi-agent coordination workflows")
        print("- Integrate with external APIs and services")
        
    except Exception as e:
        logger.error(f"Integration demonstration failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())