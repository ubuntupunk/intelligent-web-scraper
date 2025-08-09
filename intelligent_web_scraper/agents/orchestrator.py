"""
Intelligent Scraping Orchestrator Agent.

This module contains the main orchestrator agent that demonstrates advanced
atomic-agents patterns for coordinating complex multi-agent workflows.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid
import time
import logging

from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from pydantic import Field
import instructor
from openai import OpenAI

from ..config import IntelligentScrapingConfig
from ..export import ExportManager, ExportFormat, ExportConfiguration, ExportData

logger = logging.getLogger(__name__)


class IntelligentScrapingOrchestratorInputSchema(BaseIOSchema):
    """Input schema for the orchestrator agent."""
    
    scraping_request: str = Field(
        ..., 
        description="Natural language scraping request describing what to extract"
    )
    target_url: str = Field(
        ..., 
        description="URL of the website to scrape"
    )
    max_results: Optional[int] = Field(
        default=10, 
        description="Maximum number of results to extract"
    )
    quality_threshold: Optional[float] = Field(
        default=50.0, 
        description="Minimum quality score for extracted data (0-100)"
    )
    export_format: Optional[str] = Field(
        default="json", 
        description="Output format for results (json, csv, markdown, excel)"
    )
    enable_monitoring: Optional[bool] = Field(
        default=True, 
        description="Enable real-time monitoring and reporting"
    )
    concurrent_instances: Optional[int] = Field(
        default=1, 
        description="Number of concurrent scraper instances to use"
    )


class ScrapingMetadata(BaseIOSchema):
    """Metadata about the scraping operation."""
    
    url: str = Field(..., description="Target URL that was scraped")
    timestamp: datetime = Field(..., description="When the scraping operation started")
    strategy_used: str = Field(..., description="Scraping strategy that was employed")
    pages_processed: int = Field(..., description="Number of pages that were processed")
    items_extracted: int = Field(..., description="Number of items successfully extracted")
    quality_score: float = Field(..., description="Overall quality score of the extraction")
    processing_time: float = Field(..., description="Total processing time in seconds")
    errors_encountered: List[str] = Field(
        default_factory=list, 
        description="List of errors encountered during scraping"
    )
    instance_id: str = Field(..., description="ID of the scraper instance used")
    monitoring_enabled: bool = Field(..., description="Whether monitoring was enabled")


class ScraperInstanceStats(BaseIOSchema):
    """Statistics for individual scraper instances."""
    
    instance_id: str = Field(..., description="Unique instance identifier")
    status: str = Field(..., description="Current instance status")
    uptime: float = Field(..., description="Instance uptime in seconds")
    requests_processed: int = Field(..., description="Total requests processed")
    success_rate: float = Field(..., description="Success rate percentage")
    error_rate: float = Field(..., description="Error rate percentage")
    average_response_time: float = Field(..., description="Average response time in seconds")
    memory_usage_mb: float = Field(..., description="Current memory usage in MB")
    cpu_usage_percent: float = Field(..., description="Current CPU usage percentage")
    last_activity: datetime = Field(..., description="Timestamp of last activity")
    current_task: Optional[str] = Field(None, description="Description of current task")


class ScrapingMonitoringReport(BaseIOSchema):
    """Comprehensive monitoring report for scraping operations."""
    
    report_id: str = Field(..., description="Unique report identifier")
    generated_at: datetime = Field(..., description="Report generation timestamp")
    total_instances: int = Field(..., description="Total number of scraper instances")
    active_instances: int = Field(..., description="Number of currently active instances")
    overall_throughput: float = Field(
        ..., 
        description="Overall system throughput in requests per second"
    )
    overall_success_rate: float = Field(
        ..., 
        description="Overall success rate across all instances"
    )
    overall_error_rate: float = Field(
        ..., 
        description="Overall error rate across all instances"
    )
    resource_utilization: Dict[str, float] = Field(
        ..., 
        description="System resource utilization metrics"
    )
    performance_trends: Dict[str, List[float]] = Field(
        ..., 
        description="Performance trends over time"
    )
    alerts: List[str] = Field(
        default_factory=list, 
        description="Active alerts and warnings"
    )
    recommendations: List[str] = Field(
        default_factory=list, 
        description="Performance optimization recommendations"
    )
    detailed_metrics: Dict[str, Any] = Field(
        ..., 
        description="Detailed metrics for analysis"
    )


class IntelligentScrapingOrchestratorOutputSchema(BaseIOSchema):
    """Output schema for the orchestrator agent."""
    
    scraping_plan: str = Field(
        ..., 
        description="Human-readable explanation of the scraping plan and strategy"
    )
    extracted_data: List[Dict[str, Any]] = Field(
        ..., 
        description="List of extracted structured data items"
    )
    metadata: ScrapingMetadata = Field(
        ..., 
        description="Comprehensive metadata about the scraping operation"
    )
    quality_score: float = Field(
        ..., 
        description="Overall quality score of the extracted data (0-100)"
    )
    reasoning: str = Field(
        ..., 
        description="Detailed explanation of scraping decisions and approach"
    )
    export_options: Dict[str, str] = Field(
        ..., 
        description="Available export formats and their file paths"
    )
    monitoring_report: ScrapingMonitoringReport = Field(
        ..., 
        description="Detailed monitoring and performance report"
    )
    instance_statistics: List[ScraperInstanceStats] = Field(
        ..., 
        description="Statistics for each scraper instance used"
    )


class IntelligentScrapingOrchestrator(BaseAgent):
    """
    Intelligent Scraping Orchestrator Agent.
    
    This agent demonstrates advanced atomic-agents patterns by orchestrating
    complex web scraping workflows with AI-powered strategy planning,
    real-time monitoring, and production-ready error handling.
    """
    
    def __init__(self, config: IntelligentScrapingConfig, client: Optional[instructor.client.Instructor] = None):
        """Initialize the orchestrator with configuration."""
        self.config = config
        
        # Create instructor client if not provided
        if client is None:
            client = instructor.from_openai(OpenAI())
        
        # Create system prompt generator with proper atomic-agents patterns
        system_prompt_generator = self._create_system_prompt_generator()
        
        # Create BaseAgentConfig
        agent_config = BaseAgentConfig(
            client=client,
            model=config.orchestrator_model,
            input_schema=IntelligentScrapingOrchestratorInputSchema,
            output_schema=IntelligentScrapingOrchestratorOutputSchema,
            system_prompt_generator=system_prompt_generator
        )
        
        # Initialize the base agent
        super().__init__(agent_config)
        
        # Initialize orchestrator state
        self.is_running = False
        self.active_instances: Dict[str, Any] = {}
        self.monitoring_data: Dict[str, Any] = {}
        
        # Store references to context providers for dynamic updates
        self.website_analysis_provider = self.system_prompt_generator.context_providers["website_analysis"]
        self.scraping_results_provider = self.system_prompt_generator.context_providers["scraping_results"]
        self.configuration_provider = self.system_prompt_generator.context_providers["configuration"]
    
    def _create_system_prompt_generator(self) -> SystemPromptGenerator:
        """Create the system prompt generator with proper atomic-agents patterns."""
        background = [
            "You are an Intelligent Scraping Orchestrator, an advanced AI agent that coordinates sophisticated web scraping operations using the Atomic Agents framework.",
            "You demonstrate advanced patterns for building production-ready, multi-agent workflows with intelligent coordination, real-time monitoring, and educational value.",
            "Your primary purpose is to serve as both a practical scraping solution and an educational example of complex agent orchestration patterns."
        ]
        
        steps = [
            "Parse and analyze the natural language scraping request to understand user intent and requirements",
            "Evaluate the target URL and determine the most appropriate scraping strategy based on website structure and content patterns",
            "Coordinate with planning agents to develop detailed extraction schemas and scraping approaches",
            "Manage scraper instances with proper resource allocation, monitoring, and performance tracking",
            "Execute the scraping operation with real-time quality assessment and error handling",
            "Process and validate extracted data against quality thresholds and user requirements",
            "Generate comprehensive reports with monitoring data, performance metrics, and educational explanations",
            "Provide clear reasoning for all decisions and demonstrate best practices for production systems"
        ]
        
        output_instructions = [
            "Always provide a detailed scraping plan that explains your strategy and approach",
            "Include comprehensive reasoning that demonstrates your decision-making process",
            "Generate structured data with proper validation and quality scoring",
            "Provide monitoring reports with real-time performance metrics and instance statistics",
            "Include educational explanations that help users understand advanced scraping patterns",
            "Ensure all outputs follow the defined schema structure with complete metadata",
            f"Apply quality thresholds (minimum: {self.config.default_quality_threshold}%) and provide confidence scores",
            "Handle errors gracefully and provide actionable recommendations for improvement",
            "Demonstrate respect for robots.txt, rate limiting, and ethical scraping practices",
            "Focus on educational value by explaining complex concepts and production-ready patterns"
        ]
        
        # Initialize context providers for dynamic context injection
        from ..context_providers import (
            WebsiteAnalysisContextProvider,
            ScrapingResultsContextProvider,
            ConfigurationContextProvider
        )
        
        context_providers = {
            "website_analysis": WebsiteAnalysisContextProvider(),
            "scraping_results": ScrapingResultsContextProvider(),
            "configuration": ConfigurationContextProvider()
        }
        
        # Set initial configuration context
        context_providers["configuration"].set_configuration(self.config)
        
        return SystemPromptGenerator(
            background=background,
            steps=steps,
            output_instructions=output_instructions,
            context_providers=context_providers
        )
    
    async def run(self, input_data: Dict[str, Any]) -> IntelligentScrapingOrchestratorOutputSchema:
        """
        Execute the intelligent scraping orchestration with agent coordination.
        
        This method demonstrates advanced atomic-agents patterns by coordinating
        between planning agents and scraper tools with proper schema transformation.
        """
        start_time = datetime.utcnow()
        
        # Parse input
        request = IntelligentScrapingOrchestratorInputSchema(**input_data)
        
        # Generate unique operation ID
        operation_id = str(uuid.uuid4())
        
        try:
            # Step 1: Coordinate with planning agent to generate strategy
            planning_result = await self._coordinate_with_planning_agent(request)
            
            # Step 2: Transform planning result to scraper tool input
            scraper_input = self._transform_planning_to_scraper_input(
                planning_result, request
            )
            
            # Step 3: Coordinate with scraper tool to execute scraping
            scraping_result = await self._coordinate_with_scraper_tool(scraper_input)
            
            # Step 4: Update scraping results context
            self._update_scraping_results_context(
                scraping_result.get("results", {}).get("items", []),
                scraping_result.get("quality_metrics", {}),
                {
                    "operation_id": operation_id,
                    "target_url": request.target_url,
                    "strategy": planning_result.get("strategy", {}),
                    "processing_time": (datetime.utcnow() - start_time).total_seconds()
                }
            )
            
            # Step 5: Transform scraper output to orchestrator output
            orchestrator_output = self._transform_scraper_to_orchestrator_output(
                scraping_result, planning_result, request, operation_id, start_time
            )
            
            # Step 6: Update workflow state
            self._update_workflow_state(operation_id, "completed", orchestrator_output)
            
            return orchestrator_output
            
        except Exception as e:
            # Handle coordination errors gracefully
            error_output = self._handle_coordination_error(
                e, request, operation_id, start_time
            )
            self._update_workflow_state(operation_id, "failed", error_output)
            return error_output
    
    async def _update_website_analysis_context(self, url: str) -> None:
        """
        Update website analysis context for dynamic context injection.
        
        This method demonstrates dynamic context injection patterns by
        updating context providers with real-time analysis data.
        """
        try:
            # Check if we have cached analysis
            cached_analysis = self.website_analysis_provider.get_cached_analysis(url)
            if cached_analysis and self.website_analysis_provider.is_analysis_fresh():
                return  # Use cached analysis
            
            # Perform basic website analysis (simplified for demonstration)
            from ..context_providers.website_analysis import (
                WebsiteStructureAnalysis, 
                ContentPattern, 
                NavigationInfo
            )
            
            # Create mock analysis (in production, this would use actual website analyzer)
            analysis = WebsiteStructureAnalysis(url, "Website Analysis")
            analysis.quality_score = 75.0
            analysis.analysis_confidence = 80.0
            
            # Add some mock content patterns
            pattern1 = ContentPattern("article", "article, .post, .item", 5, 85.0)
            pattern1.add_example("Main content articles")
            analysis.add_content_pattern(pattern1)
            
            pattern2 = ContentPattern("navigation", ".nav, nav, .menu", 3, 90.0)
            pattern2.add_example("Navigation menus")
            analysis.add_content_pattern(pattern2)
            
            # Set navigation info
            analysis.navigation_info.main_menu_selectors = [".nav", "nav", ".menu"]
            analysis.navigation_info.pagination_selectors = [".pagination", ".pager"]
            
            # Update context provider
            self.website_analysis_provider.set_analysis_results(analysis)
            
        except Exception as e:
            # Log error but don't fail the operation
            print(f"Warning: Failed to update website analysis context: {e}")
    
    def _update_scraping_results_context(
        self, 
        items: List[Dict[str, Any]], 
        quality_metrics: Dict[str, Any],
        operation_metadata: Dict[str, Any]
    ) -> None:
        """
        Update scraping results context for dynamic context injection.
        
        This method demonstrates how to inject real-time scraping results
        into agent context for enhanced decision-making.
        """
        try:
            # Extract quality scores if available
            quality_scores = []
            for item in items:
                # In a real implementation, quality scores would be calculated
                quality_scores.append(quality_metrics.get("average_quality_score", 0.0))
            
            # Update results context provider
            self.scraping_results_provider.set_results(items, quality_scores)
            self.scraping_results_provider.set_operation_metadata(operation_metadata)
            
            # Update extraction statistics
            stats = self.scraping_results_provider.extraction_statistics
            stats.total_items_found = quality_metrics.get("total_items_found", len(items))
            stats.total_items_extracted = len(items)
            stats.processing_time_seconds = quality_metrics.get("execution_time", 0.0)
            stats.pages_processed = 1  # Would be extracted from actual results
            
        except Exception as e:
            # Log error but don't fail the operation
            print(f"Warning: Failed to update scraping results context: {e}")
    
    def update_configuration_context(self, config_updates: Dict[str, Any]) -> None:
        """
        Update configuration context with new settings.
        
        This method demonstrates configuration context management
        and dynamic context updates.
        """
        try:
            # Apply configuration overrides
            for key, value in config_updates.items():
                self.configuration_provider.add_config_override(key, value)
            
            # Update the main configuration if needed
            if hasattr(self.config, 'model_dump'):
                current_config = self.config.model_dump()
                current_config.update(config_updates)
                
                # Create new config instance (in production, would use proper config management)
                from ..config import IntelligentScrapingConfig
                updated_config = IntelligentScrapingConfig(**current_config)
                self.configuration_provider.set_configuration(updated_config)
            
        except Exception as e:
            # Log error but don't fail the operation
            print(f"Warning: Failed to update configuration context: {e}")
    
    def get_context_provider_status(self) -> Dict[str, Any]:
        """
        Get status of all context providers for monitoring.
        
        Returns:
            Dictionary containing status of each context provider
        """
        status = {}
        
        try:
            # Website analysis provider status
            website_status = {
                "has_analysis": self.website_analysis_provider.analysis_results is not None,
                "analysis_age": self.website_analysis_provider.get_analysis_age(),
                "is_fresh": self.website_analysis_provider.is_analysis_fresh(),
                "pattern_count": len(self.website_analysis_provider.content_patterns),
                "cache_size": len(self.website_analysis_provider.analysis_cache)
            }
            status["website_analysis"] = website_status
            
            # Scraping results provider status
            results_status = {
                "result_count": len(self.scraping_results_provider.results),
                "average_quality": self.scraping_results_provider.quality_metrics.overall_quality_score,
                "has_metadata": bool(self.scraping_results_provider.operation_metadata),
                "analysis_age": (datetime.utcnow() - self.scraping_results_provider.analysis_timestamp).total_seconds()
            }
            status["scraping_results"] = results_status
            
            # Configuration provider status
            config_status = {
                "has_config": self.configuration_provider.config is not None,
                "is_valid": self.configuration_provider.validation_result.is_valid if self.configuration_provider.validation_result else False,
                "override_count": len(self.configuration_provider.config_overrides),
                "last_updated": (datetime.utcnow() - self.configuration_provider.last_updated).total_seconds()
            }
            status["configuration"] = config_status
            
        except Exception as e:
            status["error"] = f"Failed to get context provider status: {e}"
        
        return status
    
    def refresh_all_contexts(self) -> None:
        """
        Refresh all context providers with current data.
        
        This method demonstrates how to refresh context providers
        to ensure agents have the most current information.
        """
        try:
            # Refresh configuration context
            if self.config:
                self.configuration_provider.set_configuration(self.config)
            
            # Clear stale caches
            self.website_analysis_provider.clear_cache()
            
            # Reset results context if no active operations
            if not any(state["status"] == "running" for state in self.monitoring_data.values()):
                self.scraping_results_provider.clear_results()
            
        except Exception as e:
            print(f"Warning: Failed to refresh contexts: {e}")
    
    def inject_custom_context(self, provider_name: str, context_data: Dict[str, Any]) -> None:
        """
        Inject custom context data into a specific provider.
        
        This method demonstrates how to add custom context information
        for specialized use cases.
        """
        try:
            if provider_name == "website_analysis":
                self.website_analysis_provider.update_analysis(
                    context_data.get("url", ""), 
                    context_data
                )
            elif provider_name == "scraping_results":
                if "operation_metadata" in context_data:
                    self.scraping_results_provider.set_operation_metadata(
                        context_data["operation_metadata"]
                    )
            elif provider_name == "configuration":
                if "overrides" in context_data:
                    for key, value in context_data["overrides"].items():
                        self.configuration_provider.add_config_override(key, value)
            
        except Exception as e:
            print(f"Warning: Failed to inject custom context for {provider_name}: {e}")
    
    async def _coordinate_with_planning_agent(self, request: IntelligentScrapingOrchestratorInputSchema) -> Dict[str, Any]:
        """
        Coordinate with the enhanced planning agent to generate scraping strategy.
        
        This method demonstrates advanced agent coordination patterns by interfacing
        with the IntelligentWebScraperPlanningAgent for optimal strategies with
        educational explanations and seamless orchestrator integration.
        """
        try:
            # Update website analysis context if available
            await self._update_website_analysis_context(request.target_url)
            
            # Import the enhanced planning agent
            from .planning_agent import (
                IntelligentWebScraperPlanningAgent, 
                IntelligentPlanningAgentInputSchema
            )
            
            # Create planning agent with same client configuration
            planning_config = BaseAgentConfig(
                client=self.client,
                model=self.config.planning_agent_model
            )
            planning_agent = IntelligentWebScraperPlanningAgent(planning_config)
            
            # Transform orchestrator input to enhanced planning agent input
            orchestrator_context = {
                'educational_mode': True,
                'monitoring_enabled': request.enable_monitoring,
                'concurrent_instances': request.concurrent_instances,
                'export_format': request.export_format
            }
            
            planning_input = IntelligentPlanningAgentInputSchema(
                scraping_request=request.scraping_request,
                target_url=request.target_url,
                max_results=request.max_results,
                quality_threshold=request.quality_threshold,
                orchestrator_context=orchestrator_context
            )
            
            # Execute enhanced planning agent
            planning_result = planning_agent.run(planning_input)
            
            return {
                "scraping_plan": planning_result.scraping_plan,
                "strategy": planning_result.strategy,
                "schema_recipe": planning_result.schema_recipe,
                "reasoning": planning_result.reasoning,
                "confidence": planning_result.confidence,
                "orchestrator_metadata": planning_result.orchestrator_metadata,
                "educational_insights": planning_result.educational_insights
            }
            
        except Exception as e:
            # Fallback to basic strategy if planning agent fails
            return self._create_fallback_strategy(request, str(e))
    
    def _transform_planning_to_scraper_input(
        self, 
        planning_result: Dict[str, Any], 
        request: IntelligentScrapingOrchestratorInputSchema
    ) -> Dict[str, Any]:
        """
        Transform planning agent output to scraper tool input.
        
        This demonstrates schema transformation patterns for data flow
        between different components in the atomic-agents ecosystem.
        """
        return {
            "target_url": request.target_url,
            "strategy": planning_result["strategy"],
            "schema_recipe": planning_result["schema_recipe"],
            "max_results": request.max_results
        }
    
    async def _coordinate_with_scraper_tool(self, scraper_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate with the scraper tool to execute scraping operation.
        
        This method demonstrates tool coordination patterns and proper
        error handling in multi-component workflows.
        """
        try:
            # Import the scraper tool and schema
            from ..tools.atomic_scraper_tool import AtomicScraperTool, AtomicScraperInputSchema, AtomicScraperToolConfig
            
            # Create scraper tool configuration
            scraper_config = AtomicScraperToolConfig.from_intelligent_config(
                base_url=scraper_input["target_url"],
                intelligent_config=self.config,
                instance_id=f"orchestrator_{int(time.time())}"
            )
            
            # Create scraper tool
            scraper_tool = AtomicScraperTool(
                config=scraper_config,
                intelligent_config=self.config
            )
            
            # Create proper input schema object with dictionaries (not objects)
            scraper_input_schema = AtomicScraperInputSchema(
                target_url=scraper_input["target_url"],
                strategy=scraper_input.get("strategy", {}),
                schema_recipe=scraper_input.get("schema_recipe", {}),
                max_results=scraper_input.get("max_results", 10),
                quality_threshold=scraper_input.get("quality_threshold", self.config.default_quality_threshold),
                enable_monitoring=scraper_input.get("enable_monitoring", True)
            )
            
            # Execute scraping operation
            scraping_result = scraper_tool.run(scraper_input_schema)
            
            return {
                "results": scraping_result.results,
                "summary": scraping_result.summary,
                "quality_metrics": scraping_result.quality_metrics
            }
            
        except Exception as e:
            import traceback
            logger.error(f"Scraper tool coordination failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return error result with proper structure
            return {
                "results": {
                    "items": [],
                    "total_found": 0,
                    "total_scraped": 0,
                    "strategy_used": scraper_input.get("strategy", "fallback"),
                    "errors": [f"Scraper tool coordination failed: {str(e)}"]
                },
                "summary": f"Scraping failed due to coordination error: {str(e)}",
                "quality_metrics": {
                    "average_quality_score": 0.0,
                    "success_rate": 0.0,
                    "total_items_found": 0.0,
                    "total_items_scraped": 0.0,
                    "execution_time": 0.0
                }
            }
    
    def _transform_scraper_to_orchestrator_output(
        self,
        scraping_result: Dict[str, Any],
        planning_result: Dict[str, Any],
        request: IntelligentScrapingOrchestratorInputSchema,
        operation_id: str,
        start_time: datetime
    ) -> IntelligentScrapingOrchestratorOutputSchema:
        """
        Transform scraper tool output to orchestrator output schema.
        
        This demonstrates advanced schema transformation and data aggregation
        patterns for complex multi-agent workflows.
        """
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Extract data from scraping result
        results = scraping_result.get("results", {})
        items = results.get("items", [])
        quality_metrics = scraping_result.get("quality_metrics", {})
        
        # Create metadata
        metadata = ScrapingMetadata(
            url=request.target_url,
            timestamp=start_time,
            strategy_used=planning_result.get("strategy", {}).get("scrape_type", "unknown"),
            pages_processed=1,  # Would be extracted from actual results
            items_extracted=len(items),
            quality_score=quality_metrics.get("average_quality_score", 0.0),
            processing_time=processing_time,
            errors_encountered=results.get("errors", []),
            instance_id=operation_id,
            monitoring_enabled=request.enable_monitoring
        )
        
        # Create monitoring report
        monitoring_report = self._create_monitoring_report(
            operation_id, quality_metrics, processing_time
        )
        
        # Create instance statistics
        instance_stats = [self._create_instance_stats(operation_id, quality_metrics, processing_time)]
        
        # Export data if requested
        export_options = self._export_results(
            items, metadata, quality_metrics, request.export_format, operation_id
        )
        
        return IntelligentScrapingOrchestratorOutputSchema(
            scraping_plan=planning_result.get("scraping_plan", "No plan available"),
            extracted_data=items,
            metadata=metadata,
            quality_score=quality_metrics.get("average_quality_score", 0.0),
            reasoning=planning_result.get("reasoning", "No reasoning available"),
            export_options=export_options,
            monitoring_report=monitoring_report,
            instance_statistics=instance_stats
        )
    
    def _create_fallback_strategy(self, request: IntelligentScrapingOrchestratorInputSchema, error: str) -> Dict[str, Any]:
        """Create a fallback strategy when planning agent coordination fails."""
        return {
            "scraping_plan": f"Fallback strategy for {request.target_url} due to planning agent error",
            "strategy": {
                "scrape_type": "list",
                "target_selectors": ["article", ".item", ".product", ".listing"],
                "max_pages": 1,
                "request_delay": 1.0,
                "pagination_strategy": None,
                "extraction_rules": {}
            },
            "schema_recipe": {
                "name": "fallback_schema",
                "description": "Basic fallback schema for general content extraction",
                "fields": {
                    "title": {
                        "field_type": "string",
                        "description": "Title or heading",
                        "extraction_selector": "h1, h2, h3, .title",
                        "required": True,
                        "quality_weight": 0.9,
                        "post_processing": ["trim", "clean"]
                    },
                    "content": {
                        "field_type": "string",
                        "description": "Main content",
                        "extraction_selector": "p, .content, .description",
                        "required": False,
                        "quality_weight": 0.7,
                        "post_processing": ["trim", "clean"]
                    }
                },
                "validation_rules": ["normalize_whitespace"],
                "quality_weights": {"completeness": 0.5, "accuracy": 0.3, "consistency": 0.2},
                "version": "1.0"
            },
            "reasoning": f"Using fallback strategy due to planning agent error: {error}",
            "confidence": 0.3
        }
    
    def _create_monitoring_report(
        self, 
        operation_id: str, 
        quality_metrics: Dict[str, Any], 
        processing_time: float
    ) -> ScrapingMonitoringReport:
        """Create a monitoring report for the operation."""
        return ScrapingMonitoringReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.utcnow(),
            total_instances=1,
            active_instances=0,
            overall_throughput=1.0 / processing_time if processing_time > 0 else 0.0,
            overall_success_rate=quality_metrics.get("success_rate", 0.0),
            overall_error_rate=100.0 - quality_metrics.get("success_rate", 0.0),
            resource_utilization={
                "memory_mb": 50.0,  # Mock value
                "cpu_percent": 25.0,  # Mock value
                "network_mbps": 1.0  # Mock value
            },
            performance_trends={
                "throughput": [1.0 / processing_time if processing_time > 0 else 0.0],
                "success_rate": [quality_metrics.get("success_rate", 0.0)],
                "response_time": [processing_time]
            },
            alerts=[],
            recommendations=self._generate_recommendations(quality_metrics),
            detailed_metrics={
                "operation_id": operation_id,
                "total_processing_time": processing_time,
                "items_processed": quality_metrics.get("total_items_scraped", 0.0)
            }
        )
    
    def _create_instance_stats(
        self, 
        operation_id: str, 
        quality_metrics: Dict[str, Any], 
        processing_time: float
    ) -> ScraperInstanceStats:
        """Create instance statistics for the operation."""
        return ScraperInstanceStats(
            instance_id=operation_id,
            status="completed",
            uptime=processing_time,
            requests_processed=1,
            success_rate=quality_metrics.get("success_rate", 0.0),
            error_rate=100.0 - quality_metrics.get("success_rate", 0.0),
            average_response_time=processing_time,
            memory_usage_mb=50.0,  # Mock value
            cpu_usage_percent=25.0,  # Mock value
            last_activity=datetime.utcnow(),
            current_task=None
        )
    
    def _generate_recommendations(self, quality_metrics: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on metrics."""
        recommendations = []
        
        success_rate = quality_metrics.get("success_rate", 0.0)
        if success_rate < 50.0:
            recommendations.append("Consider adjusting quality thresholds or extraction selectors")
        
        avg_quality = quality_metrics.get("average_quality_score", 0.0)
        if avg_quality < 70.0:
            recommendations.append("Review schema recipe and extraction rules for better data quality")
        
        execution_time = quality_metrics.get("execution_time", 0.0)
        if execution_time > 30.0:
            recommendations.append("Consider optimizing request delays or implementing concurrent processing")
        
        if not recommendations:
            recommendations.append("Performance metrics look good - no immediate optimizations needed")
        
        return recommendations
    
    def _update_workflow_state(self, operation_id: str, status: str, result: Any) -> None:
        """
        Update workflow state for progress tracking.
        
        This demonstrates workflow state management patterns for
        complex multi-step operations.
        """
        workflow_state = {
            "operation_id": operation_id,
            "status": status,
            "timestamp": datetime.utcnow(),
            "result_summary": {
                "items_extracted": len(result.extracted_data) if hasattr(result, 'extracted_data') else 0,
                "quality_score": result.quality_score if hasattr(result, 'quality_score') else 0.0,
                "processing_time": result.metadata.processing_time if hasattr(result, 'metadata') else 0.0
            }
        }
        
        # Store workflow state (in production, this would go to a database)
        self.monitoring_data[operation_id] = workflow_state
    
    def _handle_coordination_error(
        self,
        error: Exception,
        request: IntelligentScrapingOrchestratorInputSchema,
        operation_id: str,
        start_time: datetime
    ) -> IntelligentScrapingOrchestratorOutputSchema:
        """Handle coordination errors gracefully with proper error reporting."""
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        error_message = f"Orchestration failed: {str(error)}"
        
        # Create error metadata
        error_metadata = ScrapingMetadata(
            url=request.target_url,
            timestamp=start_time,
            strategy_used="error_fallback",
            pages_processed=0,
            items_extracted=0,
            quality_score=0.0,
            processing_time=processing_time,
            errors_encountered=[error_message],
            instance_id=operation_id,
            monitoring_enabled=request.enable_monitoring
        )
        
        # Create error monitoring report
        error_monitoring_report = ScrapingMonitoringReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.utcnow(),
            total_instances=1,
            active_instances=0,
            overall_throughput=0.0,
            overall_success_rate=0.0,
            overall_error_rate=100.0,
            resource_utilization={"memory_mb": 0.0, "cpu_percent": 0.0},
            performance_trends={"throughput": [0.0], "success_rate": [0.0]},
            alerts=[f"Critical error in operation {operation_id}"],
            recommendations=["Check system configuration and retry the operation"],
            detailed_metrics={"error": error_message, "operation_id": operation_id}
        )
        
        return IntelligentScrapingOrchestratorOutputSchema(
            scraping_plan="Operation failed during coordination",
            extracted_data=[],
            metadata=error_metadata,
            quality_score=0.0,
            reasoning=f"Coordination error prevented successful execution: {error_message}",
            export_options={},
            monitoring_report=error_monitoring_report,
            instance_statistics=[]
        )
    
    async def get_monitoring_report(self) -> ScrapingMonitoringReport:
        """Get current monitoring report with real workflow state."""
        active_operations = [
            state for state in self.monitoring_data.values() 
            if state["status"] == "running"
        ]
        
        completed_operations = [
            state for state in self.monitoring_data.values() 
            if state["status"] == "completed"
        ]
        
        failed_operations = [
            state for state in self.monitoring_data.values() 
            if state["status"] == "failed"
        ]
        
        # Calculate aggregate metrics
        total_items = sum(
            op["result_summary"]["items_extracted"] 
            for op in completed_operations
        )
        
        avg_quality = (
            sum(op["result_summary"]["quality_score"] for op in completed_operations) / 
            len(completed_operations)
        ) if completed_operations else 0.0
        
        success_rate = (
            len(completed_operations) / 
            (len(completed_operations) + len(failed_operations))
        ) * 100.0 if (completed_operations or failed_operations) else 0.0
        
        return ScrapingMonitoringReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.utcnow(),
            total_instances=len(self.monitoring_data),
            active_instances=len(active_operations),
            overall_throughput=len(completed_operations) / 60.0,  # Operations per minute
            overall_success_rate=success_rate,
            overall_error_rate=100.0 - success_rate,
            resource_utilization={"memory_mb": 100.0, "cpu_percent": 30.0},
            performance_trends={
                "throughput": [len(completed_operations) / 60.0],
                "success_rate": [success_rate]
            },
            alerts=[f"{len(failed_operations)} failed operations"] if failed_operations else [],
            recommendations=["System operating normally"] if not failed_operations else ["Review failed operations"],
            detailed_metrics={
                "total_operations": len(self.monitoring_data),
                "total_items_extracted": total_items,
                "average_quality_score": avg_quality
            }
        )
    
    def _export_results(
        self, 
        items: List[Dict[str, Any]], 
        metadata: ScrapingMetadata, 
        quality_metrics: Dict[str, Any],
        export_format: str,
        operation_id: str
    ) -> Dict[str, str]:
        """
        Export scraping results using the ExportManager.
        
        Args:
            items: Scraped data items
            metadata: Scraping metadata
            quality_metrics: Quality metrics
            export_format: Requested export format
            operation_id: Unique operation identifier
            
        Returns:
            Dictionary mapping format to file path
        """
        try:
            # Map string format to ExportFormat enum
            format_mapping = {
                "json": ExportFormat.JSON,
                "csv": ExportFormat.CSV,
                "markdown": ExportFormat.MARKDOWN,
                "excel": ExportFormat.EXCEL
            }
            
            export_format_enum = format_mapping.get(export_format.lower(), ExportFormat.JSON)
            
            # Create export configuration
            config = ExportConfiguration(
                format=export_format_enum,
                output_directory="./exports",
                filename_prefix=f"scraping_results_{operation_id}",
                include_timestamp=True,
                include_metadata=True,
                overwrite_existing=True
            )
            
            # Prepare export data
            export_data = ExportData(
                results=items,
                metadata={
                    "operation_id": operation_id,
                    "target_url": metadata.url,
                    "strategy_used": metadata.strategy_used,
                    "items_extracted": metadata.items_extracted,
                    "processing_time": metadata.processing_time,
                    "timestamp": metadata.timestamp.isoformat(),
                    "instance_id": metadata.instance_id
                },
                quality_metrics=quality_metrics
            )
            
            # Export data
            export_manager = ExportManager()
            result = export_manager.export_data(export_data, config)
            
            if result.success:
                return {export_format: result.file_path}
            else:
                # Return placeholder path if export fails
                return {export_format: f"./exports/export_failed_{operation_id}.{export_format}"}
                
        except Exception as e:
            # Return placeholder path if export fails
            return {export_format: f"./exports/export_error_{operation_id}.{export_format}"}