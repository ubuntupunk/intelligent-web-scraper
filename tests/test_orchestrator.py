"""
Unit tests for the IntelligentScrapingOrchestrator agent.

This module tests the orchestrator agent initialization, configuration,
and basic functionality according to atomic-agents patterns.
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

import instructor
from openai import OpenAI

from intelligent_web_scraper.agents.orchestrator import (
    IntelligentScrapingOrchestrator,
    IntelligentScrapingOrchestratorInputSchema,
    IntelligentScrapingOrchestratorOutputSchema,
    ScrapingMetadata,
    ScraperInstanceStats,
    ScrapingMonitoringReport
)
from intelligent_web_scraper.config import IntelligentScrapingConfig


class TestIntelligentScrapingOrchestratorInputSchema:
    """Test the input schema for the orchestrator agent."""
    
    def test_input_schema_required_fields(self):
        """Test that required fields are properly validated."""
        # Test with all required fields
        valid_input = IntelligentScrapingOrchestratorInputSchema(
            scraping_request="Extract product information",
            target_url="https://example.com"
        )
        assert valid_input.scraping_request == "Extract product information"
        assert valid_input.target_url == "https://example.com"
        assert valid_input.max_results == 10  # default value
        assert valid_input.quality_threshold == 50.0  # default value
        assert valid_input.export_format == "json"  # default value
        assert valid_input.enable_monitoring is True  # default value
        assert valid_input.concurrent_instances == 1  # default value
    
    def test_input_schema_missing_required_fields(self):
        """Test that missing required fields raise validation errors."""
        with pytest.raises(ValueError):
            IntelligentScrapingOrchestratorInputSchema(
                scraping_request="Extract product information"
                # missing target_url
            )
        
        with pytest.raises(ValueError):
            IntelligentScrapingOrchestratorInputSchema(
                target_url="https://example.com"
                # missing scraping_request
            )
    
    def test_input_schema_optional_fields(self):
        """Test that optional fields work correctly."""
        input_data = IntelligentScrapingOrchestratorInputSchema(
            scraping_request="Extract product information",
            target_url="https://example.com",
            max_results=20,
            quality_threshold=75.0,
            export_format="csv",
            enable_monitoring=False,
            concurrent_instances=3
        )
        assert input_data.max_results == 20
        assert input_data.quality_threshold == 75.0
        assert input_data.export_format == "csv"
        assert input_data.enable_monitoring is False
        assert input_data.concurrent_instances == 3


class TestIntelligentScrapingOrchestratorOutputSchema:
    """Test the output schema for the orchestrator agent."""
    
    def test_output_schema_creation(self):
        """Test that output schema can be created with all required fields."""
        metadata = ScrapingMetadata(
            url="https://example.com",
            timestamp=datetime.utcnow(),
            strategy_used="test_strategy",
            pages_processed=1,
            items_extracted=5,
            quality_score=85.0,
            processing_time=2.5,
            errors_encountered=[],
            instance_id="test-instance",
            monitoring_enabled=True
        )
        
        monitoring_report = ScrapingMonitoringReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.utcnow(),
            total_instances=1,
            active_instances=1,
            overall_throughput=2.0,
            overall_success_rate=95.0,
            overall_error_rate=5.0,
            resource_utilization={"memory_mb": 100.0, "cpu_percent": 25.0},
            performance_trends={"throughput": [1.5, 2.0], "success_rate": [90.0, 95.0]},
            alerts=[],
            recommendations=["Consider increasing concurrent instances"],
            detailed_metrics={"avg_response_time": 1.2}
        )
        
        instance_stats = ScraperInstanceStats(
            instance_id="test-instance",
            status="running",
            uptime=120.0,
            requests_processed=10,
            success_rate=95.0,
            error_rate=5.0,
            average_response_time=1.2,
            memory_usage_mb=50.0,
            cpu_usage_percent=15.0,
            last_activity=datetime.utcnow(),
            current_task="Extracting product data"
        )
        
        output = IntelligentScrapingOrchestratorOutputSchema(
            scraping_plan="Test scraping plan",
            extracted_data=[{"title": "Test Product", "price": "$10.00"}],
            metadata=metadata,
            quality_score=85.0,
            reasoning="Test reasoning for scraping decisions",
            export_options={"json": "./results/test.json"},
            monitoring_report=monitoring_report,
            instance_statistics=[instance_stats]
        )
        
        assert output.scraping_plan == "Test scraping plan"
        assert len(output.extracted_data) == 1
        assert output.quality_score == 85.0
        assert output.metadata.url == "https://example.com"
        assert len(output.instance_statistics) == 1


class TestIntelligentScrapingOrchestrator:
    """Test the main orchestrator agent class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return IntelligentScrapingConfig(
            orchestrator_model="gpt-4o-mini",
            planning_agent_model="gpt-4o-mini",
            default_quality_threshold=60.0,
            max_concurrent_requests=3,
            enable_monitoring=True
        )
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock instructor client."""
        mock_openai_client = Mock(spec=OpenAI)
        return instructor.from_openai(mock_openai_client)
    
    def test_orchestrator_initialization(self, mock_config, mock_client):
        """Test that the orchestrator initializes correctly."""
        orchestrator = IntelligentScrapingOrchestrator(mock_config, client=mock_client)
        
        # Test that the orchestrator was initialized with correct configuration
        assert orchestrator.config == mock_config
        assert orchestrator.is_running is False
        assert isinstance(orchestrator.active_instances, dict)
        assert isinstance(orchestrator.monitoring_data, dict)
        
        # Test that the base agent was initialized correctly
        assert orchestrator.input_schema == IntelligentScrapingOrchestratorInputSchema
        assert orchestrator.output_schema == IntelligentScrapingOrchestratorOutputSchema
        assert orchestrator.model == mock_config.orchestrator_model
    
    def test_system_prompt_generator_creation(self, mock_config, mock_client):
        """Test that the system prompt generator is created correctly."""
        orchestrator = IntelligentScrapingOrchestrator(mock_config, client=mock_client)
        
        # Test that system prompt generator exists and has correct structure
        assert orchestrator.system_prompt_generator is not None
        
        # Generate a prompt to test the structure
        prompt = orchestrator.system_prompt_generator.generate_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        
        # Check that key concepts are mentioned in the prompt
        assert "Intelligent Scraping Orchestrator" in prompt
        assert "Atomic Agents" in prompt
        assert "scraping" in prompt.lower()
        assert "monitoring" in prompt.lower()
        assert "quality" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_run_method_basic_functionality(self, mock_config, mock_client):
        """Test the basic functionality of the run method."""
        orchestrator = IntelligentScrapingOrchestrator(mock_config, client=mock_client)
        
        # Test input data
        input_data = {
            "scraping_request": "Extract product information from the homepage",
            "target_url": "https://example.com",
            "max_results": 5,
            "quality_threshold": 70.0,
            "export_format": "json",
            "enable_monitoring": True,
            "concurrent_instances": 1
        }
        
        # Run the orchestrator
        result = await orchestrator.run(input_data)
        
        # Verify the result structure
        assert isinstance(result, IntelligentScrapingOrchestratorOutputSchema)
        assert result.scraping_plan is not None
        assert isinstance(result.extracted_data, list)
        assert isinstance(result.metadata, ScrapingMetadata)
        assert isinstance(result.quality_score, float)
        assert result.reasoning is not None
        assert isinstance(result.export_options, dict)
        assert isinstance(result.monitoring_report, ScrapingMonitoringReport)
        assert isinstance(result.instance_statistics, list)
        
        # Verify metadata content
        assert result.metadata.url == "https://example.com"
        assert result.metadata.monitoring_enabled is True
        assert isinstance(result.metadata.timestamp, datetime)
        assert isinstance(result.metadata.instance_id, str)
    
    @pytest.mark.asyncio
    async def test_agent_coordination_workflow(self, mock_config, mock_client):
        """Test the complete agent coordination workflow."""
        orchestrator = IntelligentScrapingOrchestrator(mock_config, client=mock_client)
        
        # Mock the coordination methods to test workflow
        mock_planning_result = {
            "scraping_plan": "Test scraping plan",
            "strategy": {"scrape_type": "list", "target_selectors": [".item"]},
            "schema_recipe": {"name": "test_schema", "fields": {}},
            "reasoning": "Test reasoning",
            "confidence": 0.8
        }
        
        mock_scraping_result = {
            "results": {
                "items": [{"title": "Test Item", "price": "$10.00"}],
                "total_found": 1,
                "total_scraped": 1,
                "strategy_used": {"scrape_type": "list"},
                "errors": []
            },
            "summary": "Successfully scraped 1 item",
            "quality_metrics": {
                "average_quality_score": 85.0,
                "success_rate": 100.0,
                "total_items_found": 1.0,
                "total_items_scraped": 1.0,
                "execution_time": 2.5
            }
        }
        
        # Mock the coordination methods
        orchestrator._coordinate_with_planning_agent = AsyncMock(return_value=mock_planning_result)
        orchestrator._coordinate_with_scraper_tool = AsyncMock(return_value=mock_scraping_result)
        
        # Test input
        input_data = {
            "scraping_request": "Extract product information",
            "target_url": "https://example.com",
            "max_results": 10,
            "quality_threshold": 70.0
        }
        
        # Run orchestrator
        result = await orchestrator.run(input_data)
        
        # Verify coordination was called
        orchestrator._coordinate_with_planning_agent.assert_called_once()
        orchestrator._coordinate_with_scraper_tool.assert_called_once()
        
        # Verify result contains coordinated data
        assert result.scraping_plan == "Test scraping plan"
        assert len(result.extracted_data) == 1
        assert result.extracted_data[0]["title"] == "Test Item"
        assert result.quality_score == 85.0
        assert result.reasoning == "Test reasoning"
    
    @pytest.mark.asyncio
    async def test_schema_transformation_methods(self, mock_config, mock_client):
        """Test schema transformation between components."""
        orchestrator = IntelligentScrapingOrchestrator(mock_config, client=mock_client)
        
        # Test planning to scraper transformation
        planning_result = {
            "strategy": {"scrape_type": "list", "target_selectors": [".item"]},
            "schema_recipe": {"name": "test_schema", "fields": {"title": {"field_type": "string"}}},
            "reasoning": "Test reasoning"
        }
        
        request = IntelligentScrapingOrchestratorInputSchema(
            scraping_request="Test request",
            target_url="https://example.com",
            max_results=5
        )
        
        scraper_input = orchestrator._transform_planning_to_scraper_input(planning_result, request)
        
        # Verify transformation
        assert scraper_input["target_url"] == "https://example.com"
        assert scraper_input["strategy"] == planning_result["strategy"]
        assert scraper_input["schema_recipe"] == planning_result["schema_recipe"]
        assert scraper_input["max_results"] == 5
        
        # Test scraper to orchestrator transformation
        scraping_result = {
            "results": {
                "items": [{"title": "Test Item"}],
                "total_found": 1,
                "total_scraped": 1,
                "errors": []
            },
            "quality_metrics": {
                "average_quality_score": 90.0,
                "success_rate": 100.0
            }
        }
        
        operation_id = "test-operation-123"
        start_time = datetime.utcnow()
        
        orchestrator_output = orchestrator._transform_scraper_to_orchestrator_output(
            scraping_result, planning_result, request, operation_id, start_time
        )
        
        # Verify transformation
        assert isinstance(orchestrator_output, IntelligentScrapingOrchestratorOutputSchema)
        assert len(orchestrator_output.extracted_data) == 1
        assert orchestrator_output.extracted_data[0]["title"] == "Test Item"
        assert orchestrator_output.quality_score == 90.0
        assert orchestrator_output.metadata.instance_id == operation_id
    
    @pytest.mark.asyncio
    async def test_workflow_state_management(self, mock_config, mock_client):
        """Test workflow state management and progress tracking."""
        orchestrator = IntelligentScrapingOrchestrator(mock_config, client=mock_client)
        
        # Create mock result
        mock_result = Mock()
        mock_result.extracted_data = [{"title": "Test Item"}]
        mock_result.quality_score = 85.0
        mock_result.metadata = Mock()
        mock_result.metadata.processing_time = 2.5
        
        operation_id = "test-operation-456"
        
        # Test workflow state update
        orchestrator._update_workflow_state(operation_id, "completed", mock_result)
        
        # Verify state was stored
        assert operation_id in orchestrator.monitoring_data
        workflow_state = orchestrator.monitoring_data[operation_id]
        
        assert workflow_state["operation_id"] == operation_id
        assert workflow_state["status"] == "completed"
        assert isinstance(workflow_state["timestamp"], datetime)
        assert workflow_state["result_summary"]["items_extracted"] == 1
        assert workflow_state["result_summary"]["quality_score"] == 85.0
        assert workflow_state["result_summary"]["processing_time"] == 2.5
    
    @pytest.mark.asyncio
    async def test_coordination_error_handling(self, mock_config, mock_client):
        """Test error handling during agent coordination."""
        orchestrator = IntelligentScrapingOrchestrator(mock_config, client=mock_client)
        
        # Mock coordination method to raise an error
        orchestrator._coordinate_with_planning_agent = AsyncMock(
            side_effect=Exception("Planning agent connection failed")
        )
        
        input_data = {
            "scraping_request": "Test request",
            "target_url": "https://example.com"
        }
        
        # Run orchestrator and expect graceful error handling
        result = await orchestrator.run(input_data)
        
        # Verify error was handled gracefully
        assert isinstance(result, IntelligentScrapingOrchestratorOutputSchema)
        assert result.scraping_plan == "Operation failed during coordination"
        assert len(result.extracted_data) == 0
        assert result.quality_score == 0.0
        assert "Orchestration failed" in result.reasoning
        assert len(result.monitoring_report.alerts) > 0
        assert "Critical error" in result.monitoring_report.alerts[0]
    
    @pytest.mark.asyncio
    async def test_fallback_strategy_creation(self, mock_config, mock_client):
        """Test fallback strategy creation when planning fails."""
        orchestrator = IntelligentScrapingOrchestrator(mock_config, client=mock_client)
        
        request = IntelligentScrapingOrchestratorInputSchema(
            scraping_request="Extract product data",
            target_url="https://example.com"
        )
        
        error_message = "Planning agent unavailable"
        
        # Test fallback strategy creation
        fallback_strategy = orchestrator._create_fallback_strategy(request, error_message)
        
        # Verify fallback strategy structure
        assert "Fallback strategy" in fallback_strategy["scraping_plan"]
        assert fallback_strategy["strategy"]["scrape_type"] == "list"
        assert len(fallback_strategy["strategy"]["target_selectors"]) > 0
        assert fallback_strategy["schema_recipe"]["name"] == "fallback_schema"
        assert "title" in fallback_strategy["schema_recipe"]["fields"]
        assert "content" in fallback_strategy["schema_recipe"]["fields"]
        assert error_message in fallback_strategy["reasoning"]
        assert fallback_strategy["confidence"] == 0.3
    
    @pytest.mark.asyncio
    async def test_get_monitoring_report(self, mock_config, mock_client):
        """Test the get_monitoring_report method."""
        orchestrator = IntelligentScrapingOrchestrator(mock_config, client=mock_client)
        
        # Add some mock workflow states
        orchestrator.monitoring_data["operation-1"] = {
            "operation_id": "operation-1",
            "status": "completed",
            "timestamp": datetime.utcnow(),
            "result_summary": {
                "items_extracted": 5,
                "quality_score": 85.0,
                "processing_time": 2.5
            }
        }
        orchestrator.monitoring_data["operation-2"] = {
            "operation_id": "operation-2",
            "status": "running",
            "timestamp": datetime.utcnow(),
            "result_summary": {
                "items_extracted": 0,
                "quality_score": 0.0,
                "processing_time": 0.0
            }
        }
        
        # Get monitoring report
        report = await orchestrator.get_monitoring_report()
        
        # Verify report structure
        assert isinstance(report, ScrapingMonitoringReport)
        assert isinstance(report.report_id, str)
        assert isinstance(report.generated_at, datetime)
        assert report.total_instances == len(orchestrator.monitoring_data)
        assert report.active_instances == 1  # One running operation
        assert isinstance(report.resource_utilization, dict)
        assert isinstance(report.performance_trends, dict)
        assert isinstance(report.alerts, list)
        assert isinstance(report.recommendations, list)
        assert isinstance(report.detailed_metrics, dict)
        
        # Verify calculated metrics
        assert report.overall_success_rate == 100.0  # 1 completed, 0 failed
        assert report.detailed_metrics["total_items_extracted"] == 5
    
    def test_orchestrator_configuration_integration(self, mock_client):
        """Test that the orchestrator properly integrates with different configurations."""
        # Test with custom configuration
        custom_config = IntelligentScrapingConfig(
            orchestrator_model="gpt-4",
            default_quality_threshold=80.0,
            max_concurrent_requests=10,
            enable_monitoring=False
        )
        
        orchestrator = IntelligentScrapingOrchestrator(custom_config, client=mock_client)
        
        # Verify configuration is properly stored
        assert orchestrator.config.orchestrator_model == "gpt-4"
        assert orchestrator.config.default_quality_threshold == 80.0
        assert orchestrator.config.max_concurrent_requests == 10
        assert orchestrator.config.enable_monitoring is False
        
        # Verify model is set correctly
        assert orchestrator.model == "gpt-4"
    
    @pytest.mark.asyncio
    async def test_run_method_input_validation(self, mock_config, mock_client):
        """Test that the run method properly validates input."""
        orchestrator = IntelligentScrapingOrchestrator(mock_config, client=mock_client)
        
        # Test with invalid input (missing required fields)
        with pytest.raises(ValueError):
            await orchestrator.run({
                "scraping_request": "Extract data"
                # missing target_url
            })
        
        # Test with invalid input (wrong types)
        with pytest.raises(ValueError):
            await orchestrator.run({
                "scraping_request": "Extract data",
                "target_url": "https://example.com",
                "max_results": "invalid"  # should be int
            })
    
    def test_orchestrator_inheritance(self, mock_config, mock_client):
        """Test that the orchestrator properly inherits from BaseAgent."""
        orchestrator = IntelligentScrapingOrchestrator(mock_config, client=mock_client)
        
        # Test that it has BaseAgent methods
        assert hasattr(orchestrator, 'get_response')
        assert hasattr(orchestrator, 'reset_memory')
        assert hasattr(orchestrator, 'register_context_provider')
        assert hasattr(orchestrator, 'unregister_context_provider')
        assert hasattr(orchestrator, 'get_context_provider')
        
        # Test that memory is initialized
        assert orchestrator.memory is not None
        assert orchestrator.initial_memory is not None


class TestScrapingMetadata:
    """Test the ScrapingMetadata schema."""
    
    def test_metadata_creation(self):
        """Test that metadata can be created with all required fields."""
        metadata = ScrapingMetadata(
            url="https://example.com",
            timestamp=datetime.utcnow(),
            strategy_used="intelligent_analysis",
            pages_processed=3,
            items_extracted=15,
            quality_score=92.5,
            processing_time=5.2,
            errors_encountered=["Minor timeout on page 2"],
            instance_id="instance-123",
            monitoring_enabled=True
        )
        
        assert metadata.url == "https://example.com"
        assert isinstance(metadata.timestamp, datetime)
        assert metadata.strategy_used == "intelligent_analysis"
        assert metadata.pages_processed == 3
        assert metadata.items_extracted == 15
        assert metadata.quality_score == 92.5
        assert metadata.processing_time == 5.2
        assert len(metadata.errors_encountered) == 1
        assert metadata.instance_id == "instance-123"
        assert metadata.monitoring_enabled is True


class TestScraperInstanceStats:
    """Test the ScraperInstanceStats schema."""
    
    def test_instance_stats_creation(self):
        """Test that instance stats can be created with all required fields."""
        stats = ScraperInstanceStats(
            instance_id="instance-456",
            status="running",
            uptime=300.5,
            requests_processed=25,
            success_rate=96.0,
            error_rate=4.0,
            average_response_time=1.8,
            memory_usage_mb=75.2,
            cpu_usage_percent=22.5,
            last_activity=datetime.utcnow(),
            current_task="Processing product catalog"
        )
        
        assert stats.instance_id == "instance-456"
        assert stats.status == "running"
        assert stats.uptime == 300.5
        assert stats.requests_processed == 25
        assert stats.success_rate == 96.0
        assert stats.error_rate == 4.0
        assert stats.average_response_time == 1.8
        assert stats.memory_usage_mb == 75.2
        assert stats.cpu_usage_percent == 22.5
        assert isinstance(stats.last_activity, datetime)
        assert stats.current_task == "Processing product catalog"


class TestScrapingMonitoringReport:
    """Test the ScrapingMonitoringReport schema."""
    
    def test_monitoring_report_creation(self):
        """Test that monitoring report can be created with all required fields."""
        report = ScrapingMonitoringReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.utcnow(),
            total_instances=3,
            active_instances=2,
            overall_throughput=4.5,
            overall_success_rate=94.2,
            overall_error_rate=5.8,
            resource_utilization={
                "memory_mb": 150.0,
                "cpu_percent": 35.0,
                "network_mbps": 2.1
            },
            performance_trends={
                "throughput": [3.2, 4.1, 4.5],
                "success_rate": [92.0, 93.5, 94.2],
                "response_time": [1.5, 1.3, 1.2]
            },
            alerts=["High memory usage on instance-2"],
            recommendations=[
                "Consider scaling up to handle increased load",
                "Monitor instance-2 for memory leaks"
            ],
            detailed_metrics={
                "avg_response_time": 1.2,
                "peak_memory_usage": 180.0,
                "total_requests": 150
            }
        )
        
        assert isinstance(report.report_id, str)
        assert isinstance(report.generated_at, datetime)
        assert report.total_instances == 3
        assert report.active_instances == 2
        assert report.overall_throughput == 4.5
        assert report.overall_success_rate == 94.2
        assert report.overall_error_rate == 5.8
        assert "memory_mb" in report.resource_utilization
        assert "throughput" in report.performance_trends
        assert len(report.alerts) == 1
        assert len(report.recommendations) == 2
        assert "avg_response_time" in report.detailed_metrics


if __name__ == "__main__":
    pytest.main([__file__])