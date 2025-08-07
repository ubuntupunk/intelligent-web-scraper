"""Integration tests for export functionality with orchestrator."""

import os
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock
import pytest

from intelligent_web_scraper.agents.orchestrator import (
    IntelligentScrapingOrchestrator,
    IntelligentScrapingOrchestratorInputSchema
)
from intelligent_web_scraper.config import IntelligentScrapingConfig
from intelligent_web_scraper.export import ExportFormat


class TestExportIntegration:
    """Test export functionality integration with orchestrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = IntelligentScrapingConfig(
            openai_api_key="test_key",
            request_delay=0.1,
            default_quality_threshold=50.0
        )
        
        # Create a mock orchestrator with just the export method
        self.orchestrator = Mock()
        # Import the actual method to test
        from intelligent_web_scraper.agents.orchestrator import IntelligentScrapingOrchestrator
        self.orchestrator._export_results = IntelligentScrapingOrchestrator._export_results.__get__(self.orchestrator)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_export_results_json_format(self):
        """Test export functionality with JSON format."""
        # Sample data
        items = [
            {"title": "Test Article 1", "url": "https://example.com/1", "content": "Content 1"},
            {"title": "Test Article 2", "url": "https://example.com/2", "content": "Content 2"}
        ]
        
        metadata = Mock()
        metadata.url = "https://example.com"
        metadata.strategy_used = "list"
        metadata.items_extracted = 2
        metadata.processing_time = 1.5
        metadata.timestamp.isoformat.return_value = "2023-01-01T00:00:00"
        metadata.instance_id = "test_instance"
        
        quality_metrics = {"accuracy": 0.95, "completeness": 0.88}
        
        # Test export
        with patch('intelligent_web_scraper.agents.orchestrator.ExportManager') as mock_export_manager:
            mock_manager = Mock()
            mock_export_manager.return_value = mock_manager
            
            mock_result = Mock()
            mock_result.success = True
            mock_result.file_path = f"{self.temp_dir}/test_export.json"
            mock_manager.export_data.return_value = mock_result
            
            export_options = self.orchestrator._export_results(
                items, metadata, quality_metrics, "json", "test_op_123"
            )
            
            assert "json" in export_options
            assert export_options["json"] == f"{self.temp_dir}/test_export.json"
            
            # Verify export manager was called correctly
            mock_export_manager.assert_called_once()
            mock_manager.export_data.assert_called_once()
            
            # Check the export data passed to the manager
            call_args = mock_manager.export_data.call_args
            export_data = call_args[0][0]  # First positional argument
            export_config = call_args[0][1]  # Second positional argument
            
            assert export_data.results == items
            assert export_config.format == ExportFormat.JSON
            assert "test_op_123" in export_config.filename_prefix
    
    def test_export_results_csv_format(self):
        """Test export functionality with CSV format."""
        items = [{"title": "Test", "url": "https://example.com"}]
        metadata = Mock()
        metadata.url = "https://example.com"
        metadata.strategy_used = "list"
        metadata.items_extracted = 1
        metadata.processing_time = 1.0
        metadata.timestamp.isoformat.return_value = "2023-01-01T00:00:00"
        metadata.instance_id = "test_instance"
        
        quality_metrics = {"accuracy": 0.90}
        
        with patch('intelligent_web_scraper.agents.orchestrator.ExportManager') as mock_export_manager:
            mock_manager = Mock()
            mock_export_manager.return_value = mock_manager
            
            mock_result = Mock()
            mock_result.success = True
            mock_result.file_path = f"{self.temp_dir}/test_export.csv"
            mock_manager.export_data.return_value = mock_result
            
            export_options = self.orchestrator._export_results(
                items, metadata, quality_metrics, "csv", "test_op_456"
            )
            
            assert "csv" in export_options
            assert export_options["csv"] == f"{self.temp_dir}/test_export.csv"
            
            # Check export configuration
            call_args = mock_manager.export_data.call_args
            export_config = call_args[0][1]
            assert export_config.format == ExportFormat.CSV
    
    def test_export_results_excel_format(self):
        """Test export functionality with Excel format."""
        items = [{"title": "Test", "url": "https://example.com"}]
        metadata = Mock()
        metadata.url = "https://example.com"
        metadata.strategy_used = "list"
        metadata.items_extracted = 1
        metadata.processing_time = 1.0
        metadata.timestamp.isoformat.return_value = "2023-01-01T00:00:00"
        metadata.instance_id = "test_instance"
        
        quality_metrics = {"accuracy": 0.90}
        
        with patch('intelligent_web_scraper.agents.orchestrator.ExportManager') as mock_export_manager:
            mock_manager = Mock()
            mock_export_manager.return_value = mock_manager
            
            mock_result = Mock()
            mock_result.success = True
            mock_result.file_path = f"{self.temp_dir}/test_export.xlsx"
            mock_manager.export_data.return_value = mock_result
            
            export_options = self.orchestrator._export_results(
                items, metadata, quality_metrics, "excel", "test_op_789"
            )
            
            assert "excel" in export_options
            assert export_options["excel"] == f"{self.temp_dir}/test_export.xlsx"
            
            # Check export configuration
            call_args = mock_manager.export_data.call_args
            export_config = call_args[0][1]
            assert export_config.format == ExportFormat.EXCEL
    
    def test_export_results_markdown_format(self):
        """Test export functionality with Markdown format."""
        items = [{"title": "Test", "url": "https://example.com"}]
        metadata = Mock()
        metadata.url = "https://example.com"
        metadata.strategy_used = "list"
        metadata.items_extracted = 1
        metadata.processing_time = 1.0
        metadata.timestamp.isoformat.return_value = "2023-01-01T00:00:00"
        metadata.instance_id = "test_instance"
        
        quality_metrics = {"accuracy": 0.90}
        
        with patch('intelligent_web_scraper.agents.orchestrator.ExportManager') as mock_export_manager:
            mock_manager = Mock()
            mock_export_manager.return_value = mock_manager
            
            mock_result = Mock()
            mock_result.success = True
            mock_result.file_path = f"{self.temp_dir}/test_export.md"
            mock_manager.export_data.return_value = mock_result
            
            export_options = self.orchestrator._export_results(
                items, metadata, quality_metrics, "markdown", "test_op_md"
            )
            
            assert "markdown" in export_options
            assert export_options["markdown"] == f"{self.temp_dir}/test_export.md"
            
            # Check export configuration
            call_args = mock_manager.export_data.call_args
            export_config = call_args[0][1]
            assert export_config.format == ExportFormat.MARKDOWN
    
    def test_export_results_invalid_format(self):
        """Test export functionality with invalid format defaults to JSON."""
        items = [{"title": "Test", "url": "https://example.com"}]
        metadata = Mock()
        metadata.url = "https://example.com"
        metadata.strategy_used = "list"
        metadata.items_extracted = 1
        metadata.processing_time = 1.0
        metadata.timestamp.isoformat.return_value = "2023-01-01T00:00:00"
        metadata.instance_id = "test_instance"
        
        quality_metrics = {"accuracy": 0.90}
        
        with patch('intelligent_web_scraper.agents.orchestrator.ExportManager') as mock_export_manager:
            mock_manager = Mock()
            mock_export_manager.return_value = mock_manager
            
            mock_result = Mock()
            mock_result.success = True
            mock_result.file_path = f"{self.temp_dir}/test_export.json"
            mock_manager.export_data.return_value = mock_result
            
            export_options = self.orchestrator._export_results(
                items, metadata, quality_metrics, "invalid_format", "test_op_invalid"
            )
            
            assert "invalid_format" in export_options
            
            # Should default to JSON format
            call_args = mock_manager.export_data.call_args
            export_config = call_args[0][1]
            assert export_config.format == ExportFormat.JSON
    
    def test_export_results_export_failure(self):
        """Test export functionality when export fails."""
        items = [{"title": "Test", "url": "https://example.com"}]
        metadata = Mock()
        metadata.url = "https://example.com"
        metadata.strategy_used = "list"
        metadata.items_extracted = 1
        metadata.processing_time = 1.0
        metadata.timestamp.isoformat.return_value = "2023-01-01T00:00:00"
        metadata.instance_id = "test_instance"
        
        quality_metrics = {"accuracy": 0.90}
        
        with patch('intelligent_web_scraper.agents.orchestrator.ExportManager') as mock_export_manager:
            mock_manager = Mock()
            mock_export_manager.return_value = mock_manager
            
            # Simulate export failure
            mock_result = Mock()
            mock_result.success = False
            mock_result.error_message = "Export failed"
            mock_manager.export_data.return_value = mock_result
            
            export_options = self.orchestrator._export_results(
                items, metadata, quality_metrics, "json", "test_op_fail"
            )
            
            assert "json" in export_options
            assert "export_failed_test_op_fail" in export_options["json"]
    
    def test_export_results_exception_handling(self):
        """Test export functionality exception handling."""
        items = [{"title": "Test", "url": "https://example.com"}]
        metadata = Mock()
        metadata.url = "https://example.com"
        metadata.strategy_used = "list"
        metadata.items_extracted = 1
        metadata.processing_time = 1.0
        metadata.timestamp.isoformat.return_value = "2023-01-01T00:00:00"
        metadata.instance_id = "test_instance"
        
        quality_metrics = {"accuracy": 0.90}
        
        with patch('intelligent_web_scraper.agents.orchestrator.ExportManager') as mock_export_manager:
            # Simulate exception during export
            mock_export_manager.side_effect = Exception("Export manager error")
            
            export_options = self.orchestrator._export_results(
                items, metadata, quality_metrics, "json", "test_op_error"
            )
            
            assert "json" in export_options
            assert "export_error_test_op_error" in export_options["json"]
    
    def test_export_metadata_structure(self):
        """Test that export metadata contains all required fields."""
        items = [{"title": "Test", "url": "https://example.com"}]
        metadata = Mock()
        metadata.url = "https://example.com"
        metadata.strategy_used = "list"
        metadata.items_extracted = 1
        metadata.processing_time = 1.5
        metadata.timestamp.isoformat.return_value = "2023-01-01T00:00:00"
        metadata.instance_id = "test_instance"
        
        quality_metrics = {"accuracy": 0.90, "completeness": 0.85}
        
        with patch('intelligent_web_scraper.agents.orchestrator.ExportManager') as mock_export_manager:
            mock_manager = Mock()
            mock_export_manager.return_value = mock_manager
            
            mock_result = Mock()
            mock_result.success = True
            mock_result.file_path = f"{self.temp_dir}/test_export.json"
            mock_manager.export_data.return_value = mock_result
            
            self.orchestrator._export_results(
                items, metadata, quality_metrics, "json", "test_op_metadata"
            )
            
            # Check the export data metadata
            call_args = mock_manager.export_data.call_args
            export_data = call_args[0][0]
            
            expected_metadata_keys = [
                "operation_id", "target_url", "strategy_used", 
                "items_extracted", "processing_time", "timestamp", "instance_id"
            ]
            
            for key in expected_metadata_keys:
                assert key in export_data.metadata
            
            assert export_data.metadata["operation_id"] == "test_op_metadata"
            assert export_data.metadata["target_url"] == "https://example.com"
            assert export_data.metadata["strategy_used"] == "list"
            assert export_data.metadata["items_extracted"] == 1
            assert export_data.metadata["processing_time"] == 1.5
            assert export_data.quality_metrics == quality_metrics


if __name__ == "__main__":
    pytest.main([__file__])