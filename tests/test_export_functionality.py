"""Unit tests for export functionality."""

import json
import csv
import os
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from intelligent_web_scraper.export import (
    ExportManager, 
    ExportFormat, 
    ExportConfiguration, 
    ExportData, 
    ExportResult,
    ExportValidator,
    ValidationError
)


class TestExportFormats:
    """Test export format enums and data models."""
    
    def test_export_format_enum(self):
        """Test ExportFormat enum values."""
        assert ExportFormat.JSON == "json"
        assert ExportFormat.CSV == "csv"
        assert ExportFormat.MARKDOWN == "markdown"
        assert ExportFormat.EXCEL == "excel"
    
    def test_export_configuration_defaults(self):
        """Test ExportConfiguration default values."""
        config = ExportConfiguration(format=ExportFormat.JSON)
        assert config.format == ExportFormat.JSON
        assert config.output_directory == "./exports"
        assert config.filename_prefix == "scraping_results"
        assert config.include_timestamp is True
        assert config.include_metadata is True
        assert config.overwrite_existing is False
    
    def test_export_data_model(self):
        """Test ExportData model."""
        test_results = [{"title": "Test", "url": "http://example.com"}]
        data = ExportData(results=test_results)
        assert data.results == test_results
        assert isinstance(data.export_timestamp, datetime)


class TestExportValidator:
    """Test export validation functionality."""
    
    def test_validate_export_configuration_valid(self):
        """Test validation of valid configuration."""
        config = ExportConfiguration(
            format=ExportFormat.JSON,
            output_directory="./test_exports",
            filename_prefix="test_results"
        )
        errors = ExportValidator.validate_export_configuration(config)
        assert errors == []
    
    def test_validate_export_configuration_invalid_directory(self):
        """Test validation with invalid directory."""
        config = ExportConfiguration(
            format=ExportFormat.JSON,
            output_directory="",
            filename_prefix="test_results"
        )
        errors = ExportValidator.validate_export_configuration(config)
        assert "Output directory cannot be empty" in errors
    
    def test_validate_export_configuration_invalid_filename(self):
        """Test validation with invalid filename prefix."""
        config = ExportConfiguration(
            format=ExportFormat.JSON,
            output_directory="./test_exports",
            filename_prefix=""
        )
        errors = ExportValidator.validate_export_configuration(config)
        assert "Filename prefix cannot be empty" in errors
    
    def test_validate_export_configuration_csv_options(self):
        """Test validation of CSV-specific options."""
        config = ExportConfiguration(
            format=ExportFormat.CSV,
            csv_delimiter="||",  # Invalid - should be single character
            csv_quote_char='""'  # Invalid - should be single character
        )
        errors = ExportValidator.validate_export_configuration(config)
        assert "CSV delimiter must be a single character" in errors
        assert "CSV quote character must be a single character" in errors
    
    def test_validate_export_configuration_excel_options(self):
        """Test validation of Excel-specific options."""
        config = ExportConfiguration(
            format=ExportFormat.EXCEL,
            excel_sheet_name=""  # Invalid - cannot be empty
        )
        errors = ExportValidator.validate_export_configuration(config)
        assert "Excel sheet name cannot be empty" in errors
        
        # Test sheet name too long
        config.excel_sheet_name = "a" * 32  # Too long
        errors = ExportValidator.validate_export_configuration(config)
        assert "Excel sheet name too long (max 31 characters)" in errors
    
    def test_validate_export_data_empty(self):
        """Test validation of empty export data."""
        data = ExportData(results=[])
        errors = ExportValidator.validate_export_data(data)
        assert "No results to export" in errors
    
    def test_validate_export_data_inconsistent_structure(self):
        """Test validation of inconsistent data structure."""
        data = ExportData(results=[
            {"title": "Test 1", "url": "http://example1.com"},
            {"title": "Test 2", "description": "Different structure"}  # Missing url, has description
        ])
        errors = ExportValidator.validate_export_data(data)
        assert any("Inconsistent data structure" in error for error in errors)
    
    def test_validate_file_path_valid(self):
        """Test validation of valid file path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_file.json")
            errors = ExportValidator.validate_file_path(file_path)
            assert errors == []
    
    def test_validate_format_compatibility_csv_nested_data(self):
        """Test CSV format compatibility with nested data."""
        data = ExportData(results=[
            {"title": "Test", "metadata": {"author": "John", "date": "2023-01-01"}}
        ])
        errors = ExportValidator.validate_export_format_compatibility(data, ExportFormat.CSV)
        assert any("CSV format doesn't support nested data" in error for error in errors)


class TestExportManager:
    """Test export manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = ExportData(
            results=[
                {"title": "Test Article 1", "url": "http://example1.com", "content": "Content 1"},
                {"title": "Test Article 2", "url": "http://example2.com", "content": "Content 2"}
            ],
            metadata={"source": "test", "scraping_date": "2023-01-01"},
            quality_metrics={"accuracy": 0.95, "completeness": 0.88}
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_export_manager_initialization(self):
        """Test ExportManager initialization."""
        manager = ExportManager()
        assert manager.config.format == ExportFormat.JSON
        assert isinstance(manager.validator, ExportValidator)
    
    def test_export_json_format(self):
        """Test JSON export functionality."""
        config = ExportConfiguration(
            format=ExportFormat.JSON,
            output_directory=self.temp_dir,
            filename_prefix="test_json",
            include_timestamp=False
        )
        
        manager = ExportManager(config)
        result = manager.export_data(self.test_data, config)
        
        assert result.success is True
        assert result.format == ExportFormat.JSON
        assert result.records_exported == 2
        assert os.path.exists(result.file_path)
        
        # Verify JSON content
        with open(result.file_path, 'r') as f:
            exported_data = json.load(f)
        
        assert "results" in exported_data
        assert len(exported_data["results"]) == 2
        assert "metadata" in exported_data
        assert "quality_metrics" in exported_data
    
    def test_export_csv_format(self):
        """Test CSV export functionality."""
        config = ExportConfiguration(
            format=ExportFormat.CSV,
            output_directory=self.temp_dir,
            filename_prefix="test_csv",
            include_timestamp=False
        )
        
        manager = ExportManager(config)
        result = manager.export_data(self.test_data, config)
        
        assert result.success is True
        assert result.format == ExportFormat.CSV
        assert result.records_exported == 2
        assert os.path.exists(result.file_path)
        
        # Verify CSV content
        with open(result.file_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 2
        assert "title" in rows[0]
        assert "url" in rows[0]
        assert "content" in rows[0]
    
    def test_export_markdown_format(self):
        """Test Markdown export functionality."""
        config = ExportConfiguration(
            format=ExportFormat.MARKDOWN,
            output_directory=self.temp_dir,
            filename_prefix="test_markdown",
            include_timestamp=False,
            markdown_table_format=True
        )
        
        manager = ExportManager(config)
        result = manager.export_data(self.test_data, config)
        
        assert result.success is True
        assert result.format == ExportFormat.MARKDOWN
        assert result.records_exported == 2
        assert os.path.exists(result.file_path)
        
        # Verify Markdown content
        with open(result.file_path, 'r') as f:
            content = f.read()
        
        assert "# Scraping Results" in content
        assert "## Metadata" in content
        assert "## Quality Metrics" in content
        assert "| title |" in content  # Table format
    
    @patch('intelligent_web_scraper.export.export_manager.PANDAS_AVAILABLE', True)
    @patch('pandas.DataFrame')
    @patch('pandas.ExcelWriter')
    def test_export_excel_format(self, mock_excel_writer, mock_dataframe):
        """Test Excel export functionality."""
        # Mock pandas components
        mock_df = MagicMock()
        mock_dataframe.return_value = mock_df
        mock_writer = MagicMock()
        mock_excel_writer.return_value.__enter__.return_value = mock_writer
        
        config = ExportConfiguration(
            format=ExportFormat.EXCEL,
            output_directory=self.temp_dir,
            filename_prefix="test_excel",
            include_timestamp=False,
            overwrite_existing=True
        )
        
        manager = ExportManager(config)
        
        # Mock file creation for size validation
        test_file_path = os.path.join(self.temp_dir, "test_excel.xlsx")
        with open(test_file_path, 'w') as f:
            f.write("mock excel content")
        
        with patch.object(manager, '_generate_file_path', return_value=test_file_path):
            result = manager.export_data(self.test_data, config)
        
        assert result.success is True
        assert result.format == ExportFormat.EXCEL
        assert result.records_exported == 2
    
    def test_export_with_validation_error(self):
        """Test export with validation errors."""
        # Create invalid configuration
        config = ExportConfiguration(
            format=ExportFormat.JSON,
            output_directory="",  # Invalid empty directory
            filename_prefix="test"
        )
        
        manager = ExportManager(config)
        
        with pytest.raises(ValidationError):
            manager.export_data(self.test_data, config)
    
    def test_export_with_empty_data(self):
        """Test export with empty data."""
        empty_data = ExportData(results=[])
        config = ExportConfiguration(
            format=ExportFormat.JSON,
            output_directory=self.temp_dir
        )
        
        manager = ExportManager(config)
        
        with pytest.raises(ValidationError):
            manager.export_data(empty_data, config)
    
    def test_generate_file_path_with_timestamp(self):
        """Test file path generation with timestamp."""
        config = ExportConfiguration(
            format=ExportFormat.JSON,
            output_directory=self.temp_dir,
            filename_prefix="test_results",
            include_timestamp=True
        )
        
        manager = ExportManager(config)
        file_path = manager._generate_file_path(config)
        
        assert self.temp_dir in file_path
        assert "test_results" in file_path
        assert file_path.endswith(".json")
        # Should contain timestamp pattern (YYYYMMDD_HHMMSS)
        assert len(Path(file_path).stem.split('_')) >= 3
    
    def test_generate_file_path_without_timestamp(self):
        """Test file path generation without timestamp."""
        config = ExportConfiguration(
            format=ExportFormat.CSV,
            output_directory=self.temp_dir,
            filename_prefix="test_results",
            include_timestamp=False
        )
        
        manager = ExportManager(config)
        file_path = manager._generate_file_path(config)
        
        expected_path = os.path.join(self.temp_dir, "test_results.csv")
        assert file_path == expected_path
    
    def test_flatten_dict(self):
        """Test dictionary flattening for CSV export."""
        manager = ExportManager()
        
        nested_dict = {
            "title": "Test",
            "metadata": {
                "author": "John",
                "tags": ["tag1", "tag2"]
            },
            "stats": {
                "views": 100,
                "likes": {
                    "count": 50,
                    "recent": 10
                }
            }
        }
        
        flattened = manager._flatten_dict(nested_dict)
        
        assert flattened["title"] == "Test"
        assert flattened["metadata_author"] == "John"
        assert flattened["metadata_tags"] == "['tag1', 'tag2']"
        assert flattened["stats_views"] == 100
        assert flattened["stats_likes_count"] == 50
        assert flattened["stats_likes_recent"] == 10
    
    def test_get_supported_formats(self):
        """Test getting supported formats."""
        manager = ExportManager()
        formats = manager.get_supported_formats()
        
        assert ExportFormat.JSON in formats
        assert ExportFormat.CSV in formats
        assert ExportFormat.MARKDOWN in formats
        # Excel support depends on pandas availability
    
    def test_validate_format_support(self):
        """Test format support validation."""
        manager = ExportManager()
        
        assert manager.validate_format_support(ExportFormat.JSON) is True
        assert manager.validate_format_support(ExportFormat.CSV) is True
        assert manager.validate_format_support(ExportFormat.MARKDOWN) is True
        # Excel support depends on pandas availability
    
    def test_export_with_overwrite_existing(self):
        """Test export with overwrite existing files."""
        config = ExportConfiguration(
            format=ExportFormat.JSON,
            output_directory=self.temp_dir,
            filename_prefix="test_overwrite",
            include_timestamp=False,
            overwrite_existing=True
        )
        
        manager = ExportManager(config)
        
        # First export
        result1 = manager.export_data(self.test_data, config)
        assert result1.success is True
        
        # Second export with overwrite
        result2 = manager.export_data(self.test_data, config)
        assert result2.success is True
        assert result1.file_path == result2.file_path
    
    def test_export_error_handling(self):
        """Test export error handling."""
        config = ExportConfiguration(
            format=ExportFormat.JSON,
            output_directory=self.temp_dir,
            filename_prefix="test_error",
            overwrite_existing=True  # Allow overwrite to bypass file path validation
        )
        
        manager = ExportManager(config)
        
        # Should handle the error gracefully and return failed result
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            result = manager.export_data(self.test_data, config)
            assert result.success is False
            assert result.error_message is not None
            assert "Permission denied" in result.error_message


if __name__ == "__main__":
    pytest.main([__file__])