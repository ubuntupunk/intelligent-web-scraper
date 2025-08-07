"""
Unit tests for export validation functionality.

This module tests the ExportValidator class and validation methods.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from intelligent_web_scraper.export.export_validators import (
    ExportValidator,
    ValidationError
)
from intelligent_web_scraper.export.export_formats import (
    ExportFormat,
    ExportConfiguration,
    ExportData
)


class TestExportValidator:
    """Test the ExportValidator class."""
    
    def test_validator_constants(self):
        """Test that validator constants are properly defined."""
        assert hasattr(ExportValidator, 'VALID_FILENAME_PATTERN')
        assert hasattr(ExportValidator, 'MAX_FILENAME_LENGTH')
        assert hasattr(ExportValidator, 'MAX_FILE_SIZE_BYTES')
        
        assert ExportValidator.MAX_FILENAME_LENGTH == 255
        assert ExportValidator.MAX_FILE_SIZE_BYTES == 100 * 1024 * 1024  # 100MB
    
    def test_valid_filename_pattern(self):
        """Test the valid filename pattern regex."""
        pattern = ExportValidator.VALID_FILENAME_PATTERN
        
        # Valid filenames
        assert pattern.match("valid_filename")
        assert pattern.match("file123")
        assert pattern.match("my-file.txt")
        assert pattern.match("data_export_2023")
        assert pattern.match("file.with.dots")
        assert pattern.match("file-with-dashes")
        assert pattern.match("file_with_underscores")
        
        # Invalid filenames
        assert not pattern.match("file with spaces")
        assert not pattern.match("file/with/slashes")
        assert not pattern.match("file\\with\\backslashes")
        assert not pattern.match("file:with:colons")
        assert not pattern.match("file*with*asterisks")
        assert not pattern.match("file?with?questions")
        assert not pattern.match("file<with>brackets")
        assert not pattern.match("file|with|pipes")
        assert not pattern.match("")  # Empty string
    
    def test_validate_export_configuration_valid(self):
        """Test validation of valid export configuration."""
        config = ExportConfiguration(
            format=ExportFormat.JSON,
            output_directory="./test_output",
            filename_prefix="test_export",
            include_metadata=True,
            json_indent=2
        )
        
        errors = ExportValidator.validate_export_configuration(config)
        
        assert isinstance(errors, list)
        assert len(errors) == 0
    
    def test_validate_export_configuration_empty_directory(self):
        """Test validation with empty output directory."""
        config = ExportConfiguration(
            format=ExportFormat.JSON,
            output_directory="",
            filename_prefix="test_export"
        )
        
        errors = ExportValidator.validate_export_configuration(config)
        
        assert len(errors) > 0
        assert any("Output directory cannot be empty" in error for error in errors)
    
    def test_validate_export_configuration_invalid_directory(self):
        """Test validation with invalid output directory."""
        config = ExportConfiguration(
            format=ExportFormat.JSON,
            output_directory="\x00invalid\x00path",  # Null bytes are invalid
            filename_prefix="test_export"
        )
        
        errors = ExportValidator.validate_export_configuration(config)
        
        assert len(errors) > 0
        assert any("Invalid output directory path" in error for error in errors)
    
    def test_validate_export_configuration_empty_prefix(self):
        """Test validation with empty filename prefix."""
        config = ExportConfiguration(
            format=ExportFormat.JSON,
            output_directory="./test_output",
            filename_prefix=""
        )
        
        errors = ExportValidator.validate_export_configuration(config)
        
        assert len(errors) > 0
        assert any("Filename prefix cannot be empty" in error for error in errors)
    
    def test_validate_export_configuration_invalid_prefix(self):
        """Test validation with invalid filename prefix."""
        config = ExportConfiguration(
            format=ExportFormat.JSON,
            output_directory="./test_output",
            filename_prefix="invalid filename with spaces"
        )
        
        errors = ExportValidator.validate_export_configuration(config)
        
        assert len(errors) > 0
        assert any("Filename prefix contains invalid characters" in error for error in errors)
    
    def test_validate_export_configuration_long_prefix(self):
        """Test validation with overly long filename prefix."""
        long_prefix = "a" * (ExportValidator.MAX_FILENAME_LENGTH - 40)  # Too long
        config = ExportConfiguration(
            format=ExportFormat.JSON,
            output_directory="./test_output",
            filename_prefix=long_prefix
        )
        
        errors = ExportValidator.validate_export_configuration(config)
        
        assert len(errors) > 0
        assert any("Filename prefix too long" in error for error in errors)
    
    def test_validate_export_configuration_csv_options(self):
        """Test validation of CSV-specific options."""
        # Valid CSV configuration
        config = ExportConfiguration(
            format=ExportFormat.CSV,
            output_directory="./test_output",
            filename_prefix="test_export",
            csv_delimiter=",",
            csv_quote_char='"'
        )
        
        errors = ExportValidator.validate_export_configuration(config)
        assert len(errors) == 0
        
        # Invalid CSV delimiter (multiple characters)
        config.csv_delimiter = ",;"
        errors = ExportValidator.validate_export_configuration(config)
        assert any("CSV delimiter must be a single character" in error for error in errors)
        
        # Invalid CSV quote character (multiple characters)
        config.csv_delimiter = ","
        config.csv_quote_char = '""'
        errors = ExportValidator.validate_export_configuration(config)
        assert any("CSV quote character must be a single character" in error for error in errors)
    
    def test_validate_export_configuration_excel_options(self):
        """Test validation of Excel-specific options."""
        # Valid Excel configuration
        config = ExportConfiguration(
            format=ExportFormat.EXCEL,
            output_directory="./test_output",
            filename_prefix="test_export",
            excel_sheet_name="Data"
        )
        
        errors = ExportValidator.validate_export_configuration(config)
        assert len(errors) == 0
        
        # Empty Excel sheet name
        config.excel_sheet_name = ""
        errors = ExportValidator.validate_export_configuration(config)
        assert any("Excel sheet name cannot be empty" in error for error in errors)
        
        # Excel sheet name too long
        config.excel_sheet_name = "a" * 32  # Excel limit is 31 characters
        errors = ExportValidator.validate_export_configuration(config)
        assert any("Excel sheet name too long" in error for error in errors)
    
    def test_validate_export_configuration_json_options(self):
        """Test validation of JSON-specific options."""
        # Valid JSON configuration
        config = ExportConfiguration(
            format=ExportFormat.JSON,
            output_directory="./test_output",
            filename_prefix="test_export",
            json_indent=4
        )
        
        errors = ExportValidator.validate_export_configuration(config)
        assert len(errors) == 0
        
        # Invalid JSON indent (negative)
        config.json_indent = -1
        errors = ExportValidator.validate_export_configuration(config)
        assert any("JSON indent must be between 0 and 10" in error for error in errors)
        
        # Invalid JSON indent (too large)
        config.json_indent = 15
        errors = ExportValidator.validate_export_configuration(config)
        assert any("JSON indent must be between 0 and 10" in error for error in errors)
    
    def test_validate_export_data_valid(self):
        """Test validation of valid export data."""
        data = ExportData(
            results=[
                {"title": "Item 1", "url": "http://example.com/1", "content": "Content 1"},
                {"title": "Item 2", "url": "http://example.com/2", "content": "Content 2"},
                {"title": "Item 3", "url": "http://example.com/3", "content": "Content 3"}
            ],
            metadata={"total_items": 3, "source": "test"}
        )
        
        errors = ExportValidator.validate_export_data(data)
        
        assert isinstance(errors, list)
        assert len(errors) == 0
    
    def test_validate_export_data_empty_results(self):
        """Test validation with empty results."""
        data = ExportData(
            results=[],
            metadata={"total_items": 0}
        )
        
        errors = ExportValidator.validate_export_data(data)
        
        assert len(errors) > 0
        assert any("No results to export" in error for error in errors)
    
    def test_validate_export_data_inconsistent_structure(self):
        """Test validation with inconsistent data structure."""
        data = ExportData(
            results=[
                {"title": "Item 1", "url": "http://example.com/1", "content": "Content 1"},
                {"title": "Item 2", "description": "Different structure"},  # Missing url, content; has description
                {"title": "Item 3", "url": "http://example.com/3", "content": "Content 3"}
            ],
            metadata={"total_items": 3}
        )
        
        errors = ExportValidator.validate_export_data(data)
        
        assert len(errors) > 0
        assert any("Inconsistent data structure" in error for error in errors)
    
    def test_validate_export_data_empty_critical_fields(self):
        """Test validation with empty critical fields."""
        data = ExportData(
            results=[
                {"title": "Item 1", "url": "http://example.com/1", "content": "Content 1"},
                {"title": "", "url": "http://example.com/2", "content": "Content 2"},  # Empty title
                {"title": "Item 3", "url": "", "content": "Content 3"}  # Empty URL
            ],
            metadata={"total_items": 3}
        )
        
        errors = ExportValidator.validate_export_data(data)
        
        assert len(errors) > 0
        assert any("Empty critical field 'title'" in error for error in errors)
        assert any("Empty critical field 'url'" in error for error in errors)
    
    def test_validate_file_path_valid(self):
        """Test validation of valid file path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_export.json")
            
            errors = ExportValidator.validate_file_path(file_path)
            
            assert len(errors) == 0
    
    def test_validate_file_path_nonexistent_directory(self):
        """Test validation with non-existent directory that can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "new_subdir", "test_export.json")
            
            errors = ExportValidator.validate_file_path(file_path)
            
            # Should succeed because directory can be created
            assert len(errors) == 0
            
            # Directory should have been created
            assert os.path.exists(os.path.dirname(file_path))
    
    def test_validate_file_path_invalid_directory(self):
        """Test validation with invalid directory path."""
        # Use a path that cannot be created (permission denied or invalid)
        if os.name == 'nt':  # Windows
            invalid_path = "C:\\invalid\\path\\that\\cannot\\be\\created\\file.json"
        else:  # Unix-like
            invalid_path = "/root/invalid/path/file.json"  # Assuming no root access
        
        errors = ExportValidator.validate_file_path(invalid_path)
        
        # May have errors depending on system permissions
        # This test is system-dependent, so we just check it doesn't crash
        assert isinstance(errors, list)
    
    def test_validate_file_path_long_filename(self):
        """Test validation with overly long filename."""
        with tempfile.TemporaryDirectory() as temp_dir:
            long_filename = "a" * (ExportValidator.MAX_FILENAME_LENGTH + 1) + ".json"
            file_path = os.path.join(temp_dir, long_filename)
            
            errors = ExportValidator.validate_file_path(file_path)
            
            assert len(errors) > 0
            assert any("Filename too long" in error for error in errors)
    
    def test_validate_file_path_invalid_characters(self):
        """Test validation with invalid filename characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use filename with spaces (invalid according to the pattern)
            invalid_filename = "file with spaces.json"
            file_path = os.path.join(temp_dir, invalid_filename)
            
            errors = ExportValidator.validate_file_path(file_path)
            
            assert len(errors) > 0
            assert any("Filename contains invalid characters" in error for error in errors)
    
    def test_validate_file_path_existing_file(self):
        """Test validation with existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            try:
                errors = ExportValidator.validate_file_path(temp_file.name)
                
                assert len(errors) > 0
                assert any("File already exists" in error for error in errors)
            finally:
                os.unlink(temp_file.name)
    
    def test_validate_file_size_valid(self):
        """Test validation of valid file size."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            try:
                # Write some data to the file
                temp_file.write(b"Test data for file size validation")
                temp_file.flush()
                
                errors = ExportValidator.validate_file_size(temp_file.name)
                
                assert len(errors) == 0
            finally:
                os.unlink(temp_file.name)
    
    def test_validate_file_size_empty_file(self):
        """Test validation of empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            try:
                # File is empty by default
                temp_file.flush()
                
                errors = ExportValidator.validate_file_size(temp_file.name)
                
                assert len(errors) > 0
                assert any("Exported file is empty" in error for error in errors)
            finally:
                os.unlink(temp_file.name)
    
    def test_validate_file_size_nonexistent_file(self):
        """Test validation of non-existent file."""
        nonexistent_path = "/path/that/does/not/exist/file.json"
        
        errors = ExportValidator.validate_file_size(nonexistent_path)
        
        # Should not raise errors for non-existent files (they haven't been created yet)
        assert len(errors) == 0
    
    @patch('os.path.getsize')
    def test_validate_file_size_too_large(self, mock_getsize):
        """Test validation of file that's too large."""
        # Mock file size to be larger than limit
        mock_getsize.return_value = ExportValidator.MAX_FILE_SIZE_BYTES + 1
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            try:
                errors = ExportValidator.validate_file_size(temp_file.name)
                
                assert len(errors) > 0
                assert any("Exported file too large" in error for error in errors)
            finally:
                os.unlink(temp_file.name)
    
    def test_validate_export_format_compatibility_json(self):
        """Test format compatibility validation for JSON."""
        # JSON supports nested data
        data = ExportData(
            results=[
                {
                    "title": "Item 1",
                    "metadata": {"tags": ["tag1", "tag2"], "nested": {"key": "value"}},
                    "items": [1, 2, 3]
                }
            ]
        )
        
        errors = ExportValidator.validate_export_format_compatibility(data, ExportFormat.JSON)
        
        assert len(errors) == 0
    
    def test_validate_export_format_compatibility_csv_nested_data(self):
        """Test format compatibility validation for CSV with nested data."""
        # CSV doesn't support nested data
        data = ExportData(
            results=[
                {
                    "title": "Item 1",
                    "metadata": {"tags": ["tag1", "tag2"]},  # Nested dict
                    "items": [1, 2, 3]  # List
                }
            ]
        )
        
        errors = ExportValidator.validate_export_format_compatibility(data, ExportFormat.CSV)
        
        assert len(errors) > 0
        assert any("CSV format doesn't support nested data" in error for error in errors)
    
    def test_validate_export_format_compatibility_csv_flat_data(self):
        """Test format compatibility validation for CSV with flat data."""
        # CSV supports flat data
        data = ExportData(
            results=[
                {"title": "Item 1", "url": "http://example.com/1", "score": 85.5},
                {"title": "Item 2", "url": "http://example.com/2", "score": 92.0}
            ]
        )
        
        errors = ExportValidator.validate_export_format_compatibility(data, ExportFormat.CSV)
        
        assert len(errors) == 0
    
    def test_validate_export_format_compatibility_excel_long_column_names(self):
        """Test format compatibility validation for Excel with long column names."""
        # Excel has column name length restrictions
        long_column_name = "a" * 256  # Longer than Excel's 255 character limit
        data = ExportData(
            results=[
                {"title": "Item 1", long_column_name: "value"}
            ]
        )
        
        errors = ExportValidator.validate_export_format_compatibility(data, ExportFormat.EXCEL)
        
        assert len(errors) > 0
        assert any("Column name too long for Excel" in error for error in errors)
    
    def test_validate_export_format_compatibility_excel_bracket_column_names(self):
        """Test format compatibility validation for Excel with bracket column names."""
        # Excel doesn't support column names with brackets
        data = ExportData(
            results=[
                {"title": "Item 1", "[bracketed_column]": "value"}
            ]
        )
        
        errors = ExportValidator.validate_export_format_compatibility(data, ExportFormat.EXCEL)
        
        assert len(errors) > 0
        assert any("Excel doesn't support column names with brackets" in error for error in errors)
    
    def test_validate_export_format_compatibility_empty_data(self):
        """Test format compatibility validation with empty data."""
        data = ExportData(results=[])
        
        errors = ExportValidator.validate_export_format_compatibility(data, ExportFormat.JSON)
        
        assert len(errors) > 0
        assert any("No data to validate" in error for error in errors)


class TestValidationError:
    """Test the ValidationError exception."""
    
    def test_validation_error_creation(self):
        """Test ValidationError exception creation."""
        error = ValidationError("Test validation error")
        
        assert isinstance(error, Exception)
        assert str(error) == "Test validation error"
    
    def test_validation_error_inheritance(self):
        """Test ValidationError inheritance."""
        error = ValidationError("Test error")
        
        assert isinstance(error, Exception)
        assert issubclass(ValidationError, Exception)


if __name__ == "__main__":
    pytest.main([__file__])