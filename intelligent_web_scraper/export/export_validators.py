"""Export validation functionality."""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from .export_formats import ExportFormat, ExportConfiguration, ExportData


class ValidationError(Exception):
    """Exception raised when export validation fails."""
    pass


class ExportValidator:
    """Validates export operations and data."""
    
    # Valid filename characters (excluding path separators and special chars)
    VALID_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9._-]+$')
    
    # Maximum filename length
    MAX_FILENAME_LENGTH = 255
    
    # Maximum file size (100MB)
    MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024
    
    @classmethod
    def validate_export_configuration(cls, config: ExportConfiguration) -> List[str]:
        """Validate export configuration and return list of validation errors."""
        errors = []
        
        # Validate output directory
        if not config.output_directory:
            errors.append("Output directory cannot be empty")
        else:
            try:
                Path(config.output_directory).resolve()
            except (OSError, ValueError) as e:
                errors.append(f"Invalid output directory path: {e}")
        
        # Validate filename prefix
        if not config.filename_prefix:
            errors.append("Filename prefix cannot be empty")
        elif not cls.VALID_FILENAME_PATTERN.match(config.filename_prefix):
            errors.append("Filename prefix contains invalid characters")
        elif len(config.filename_prefix) > cls.MAX_FILENAME_LENGTH - 50:  # Leave room for timestamp and extension
            errors.append(f"Filename prefix too long (max {cls.MAX_FILENAME_LENGTH - 50} characters)")
        
        # Validate format-specific options
        if config.format == ExportFormat.CSV:
            if len(config.csv_delimiter) != 1:
                errors.append("CSV delimiter must be a single character")
            if len(config.csv_quote_char) != 1:
                errors.append("CSV quote character must be a single character")
        
        elif config.format == ExportFormat.EXCEL:
            if not config.excel_sheet_name:
                errors.append("Excel sheet name cannot be empty")
            elif len(config.excel_sheet_name) > 31:  # Excel sheet name limit
                errors.append("Excel sheet name too long (max 31 characters)")
        
        elif config.format == ExportFormat.JSON:
            if config.json_indent < 0 or config.json_indent > 10:
                errors.append("JSON indent must be between 0 and 10")
        
        return errors
    
    @classmethod
    def validate_export_data(cls, data: ExportData) -> List[str]:
        """Validate export data and return list of validation errors."""
        errors = []
        
        # Check if results exist
        if not data.results:
            errors.append("No results to export")
        
        # Validate data structure consistency
        if data.results:
            first_keys = set(data.results[0].keys()) if data.results else set()
            for i, result in enumerate(data.results[1:], 1):
                if set(result.keys()) != first_keys:
                    errors.append(f"Inconsistent data structure at record {i}")
                    break
        
        # Check for empty or None values in critical fields
        critical_fields = ['url', 'title', 'content']  # Common critical fields
        for i, result in enumerate(data.results):
            for field in critical_fields:
                if field in result and not result[field]:
                    errors.append(f"Empty critical field '{field}' at record {i}")
        
        return errors
    
    @classmethod
    def validate_file_path(cls, file_path: str) -> List[str]:
        """Validate file path and return list of validation errors."""
        errors = []
        
        try:
            path = Path(file_path)
            
            # Check if parent directory exists or can be created
            parent_dir = path.parent
            if not parent_dir.exists():
                try:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                except (OSError, PermissionError) as e:
                    errors.append(f"Cannot create output directory: {e}")
            
            # Check filename length
            if len(path.name) > cls.MAX_FILENAME_LENGTH:
                errors.append(f"Filename too long (max {cls.MAX_FILENAME_LENGTH} characters)")
            
            # Check for valid filename characters
            if not cls.VALID_FILENAME_PATTERN.match(path.stem) and path.stem:
                errors.append("Filename contains invalid characters")
            
            # Check if file already exists (if overwrite is not allowed)
            if path.exists():
                errors.append(f"File already exists: {file_path}")
            
            # Check write permissions
            try:
                # Try to create a temporary file to test permissions
                temp_file = parent_dir / f".temp_write_test_{os.getpid()}"
                temp_file.touch()
                temp_file.unlink()
            except (OSError, PermissionError) as e:
                errors.append(f"No write permission for directory: {e}")
        
        except (OSError, ValueError) as e:
            errors.append(f"Invalid file path: {e}")
        
        return errors
    
    @classmethod
    def validate_file_size(cls, file_path: str) -> List[str]:
        """Validate file size after export."""
        errors = []
        
        try:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > cls.MAX_FILE_SIZE_BYTES:
                    errors.append(f"Exported file too large: {file_size} bytes (max {cls.MAX_FILE_SIZE_BYTES})")
                elif file_size == 0:
                    errors.append("Exported file is empty")
        except OSError as e:
            errors.append(f"Cannot check file size: {e}")
        
        return errors
    
    @classmethod
    def validate_export_format_compatibility(cls, data: ExportData, format: ExportFormat) -> List[str]:
        """Validate data compatibility with export format."""
        errors = []
        
        if not data.results:
            return ["No data to validate"]
        
        # Check format-specific requirements
        if format == ExportFormat.CSV:
            # CSV requires flat data structure
            for i, result in enumerate(data.results):
                for key, value in result.items():
                    if isinstance(value, (dict, list)):
                        errors.append(f"CSV format doesn't support nested data at record {i}, field '{key}'")
        
        elif format == ExportFormat.EXCEL:
            # Excel has column name restrictions
            if data.results:
                for key in data.results[0].keys():
                    if len(key) > 255:  # Excel column name limit
                        errors.append(f"Column name too long for Excel: '{key[:50]}...'")
                    if key.startswith('[') or key.endswith(']'):
                        errors.append(f"Excel doesn't support column names with brackets: '{key}'")
        
        return errors