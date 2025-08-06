# Export Functionality

The Intelligent Web Scraper includes comprehensive export functionality that supports multiple formats and provides robust validation and error handling.

## Supported Formats

- **JSON**: Structured data with metadata and quality metrics
- **CSV**: Tabular format with automatic data flattening
- **Markdown**: Human-readable format with tables and metadata
- **Excel**: Multi-sheet workbooks with data, metadata, and export info

## Key Features

### Export Manager
- Centralized export management with configuration-based approach
- Automatic format validation and compatibility checking
- Comprehensive error handling and graceful degradation
- File path management with timestamp support

### Export Validation
- Configuration validation (directories, filenames, format-specific options)
- Data structure validation and consistency checking
- File path validation with permission checking
- Format compatibility validation (e.g., CSV doesn't support nested data)

### Export Configuration
- Flexible configuration with format-specific options
- Support for custom output directories and filename patterns
- Metadata inclusion control
- Overwrite protection with optional override

## Usage Example

```python
from intelligent_web_scraper.export import (
    ExportManager, 
    ExportFormat, 
    ExportConfiguration, 
    ExportData
)

# Create export configuration
config = ExportConfiguration(
    format=ExportFormat.JSON,
    output_directory="./exports",
    filename_prefix="scraping_results",
    include_timestamp=True,
    include_metadata=True
)

# Prepare data for export
data = ExportData(
    results=[
        {"title": "Article 1", "url": "http://example.com/1"},
        {"title": "Article 2", "url": "http://example.com/2"}
    ],
    metadata={"source": "example.com", "date": "2023-12-01"},
    quality_metrics={"accuracy": 0.95, "completeness": 0.88}
)

# Export data
manager = ExportManager(config)
result = manager.export_data(data)

if result.success:
    print(f"Export successful: {result.file_path}")
    print(f"Records exported: {result.records_exported}")
else:
    print(f"Export failed: {result.error_message}")
```

## Format-Specific Features

### JSON Export
- Pretty-printed with configurable indentation
- Includes complete metadata and quality metrics
- Preserves nested data structures
- UTF-8 encoding support

### CSV Export
- Automatic flattening of nested data structures
- Configurable delimiters and quote characters
- Header row with all field names
- Handles missing fields gracefully

### Markdown Export
- Table format or list format options
- Metadata and quality metrics sections
- Proper escaping of special characters
- Human-readable structure

### Excel Export
- Multiple sheets (data, metadata, export info)
- Proper column formatting
- Metadata preservation
- Requires pandas and openpyxl dependencies

## Validation and Error Handling

The export system includes comprehensive validation:

1. **Configuration Validation**: Checks output directories, filename patterns, and format-specific options
2. **Data Validation**: Ensures data consistency and completeness
3. **Format Compatibility**: Validates data structure compatibility with target format
4. **File System Validation**: Checks permissions, file sizes, and path validity

All validation errors are collected and reported with actionable error messages.

## Dependencies

- **Core**: No additional dependencies for JSON, CSV, and Markdown
- **Excel**: Requires `pandas` and `openpyxl` packages
- **Validation**: Uses built-in Python libraries for file system operations

## Testing

The export functionality includes comprehensive unit tests covering:
- All export formats
- Validation scenarios
- Error handling
- Configuration management
- File path generation
- Data flattening for CSV

Run tests with:
```bash
poetry run pytest tests/test_export_functionality.py -v
```

## Integration

The export functionality is designed to integrate seamlessly with the intelligent web scraper's orchestrator and can be used independently for any data export needs.