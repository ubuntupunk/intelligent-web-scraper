"""Export format definitions and enums."""

from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ExportFormat(str, Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    EXCEL = "excel"


class ExportResult(BaseModel):
    """Result of an export operation."""
    
    success: bool = Field(..., description="Whether the export was successful")
    file_path: str = Field(..., description="Path to the exported file")
    format: ExportFormat = Field(..., description="Export format used")
    records_exported: int = Field(..., description="Number of records exported")
    file_size_bytes: int = Field(..., description="Size of exported file in bytes")
    export_time: datetime = Field(..., description="Timestamp when export was completed")
    error_message: Optional[str] = Field(None, description="Error message if export failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional export metadata")


class ExportConfiguration(BaseModel):
    """Configuration for export operations."""
    
    format: ExportFormat = Field(..., description="Export format")
    output_directory: str = Field("./exports", description="Directory for exported files")
    filename_prefix: str = Field("scraping_results", description="Prefix for exported filenames")
    include_timestamp: bool = Field(True, description="Include timestamp in filename")
    include_metadata: bool = Field(True, description="Include metadata in export")
    overwrite_existing: bool = Field(False, description="Overwrite existing files")
    
    # Format-specific options
    csv_delimiter: str = Field(",", description="CSV delimiter character")
    csv_quote_char: str = Field('"', description="CSV quote character")
    excel_sheet_name: str = Field("Scraping Results", description="Excel sheet name")
    json_indent: int = Field(2, description="JSON indentation")
    markdown_table_format: bool = Field(True, description="Use table format for Markdown")


class ExportData(BaseModel):
    """Data structure for export operations."""
    
    results: List[Dict[str, Any]] = Field(..., description="Scraped data results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Export metadata")
    schema_info: Optional[Dict[str, Any]] = Field(None, description="Schema information")
    quality_metrics: Optional[Dict[str, float]] = Field(None, description="Quality metrics")
    export_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Export timestamp")