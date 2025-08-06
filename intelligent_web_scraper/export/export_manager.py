"""Export manager for handling different export formats."""

import json
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from .export_formats import ExportFormat, ExportConfiguration, ExportData, ExportResult
from .export_validators import ExportValidator, ValidationError

logger = logging.getLogger(__name__)


class ExportManager:
    """Manages export operations for different formats."""
    
    def __init__(self, config: Optional[ExportConfiguration] = None):
        """Initialize export manager with configuration."""
        self.config = config or ExportConfiguration(format=ExportFormat.JSON)
        self.validator = ExportValidator()
    
    def export_data(self, data: ExportData, config: Optional[ExportConfiguration] = None) -> ExportResult:
        """Export data using specified configuration."""
        export_config = config or self.config
        
        # Validate configuration
        config_errors = self.validator.validate_export_configuration(export_config)
        if config_errors:
            raise ValidationError(f"Configuration validation failed: {'; '.join(config_errors)}")
        
        # Validate data
        data_errors = self.validator.validate_export_data(data)
        if data_errors:
            raise ValidationError(f"Data validation failed: {'; '.join(data_errors)}")
        
        # Validate format compatibility
        format_errors = self.validator.validate_export_format_compatibility(data, export_config.format)
        if format_errors:
            raise ValidationError(f"Format compatibility validation failed: {'; '.join(format_errors)}")
        
        # Generate file path
        file_path = self._generate_file_path(export_config)
        
        # Validate file path
        if not export_config.overwrite_existing:
            path_errors = self.validator.validate_file_path(file_path)
            if path_errors:
                raise ValidationError(f"File path validation failed: {'; '.join(path_errors)}")
        
        try:
            # Ensure output directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Export based on format
            if export_config.format == ExportFormat.JSON:
                records_exported = self._export_json(data, file_path, export_config)
            elif export_config.format == ExportFormat.CSV:
                records_exported = self._export_csv(data, file_path, export_config)
            elif export_config.format == ExportFormat.MARKDOWN:
                records_exported = self._export_markdown(data, file_path, export_config)
            elif export_config.format == ExportFormat.EXCEL:
                records_exported = self._export_excel(data, file_path, export_config)
            else:
                raise ValueError(f"Unsupported export format: {export_config.format}")
            
            # Validate file size after export
            size_errors = self.validator.validate_file_size(file_path)
            if size_errors:
                # Clean up the file if it's too large
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise ValidationError(f"File size validation failed: {'; '.join(size_errors)}")
            
            # Get file size
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            return ExportResult(
                success=True,
                file_path=file_path,
                format=export_config.format,
                records_exported=records_exported,
                file_size_bytes=file_size,
                export_time=datetime.utcnow(),
                metadata={
                    "configuration": export_config.model_dump(),
                    "data_metadata": data.metadata
                }
            )
        
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ExportResult(
                success=False,
                file_path=file_path,
                format=export_config.format,
                records_exported=0,
                file_size_bytes=0,
                export_time=datetime.utcnow(),
                error_message=str(e)
            )
    
    def _generate_file_path(self, config: ExportConfiguration) -> str:
        """Generate file path based on configuration."""
        # Create filename
        filename_parts = [config.filename_prefix]
        
        if config.include_timestamp:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename_parts.append(timestamp)
        
        filename = "_".join(filename_parts)
        
        # Add extension based on format
        extensions = {
            ExportFormat.JSON: ".json",
            ExportFormat.CSV: ".csv",
            ExportFormat.MARKDOWN: ".md",
            ExportFormat.EXCEL: ".xlsx"
        }
        
        filename += extensions[config.format]
        
        # Combine with output directory
        return os.path.join(config.output_directory, filename)
    
    def _export_json(self, data: ExportData, file_path: str, config: ExportConfiguration) -> int:
        """Export data to JSON format."""
        export_data = {
            "results": data.results,
            "export_info": {
                "timestamp": data.export_timestamp.isoformat(),
                "format": "json",
                "records_count": len(data.results)
            }
        }
        
        if config.include_metadata:
            export_data["metadata"] = data.metadata
            if data.schema_info:
                export_data["schema_info"] = data.schema_info
            if data.quality_metrics:
                export_data["quality_metrics"] = data.quality_metrics
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=config.json_indent, ensure_ascii=False, default=str)
        
        return len(data.results)
    
    def _export_csv(self, data: ExportData, file_path: str, config: ExportConfiguration) -> int:
        """Export data to CSV format."""
        if not data.results:
            return 0
        
        # Get all unique keys from all records
        all_keys = set()
        for result in data.results:
            all_keys.update(result.keys())
        
        fieldnames = sorted(all_keys)
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=fieldnames,
                delimiter=config.csv_delimiter,
                quotechar=config.csv_quote_char,
                quoting=csv.QUOTE_MINIMAL
            )
            
            writer.writeheader()
            
            for result in data.results:
                # Flatten nested data for CSV compatibility
                flattened_result = self._flatten_dict(result)
                writer.writerow(flattened_result)
        
        return len(data.results)
    
    def _export_markdown(self, data: ExportData, file_path: str, config: ExportConfiguration) -> int:
        """Export data to Markdown format."""
        with open(file_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("# Scraping Results\n\n")
            f.write(f"**Export Date:** {data.export_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Records Count:** {len(data.results)}\n\n")
            
            # Write metadata if included
            if config.include_metadata and data.metadata:
                f.write("## Metadata\n\n")
                for key, value in data.metadata.items():
                    f.write(f"- **{key}:** {value}\n")
                f.write("\n")
            
            # Write quality metrics if available
            if config.include_metadata and data.quality_metrics:
                f.write("## Quality Metrics\n\n")
                for metric, value in data.quality_metrics.items():
                    f.write(f"- **{metric}:** {value}\n")
                f.write("\n")
            
            # Write results
            if data.results:
                f.write("## Results\n\n")
                
                if config.markdown_table_format and len(data.results) > 0:
                    # Table format
                    headers = list(data.results[0].keys())
                    
                    # Write table header
                    f.write("| " + " | ".join(headers) + " |\n")
                    f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
                    
                    # Write table rows
                    for result in data.results:
                        row_values = []
                        for header in headers:
                            value = result.get(header, "")
                            # Escape markdown characters and handle newlines
                            if isinstance(value, str):
                                value = value.replace("|", "\\|").replace("\n", "<br>")
                            row_values.append(str(value))
                        f.write("| " + " | ".join(row_values) + " |\n")
                else:
                    # List format
                    for i, result in enumerate(data.results, 1):
                        f.write(f"### Record {i}\n\n")
                        for key, value in result.items():
                            f.write(f"**{key}:** {value}\n\n")
        
        return len(data.results)
    
    def _export_excel(self, data: ExportData, file_path: str, config: ExportConfiguration) -> int:
        """Export data to Excel format."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas and openpyxl are required for Excel export. Install with: pip install pandas openpyxl")
        
        # Convert data to DataFrame
        df = pd.DataFrame(data.results)
        
        # Create Excel writer
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Write main data
            df.to_excel(writer, sheet_name=config.excel_sheet_name, index=False)
            
            # Write metadata if included
            if config.include_metadata and (data.metadata or data.quality_metrics):
                metadata_data = []
                
                if data.metadata:
                    for key, value in data.metadata.items():
                        metadata_data.append({"Type": "Metadata", "Key": key, "Value": str(value)})
                
                if data.quality_metrics:
                    for key, value in data.quality_metrics.items():
                        metadata_data.append({"Type": "Quality Metric", "Key": key, "Value": str(value)})
                
                if metadata_data:
                    metadata_df = pd.DataFrame(metadata_data)
                    metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
            
            # Write export info
            export_info = pd.DataFrame([{
                "Export Date": data.export_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "Records Count": len(data.results),
                "Format": "Excel",
                "Sheet Name": config.excel_sheet_name
            }])
            export_info.to_excel(writer, sheet_name="Export Info", index=False)
        
        return len(data.results)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert list to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get_supported_formats(self) -> List[ExportFormat]:
        """Get list of supported export formats."""
        formats = [ExportFormat.JSON, ExportFormat.CSV, ExportFormat.MARKDOWN]
        if PANDAS_AVAILABLE:
            formats.append(ExportFormat.EXCEL)
        return formats
    
    def validate_format_support(self, format: ExportFormat) -> bool:
        """Check if a format is supported in current environment."""
        if format == ExportFormat.EXCEL:
            return PANDAS_AVAILABLE
        return format in [ExportFormat.JSON, ExportFormat.CSV, ExportFormat.MARKDOWN]