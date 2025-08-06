#!/usr/bin/env python3
"""
Example demonstrating the export functionality of the intelligent web scraper.

This example shows how to use the ExportManager to export scraped data
in different formats (JSON, CSV, Markdown, Excel).
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

from intelligent_web_scraper.export import (
    ExportManager,
    ExportFormat,
    ExportConfiguration,
    ExportData,
    ValidationError
)


def create_sample_data() -> ExportData:
    """Create sample scraped data for demonstration."""
    sample_results = [
        {
            "title": "Advanced Web Scraping Techniques",
            "url": "https://example.com/article1",
            "content": "Learn about modern web scraping approaches using AI and machine learning.",
            "author": "John Doe",
            "publish_date": "2023-12-01",
            "tags": "web-scraping,ai,automation",
            "word_count": 1250,
            "quality_score": 95.5
        },
        {
            "title": "Building Scalable Data Pipelines",
            "url": "https://example.com/article2", 
            "content": "Discover how to build robust data processing pipelines for large-scale operations.",
            "author": "Jane Smith",
            "publish_date": "2023-12-02",
            "tags": "data-engineering,scalability,pipelines",
            "word_count": 2100,
            "quality_score": 88.2
        },
        {
            "title": "Machine Learning for Content Analysis",
            "url": "https://example.com/article3",
            "content": "Explore ML techniques for analyzing and categorizing web content automatically.",
            "author": "Bob Johnson",
            "publish_date": "2023-12-03",
            "tags": "machine-learning,nlp,content-analysis",
            "word_count": 1800,
            "quality_score": 92.1
        }
    ]
    
    metadata = {
        "scraping_session_id": "session_123",
        "target_website": "example.com",
        "scraping_strategy": "intelligent_extraction",
        "total_pages_processed": 3,
        "scraping_duration_seconds": 45.2,
        "user_agent": "IntelligentScraper/1.0"
    }
    
    quality_metrics = {
        "average_quality_score": 91.9,
        "content_completeness": 0.95,
        "data_accuracy": 0.92,
        "extraction_success_rate": 1.0,
        "duplicate_detection_rate": 0.0
    }
    
    return ExportData(
        results=sample_results,
        metadata=metadata,
        quality_metrics=quality_metrics
    )


def demonstrate_json_export(data: ExportData, output_dir: str):
    """Demonstrate JSON export functionality."""
    print("\n=== JSON Export Demo ===")
    
    config = ExportConfiguration(
        format=ExportFormat.JSON,
        output_directory=output_dir,
        filename_prefix="scraped_articles_json",
        include_timestamp=True,
        include_metadata=True,
        json_indent=2
    )
    
    manager = ExportManager(config)
    result = manager.export_data(data)
    
    if result.success:
        print(f"✅ JSON export successful!")
        print(f"   File: {result.file_path}")
        print(f"   Records: {result.records_exported}")
        print(f"   Size: {result.file_size_bytes} bytes")
        
        # Show a snippet of the exported file
        with open(result.file_path, 'r') as f:
            content = f.read()[:500]
            print(f"   Preview: {content}...")
    else:
        print(f"❌ JSON export failed: {result.error_message}")


def demonstrate_csv_export(data: ExportData, output_dir: str):
    """Demonstrate CSV export functionality."""
    print("\n=== CSV Export Demo ===")
    
    config = ExportConfiguration(
        format=ExportFormat.CSV,
        output_directory=output_dir,
        filename_prefix="scraped_articles_csv",
        include_timestamp=True,
        csv_delimiter=",",
        csv_quote_char='"'
    )
    
    manager = ExportManager(config)
    result = manager.export_data(data)
    
    if result.success:
        print(f"✅ CSV export successful!")
        print(f"   File: {result.file_path}")
        print(f"   Records: {result.records_exported}")
        print(f"   Size: {result.file_size_bytes} bytes")
        
        # Show the header and first row
        with open(result.file_path, 'r') as f:
            lines = f.readlines()[:2]
            print(f"   Header: {lines[0].strip()}")
            if len(lines) > 1:
                print(f"   First row: {lines[1].strip()[:100]}...")
    else:
        print(f"❌ CSV export failed: {result.error_message}")


def demonstrate_markdown_export(data: ExportData, output_dir: str):
    """Demonstrate Markdown export functionality."""
    print("\n=== Markdown Export Demo ===")
    
    config = ExportConfiguration(
        format=ExportFormat.MARKDOWN,
        output_directory=output_dir,
        filename_prefix="scraped_articles_md",
        include_timestamp=True,
        include_metadata=True,
        markdown_table_format=True
    )
    
    manager = ExportManager(config)
    result = manager.export_data(data)
    
    if result.success:
        print(f"✅ Markdown export successful!")
        print(f"   File: {result.file_path}")
        print(f"   Records: {result.records_exported}")
        print(f"   Size: {result.file_size_bytes} bytes")
        
        # Show a snippet
        with open(result.file_path, 'r') as f:
            lines = f.readlines()[:10]
            print("   Preview:")
            for line in lines:
                print(f"     {line.rstrip()}")
    else:
        print(f"❌ Markdown export failed: {result.error_message}")


def demonstrate_excel_export(data: ExportData, output_dir: str):
    """Demonstrate Excel export functionality."""
    print("\n=== Excel Export Demo ===")
    
    try:
        config = ExportConfiguration(
            format=ExportFormat.EXCEL,
            output_directory=output_dir,
            filename_prefix="scraped_articles_excel",
            include_timestamp=True,
            include_metadata=True,
            excel_sheet_name="Scraped Articles"
        )
        
        manager = ExportManager(config)
        
        # Check if Excel export is supported
        if not manager.validate_format_support(ExportFormat.EXCEL):
            print("❌ Excel export not supported (pandas/openpyxl not available)")
            return
        
        result = manager.export_data(data)
        
        if result.success:
            print(f"✅ Excel export successful!")
            print(f"   File: {result.file_path}")
            print(f"   Records: {result.records_exported}")
            print(f"   Size: {result.file_size_bytes} bytes")
            print(f"   Sheet: {config.excel_sheet_name}")
        else:
            print(f"❌ Excel export failed: {result.error_message}")
            
    except ImportError as e:
        print(f"❌ Excel export not available: {e}")


def demonstrate_validation_errors():
    """Demonstrate validation error handling."""
    print("\n=== Validation Error Demo ===")
    
    # Try to export with invalid configuration
    invalid_config = ExportConfiguration(
        format=ExportFormat.JSON,
        output_directory="",  # Invalid empty directory
        filename_prefix=""    # Invalid empty prefix
    )
    
    sample_data = ExportData(results=[])  # Empty data
    
    manager = ExportManager(invalid_config)
    
    try:
        result = manager.export_data(sample_data)
        print("❌ Expected validation error but export succeeded")
    except ValidationError as e:
        print(f"✅ Validation error caught as expected: {e}")


def main():
    """Main demonstration function."""
    print("🚀 Intelligent Web Scraper - Export Functionality Demo")
    print("=" * 60)
    
    # Create temporary directory for exports
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Create sample data
        sample_data = create_sample_data()
        print(f"📊 Created sample data with {len(sample_data.results)} records")
        
        # Demonstrate different export formats
        demonstrate_json_export(sample_data, temp_dir)
        demonstrate_csv_export(sample_data, temp_dir)
        demonstrate_markdown_export(sample_data, temp_dir)
        demonstrate_excel_export(sample_data, temp_dir)
        
        # Demonstrate validation
        demonstrate_validation_errors()
        
        # Show all exported files
        print(f"\n📋 All exported files in {temp_dir}:")
        for file_path in Path(temp_dir).glob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"   {file_path.name} ({size} bytes)")
        
        print("\n✨ Export functionality demonstration complete!")
        print("Press Enter to continue (files will be cleaned up automatically)...")
        input()


if __name__ == "__main__":
    main()