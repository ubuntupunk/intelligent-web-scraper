"""
Command-line interface for the Intelligent Web Scraper.

This module provides CLI commands that integrate with the atomic-agents ecosystem
and follow established patterns for tool discovery and usage.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .config import IntelligentScrapingConfig
from .agents.orchestrator import IntelligentScrapingOrchestrator
from .main import IntelligentScrapingApp


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="intelligent-web-scraper",
        description="Intelligent Web Scraper - An Advanced Atomic Agents Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  intelligent-web-scraper

  # Direct scraping with single URL
  intelligent-web-scraper --direct --url "https://example.com" --request "Extract all product names"

  # Batch scraping with multiple URLs
  intelligent-web-scraper --direct --urls "https://site1.com" "https://site2.com" --request "Extract titles"

  # Batch scraping from file
  intelligent-web-scraper --direct --urls-file urls.txt --request "Extract product information"

  # Batch processing from config file
  intelligent-web-scraper --config scraping_config.json

  # Show version and exit
  intelligent-web-scraper --version

For more information, visit: https://github.com/atomic-agents/intelligent-web-scraper
        """
    )
    
    # Version
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    # Interactive vs direct mode
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Run in interactive mode (default)"
    )
    
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Run in direct mode with provided parameters"
    )
    
    # Direct mode parameters
    parser.add_argument(
        "--url",
        type=str,
        help="Target URL to scrape"
    )
    
    parser.add_argument(
        "--urls",
        type=str,
        nargs="+",
        help="Multiple target URLs for batch scraping"
    )
    
    parser.add_argument(
        "--urls-file",
        type=str,
        help="File containing URLs (one per line) for batch scraping"
    )
    
    parser.add_argument(
        "--request",
        type=str,
        help="Natural language scraping request"
    )
    
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum number of results to extract (default: 10)"
    )
    
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=50.0,
        help="Quality threshold for extracted data (0-100, default: 50.0)"
    )
    
    parser.add_argument(
        "--export-format",
        choices=["json", "csv", "markdown", "excel"],
        default="json",
        help="Export format for results (default: json)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (default: ./results)"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (JSON format)"
    )
    
    parser.add_argument(
        "--env-file",
        type=str,
        help="Path to environment file (.env format)"
    )
    
    # Monitoring and performance
    parser.add_argument(
        "--enable-monitoring",
        action="store_true",
        default=True,
        help="Enable real-time monitoring (default: enabled)"
    )
    
    parser.add_argument(
        "--disable-monitoring",
        action="store_true",
        help="Disable real-time monitoring"
    )
    
    parser.add_argument(
        "--concurrent-instances",
        type=int,
        default=1,
        help="Number of concurrent scraper instances (default: 1)"
    )
    
    # Debugging and verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-essential output"
    )
    
    return parser


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")


def load_urls_from_file(file_path: str) -> list[str]:
    """Load URLs from a file (one per line)."""
    try:
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        return urls
    except FileNotFoundError:
        raise FileNotFoundError(f"URLs file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading URLs file: {e}")


def validate_url(url: str) -> bool:
    """Validate a single URL format."""
    return url.startswith('http://') or url.startswith('https://')


def get_target_urls(args: argparse.Namespace) -> list[str]:
    """Get target URLs from various sources."""
    urls = []
    
    # Single URL
    if args.url:
        urls.append(args.url)
    
    # Multiple URLs from command line
    if args.urls:
        urls.extend(args.urls)
    
    # URLs from file
    if args.urls_file:
        file_urls = load_urls_from_file(args.urls_file)
        urls.extend(file_urls)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    return unique_urls


def validate_direct_mode_args(args: argparse.Namespace) -> None:
    """Validate arguments for direct mode."""
    if args.direct:
        # Get target URLs
        target_urls = get_target_urls(args)
        
        if not target_urls:
            raise ValueError("At least one URL is required in direct mode (use --url, --urls, or --urls-file)")
        
        if not args.request:
            raise ValueError("--request is required in direct mode")
        
        # Validate URL formats
        for url in target_urls:
            if not validate_url(url):
                raise ValueError(f"Invalid URL format: {url} (must start with http:// or https://)")
        
        # Validate quality threshold
        if not (0 <= args.quality_threshold <= 100):
            raise ValueError("Quality threshold must be between 0 and 100")
        
        # Validate max results
        if not (1 <= args.max_results <= 10000):
            raise ValueError("Max results must be between 1 and 10000")


async def run_direct_mode(args: argparse.Namespace, config: IntelligentScrapingConfig) -> None:
    """Run scraping in direct mode with provided parameters."""
    console = Console()
    
    # Get target URLs
    target_urls = get_target_urls(args)
    
    if not args.quiet:
        console.print("[bold blue]🤖 Intelligent Web Scraper - Direct Mode[/bold blue]")
        if len(target_urls) == 1:
            console.print(f"[cyan]URL:[/cyan] {target_urls[0]}")
        else:
            console.print(f"[cyan]URLs:[/cyan] {len(target_urls)} URLs for batch processing")
        console.print(f"[cyan]Request:[/cyan] {args.request}")
        console.print()
    
    # Create orchestrator
    orchestrator = IntelligentScrapingOrchestrator(config=config)
    
    # Process each URL
    all_results = []
    successful_urls = 0
    failed_urls = 0
    total_items = 0
    
    for i, target_url in enumerate(target_urls, 1):
        try:
            if not args.quiet and len(target_urls) > 1:
                console.print(f"[blue]Processing URL {i}/{len(target_urls)}:[/blue] {target_url}")
            
            # Prepare request data
            request_data = {
                "scraping_request": args.request,
                "target_url": target_url,
                "max_results": args.max_results,
                "quality_threshold": args.quality_threshold,
                "export_format": args.export_format,
                "enable_monitoring": args.enable_monitoring and not args.disable_monitoring,
                "concurrent_instances": args.concurrent_instances,
                "batch_index": i,
                "batch_total": len(target_urls)
            }
            
            if not args.quiet:
                console.print("[yellow]🔄 Processing scraping request...[/yellow]")
            
            # Execute scraping
            result = await orchestrator.run(request_data)
            
            # Store result with URL info
            result.source_url = target_url
            result.batch_index = i
            all_results.append(result)
            successful_urls += 1
            
            # Count items
            items_count = len(result.extracted_data) if hasattr(result, 'extracted_data') and result.extracted_data else 0
            total_items += items_count
            
            if not args.quiet:
                if len(target_urls) == 1:
                    console.print("[green]✅ Scraping completed successfully![/green]")
                else:
                    console.print(f"[green]✅ URL {i} completed: {items_count} items extracted[/green]")
            
        except Exception as e:
            failed_urls += 1
            if not args.quiet:
                console.print(f"[red]❌ URL {i} failed: {str(e)}[/red]")
            continue
    
    # Display results summary
    if not args.quiet:
        if len(target_urls) > 1:
            # Batch summary
            console.print(f"\n[bold green]📊 Batch Results Summary[/bold green]")
            console.print(f"[cyan]URLs processed:[/cyan] {successful_urls}/{len(target_urls)}")
            console.print(f"[cyan]Total items extracted:[/cyan] {total_items}")
            console.print(f"[cyan]Success rate:[/cyan] {(successful_urls/len(target_urls)*100):.1f}%")
        else:
            # Single URL summary
            if all_results:
                result = all_results[0]
                items_count = len(result.extracted_data) if hasattr(result, 'extracted_data') and result.extracted_data else 0
                console.print(f"[cyan]Items extracted:[/cyan] {items_count}")
                
                if hasattr(result, 'quality_score'):
                    console.print(f"[cyan]Quality score:[/cyan] {result.quality_score:.1f}%")
        
        # Export files
        if all_results:
            console.print("[cyan]Export files:[/cyan]")
            for i, result in enumerate(all_results, 1):
                if hasattr(result, 'export_options') and result.export_options:
                    if len(target_urls) > 1:
                        console.print(f"  URL {i}:")
                    for format_type, path in result.export_options.items():
                        prefix = "    " if len(target_urls) > 1 else "  "
                        console.print(f"{prefix}• {format_type.upper()}: {path}")
    
    # Output JSON result for programmatic use
    if args.verbose or args.debug:
        if len(target_urls) == 1 and all_results:
            # Single URL output
            result = all_results[0]
            output_data = {
                "success": True,
                "url": target_urls[0],
                "items_extracted": len(result.extracted_data) if hasattr(result, 'extracted_data') and result.extracted_data else 0,
                "quality_score": result.quality_score if hasattr(result, 'quality_score') else None,
                "export_options": result.export_options if hasattr(result, 'export_options') else {},
                "metadata": result.metadata.__dict__ if hasattr(result, 'metadata') else {}
            }
            
            if args.debug:
                output_data["extracted_data"] = result.extracted_data if hasattr(result, 'extracted_data') else []
                output_data["scraping_plan"] = result.scraping_plan if hasattr(result, 'scraping_plan') else ""
                output_data["reasoning"] = result.reasoning if hasattr(result, 'reasoning') else ""
        else:
            # Batch output
            output_data = {
                "success": successful_urls > 0,
                "batch_summary": {
                    "total_urls": len(target_urls),
                    "successful_urls": successful_urls,
                    "failed_urls": failed_urls,
                    "total_items_extracted": total_items,
                    "success_rate": (successful_urls/len(target_urls)*100) if target_urls else 0
                },
                "results": []
            }
            
            for result in all_results:
                result_data = {
                    "url": result.source_url,
                    "batch_index": result.batch_index,
                    "items_extracted": len(result.extracted_data) if hasattr(result, 'extracted_data') and result.extracted_data else 0,
                    "quality_score": result.quality_score if hasattr(result, 'quality_score') else None,
                    "export_options": result.export_options if hasattr(result, 'export_options') else {}
                }
                
                if args.debug:
                    result_data["extracted_data"] = result.extracted_data if hasattr(result, 'extracted_data') else []
                    result_data["scraping_plan"] = result.scraping_plan if hasattr(result, 'scraping_plan') else ""
                    result_data["reasoning"] = result.reasoning if hasattr(result, 'reasoning') else ""
                
                output_data["results"].append(result_data)
        
        console.print_json(data=output_data)
    
    # Exit with error if all URLs failed
    if successful_urls == 0:
        if not args.quiet:
            console.print(f"[red]❌ All {len(target_urls)} URL(s) failed to process[/red]")
        sys.exit(1)


async def run_interactive_mode(args: argparse.Namespace, config: IntelligentScrapingConfig) -> None:
    """Run scraping in interactive mode."""
    app = IntelligentScrapingApp(config=config)
    await app.run_interactive()


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    console = Console()
    
    try:
        # Load configuration
        config_data = {}
        
        # Load from config file if provided
        if args.config:
            config_data.update(load_config_from_file(args.config))
        
        # Override with command line arguments
        if args.output_dir:
            config_data["results_directory"] = args.output_dir
        
        if args.disable_monitoring:
            config_data["enable_monitoring"] = False
        elif args.enable_monitoring:
            config_data["enable_monitoring"] = True
        
        # Create configuration
        if config_data:
            config = IntelligentScrapingConfig(**config_data)
        else:
            config = IntelligentScrapingConfig.from_env()
        
        # Determine mode
        if args.direct:
            validate_direct_mode_args(args)
            asyncio.run(run_direct_mode(args, config))
        else:
            asyncio.run(run_interactive_mode(args, config))
            
    except KeyboardInterrupt:
        if not args.quiet:
            console.print("\n[yellow]👋 Goodbye![/yellow]")
    except Exception as e:
        if not args.quiet:
            console.print(f"[red]❌ Error: {str(e)}[/red]")
        
        if args.debug:
            import traceback
            console.print("[red]Debug traceback:[/red]")
            console.print(traceback.format_exc())
        
        sys.exit(1)


if __name__ == "__main__":
    main()