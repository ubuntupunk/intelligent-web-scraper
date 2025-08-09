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

  # Direct scraping with parameters
  intelligent-web-scraper --url "https://example.com" --request "Extract all product names"

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


def validate_direct_mode_args(args: argparse.Namespace) -> None:
    """Validate arguments for direct mode."""
    if args.direct:
        if not args.url:
            raise ValueError("--url is required in direct mode")
        if not args.request:
            raise ValueError("--request is required in direct mode")
        
        # Validate URL format
        if not (args.url.startswith('http://') or args.url.startswith('https://')):
            raise ValueError("URL must start with http:// or https://")
        
        # Validate quality threshold
        if not (0 <= args.quality_threshold <= 100):
            raise ValueError("Quality threshold must be between 0 and 100")
        
        # Validate max results
        if not (1 <= args.max_results <= 10000):
            raise ValueError("Max results must be between 1 and 10000")


async def run_direct_mode(args: argparse.Namespace, config: IntelligentScrapingConfig) -> None:
    """Run scraping in direct mode with provided parameters."""
    console = Console()
    
    if not args.quiet:
        console.print("[bold blue]🤖 Intelligent Web Scraper - Direct Mode[/bold blue]")
        console.print(f"[cyan]URL:[/cyan] {args.url}")
        console.print(f"[cyan]Request:[/cyan] {args.request}")
        console.print()
    
    # Create orchestrator
    orchestrator = IntelligentScrapingOrchestrator(config=config)
    
    # Prepare request data
    request_data = {
        "scraping_request": args.request,
        "target_url": args.url,
        "max_results": args.max_results,
        "quality_threshold": args.quality_threshold,
        "export_format": args.export_format,
        "enable_monitoring": args.enable_monitoring and not args.disable_monitoring,
        "concurrent_instances": args.concurrent_instances
    }
    
    try:
        if not args.quiet:
            console.print("[yellow]🔄 Processing scraping request...[/yellow]")
        
        # Execute scraping
        result = await orchestrator.run(request_data)
        
        if not args.quiet:
            console.print("[green]✅ Scraping completed successfully![/green]")
            
            # Display summary
            items_count = len(result.extracted_data) if hasattr(result, 'extracted_data') and result.extracted_data else 0
            console.print(f"[cyan]Items extracted:[/cyan] {items_count}")
            
            if hasattr(result, 'quality_score'):
                console.print(f"[cyan]Quality score:[/cyan] {result.quality_score:.1f}%")
            
            if hasattr(result, 'export_options') and result.export_options:
                console.print("[cyan]Export files:[/cyan]")
                for format_type, path in result.export_options.items():
                    console.print(f"  • {format_type.upper()}: {path}")
        
        # Output JSON result for programmatic use
        if args.verbose or args.debug:
            output_data = {
                "success": True,
                "items_extracted": len(result.extracted_data) if hasattr(result, 'extracted_data') and result.extracted_data else 0,
                "quality_score": result.quality_score if hasattr(result, 'quality_score') else None,
                "export_options": result.export_options if hasattr(result, 'export_options') else {},
                "metadata": result.metadata.__dict__ if hasattr(result, 'metadata') else {}
            }
            
            if args.debug:
                output_data["extracted_data"] = result.extracted_data if hasattr(result, 'extracted_data') else []
                output_data["scraping_plan"] = result.scraping_plan if hasattr(result, 'scraping_plan') else ""
                output_data["reasoning"] = result.reasoning if hasattr(result, 'reasoning') else ""
            
            console.print_json(data=output_data)
        
    except Exception as e:
        if not args.quiet:
            console.print(f"[red]❌ Scraping failed: {str(e)}[/red]")
        
        if args.verbose or args.debug:
            console.print_json(data={
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            })
        
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