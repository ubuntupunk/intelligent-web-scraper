"""
Main entry point for the Intelligent Web Scraper application.

This module provides the interactive command-line interface and demonstrates
how to use the orchestrator agent in practice.
"""

import asyncio
import os
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text

from .config import IntelligentScrapingConfig
from .agents.orchestrator import IntelligentScrapingOrchestrator


class IntelligentScrapingApp:
    """Main application class for the Intelligent Web Scraper."""
    
    def __init__(self, config: Optional[IntelligentScrapingConfig] = None):
        self.config = config or IntelligentScrapingConfig.from_env()
        self.console = Console()
        self.orchestrator = IntelligentScrapingOrchestrator(config=self.config)
    
    def display_welcome(self) -> None:
        """Display welcome message and application information."""
        welcome_text = Text()
        welcome_text.append("🤖 Intelligent Web Scraper\n", style="bold blue")
        welcome_text.append("An Advanced Atomic Agents Example\n\n", style="italic")
        welcome_text.append("This application demonstrates sophisticated AI-powered web scraping\n")
        welcome_text.append("orchestration using the Atomic Agents framework.\n\n")
        welcome_text.append("Features:\n", style="bold")
        welcome_text.append("• Natural language scraping requests\n")
        welcome_text.append("• Intelligent strategy planning\n") 
        welcome_text.append("• Real-time monitoring\n")
        welcome_text.append("• Production-ready patterns\n")
        
        panel = Panel(
            welcome_text,
            title="Welcome",
            border_style="bright_blue",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def display_configuration(self) -> None:
        """Display current configuration settings."""
        config_table = Table(title="Current Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")
        
        config_table.add_row("Orchestrator Model", self.config.orchestrator_model)
        config_table.add_row("Planning Agent Model", self.config.planning_agent_model)
        config_table.add_row("Quality Threshold", f"{self.config.default_quality_threshold}%")
        config_table.add_row("Max Concurrent Requests", str(self.config.max_concurrent_requests))
        config_table.add_row("Export Format", self.config.default_export_format)
        config_table.add_row("Results Directory", self.config.results_directory)
        config_table.add_row("Monitoring Enabled", str(self.config.enable_monitoring))
        
        self.console.print(config_table)
    
    async def run_interactive(self) -> None:
        """Run the interactive scraping interface."""
        self.display_welcome()
        
        if Confirm.ask("Would you like to see the current configuration?"):
            self.display_configuration()
        
        self.console.print("\n[bold green]Ready to start scraping![/bold green]\n")
        
        while True:
            try:
                # Get scraping request from user
                scraping_request = Prompt.ask(
                    "[bold cyan]Enter your scraping request[/bold cyan]",
                    default="Extract all product information from this page"
                )
                
                target_url = Prompt.ask(
                    "[bold cyan]Enter the target URL[/bold cyan]"
                )
                
                if not target_url:
                    self.console.print("[red]URL is required![/red]")
                    continue
                
                # Optional parameters
                max_results = Prompt.ask(
                    "[cyan]Maximum results[/cyan]",
                    default="10"
                )
                
                quality_threshold = Prompt.ask(
                    "[cyan]Quality threshold (0-100)[/cyan]",
                    default=str(self.config.default_quality_threshold)
                )
                
                export_format = Prompt.ask(
                    "[cyan]Export format[/cyan]",
                    choices=["json", "csv", "markdown", "excel"],
                    default=self.config.default_export_format
                )
                
                # Prepare request
                request_data = {
                    "scraping_request": scraping_request,
                    "target_url": target_url,
                    "max_results": int(max_results),
                    "quality_threshold": float(quality_threshold),
                    "export_format": export_format,
                    "enable_monitoring": self.config.enable_monitoring
                }
                
                # Execute scraping
                self.console.print("\n[yellow]Starting intelligent scraping...[/yellow]")
                
                with self.console.status("[bold green]Processing...") as status:
                    result = await self.orchestrator.run(request_data)
                
                # Display results
                self.display_results(result)
                
                # Ask if user wants to continue
                if not Confirm.ask("\nWould you like to perform another scraping operation?"):
                    break
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Operation cancelled by user.[/yellow]")
                break
            except Exception as e:
                self.console.print(f"\n[red]Error: {str(e)}[/red]")
                if not Confirm.ask("Would you like to try again?"):
                    break
        
        self.console.print("\n[bold blue]Thank you for using Intelligent Web Scraper![/bold blue]")
    
    def display_results(self, result) -> None:
        """Display scraping results in a formatted way."""
        # Results summary
        summary_table = Table(title="Scraping Results Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Items Extracted", str(len(result.extracted_data)))
        summary_table.add_row("Quality Score", f"{result.quality_score:.1f}%")
        summary_table.add_row("Processing Time", f"{result.metadata.processing_time:.2f}s")
        summary_table.add_row("Pages Processed", str(result.metadata.pages_processed))
        
        self.console.print(summary_table)
        
        # Scraping plan and reasoning
        if hasattr(result, 'scraping_plan') and result.scraping_plan:
            plan_panel = Panel(
                result.scraping_plan,
                title="Scraping Plan",
                border_style="blue"
            )
            self.console.print(plan_panel)
        
        if hasattr(result, 'reasoning') and result.reasoning:
            reasoning_panel = Panel(
                result.reasoning,
                title="AI Reasoning",
                border_style="green"
            )
            self.console.print(reasoning_panel)
        
        # Export information
        if hasattr(result, 'export_options') and result.export_options:
            export_table = Table(title="Export Options")
            export_table.add_column("Format", style="cyan")
            export_table.add_column("Path", style="green")
            
            for format_type, path in result.export_options.items():
                export_table.add_row(format_type.upper(), path)
            
            self.console.print(export_table)
        
        # Sample data preview
        if result.extracted_data and len(result.extracted_data) > 0:
            self.console.print("\n[bold]Sample Data Preview:[/bold]")
            sample_data = result.extracted_data[:3]  # Show first 3 items
            
            for i, item in enumerate(sample_data, 1):
                item_panel = Panel(
                    str(item),
                    title=f"Item {i}",
                    border_style="dim"
                )
                self.console.print(item_panel)
            
            if len(result.extracted_data) > 3:
                self.console.print(f"[dim]... and {len(result.extracted_data) - 3} more items[/dim]")


async def main() -> None:
    """Main entry point for the application."""
    # Load configuration from environment
    config = IntelligentScrapingConfig.from_env()
    
    # Create and run the application
    app = IntelligentScrapingApp(config=config)
    await app.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())