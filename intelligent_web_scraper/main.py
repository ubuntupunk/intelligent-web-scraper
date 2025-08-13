"""
Main entry point for the Intelligent Web Scraper application.

This module provides the interactive command-line interface and demonstrates
how to use the orchestrator agent in practice.
"""

import asyncio
import os
import sys
from typing import Optional, Dict, Any
import re
from urllib.parse import urlparse

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.align import Align
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich.syntax import Syntax

from .config import IntelligentScrapingConfig
from .agents.orchestrator import IntelligentScrapingOrchestrator


class IntelligentScrapingApp:
    """
    Main application class for the Intelligent Web Scraper.
    
    This class provides a rich, interactive command-line interface that guides
    users through the scraping process with proper input validation, help text,
    and beautiful formatted output.
    """
    
    def __init__(self, config: Optional[IntelligentScrapingConfig] = None):
        self.config = config or IntelligentScrapingConfig.from_env()
        self.console = Console()
        self.orchestrator = IntelligentScrapingOrchestrator(config=self.config)
        
        # Application state
        self.session_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_items_extracted": 0
        }
    
    def display_welcome(self) -> None:
        """Display welcome message and application information."""
        # Create main welcome text
        welcome_text = Text()
        welcome_text.append("🤖 Intelligent Web Scraper\n", style="bold blue")
        welcome_text.append("An Advanced Atomic Agents Example\n\n", style="italic cyan")
        welcome_text.append("This application demonstrates sophisticated AI-powered web scraping\n")
        welcome_text.append("orchestration using the Atomic Agents framework.\n\n")
        
        # Features section
        welcome_text.append("✨ Key Features:\n", style="bold yellow")
        welcome_text.append("• Natural language scraping requests\n", style="green")
        welcome_text.append("• Intelligent strategy planning\n", style="green") 
        welcome_text.append("• Real-time monitoring & progress tracking\n", style="green")
        welcome_text.append("• Multiple export formats (JSON, CSV, Excel, Markdown)\n", style="green")
        welcome_text.append("• Production-ready patterns & error handling\n", style="green")
        
        # Create welcome panel
        welcome_panel = Panel(
            welcome_text,
            title="🚀 Welcome to Intelligent Web Scraper",
            title_align="center",
            border_style="bright_blue",
            padding=(1, 2)
        )
        
        # Create usage examples
        examples_text = Text()
        examples_text.append("📝 Example Requests:\n", style="bold yellow")
        examples_text.append("• 'Extract all product names and prices from this e-commerce page'\n", style="dim")
        examples_text.append("• 'Get all article titles and publication dates from this news site'\n", style="dim")
        examples_text.append("• 'Scrape contact information from company directory pages'\n", style="dim")
        examples_text.append("• 'Find all job listings with salary information'\n", style="dim")
        
        examples_panel = Panel(
            examples_text,
            title="💡 Usage Examples",
            title_align="center",
            border_style="green",
            padding=(1, 2)
        )
        
        # Display panels
        self.console.print(welcome_panel)
        self.console.print(examples_panel)
        
        # Display help information
        self.display_help_info()
    
    def display_help_info(self) -> None:
        """Display help information and keyboard shortcuts."""
        help_text = Text()
        help_text.append("⌨️  Keyboard Shortcuts:\n", style="bold cyan")
        help_text.append("• Ctrl+C: Cancel current operation\n", style="dim")
        help_text.append("• Enter: Use default value (shown in brackets)\n", style="dim")
        help_text.append("• Type 'help' at any prompt for detailed guidance\n", style="dim")
        help_text.append("• Type 'config' to view/modify configuration\n", style="dim")
        help_text.append("• Type 'stats' to view session statistics\n", style="dim")
        
        help_panel = Panel(
            help_text,
            title="ℹ️  Help & Tips",
            title_align="center",
            border_style="yellow",
            padding=(1, 2)
        )
        self.console.print(help_panel)
    
    def validate_url(self, url: str) -> bool:
        """Validate URL format and accessibility."""
        if not url or url.strip() == "":
            return False
            
        # Basic URL format validation
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def validate_quality_threshold(self, threshold: str) -> tuple[bool, float]:
        """Validate quality threshold input."""
        try:
            value = float(threshold)
            if 0 <= value <= 100:
                return True, value
            return False, 0.0
        except ValueError:
            return False, 0.0
    
    def validate_max_results(self, max_results: str) -> tuple[bool, int]:
        """Validate max results input."""
        try:
            value = int(max_results)
            if 1 <= value <= 10000:
                return True, value
            return False, 0
        except ValueError:
            return False, 0
    
    def get_batch_urls(self) -> list[str]:
        """Get multiple URLs for batch processing."""
        urls = []
        self.console.print("[dim]Enter URLs one by one. Press Enter with empty input to finish.[/dim]")
        
        while True:
            try:
                url_input = Prompt.ask(
                    f"[cyan]URL #{len(urls) + 1}[/cyan] [dim](or press Enter to finish)[/dim]",
                    default=""
                )
                
                if not url_input.strip():
                    break
                
                # Handle special commands
                if url_input.lower() == 'help':
                    self.display_detailed_help()
                    continue
                elif url_input.lower() == 'config':
                    self.display_configuration()
                    continue
                elif url_input.lower() == 'stats':
                    self.display_session_stats()
                    continue
                
                # Validate URL
                if self.validate_url(url_input):
                    urls.append(url_input)
                    self.console.print(f"[green]✓ Added: {url_input}[/green]")
                else:
                    self.console.print("[red]❌ Invalid URL format. Please include http:// or https://[/red]")
                
            except KeyboardInterrupt:
                if Confirm.ask("[yellow]Cancel batch URL entry?[/yellow]"):
                    return []
                continue
        
        if not urls:
            self.console.print("[yellow]No URLs entered.[/yellow]")
            return []
        
        # Display summary
        self.console.print(f"\n[bold green]📋 Batch Summary: {len(urls)} URLs added[/bold green]")
        for i, url in enumerate(urls, 1):
            self.console.print(f"  {i}. {url}")
        
        if not Confirm.ask(f"\n[cyan]Proceed with these {len(urls)} URLs?[/cyan]"):
            return []
        
        return urls
    
    def get_validated_input(self, prompt_text: str, validator_func, error_message: str, default_value=None):
        """Get validated input from user with retry logic."""
        while True:
            try:
                if default_value is not None:
                    user_input = Prompt.ask(prompt_text, default=str(default_value))
                else:
                    user_input = Prompt.ask(prompt_text)
                
                # Handle special commands
                if user_input.lower() == 'help':
                    self.display_detailed_help()
                    continue
                elif user_input.lower() == 'config':
                    self.display_configuration()
                    continue
                elif user_input.lower() == 'stats':
                    self.display_session_stats()
                    continue
                
                # Validate input
                if validator_func:
                    if hasattr(validator_func, '__call__'):
                        if validator_func.__name__ in ['validate_quality_threshold', 'validate_max_results']:
                            is_valid, validated_value = validator_func(user_input)
                            if is_valid:
                                return validated_value
                        else:
                            if validator_func(user_input):
                                return user_input
                    else:
                        return user_input
                else:
                    return user_input
                
                # Show error and retry
                self.console.print(f"[red]{error_message}[/red]")
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.console.print(f"[red]Invalid input: {str(e)}[/red]")
    
    def display_detailed_help(self) -> None:
        """Display detailed help information."""
        help_sections = [
            {
                "title": "🎯 Scraping Request Tips",
                "content": [
                    "• Be specific about what data you want to extract",
                    "• Mention the type of website (e-commerce, news, directory, etc.)",
                    "• Include any specific fields you're interested in",
                    "• Use natural language - the AI will understand your intent"
                ]
            },
            {
                "title": "🌐 URL Guidelines", 
                "content": [
                    "• Use complete URLs including http:// or https://",
                    "• Ensure the website is publicly accessible",
                    "• Some sites may block automated access",
                    "• The system respects robots.txt by default"
                ]
            },
            {
                "title": "⚙️ Configuration Options",
                "content": [
                    "• Quality Threshold: 0-100 (higher = more selective)",
                    "• Max Results: 1-10000 (reasonable limits recommended)",
                    "• Export Formats: JSON, CSV, Excel, Markdown",
                    "• Monitoring: Real-time progress and performance tracking"
                ]
            }
        ]
        
        for section in help_sections:
            content_text = Text()
            for item in section["content"]:
                content_text.append(f"{item}\n", style="dim")
            
            panel = Panel(
                content_text,
                title=section["title"],
                border_style="blue",
                padding=(1, 2)
            )
            self.console.print(panel)
    
    def display_session_stats(self) -> None:
        """Display current session statistics."""
        stats_table = Table(title="📊 Session Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Requests", str(self.session_stats["total_requests"]))
        stats_table.add_row("Successful Requests", str(self.session_stats["successful_requests"]))
        stats_table.add_row("Failed Requests", str(self.session_stats["failed_requests"]))
        stats_table.add_row("Total Items Extracted", str(self.session_stats["total_items_extracted"]))
        
        if self.session_stats["total_requests"] > 0:
            success_rate = (self.session_stats["successful_requests"] / self.session_stats["total_requests"]) * 100
            stats_table.add_row("Success Rate", f"{success_rate:.1f}%")
        
        self.console.print(stats_table)
    
    def display_configuration(self) -> None:
        """Display current configuration settings with enhanced formatting."""
        # Agent Configuration
        agent_table = Table(title="🤖 Agent Configuration")
        agent_table.add_column("Setting", style="cyan", width=25)
        agent_table.add_column("Value", style="green")
        agent_table.add_column("Description", style="dim", width=40)
        
        agent_table.add_row("Orchestrator Model", self.config.orchestrator_model, "Main AI model for coordination")
        agent_table.add_row("Planning Agent Model", self.config.planning_agent_model, "AI model for strategy planning")
        
        # Scraping Configuration
        scraping_table = Table(title="🌐 Scraping Configuration")
        scraping_table.add_column("Setting", style="cyan", width=25)
        scraping_table.add_column("Value", style="green")
        scraping_table.add_column("Description", style="dim", width=40)
        
        scraping_table.add_row("Quality Threshold", f"{self.config.default_quality_threshold}%", "Minimum quality for extracted data")
        scraping_table.add_row("Max Concurrent Requests", str(self.config.max_concurrent_requests), "Parallel request limit")
        scraping_table.add_row("Request Delay", f"{self.config.request_delay}s", "Delay between requests")
        scraping_table.add_row("Respect Robots.txt", str(self.config.respect_robots_txt), "Follow robots.txt rules")
        scraping_table.add_row("Rate Limiting", str(self.config.enable_rate_limiting), "Enable request rate limiting")
        
        # Output Configuration
        output_table = Table(title="📁 Output Configuration")
        output_table.add_column("Setting", style="cyan", width=25)
        output_table.add_column("Value", style="green")
        output_table.add_column("Description", style="dim", width=40)
        
        output_table.add_row("Export Format", self.config.default_export_format, "Default export file format")
        output_table.add_row("Results Directory", self.config.results_directory, "Directory for saved results")
        
        # Performance Configuration
        perf_table = Table(title="⚡ Performance Configuration")
        perf_table.add_column("Setting", style="cyan", width=25)
        perf_table.add_column("Value", style="green")
        perf_table.add_column("Description", style="dim", width=40)
        
        perf_table.add_row("Monitoring Enabled", str(self.config.enable_monitoring), "Real-time monitoring")
        perf_table.add_row("Monitoring Interval", f"{self.config.monitoring_interval}s", "Update frequency")
        perf_table.add_row("Max Instances", str(self.config.max_instances), "Maximum scraper instances")
        perf_table.add_row("Max Workers", str(self.config.max_workers), "Maximum worker threads")
        perf_table.add_row("Max Async Tasks", str(self.config.max_async_tasks), "Maximum concurrent tasks")
        
        # Display all tables
        self.console.print(agent_table)
        self.console.print(scraping_table)
        self.console.print(output_table)
        self.console.print(perf_table)
        
        # Configuration modification option
        if Confirm.ask("\n[cyan]Would you like to modify any configuration settings?[/cyan]"):
            self.modify_configuration()
    
    def modify_configuration(self) -> None:
        """Allow user to modify configuration settings interactively."""
        self.console.print("\n[bold yellow]Configuration Modification[/bold yellow]")
        self.console.print("[dim]Leave empty to keep current value[/dim]\n")
        
        # AI Model Configuration
        self.console.print("[bold blue]🤖 AI Model Configuration[/bold blue]")
        self.console.print("[dim]Note: Ensure your API keys are configured for the selected provider[/dim]\n")
        
        # Available models
        available_models = [
            "gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo",
            "gemini-1.5-flash", "gemini-1.5-pro", 
            "claude-3-haiku-20240307", "claude-3-5-sonnet-20241022",
            "deepseek-chat"
        ]
        
        # Orchestrator model
        new_orchestrator_model = Prompt.ask(
            f"[cyan]Orchestrator model[/cyan] [dim](current: {self.config.orchestrator_model})[/dim]",
            choices=available_models + [""],
            default=""
        )
        if new_orchestrator_model:
            self.config.orchestrator_model = new_orchestrator_model
            self.console.print(f"[green]✓ Orchestrator model updated to {new_orchestrator_model}[/green]")
        
        # Planning agent model
        new_planning_model = Prompt.ask(
            f"[cyan]Planning agent model[/cyan] [dim](current: {self.config.planning_agent_model})[/dim]",
            choices=available_models + [""],
            default=""
        )
        if new_planning_model:
            self.config.planning_agent_model = new_planning_model
            self.console.print(f"[green]✓ Planning agent model updated to {new_planning_model}[/green]")
        
        # LLM Provider
        new_provider = Prompt.ask(
            f"[cyan]LLM provider[/cyan] [dim](current: {self.config.llm_provider})[/dim]",
            choices=["openai", "gemini", "anthropic", "deepseek", "openrouter", ""],
            default=""
        )
        if new_provider:
            self.config.llm_provider = new_provider
            self.console.print(f"[green]✓ LLM provider updated to {new_provider}[/green]")
        
        self.console.print("\n[bold blue]🎯 Scraping Configuration[/bold blue]")
        
        # Quality threshold
        new_threshold = Prompt.ask(
            f"[cyan]Quality threshold (0-100)[/cyan] [dim](current: {self.config.default_quality_threshold})[/dim]",
            default=""
        )
        if new_threshold:
            is_valid, value = self.validate_quality_threshold(new_threshold)
            if is_valid:
                self.config.default_quality_threshold = value
                self.console.print(f"[green]✓ Quality threshold updated to {value}%[/green]")
        
        # Max concurrent requests
        new_concurrent = Prompt.ask(
            f"[cyan]Max concurrent requests (1-20)[/cyan] [dim](current: {self.config.max_concurrent_requests})[/dim]",
            default=""
        )
        if new_concurrent:
            try:
                value = int(new_concurrent)
                if 1 <= value <= 20:
                    self.config.max_concurrent_requests = value
                    self.console.print(f"[green]✓ Max concurrent requests updated to {value}[/green]")
            except ValueError:
                self.console.print("[red]Invalid number[/red]")
        
        # Request delay
        new_delay = Prompt.ask(
            f"[cyan]Request delay (seconds)[/cyan] [dim](current: {self.config.request_delay})[/dim]",
            default=""
        )
        if new_delay:
            try:
                value = float(new_delay)
                if value >= 0:
                    self.config.request_delay = value
                    self.console.print(f"[green]✓ Request delay updated to {value}s[/green]")
            except ValueError:
                self.console.print("[red]Invalid number[/red]")
        
        self.console.print("\n[bold blue]💾 Output Configuration[/bold blue]")
        
        # Export format
        new_format = Prompt.ask(
            f"[cyan]Export format[/cyan] [dim](current: {self.config.default_export_format})[/dim]",
            choices=["json", "csv", "markdown", "excel", ""],
            default=""
        )
        if new_format:
            self.config.default_export_format = new_format
            self.console.print(f"[green]✓ Export format updated to {new_format}[/green]")
        
        # Results directory
        new_dir = Prompt.ask(
            f"[cyan]Results directory[/cyan] [dim](current: {self.config.results_directory})[/dim]",
            default=""
        )
        if new_dir:
            self.config.results_directory = new_dir
            self.console.print(f"[green]✓ Results directory updated to {new_dir}[/green]")
        
        self.console.print("\n[bold blue]📊 Monitoring Configuration[/bold blue]")
        
        # Enable monitoring
        new_monitoring = Prompt.ask(
            f"[cyan]Enable monitoring (true/false)[/cyan] [dim](current: {self.config.enable_monitoring})[/dim]",
            choices=["true", "false", ""],
            default=""
        )
        if new_monitoring:
            self.config.enable_monitoring = new_monitoring.lower() == "true"
            self.console.print(f"[green]✓ Monitoring updated to {self.config.enable_monitoring}[/green]")
        
        # Monitoring interval
        new_interval = Prompt.ask(
            f"[cyan]Monitoring interval (seconds)[/cyan] [dim](current: {self.config.monitoring_interval})[/dim]",
            default=""
        )
        if new_interval:
            try:
                value = float(new_interval)
                if value > 0:
                    self.config.monitoring_interval = value
                    self.console.print(f"[green]✓ Monitoring interval updated to {value}s[/green]")
            except ValueError:
                self.console.print("[red]Invalid number[/red]")
        
        self.console.print("\n[green]Configuration updated successfully![/green]")
    
    async def run_interactive(self) -> None:
        """Run the enhanced interactive scraping interface."""
        # Clear screen and display welcome
        self.console.clear()
        self.display_welcome()
        
        # Configuration check
        if Confirm.ask("\n[cyan]Would you like to view/modify configuration?[/cyan]"):
            self.display_configuration()
        
        # Main interaction loop
        self.console.print(Rule("[bold green]🚀 Ready to Start Scraping![/bold green]"))
        
        while True:
            try:
                self.console.print()  # Add spacing
                
                # Get target URL(s) first - supports batch processing
                self.console.print("[bold blue]🌐 Target URL Configuration[/bold blue]")
                
                # Ask if user wants single or batch processing
                batch_mode = Confirm.ask("[cyan]Do you want to scrape multiple URLs in batch?[/cyan]")
                
                if batch_mode:
                    target_urls = self.get_batch_urls()
                    if not target_urls:
                        continue
                else:
                    single_url = self.get_validated_input(
                        "[bold cyan]🌐 Enter the target URL[/bold cyan] [dim](must include http:// or https://)[/dim]",
                        self.validate_url,
                        "Please enter a valid URL (e.g., https://example.com)"
                    )
                    target_urls = [single_url]
                
                # Get scraping request after URLs are specified
                scraping_request = self.get_validated_input(
                    "[bold cyan]📝 Enter your scraping request[/bold cyan] [dim](will be applied to all URLs)[/dim]",
                    lambda x: len(x.strip()) > 0,
                    "Please enter a valid scraping request.",
                    "Extract all product information from this page"
                )
                
                # Get optional parameters with validation
                max_results = self.get_validated_input(
                    "[cyan]📊 Maximum results (1-10000)[/cyan]",
                    self.validate_max_results,
                    "Please enter a number between 1 and 10000",
                    10
                )
                
                quality_threshold = self.get_validated_input(
                    "[cyan]🎯 Quality threshold (0-100)[/cyan]",
                    self.validate_quality_threshold,
                    "Please enter a number between 0 and 100",
                    self.config.default_quality_threshold
                )
                
                export_format = Prompt.ask(
                    "[cyan]💾 Export format[/cyan]",
                    choices=["json", "csv", "markdown", "excel"],
                    default=self.config.default_export_format
                )
                
                # Confirm before processing
                self.console.print("\n[bold yellow]📋 Request Summary:[/bold yellow]")
                summary_table = Table(show_header=False, box=None)
                summary_table.add_column("Field", style="cyan")
                summary_table.add_column("Value", style="green")
                
                summary_table.add_row("Request:", scraping_request[:80] + "..." if len(scraping_request) > 80 else scraping_request)
                summary_table.add_row("URLs:", f"{len(target_urls)} URL(s)" if len(target_urls) > 1 else target_urls[0])
                summary_table.add_row("Max Results:", str(max_results))
                summary_table.add_row("Quality Threshold:", f"{quality_threshold}%")
                summary_table.add_row("Export Format:", export_format.upper())
                
                self.console.print(summary_table)
                
                if not Confirm.ask(f"\n[bold cyan]Proceed with scraping {len(target_urls)} URL(s)?[/bold cyan]"):
                    continue
                
                # Update session stats
                self.session_stats["total_requests"] += len(target_urls)
                
                # Execute scraping for each URL
                all_results = []
                successful_urls = 0
                failed_urls = 0
                
                self.console.print(Rule(f"[bold yellow]🔄 Processing {len(target_urls)} URL(s)[/bold yellow]"))
                
                for i, target_url in enumerate(target_urls, 1):
                    try:
                        self.console.print(f"\n[bold blue]Processing URL {i}/{len(target_urls)}:[/bold blue] {target_url}")
                        
                        # Prepare request data for this URL
                        request_data = {
                            "scraping_request": scraping_request,
                            "target_url": target_url,
                            "max_results": max_results,
                            "quality_threshold": quality_threshold,
                            "export_format": export_format,
                            "enable_monitoring": self.config.enable_monitoring,
                            "batch_index": i,
                            "batch_total": len(target_urls)
                        }
                        
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            console=self.console,
                            transient=False
                        ) as progress:
                            
                            # Add progress tasks
                            task = progress.add_task(f"Processing URL {i}/{len(target_urls)}...", total=None)
                            await asyncio.sleep(0.3)
                            
                            progress.update(task, description=f"Analyzing {target_url}...")
                            await asyncio.sleep(0.3)
                            
                            progress.update(task, description="Planning scraping strategy...")
                            await asyncio.sleep(0.3)
                            
                            progress.update(task, description="Extracting data...")
                            result = await self.orchestrator.run(request_data)
                            
                            progress.update(task, description=f"URL {i} complete!")
                        
                        # Store result with URL info
                        result.source_url = target_url
                        result.batch_index = i
                        all_results.append(result)
                        successful_urls += 1
                        
                        # Update success stats
                        if hasattr(result, 'extracted_data') and result.extracted_data:
                            self.session_stats["total_items_extracted"] += len(result.extracted_data)
                        
                        self.console.print(f"[green]✅ URL {i} completed successfully[/green]")
                        
                    except Exception as e:
                        failed_urls += 1
                        self.console.print(f"[red]❌ URL {i} failed: {str(e)}[/red]")
                        continue
                
                # Update final stats
                self.session_stats["successful_requests"] += successful_urls
                self.session_stats["failed_requests"] += failed_urls
                
                # Display batch results summary
                if len(target_urls) > 1:
                    self.display_batch_results(all_results, successful_urls, failed_urls)
                else:
                    # Single URL - display normal results
                    if all_results:
                        self.display_results(all_results[0])
                
            except Exception as e:
                # Update failure stats for overall operation
                self.session_stats["failed_requests"] += len(target_urls)
                
                self.console.print(f"\n[red]❌ Scraping operation failed: {str(e)}[/red]")
                
                # Offer detailed error information
                if Confirm.ask("[cyan]Would you like to see detailed error information?[/cyan]"):
                    error_panel = Panel(
                        str(e),
                        title="🔍 Error Details",
                        border_style="red"
                    )
                    self.console.print(error_panel)
                
                # Ask if user wants to continue
                if not Confirm.ask("\n[bold cyan]Would you like to perform another scraping operation?[/cyan]"):
                    break
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]⏹️ Operation cancelled by user.[/yellow]")
                if not Confirm.ask("[cyan]Exit application?[/cyan]"):
                    continue
                else:
                    break
            except Exception as e:
                self.console.print(f"\n[red]❌ Unexpected error: {str(e)}[/red]")
                if not Confirm.ask("[cyan]Would you like to try again?[/cyan]"):
                    break
        
        # Display final session summary
        self.console.print(Rule("[bold blue]📊 Session Summary[/bold blue]"))
        self.display_session_stats()
        self.console.print("\n[dim]🙏 Thank you for using Intelligent Web Scraper!")
        self.console.print("[dim]Built with ❤️  using Atomic Agents framework[/dim]")

    def display_batch_results(self, results: list, successful_urls: int, failed_urls: int) -> None:
        """Display batch scraping results summary."""
        self.console.print(Rule("[bold green]📊 Batch Scraping Results[/bold green]"))
        
        # Batch summary
        batch_table = Table(title="🎯 Batch Summary", title_style="bold green")
        batch_table.add_column("Metric", style="cyan", width=20)
        batch_table.add_column("Value", style="green", width=15)
        batch_table.add_column("Details", style="dim", width=40)
        
        total_items = sum(len(result.extracted_data) if hasattr(result, 'extracted_data') and result.extracted_data else 0 for result in results)
        avg_quality = sum(result.quality_score if hasattr(result, 'quality_score') else 0 for result in results) / len(results) if results else 0
        
        batch_table.add_row("URLs Processed", f"{successful_urls + failed_urls}", f"{successful_urls} successful, {failed_urls} failed")
        batch_table.add_row("Total Items", str(total_items), "Combined data from all successful URLs")
        batch_table.add_row("Average Quality", f"{avg_quality:.1f}%", "Average quality score across all URLs")
        batch_table.add_row("Success Rate", f"{(successful_urls/(successful_urls + failed_urls)*100):.1f}%", "Percentage of successful URL processing")
        
        self.console.print(batch_table)
        
        # Individual URL results
        if results:
            url_table = Table(title="📋 Individual URL Results", title_style="bold blue")
            url_table.add_column("#", style="cyan", width=3)
            url_table.add_column("URL", style="blue", width=40)
            url_table.add_column("Items", style="green", width=8)
            url_table.add_column("Quality", style="yellow", width=10)
            url_table.add_column("Status", style="bold", width=10)
            
            for result in results:
                items_count = len(result.extracted_data) if hasattr(result, 'extracted_data') and result.extracted_data else 0
                quality_score = result.quality_score if hasattr(result, 'quality_score') else 0
                url_display = result.source_url[:37] + "..." if len(result.source_url) > 40 else result.source_url
                
                url_table.add_row(
                    str(result.batch_index),
                    url_display,
                    str(items_count),
                    f"{quality_score:.1f}%",
                    "[green]SUCCESS[/green]"
                )
            
            self.console.print(url_table)
        
        # Export information
        if results and hasattr(results[0], 'export_options'):
            self.console.print("\n[bold cyan]💾 Batch Export Files:[/bold cyan]")
            for i, result in enumerate(results, 1):
                if hasattr(result, 'export_options') and result.export_options:
                    self.console.print(f"[dim]URL {i}:[/dim]")
                    for format_type, path in result.export_options.items():
                        self.console.print(f"  • {format_type.upper()}: {path}")
        
        # Sample data from first successful result
        if results and hasattr(results[0], 'extracted_data') and results[0].extracted_data:
            self.console.print(Rule("[bold]📋 Sample Data (from first URL)[/bold]"))
            sample_data = results[0].extracted_data[:2]  # Show first 2 items
            
            for i, item in enumerate(sample_data, 1):
                if isinstance(item, dict):
                    formatted_item = "\n".join([f"[cyan]{k}:[/cyan] {v}" for k, v in item.items()])
                else:
                    formatted_item = str(item)
                
                item_panel = Panel(
                    formatted_item,
                    title=f"📄 Sample Item {i}",
                    title_align="left",
                    border_style="dim blue",
                    padding=(1, 2)
                )
                self.console.print(item_panel)


    def display_results(self, result) -> None:
        """Display scraping results in a beautifully formatted way."""
        # Results summary with enhanced styling
        summary_table = Table(title="🎉 Scraping Results Summary", title_style="bold green")
        summary_table.add_column("Metric", style="cyan", width=20)
        summary_table.add_column("Value", style="green", width=15)
        summary_table.add_column("Details", style="dim", width=40)
        
        items_count = len(result.extracted_data) if hasattr(result, 'extracted_data') and result.extracted_data else 0
        summary_table.add_row("Items Extracted", str(items_count), "Total data items successfully extracted")
        
        if hasattr(result, 'quality_score'):
            quality_style = "green" if result.quality_score >= 70 else "yellow" if result.quality_score >= 50 else "red"
            summary_table.add_row("Quality Score", f"{result.quality_score:.1f}%", f"[{quality_style}]Data quality assessment[/{quality_style}]")
        
        if hasattr(result, 'metadata'):
            if hasattr(result.metadata, 'processing_time'):
                summary_table.add_row("Processing Time", f"{result.metadata.processing_time:.2f}s", "Total time for scraping operation")
            if hasattr(result.metadata, 'pages_processed'):
                summary_table.add_row("Pages Processed", str(result.metadata.pages_processed), "Number of web pages analyzed")
        
        self.console.print(summary_table)
        
        # Scraping plan and reasoning with better formatting
        if hasattr(result, 'scraping_plan') and result.scraping_plan:
            plan_panel = Panel(
                result.scraping_plan,
                title="🧠 AI Scraping Strategy",
                title_align="center",
                border_style="blue",
                padding=(1, 2)
            )
            self.console.print(plan_panel)
        
        if hasattr(result, 'reasoning') and result.reasoning:
            reasoning_panel = Panel(
                result.reasoning,
                title="💭 AI Decision Process",
                title_align="center",
                border_style="green",
                padding=(1, 2)
            )
            self.console.print(reasoning_panel)
        
        # Export information with file paths
        if hasattr(result, 'export_options') and result.export_options:
            export_table = Table(title="💾 Export Options", title_style="bold cyan")
            export_table.add_column("Format", style="cyan", width=10)
            export_table.add_column("File Path", style="green", width=50)
            export_table.add_column("Description", style="dim", width=30)
            
            format_descriptions = {
                "json": "Structured data for APIs",
                "csv": "Tabular data for spreadsheets", 
                "markdown": "Human-readable format",
                "excel": "Advanced analysis & charts"
            }
            
            for format_type, path in result.export_options.items():
                description = format_descriptions.get(format_type.lower(), "Data export file")
                export_table.add_row(format_type.upper(), path, description)
            
            self.console.print(export_table)
        
        # Enhanced sample data preview
        if hasattr(result, 'extracted_data') and result.extracted_data and len(result.extracted_data) > 0:
            self.console.print(Rule("[bold]📋 Sample Data Preview[/bold]"))
            sample_data = result.extracted_data[:3]  # Show first 3 items
            
            for i, item in enumerate(sample_data, 1):
                # Format item data nicely
                if isinstance(item, dict):
                    formatted_item = "\n".join([f"[cyan]{k}:[/cyan] {v}" for k, v in item.items()])
                else:
                    formatted_item = str(item)
                
                item_panel = Panel(
                    formatted_item,
                    title=f"📄 Item {i}",
                    title_align="left",
                    border_style="dim blue",
                    padding=(1, 2)
                )
                self.console.print(item_panel)
            
            if len(result.extracted_data) > 3:
                remaining = len(result.extracted_data) - 3
                self.console.print(f"[dim]... and {remaining} more items (see export files for complete data)[/dim]")


def main() -> None:
    """Main entry point for the CLI application."""
    asyncio.run(async_main())


async def async_main() -> None:
    """Async main entry point for the application."""
    try:
        # Load configuration from environment
        config = IntelligentScrapingConfig.from_env()
        
        # Create and run the application
        app = IntelligentScrapingApp(config=config)
        await app.run_interactive()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Application error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()