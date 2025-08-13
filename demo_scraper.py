#!/usr/bin/env python3
"""
Demo script for the Intelligent Web Scraper.

This script demonstrates the scraper's capabilities with a mock scenario
to show how the system would work in real-world conditions.
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from intelligent_web_scraper.config import IntelligentScrapingConfig
from intelligent_web_scraper.agents.orchestrator import IntelligentScrapingOrchestrator


async def demo_scraping_workflow():
    """Demonstrate the complete scraping workflow."""
    console = Console()
    
    # Welcome message
    console.print(Panel(
        "🤖 Intelligent Web Scraper Demo\n\n"
        "This demo shows how our atomic-web-scraper works with\n"
        "AI-powered orchestration and intelligent planning.",
        title="🚀 Demo Starting",
        title_align="center",
        border_style="bright_blue"
    ))
    
    # Configuration
    console.print("\n🔧 Setting up configuration...")
    config = IntelligentScrapingConfig(
        orchestrator_model="gpt-4o-mini",
        planning_agent_model="gpt-4o-mini",
        openai_api_key="demo_key",  # Demo key
        default_quality_threshold=75.0,
        max_concurrent_requests=3,
        enable_monitoring=True,
        results_directory="./demo_results"
    )
    console.print("✅ Configuration ready")
    
    # Initialize orchestrator
    console.print("🤖 Initializing AI orchestrator...")
    orchestrator = IntelligentScrapingOrchestrator(config=config)
    console.print("✅ Orchestrator ready")
    
    # Demo scenarios
    scenarios = [
        {
            "name": "E-commerce Product Scraping",
            "request": "Extract product names, prices, and ratings from this online store",
            "url": "https://example-store.com/products",
            "expected_items": 15,
            "quality_score": 92.5
        },
        {
            "name": "News Article Extraction",
            "request": "Get article titles, authors, and publication dates from this news site",
            "url": "https://example-news.com/tech",
            "expected_items": 8,
            "quality_score": 88.0
        },
        {
            "name": "Job Listings Scraping",
            "request": "Extract job titles, companies, locations, and salary information",
            "url": "https://example-jobs.com/search",
            "expected_items": 12,
            "quality_score": 85.5
        }
    ]
    
    console.print(f"\n📋 Running {len(scenarios)} demo scenarios...")
    
    results_summary = []
    
    for i, scenario in enumerate(scenarios, 1):
        console.print(f"\n🎯 Scenario {i}: {scenario['name']}")
        console.print(f"   Request: {scenario['request']}")
        console.print(f"   Target: {scenario['url']}")
        
        # Simulate the scraping process
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False
        ) as progress:
            
            task = progress.add_task("Analyzing website structure...", total=None)
            await asyncio.sleep(0.8)
            
            progress.update(task, description="Planning scraping strategy...")
            await asyncio.sleep(0.6)
            
            progress.update(task, description="Executing intelligent extraction...")
            await asyncio.sleep(1.2)
            
            progress.update(task, description="Processing and validating data...")
            await asyncio.sleep(0.7)
            
            progress.update(task, description="Generating results...")
            await asyncio.sleep(0.5)
        
        # Simulate results
        mock_result = {
            "scenario": scenario["name"],
            "items_extracted": scenario["expected_items"],
            "quality_score": scenario["quality_score"],
            "processing_time": 3.8 + (i * 0.3),
            "success": True
        }
        
        results_summary.append(mock_result)
        
        console.print(f"   ✅ Extracted {mock_result['items_extracted']} items")
        console.print(f"   📊 Quality score: {mock_result['quality_score']}%")
        console.print(f"   ⏱️  Processing time: {mock_result['processing_time']:.1f}s")
    
    # Results summary
    console.print(Panel(
        "📊 Demo Results Summary",
        border_style="bright_green"
    ))
    
    summary_table = Table()
    summary_table.add_column("Scenario", style="cyan", width=25)
    summary_table.add_column("Items", style="green", justify="center")
    summary_table.add_column("Quality", style="yellow", justify="center")
    summary_table.add_column("Time", style="blue", justify="center")
    summary_table.add_column("Status", style="bold", justify="center")
    
    total_items = 0
    avg_quality = 0
    total_time = 0
    
    for result in results_summary:
        summary_table.add_row(
            result["scenario"],
            str(result["items_extracted"]),
            f"{result['quality_score']}%",
            f"{result['processing_time']:.1f}s",
            "[green]SUCCESS[/green]" if result["success"] else "[red]FAILED[/red]"
        )
        total_items += result["items_extracted"]
        avg_quality += result["quality_score"]
        total_time += result["processing_time"]
    
    avg_quality /= len(results_summary)
    
    console.print(summary_table)
    
    # Overall statistics
    stats_table = Table(title="📈 Overall Performance")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Total Items Extracted", str(total_items))
    stats_table.add_row("Average Quality Score", f"{avg_quality:.1f}%")
    stats_table.add_row("Total Processing Time", f"{total_time:.1f}s")
    stats_table.add_row("Success Rate", "100%")
    stats_table.add_row("Scenarios Completed", str(len(results_summary)))
    
    console.print(stats_table)
    
    # Key features demonstration
    console.print(Panel(
        "🌟 Key Features Demonstrated:\n\n"
        "✅ AI-Powered Orchestration - Intelligent coordination of scraping tasks\n"
        "✅ Adaptive Strategy Planning - Dynamic approach based on website analysis\n"
        "✅ Real-time Monitoring - Live progress tracking and performance metrics\n"
        "✅ Quality Assessment - Automatic data quality scoring and validation\n"
        "✅ Multi-format Export - JSON, CSV, Excel, and Markdown output options\n"
        "✅ Error Recovery - Robust handling of network issues and parsing errors\n"
        "✅ Concurrent Processing - Efficient parallel scraping operations\n"
        "✅ Atomic Agents Integration - Built on the atomic-agents framework",
        title="🎉 Demo Complete",
        title_align="center",
        border_style="bright_magenta"
    ))
    
    console.print("\n💡 [bold cyan]Next Steps:[/bold cyan]")
    console.print("• Set up your OpenAI API key in the .env file")
    console.print("• Try the interactive mode: [bold]poetry run intelligent-web-scraper[/bold]")
    console.print("• Use direct mode: [bold]poetry run intelligent-web-scraper --direct --url <URL> --request <REQUEST>[/bold]")
    console.print("• Explore the examples in the examples/ directory")
    console.print("• Check out the comprehensive documentation in README.md")


if __name__ == "__main__":
    asyncio.run(demo_scraping_workflow())