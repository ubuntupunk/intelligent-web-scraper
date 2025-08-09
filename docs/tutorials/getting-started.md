# Getting Started with Intelligent Web Scraper

Welcome to the Intelligent Web Scraper! This tutorial will guide you through the basics of using this advanced Atomic Agents example application for AI-powered web scraping.

## Table of Contents

1. [What is Intelligent Web Scraper?](#what-is-intelligent-web-scraper)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Understanding the Architecture](#understanding-the-architecture)
5. [Your First Scraping Operation](#your-first-scraping-operation)
6. [Working with Different Data Types](#working-with-different-data-types)
7. [Configuration and Customization](#configuration-and-customization)
8. [Monitoring and Performance](#monitoring-and-performance)
9. [Best Practices](#best-practices)
10. [Next Steps](#next-steps)

## What is Intelligent Web Scraper?

The Intelligent Web Scraper is an advanced example application built with the Atomic Agents framework that demonstrates:

- **AI-Powered Strategy Planning**: Uses LLMs to analyze websites and generate optimal scraping strategies
- **Natural Language Requests**: Accepts human-readable scraping instructions
- **Intelligent Orchestration**: Coordinates multiple agents and tools for complex workflows
- **Production-Ready Patterns**: Implements monitoring, error handling, and scalability features
- **Educational Value**: Serves as a comprehensive example of Atomic Agents best practices

### Key Features

✨ **Natural Language Interface**: Describe what you want to scrape in plain English
🧠 **AI-Powered Planning**: Automatically analyzes websites and generates scraping strategies
📊 **Real-Time Monitoring**: Track performance, quality, and resource usage
🔄 **Concurrent Processing**: Handle multiple scraping operations simultaneously
📁 **Multiple Export Formats**: JSON, CSV, Markdown, Excel output options
🛡️ **Ethical Scraping**: Respects robots.txt and implements rate limiting
🎯 **Quality Scoring**: Evaluates and scores extracted data quality

## Installation

### Prerequisites

- Python 3.11 or higher
- OpenAI API key (or compatible LLM service)
- Poetry (recommended) or pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/atomic-agents/intelligent-web-scraper.git
cd intelligent-web-scraper
```

### Step 2: Install Dependencies

Using Poetry (recommended):
```bash
poetry install
```

Using pip:
```bash
pip install -e .
```

### Step 3: Set Up Environment

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```

3. Optionally customize other settings:
   ```bash
   ORCHESTRATOR_MODEL=gpt-4o-mini
   DEFAULT_QUALITY_THRESHOLD=60.0
   MAX_CONCURRENT_REQUESTS=5
   ```

### Step 4: Verify Installation

Run the health check:
```bash
poetry run python scripts/health-check.py --verbose
```

You should see output indicating all systems are healthy.

## Basic Usage

### Interactive Mode (Recommended for Beginners)

Start the interactive interface:
```bash
poetry run intelligent-web-scraper
```

This launches a beautiful, guided interface that will:
- Welcome you with feature overview
- Guide you through configuration options
- Help you formulate scraping requests
- Display results in formatted tables
- Offer export options

### Command Line Mode

For automation and scripting:
```bash
poetry run intelligent-web-scraper \
  --direct \
  --url "https://books.toscrape.com" \
  --request "Extract book titles, prices, and ratings" \
  --max-results 20 \
  --export-format json
```

### Programmatic Usage

```python
import asyncio
from intelligent_web_scraper import (
    IntelligentScrapingOrchestrator,
    IntelligentScrapingConfig
)

async def scrape_example():
    # Create configuration
    config = IntelligentScrapingConfig(
        orchestrator_model="gpt-4o-mini",
        default_quality_threshold=60.0,
        enable_monitoring=True
    )
    
    # Initialize orchestrator
    orchestrator = IntelligentScrapingOrchestrator(config=config)
    
    # Define request
    request = {
        "scraping_request": "Extract product names and prices",
        "target_url": "https://example-store.com",
        "max_results": 10,
        "quality_threshold": 60.0,
        "export_format": "json"
    }
    
    # Execute scraping
    result = await orchestrator.run(request)
    
    # Process results
    print(f"Extracted {len(result.extracted_data)} items")
    print(f"Quality score: {result.quality_score}")
    
    return result

# Run the example
asyncio.run(scrape_example())
```

## Understanding the Architecture

The Intelligent Web Scraper follows a sophisticated multi-agent architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface                           │
│  (Interactive CLI, Direct CLI, Programmatic API)           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│            Intelligent Scraping Orchestrator               │
│  • Coordinates the entire workflow                         │
│  • Manages agent communication                             │
│  • Handles context injection                               │
│  • Monitors performance                                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
┌───────▼──────┐            ┌──────▼──────┐
│ Planning     │            │ Scraper     │
│ Agent        │            │ Tool        │
│              │            │             │
│ • Analyzes   │            │ • Executes  │
│   websites   │            │   scraping  │
│ • Generates  │            │ • Extracts  │
│   strategies │            │   data      │
│ • Creates    │            │ • Validates │
│   schemas    │            │   quality   │
└──────────────┘            └─────────────┘
```

### Key Components

1. **Orchestrator Agent**: The main coordinator that manages the entire workflow
2. **Planning Agent**: AI-powered agent that analyzes websites and creates scraping strategies
3. **Scraper Tool**: Sophisticated tool that executes the planned scraping operations
4. **Context Providers**: Dynamic context injection for enhanced AI capabilities
5. **Monitoring System**: Real-time performance and quality tracking

## Your First Scraping Operation

Let's walk through a complete scraping operation step by step.

### Example: Scraping Book Information

1. **Start the application**:
   ```bash
   poetry run intelligent-web-scraper
   ```

2. **When prompted, enter your scraping request**:
   ```
   Extract book titles, authors, prices, and ratings from this bookstore
   ```

3. **Provide the target URL**:
   ```
   https://books.toscrape.com
   ```

4. **Configure parameters**:
   - Max results: `20`
   - Quality threshold: `60`
   - Export format: `json`

5. **Review and confirm** the request summary

6. **Watch the AI work**:
   - The system analyzes the website structure
   - Plans an optimal scraping strategy
   - Executes the scraping operation
   - Validates and scores the results

7. **Review the results**:
   - Items extracted count
   - Quality score
   - Processing time
   - Sample data preview
   - Export file locations

### Understanding the Output

The system provides comprehensive output including:

```
✅ Scraping completed successfully!
📊 Items extracted: 20
🎯 Quality score: 87.3
⏱️  Processing time: 12.45s

🧠 AI Reasoning:
The website uses a standard e-commerce layout with product cards.
I identified book titles in h3 tags, prices in .price_color classes,
and ratings in star-rating classes. The extraction strategy focused
on these consistent patterns across all product listings.

💾 Export options:
JSON: ./results/scraping_results_20250809_041847.json
CSV: ./results/scraping_results_20250809_041847.csv
```

## Working with Different Data Types

The Intelligent Web Scraper can handle various types of websites and data:

### E-commerce Sites

```python
request = {
    "scraping_request": "Extract product names, prices, descriptions, and customer ratings",
    "target_url": "https://example-store.com/products",
    "max_results": 50,
    "quality_threshold": 70.0
}
```

**Common patterns detected**:
- Product listings and grids
- Price information
- Customer reviews and ratings
- Product specifications
- Availability status

### News Websites

```python
request = {
    "scraping_request": "Get article headlines, publication dates, authors, and summaries",
    "target_url": "https://news-site.com",
    "max_results": 30,
    "quality_threshold": 80.0
}
```

**Common patterns detected**:
- Article headlines and titles
- Publication timestamps
- Author information
- Article summaries/excerpts
- Category tags

### Directory Listings

```python
request = {
    "scraping_request": "Extract business names, addresses, phone numbers, and websites",
    "target_url": "https://business-directory.com",
    "max_results": 100,
    "quality_threshold": 60.0
}
```

**Common patterns detected**:
- Business listings
- Contact information
- Address data
- Website URLs
- Business categories

### Job Boards

```python
request = {
    "scraping_request": "Find job titles, companies, locations, salaries, and requirements",
    "target_url": "https://job-board.com/search",
    "max_results": 25,
    "quality_threshold": 75.0
}
```

**Common patterns detected**:
- Job postings
- Company information
- Location data
- Salary ranges
- Job requirements

## Configuration and Customization

### Environment Variables

The most common way to configure the system:

```bash
# Core settings
ORCHESTRATOR_MODEL=gpt-4                    # Use GPT-4 for better quality
PLANNING_AGENT_MODEL=gpt-4o-mini           # Use mini for cost efficiency
DEFAULT_QUALITY_THRESHOLD=75.0             # Higher quality threshold

# Performance settings
MAX_CONCURRENT_REQUESTS=8                   # More concurrent requests
REQUEST_DELAY=0.5                          # Faster requests
MAX_INSTANCES=3                            # Multiple scraper instances

# Monitoring
ENABLE_MONITORING=true                      # Enable real-time monitoring
MONITORING_INTERVAL=2.0                    # Update every 2 seconds
```

### Configuration Files

For complex setups, use JSON configuration files:

```json
{
  "orchestrator_model": "gpt-4",
  "planning_agent_model": "gpt-4o-mini",
  "default_quality_threshold": 80.0,
  "max_concurrent_requests": 10,
  "request_delay": 1.0,
  "enable_monitoring": true,
  "monitoring_interval": 1.0,
  "max_instances": 5,
  "results_directory": "/app/results",
  "respect_robots_txt": true,
  "enable_rate_limiting": true
}
```

Use with:
```bash
intelligent-web-scraper --config production.json
```

### Programmatic Configuration

```python
from intelligent_web_scraper import IntelligentScrapingConfig

# Create custom configuration
config = IntelligentScrapingConfig(
    orchestrator_model="gpt-4",
    planning_agent_model="gpt-4o-mini",
    default_quality_threshold=85.0,
    max_concurrent_requests=12,
    request_delay=0.8,
    enable_monitoring=True,
    monitoring_interval=1.5,
    max_instances=4,
    max_workers=16,
    max_async_tasks=60,
    results_directory="./custom_results",
    default_export_format="excel"
)
```

## Monitoring and Performance

### Real-Time Monitoring Dashboard

When monitoring is enabled, you'll see a live dashboard:

```
┌─ Intelligent Scraper Monitoring Dashboard ─┐
│                                             │
│ ┌─ Scraper Instances ─────────────────────┐ │
│ │ Instance ID │ Status │ Success Rate    │ │
│ │ scraper-001 │ ACTIVE │ 94.2%          │ │
│ │ scraper-002 │ IDLE   │ 97.8%          │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ ┌─ Performance ─┐ ┌─ Resources ──────────┐ │
│ │ Throughput:   │ │ Memory: 245.3 MB    │ │
│ │ 3.2 req/sec   │ │ CPU: 23.4%          │ │
│ │ Success: 96%  │ │ Network: 1.2 Mbps   │ │
│ └───────────────┘ └─────────────────────┘ │
│                                             │
│ ┌─ Alerts ────────────────────────────────┐ │
│ │ ✅ All systems operational              │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

### Performance Metrics

The system tracks comprehensive metrics:

- **Throughput**: Requests processed per second
- **Success Rate**: Percentage of successful operations
- **Quality Scores**: Average quality of extracted data
- **Response Times**: Average processing time per request
- **Resource Usage**: Memory, CPU, and network utilization
- **Error Rates**: Types and frequency of errors

### Monitoring Reports

After each operation, you get detailed reports:

```python
# Access monitoring report
result = await orchestrator.run(request)
monitoring_report = result.monitoring_report

print(f"Active instances: {monitoring_report.active_instances}")
print(f"Overall throughput: {monitoring_report.overall_throughput:.2f} req/sec")
print(f"Success rate: {monitoring_report.overall_success_rate:.1%}")
print(f"Resource utilization: {monitoring_report.resource_utilization}")
```

## Best Practices

### 1. Ethical Scraping

Always follow ethical scraping practices:

```python
config = IntelligentScrapingConfig(
    respect_robots_txt=True,        # Always respect robots.txt
    enable_rate_limiting=True,      # Enable automatic rate limiting
    request_delay=2.0,              # Add delays between requests
    max_concurrent_requests=3,      # Limit concurrent requests
    user_agent="YourBot/1.0 (+https://yoursite.com/contact)"
)
```

### 2. Quality Over Quantity

Focus on quality rather than speed:

```python
config = IntelligentScrapingConfig(
    default_quality_threshold=75.0,  # Higher quality threshold
    orchestrator_model="gpt-4",      # Better model for quality
    request_delay=1.5,               # Slower but more reliable
)
```

### 3. Monitoring and Alerting

Always enable monitoring for production use:

```python
config = IntelligentScrapingConfig(
    enable_monitoring=True,
    monitoring_interval=5.0,         # Monitor every 5 seconds
    max_instances=5,                 # Multiple instances for reliability
)
```

### 4. Error Handling

Implement proper error handling:

```python
try:
    result = await orchestrator.run(request)
    
    # Check for errors
    if result.metadata.errors_encountered:
        print(f"Errors occurred: {result.metadata.errors_encountered}")
    
    # Validate quality
    if result.quality_score < 60.0:
        print(f"Low quality score: {result.quality_score}")
    
except Exception as e:
    logger.error(f"Scraping failed: {e}")
    # Implement retry logic or fallback
```

### 5. Resource Management

Monitor and manage resources:

```python
config = IntelligentScrapingConfig(
    max_concurrent_requests=5,       # Don't overwhelm targets
    max_workers=8,                   # Limit CPU usage
    max_async_tasks=30,              # Control memory usage
    monitoring_interval=2.0,         # Regular monitoring
)
```

### 6. Data Validation

Always validate extracted data:

```python
result = await orchestrator.run(request)

# Check data quality
if result.quality_score < config.default_quality_threshold:
    print("Warning: Low quality data detected")

# Validate data structure
for item in result.extracted_data:
    if not isinstance(item, dict):
        print("Warning: Unexpected data format")
    
    # Check for required fields
    required_fields = ['title', 'price']
    missing_fields = [f for f in required_fields if f not in item]
    if missing_fields:
        print(f"Warning: Missing fields {missing_fields}")
```

## Next Steps

Now that you understand the basics, explore these advanced topics:

### 1. Advanced Examples

Run the advanced examples to see more sophisticated usage:

```bash
# Advanced orchestration patterns
poetry run python examples/advanced_orchestration_example.py

# Real-time monitoring dashboard
poetry run python examples/monitoring_dashboard_demo.py

# Performance optimization
poetry run python examples/performance_monitoring_example.py

# Export format options
poetry run python examples/export_example.py
```

### 2. Custom Context Providers

Learn to create custom context providers for specialized use cases:

```python
from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase

class CustomContextProvider(SystemPromptContextProviderBase):
    def __init__(self, title: str = "Custom Context"):
        super().__init__(title=title)
        self.custom_data = {}
    
    def get_info(self) -> str:
        return f"Custom context: {self.custom_data}"

# Register with orchestrator
orchestrator.register_context_provider("custom", CustomContextProvider())
```

### 3. Integration Patterns

Integrate with other systems and workflows:

```python
# Database integration
import sqlite3

async def scrape_and_store():
    result = await orchestrator.run(request)
    
    # Store in database
    conn = sqlite3.connect('scraping_results.db')
    for item in result.extracted_data:
        conn.execute(
            "INSERT INTO products (title, price, quality) VALUES (?, ?, ?)",
            (item.get('title'), item.get('price'), result.quality_score)
        )
    conn.commit()
    conn.close()

# API integration
import requests

async def scrape_and_webhook():
    result = await orchestrator.run(request)
    
    # Send to webhook
    webhook_data = {
        'items': result.extracted_data,
        'quality_score': result.quality_score,
        'timestamp': result.metadata.timestamp.isoformat()
    }
    
    requests.post('https://your-webhook.com/scraping-results', json=webhook_data)
```

### 4. Deployment

Deploy to production environments:

```bash
# Docker deployment
docker build -t intelligent-web-scraper .
docker run -e OPENAI_API_KEY=your_key intelligent-web-scraper

# Kubernetes deployment
kubectl apply -f k8s-deployment.yaml

# Systemd service
sudo systemctl enable intelligent-web-scraper
sudo systemctl start intelligent-web-scraper
```

### 5. Monitoring and Observability

Set up comprehensive monitoring:

```bash
# Prometheus metrics
curl http://localhost:8001/metrics

# Grafana dashboards
# Import the provided dashboard configuration

# Health checks
python scripts/health-check.py --json
```

### 6. Contributing

Contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### 7. Community and Support

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Documentation**: Contribute to documentation improvements
- **Examples**: Share your own usage examples

## Conclusion

You now have a solid foundation for using the Intelligent Web Scraper! This powerful tool demonstrates advanced Atomic Agents patterns while providing practical web scraping capabilities.

Key takeaways:
- Start with the interactive mode for learning
- Use natural language to describe scraping needs
- Monitor quality and performance metrics
- Follow ethical scraping practices
- Leverage the monitoring and error handling features
- Explore advanced examples for complex use cases

Happy scraping! 🚀