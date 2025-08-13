# 🤖 Intelligent Web Scraper

> **AI-Powered Web Scraping with Natural Language Interface**

A production-ready, standalone web scraping solution that uses AI to understand your scraping needs in plain English and automatically extract structured data from websites. Built on the atomic-agents framework for reliability and extensibility.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue.svg)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](#testing)

## ✨ What Makes This Special?

Instead of writing complex scraping code, just tell the AI what you want:

```bash
# Interactive mode - just describe what you need
intelligent-web-scraper

# Direct mode - one command, structured results
intelligent-web-scraper --direct \
  --url "https://news.ycombinator.com" \
  --request "Extract article titles, scores, and comment counts" \
  --export-format json
```

The AI will:
- 🧠 **Analyze the website** structure automatically
- 📋 **Plan the optimal strategy** for data extraction  
- 🔍 **Extract structured data** with quality validation
- 📊 **Export results** in your preferred format (JSON, CSV, Excel, Markdown)
- 📈 **Monitor performance** in real-time

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- Poetry (for dependency management)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-org/intelligent-web-scraper.git
cd intelligent-web-scraper
```

2. **Install dependencies:**
```bash
poetry install
```

3. **Set up your API key:**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

4. **Run your first scraping job:**
```bash
# Interactive mode with guided setup
poetry run intelligent-web-scraper

# Or try the demo
poetry run python demo_scraper.py
```

## 🎯 Usage Examples

### Interactive Mode (Recommended for Beginners)
```bash
poetry run intelligent-web-scraper
```
The interactive mode guides you through:
- Describing what data you want to extract
- Entering the target URL
- Configuring quality thresholds and export options
- Real-time progress monitoring

### Direct Mode (Perfect for Automation)
```bash
# Extract product information
poetry run intelligent-web-scraper --direct \
  --url "https://example-store.com/products" \
  --request "Get product names, prices, and ratings" \
  --max-results 20 \
  --quality-threshold 80 \
  --export-format csv

# News article extraction
poetry run iws --direct \
  --url "https://example-news.com" \
  --request "Extract headlines, authors, and publication dates" \
  --export-format json

# Job listings scraping
poetry run intelligent-web-scraper --direct \
  --url "https://jobs.example.com/search" \
  --request "Get job titles, companies, locations, and salaries" \
  --max-results 50 \
  --export-format excel
```

### Python API Usage
```python
import asyncio
from intelligent_web_scraper.config import IntelligentScrapingConfig
from intelligent_web_scraper.agents.orchestrator import IntelligentScrapingOrchestrator

async def scrape_data():
    # Configure the scraper
    config = IntelligentScrapingConfig(
        orchestrator_model="gpt-4o-mini",
        openai_api_key="your-api-key",
        default_quality_threshold=75.0
    )
    
    # Create orchestrator
    orchestrator = IntelligentScrapingOrchestrator(config=config)
    
    # Define scraping request
    request = {
        "scraping_request": "Extract product names and prices",
        "target_url": "https://example.com/products",
        "max_results": 10,
        "export_format": "json"
    }
    
    # Execute scraping
    result = await orchestrator.run(request)
    return result

# Run the scraper
result = asyncio.run(scrape_data())
print(f"Extracted {len(result.extracted_data)} items")
```

## 🛠️ Configuration

### Environment Variables (.env file)
```bash
# Required: OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
ORCHESTRATOR_MODEL=gpt-4o-mini
PLANNING_AGENT_MODEL=gpt-4o-mini

# Optional: Scraping Configuration
DEFAULT_QUALITY_THRESHOLD=75.0
MAX_CONCURRENT_REQUESTS=5
REQUEST_DELAY=1.0
EXPORT_FORMAT=json
RESULTS_DIRECTORY=./results

# Optional: Monitoring
ENABLE_MONITORING=true
MONITORING_INTERVAL=1.0

# Optional: Compliance
RESPECT_ROBOTS_TXT=true
ENABLE_RATE_LIMITING=true
```

### Command Line Options
```bash
intelligent-web-scraper --help

Options:
  --direct                    Run in direct mode with provided parameters
  --url URL                   Target URL to scrape
  --request REQUEST           Natural language scraping request
  --max-results N             Maximum number of results (default: 10)
  --quality-threshold N       Quality threshold 0-100 (default: 50.0)
  --export-format FORMAT      Export format: json, csv, markdown, excel
  --output-dir DIR            Output directory for results
  --enable-monitoring         Enable real-time monitoring
  --concurrent-instances N    Number of concurrent scraper instances
  --verbose                   Enable verbose output
  --quiet                     Suppress non-essential output
```

## 📊 Features

### 🧠 AI-Powered Intelligence
- **Natural Language Processing**: Describe what you want in plain English
- **Automatic Website Analysis**: AI analyzes page structure and content patterns
- **Intelligent Strategy Planning**: Generates optimal extraction strategies
- **Quality Assessment**: Automatic data quality scoring and validation

### 🚀 Production Ready
- **Standalone Operation**: No external dependencies beyond Python packages
- **Multiple Interfaces**: CLI, Python API, and interactive modes
- **Real-time Monitoring**: Live progress tracking and performance metrics
- **Error Recovery**: Robust handling of network issues and parsing errors
- **Rate Limiting**: Respectful scraping with configurable delays
- **Export Options**: JSON, CSV, Excel, and Markdown formats

### ⚡ Performance & Scalability
- **Concurrent Processing**: Efficient parallel scraping operations
- **Resource Management**: Intelligent memory and CPU usage optimization
- **Caching**: Smart caching for repeated operations
- **Monitoring Dashboard**: Real-time performance visualization

### 🛡️ Compliance & Ethics
- **Robots.txt Respect**: Automatic robots.txt compliance checking
- **Rate Limiting**: Configurable request delays and throttling
- **User Agent**: Proper identification and contact information
- **Legal Compliance**: Built-in safeguards for responsible scraping

## 🏗️ Architecture

### System Overview
```
┌─────────────────────────────────────┐
│           USER INTERFACE            │
├─────────────────────────────────────┤
│ • CLI Commands (iws, intelligent-   │
│   web-scraper)                      │
│ • Interactive Mode                  │
│ • Python API                       │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│        AI ORCHESTRATION LAYER       │
├─────────────────────────────────────┤
│ • Natural Language Processing       │
│ • Strategy Planning Agent           │
│ • Context Providers                 │
│ • Quality Assessment                │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│       SCRAPING EXECUTION LAYER      │
├─────────────────────────────────────┤
│ • Website Analysis                  │
│ • Content Extraction                │
│ • Data Processing                   │
│ • Error Recovery                    │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│      MONITORING & EXPORT LAYER      │
├─────────────────────────────────────┤
│ • Real-time Monitoring              │
│ • Performance Metrics               │
│ • Multi-format Export               │
│ • Results Validation                │
└─────────────────────────────────────┘
```

### Key Components

1. **Intelligent Orchestrator**: Coordinates all scraping operations using AI
2. **Planning Agent**: Analyzes websites and generates extraction strategies
3. **Scraper Tool**: Executes scraping with error recovery and quality control
4. **Context Providers**: Supply dynamic context for enhanced AI decision-making
5. **Performance Monitor**: Tracks metrics and provides real-time feedback
6. **Export Manager**: Handles multiple output formats and file management

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
poetry run pytest

# Run specific test categories
poetry run pytest tests/test_config.py -v
poetry run pytest tests/test_orchestrator.py -v

# Run real-world validation tests
poetry run python test_real_world.py

# Run interactive demo
poetry run python demo_scraper.py

# Run comprehensive system test
poetry run python comprehensive_test.py
```

### Test Coverage
- ✅ Configuration management and validation
- ✅ AI orchestrator and planning agent integration
- ✅ Context providers and dynamic injection
- ✅ Performance monitoring and metrics collection
- ✅ Export functionality and format validation
- ✅ Error handling and recovery mechanisms
- ✅ CLI interface and command processing

## 🔧 Development

### Project Structure
```
intelligent-web-scraper/
├── intelligent_web_scraper/          # Main package
│   ├── agents/                       # AI agents (orchestrator, planning)
│   ├── context_providers/            # Dynamic context injection
│   ├── monitoring/                   # Performance monitoring
│   ├── tools/                        # Scraping tools integration
│   ├── config.py                     # Configuration management
│   ├── main.py                       # Main application
│   └── cli.py                        # Command-line interface
├── tests/                            # Test suite
├── examples/                         # Usage examples
├── docs/                            # Documentation
├── .env.example                     # Environment template
├── pyproject.toml                   # Poetry configuration
└── README.md                        # This file
```

### Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `poetry run pytest`
5. Submit a pull request

### Built With
- **[Atomic Agents](https://github.com/atomic-agents/atomic-agents)**: AI agent framework
- **[OpenAI GPT](https://openai.com/)**: Language model for intelligence
- **[Rich](https://github.com/Textualize/rich)**: Beautiful terminal interfaces
- **[Pydantic](https://pydantic.dev/)**: Data validation and settings
- **[Poetry](https://python-poetry.org/)**: Dependency management

## 📈 Performance

### Benchmarks
- **Response Time**: Average 2-5 seconds per page analysis
- **Throughput**: 10-50 pages per minute (depending on complexity)
- **Memory Usage**: ~100-500MB during operation
- **Accuracy**: 85-95% data extraction quality (varies by site)

### Optimization Tips
1. **Use appropriate quality thresholds** (higher = more selective)
2. **Configure concurrent requests** based on target site capacity
3. **Enable monitoring** for performance insights
4. **Use caching** for repeated operations on similar sites

## 🤝 Support

### Getting Help
- **Documentation**: Check this README and inline code documentation
- **Issues**: Report bugs and request features on GitHub Issues
- **Discussions**: Join community discussions for usage questions

### Common Issues
1. **API Key Errors**: Ensure your OpenAI API key is valid and has sufficient credits
2. **Rate Limiting**: Some sites may block requests; adjust delays and respect robots.txt
3. **Quality Issues**: Lower quality thresholds for more results, higher for better accuracy
4. **Memory Usage**: Monitor resource usage for large-scale operations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the excellent [Atomic Agents](https://github.com/atomic-agents/atomic-agents) framework
- Inspired by the need for intelligent, user-friendly web scraping tools
- Thanks to the open-source community for the amazing libraries that make this possible

---

**Ready to start scraping intelligently?** 🚀

```bash
poetry run intelligent-web-scraper
```

*Transform any website into structured data with the power of AI!*