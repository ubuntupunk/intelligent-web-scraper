# Contributing to Intelligent Web Scraper

Thank you for your interest in contributing to the Intelligent Web Scraper! This project serves as an advanced example of Atomic Agents patterns and welcomes contributions that improve its educational value, functionality, and reliability.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Types of Contributions](#types-of-contributions)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Documentation](#documentation)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)
- [Community](#community)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. By participating, you agree to:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them learn
- Focus on constructive feedback and collaboration
- Respect different viewpoints and experiences
- Report any unacceptable behavior to the maintainers

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.11 or higher
- Poetry for dependency management
- Git for version control
- A GitHub account
- Basic understanding of Atomic Agents framework
- Familiarity with async Python programming

### First Steps

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/intelligent-web-scraper.git
   cd intelligent-web-scraper
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/atomic-agents/intelligent-web-scraper.git
   ```

## Development Setup

### 1. Install Dependencies

```bash
# Install all dependencies including development tools
poetry install --with dev

# Activate the virtual environment
poetry shell
```

### 2. Set Up Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# At minimum, add your OpenAI API key
```

### 3. Verify Setup

```bash
# Run health check
poetry run python scripts/health-check.py --verbose

# Run tests
poetry run pytest

# Run linting
poetry run black --check .
poetry run isort --check-only .
poetry run flake8
```

### 4. Set Up Pre-commit Hooks

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Test the hooks
poetry run pre-commit run --all-files
```

## Contributing Guidelines

### General Principles

1. **Educational Value**: This is an example project, so code should be clear, well-documented, and demonstrate best practices
2. **Atomic Agents Patterns**: Follow established Atomic Agents conventions and patterns
3. **Production Ready**: Code should demonstrate production-ready patterns including error handling, monitoring, and testing
4. **Backwards Compatibility**: Maintain compatibility with existing APIs unless there's a compelling reason for breaking changes
5. **Performance**: Consider performance implications, especially for concurrent operations

### What We're Looking For

- **Bug fixes** with clear reproduction steps and tests
- **Feature enhancements** that demonstrate advanced Atomic Agents patterns
- **Documentation improvements** that help users understand concepts
- **Example additions** that show different use cases
- **Performance optimizations** with benchmarks
- **Test coverage improvements**
- **Monitoring and observability enhancements**

### What We're Not Looking For

- Breaking changes without strong justification
- Features that don't align with the educational mission
- Code that doesn't follow Atomic Agents patterns
- Contributions without tests or documentation
- Performance changes without benchmarks

## Types of Contributions

### 🐛 Bug Reports

When reporting bugs, please include:

```markdown
**Bug Description**
A clear description of what the bug is.

**Reproduction Steps**
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g. Ubuntu 22.04]
- Python version: [e.g. 3.11.2]
- Poetry version: [e.g. 1.7.1]
- Package version: [e.g. 0.1.0]

**Additional Context**
- Error logs
- Configuration files
- Screenshots (if applicable)
```

### 🚀 Feature Requests

For feature requests, please include:

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Use Case**
Describe the problem this feature would solve.

**Proposed Solution**
How you envision this feature working.

**Atomic Agents Alignment**
How this feature demonstrates or enhances Atomic Agents patterns.

**Educational Value**
How this feature helps users learn advanced concepts.

**Alternatives Considered**
Other approaches you've considered.
```

### 📝 Documentation

Documentation contributions are highly valued:

- **API documentation** improvements
- **Tutorial enhancements** with more examples
- **Architecture explanations** for complex components
- **Best practices guides** for different scenarios
- **Troubleshooting guides** for common issues
- **Performance tuning guides**

### 🧪 Testing

Testing contributions help ensure reliability:

- **Unit tests** for individual components
- **Integration tests** for component interactions
- **End-to-end tests** for complete workflows
- **Performance tests** with benchmarks
- **Error handling tests** for edge cases
- **Concurrency tests** for thread safety

### 📊 Examples

Example contributions demonstrate usage patterns:

- **Basic usage examples** for beginners
- **Advanced orchestration patterns** for complex workflows
- **Integration examples** with other systems
- **Performance optimization examples**
- **Custom context provider examples**
- **Monitoring and alerting examples**

## Development Workflow

### 1. Create a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clear, well-documented code
- Follow the established code style
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Test Your Changes

```bash
# Run the full test suite
poetry run pytest

# Run specific tests
poetry run pytest tests/test_your_feature.py

# Run integration tests
poetry run pytest tests/test_integration.py

# Check code style
poetry run black --check .
poetry run isort --check-only .
poetry run flake8

# Run health check
poetry run python scripts/health-check.py
```

### 4. Update Documentation

- Update docstrings for new functions/classes
- Add or update examples if needed
- Update README.md if the change affects usage
- Add entries to CHANGELOG.md

### 5. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a clear message
git commit -m "feat: add advanced monitoring dashboard

- Implement real-time metrics collection
- Add Grafana dashboard configuration
- Include performance benchmarking
- Update documentation with monitoring guide

Closes #123"
```

### Commit Message Format

Use conventional commits format:

```
<type>(<scope>): <description>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples:**
```
feat(orchestrator): add context provider registration
fix(scraper): handle timeout errors gracefully
docs(tutorial): add advanced configuration examples
test(integration): add end-to-end workflow tests
```

## Testing

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_config.py
│   ├── test_orchestrator.py
│   └── test_context_providers.py
├── integration/             # Integration tests
│   ├── test_orchestrator_planning.py
│   └── test_end_to_end.py
├── performance/             # Performance tests
│   └── test_benchmarks.py
└── fixtures/                # Test fixtures
    └── sample_data.json
```

### Writing Tests

#### Unit Tests

```python
import pytest
from unittest.mock import Mock, AsyncMock
from intelligent_web_scraper.config import IntelligentScrapingConfig

class TestIntelligentScrapingConfig:
    def test_default_configuration(self):
        """Test default configuration values."""
        config = IntelligentScrapingConfig()
        
        assert config.orchestrator_model == "gpt-4o-mini"
        assert config.default_quality_threshold == 50.0
        assert config.max_concurrent_requests == 5
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            IntelligentScrapingConfig(default_quality_threshold=150.0)
    
    def test_environment_loading(self):
        """Test loading configuration from environment."""
        with patch.dict(os.environ, {"ORCHESTRATOR_MODEL": "gpt-4"}):
            config = IntelligentScrapingConfig.from_env()
            assert config.orchestrator_model == "gpt-4"
```

#### Integration Tests

```python
import pytest
from intelligent_web_scraper import IntelligentScrapingOrchestrator

@pytest.mark.asyncio
async def test_orchestrator_integration():
    """Test orchestrator with real components."""
    config = IntelligentScrapingConfig()
    orchestrator = IntelligentScrapingOrchestrator(config=config)
    
    request = {
        "scraping_request": "Test request",
        "target_url": "https://httpbin.org/json",
        "max_results": 1
    }
    
    result = await orchestrator.run(request)
    
    assert result is not None
    assert hasattr(result, 'extracted_data')
    assert hasattr(result, 'quality_score')
```

#### Performance Tests

```python
import time
import pytest
from intelligent_web_scraper import IntelligentScrapingOrchestrator

@pytest.mark.performance
async def test_concurrent_processing_performance():
    """Test performance of concurrent processing."""
    config = IntelligentScrapingConfig(max_concurrent_requests=5)
    orchestrator = IntelligentScrapingOrchestrator(config=config)
    
    requests = [
        {"scraping_request": f"Test {i}", "target_url": "https://httpbin.org/json"}
        for i in range(10)
    ]
    
    start_time = time.time()
    results = await asyncio.gather(*[
        orchestrator.run(request) for request in requests
    ])
    end_time = time.time()
    
    # Performance assertions
    assert len(results) == 10
    assert end_time - start_time < 30  # Should complete within 30 seconds
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test categories
poetry run pytest tests/unit/
poetry run pytest tests/integration/
poetry run pytest -m performance

# Run with coverage
poetry run pytest --cov=intelligent_web_scraper --cov-report=html

# Run tests in parallel
poetry run pytest -n auto
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
async def scrape_website(
    self, 
    request: Dict[str, Any], 
    timeout: float = 30.0
) -> ScrapingResult:
    """Scrape a website using AI-powered strategy planning.
    
    This method demonstrates the complete scraping workflow including
    website analysis, strategy generation, and data extraction with
    quality scoring.
    
    Args:
        request: Scraping request containing target URL and instructions.
            Must include 'scraping_request' and 'target_url' keys.
        timeout: Maximum time to wait for completion in seconds.
            Defaults to 30.0 seconds.
    
    Returns:
        ScrapingResult containing extracted data, quality metrics,
        and processing metadata.
    
    Raises:
        ValueError: If request is missing required fields.
        TimeoutError: If scraping exceeds the specified timeout.
        ScrapingError: If scraping fails due to website issues.
    
    Example:
        >>> config = IntelligentScrapingConfig()
        >>> orchestrator = IntelligentScrapingOrchestrator(config)
        >>> request = {
        ...     "scraping_request": "Extract product names and prices",
        ...     "target_url": "https://example-store.com"
        ... }
        >>> result = await orchestrator.scrape_website(request)
        >>> print(f"Extracted {len(result.extracted_data)} items")
    
    Note:
        This method respects robots.txt and implements rate limiting
        by default. Configure the scraper settings to adjust these
        behaviors for your specific use case.
    """
```

### Documentation Structure

```
docs/
├── README.md                # Main documentation
├── tutorials/               # Step-by-step guides
│   ├── getting-started.md
│   ├── advanced-usage.md
│   └── extending-the-system.md
├── deployment/              # Deployment guides
│   ├── README.md
│   ├── docker.md
│   └── kubernetes.md
├── api/                     # API documentation
│   ├── orchestrator.md
│   ├── config.md
│   └── context-providers.md
└── architecture/            # Architecture documentation
    ├── overview.md
    ├── agent-patterns.md
    └── monitoring.md
```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 79)
- **String quotes**: Use double quotes for strings
- **Import order**: Use isort with black compatibility
- **Type hints**: Required for all public functions
- **Docstrings**: Required for all public classes and functions

### Formatting Tools

```bash
# Format code
poetry run black .

# Sort imports
poetry run isort .

# Check style
poetry run flake8

# Type checking
poetry run mypy intelligent_web_scraper/
```

### Configuration Files

The project includes configuration for all tools:

- `.flake8`: Flake8 configuration
- `pyproject.toml`: Black, isort, and mypy configuration
- `.pre-commit-config.yaml`: Pre-commit hooks

### Code Organization

```python
# File header
"""
Module description.

This module demonstrates [specific pattern or concept] and provides
[brief description of functionality].
"""

# Imports (organized by isort)
import asyncio
import logging
from typing import Dict, Any, Optional

from atomic_agents.agents.base_agent import BaseAgent
from atomic_agents.lib.base.base_io_schema import BaseIOSchema

from .config import IntelligentScrapingConfig
from .context_providers import WebsiteAnalysisContextProvider

# Constants
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3

# Logger
logger = logging.getLogger(__name__)

# Classes and functions
class ExampleClass:
    """Example class demonstrating code organization."""
    
    def __init__(self, config: IntelligentScrapingConfig):
        """Initialize the example class."""
        self.config = config
        self._internal_state = {}
    
    async def public_method(self, param: str) -> Dict[str, Any]:
        """Public method with proper documentation."""
        return await self._private_method(param)
    
    async def _private_method(self, param: str) -> Dict[str, Any]:
        """Private method for internal use."""
        # Implementation
        pass
```

## Submitting Changes

### Pull Request Process

1. **Ensure your branch is up to date**:
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-feature-branch
   git rebase main
   ```

2. **Push your changes**:
   ```bash
   git push origin your-feature-branch
   ```

3. **Create a pull request** on GitHub with:
   - Clear title describing the change
   - Detailed description of what was changed and why
   - Reference to any related issues
   - Screenshots or examples if applicable

### Pull Request Template

```markdown
## Description
Brief description of the changes and their purpose.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Related Issues
Closes #123
Related to #456

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally
- [ ] Manual testing completed

## Documentation
- [ ] Docstrings updated
- [ ] README updated (if needed)
- [ ] Examples updated (if needed)
- [ ] Changelog updated

## Checklist
- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Code is well-commented and documented
- [ ] No new warnings introduced
- [ ] Performance impact considered
- [ ] Backwards compatibility maintained (or breaking changes justified)

## Screenshots/Examples
If applicable, add screenshots or code examples demonstrating the changes.
```

## Review Process

### What Reviewers Look For

1. **Code Quality**
   - Follows established patterns and conventions
   - Proper error handling and edge cases
   - Clear and maintainable code structure
   - Appropriate use of type hints

2. **Testing**
   - Adequate test coverage for new functionality
   - Tests cover both happy path and error cases
   - Integration tests for component interactions
   - Performance tests for optimization changes

3. **Documentation**
   - Clear docstrings for public APIs
   - Updated examples and tutorials
   - Architecture documentation for complex changes
   - Changelog entries for user-facing changes

4. **Atomic Agents Alignment**
   - Follows Atomic Agents patterns and conventions
   - Demonstrates best practices appropriately
   - Maintains educational value of the project
   - Integrates well with existing components

### Review Timeline

- **Initial review**: Within 2-3 business days
- **Follow-up reviews**: Within 1-2 business days
- **Merge**: After approval from at least one maintainer

### Addressing Review Feedback

1. **Make requested changes** in your feature branch
2. **Respond to comments** explaining your changes
3. **Request re-review** when ready
4. **Be patient and collaborative** throughout the process

## Community

### Getting Help

- **GitHub Discussions**: Ask questions and share ideas
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check existing documentation first
- **Examples**: Look at existing examples for patterns

### Communication Guidelines

- **Be respectful** and professional in all interactions
- **Provide context** when asking questions or reporting issues
- **Search existing issues** before creating new ones
- **Use clear, descriptive titles** for issues and PRs
- **Follow up** on your contributions and respond to feedback

### Recognition

Contributors are recognized in:
- **CONTRIBUTORS.md**: List of all contributors
- **Release notes**: Major contributions highlighted
- **Documentation**: Examples and guides credited to authors
- **GitHub**: Contributor statistics and recognition

## Getting Started Checklist

Before making your first contribution:

- [ ] Read this contributing guide completely
- [ ] Set up the development environment
- [ ] Run the health check successfully
- [ ] Run all tests and ensure they pass
- [ ] Read the existing code to understand patterns
- [ ] Look at recent pull requests for examples
- [ ] Join GitHub discussions to introduce yourself
- [ ] Start with a small contribution (documentation, tests, or bug fix)

## Questions?

If you have questions about contributing:

1. **Check the documentation** first
2. **Search existing issues** and discussions
3. **Create a new discussion** for general questions
4. **Create an issue** for specific bugs or feature requests

Thank you for contributing to the Intelligent Web Scraper! Your contributions help make this a better example of Atomic Agents patterns and a more useful tool for the community. 🚀