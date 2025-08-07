# Code Documentation Standards

This document defines the documentation standards for the Intelligent Web Scraper project, ensuring consistent, comprehensive, and maintainable code documentation.

## Table of Contents

1. [Documentation Philosophy](#documentation-philosophy)
2. [Docstring Standards](#docstring-standards)
3. [Type Annotations](#type-annotations)
4. [Inline Comments](#inline-comments)
5. [API Documentation](#api-documentation)
6. [Examples and Usage](#examples-and-usage)
7. [Architecture Documentation](#architecture-documentation)

## Documentation Philosophy

Our documentation follows these core principles:

1. **Clarity Over Brevity**: Better to be clear and verbose than concise and confusing
2. **Educational Value**: Documentation should teach atomic-agents patterns and best practices
3. **Practical Examples**: Every public method should have usage examples
4. **Maintenance**: Documentation is code and must be maintained with the same rigor
5. **Accessibility**: Documentation should be accessible to developers of all skill levels

## Docstring Standards

We use Google-style docstrings with Pydantic-style field documentation for schemas.

### Class Docstrings

```python
class ExampleClass:
    """
    Brief one-line description of the class.
    
    Longer description explaining the purpose, design decisions, and
    how this class fits into the larger system architecture. Include
    information about atomic-agents patterns demonstrated.
    
    This class demonstrates [specific pattern] by [explanation].
    It integrates with [other components] to provide [functionality].
    
    Attributes:
        attribute_name (type): Description of the attribute, including
            its purpose, valid values, and any constraints.
        another_attribute (Optional[type]): Description with default
            value information and when it might be None.
    
    Example:
        Basic usage example:
        
        ```python
        instance = ExampleClass(
            attribute_name="value",
            another_attribute=42
        )
        result = instance.method_name()
        ```
        
        Advanced usage with context:
        
        ```python
        # Set up context
        context = SomeContext()
        
        # Use with context
        instance = ExampleClass.from_context(context)
        result = await instance.async_method()
        ```
    
    Note:
        Important implementation details, limitations, or warnings
        that users should be aware of.
    
    See Also:
        RelatedClass: For related functionality
        another_module.SimilarClass: For alternative approaches
    """
```

### Method Docstrings

```python
async def example_method(
    self,
    required_param: str,
    optional_param: Optional[int] = None,
    **kwargs: Any
) -> ExampleResult:
    """
    Brief description of what the method does.
    
    Longer description explaining the method's purpose, behavior,
    and any important implementation details. Explain how this
    method fits into the overall workflow.
    
    This method demonstrates [atomic-agents pattern] by [explanation].
    It coordinates with [other components] to achieve [goal].
    
    Args:
        required_param (str): Description of the required parameter,
            including valid values, format requirements, and examples.
        optional_param (Optional[int], optional): Description of the
            optional parameter, including default behavior when None.
            Defaults to None.
        **kwargs: Additional keyword arguments passed to underlying
            components. Common options include:
            - timeout (float): Request timeout in seconds
            - retries (int): Number of retry attempts
    
    Returns:
        ExampleResult: Description of the return value, including
            the structure of complex objects and what each field
            represents.
    
    Raises:
        ValueError: When required_param is empty or invalid format
        TimeoutError: When operation exceeds specified timeout
        CustomError: When specific business logic conditions fail
    
    Example:
        Basic usage:
        
        ```python
        result = await instance.example_method("input_value")
        print(f"Result: {result.output}")
        ```
        
        With optional parameters:
        
        ```python
        result = await instance.example_method(
            required_param="input_value",
            optional_param=42,
            timeout=30.0,
            retries=3
        )
        ```
        
        Error handling:
        
        ```python
        try:
            result = await instance.example_method("input")
        except ValueError as e:
            logger.error(f"Invalid input: {e}")
        except TimeoutError:
            logger.warning("Operation timed out, retrying...")
        ```
    
    Note:
        Important performance considerations, thread safety notes,
        or other implementation details that affect usage.
    """
```

### Schema Docstrings

```python
class ExampleInputSchema(BaseIOSchema):
    """
    Input schema for example operations.
    
    This schema demonstrates proper atomic-agents input validation
    patterns with comprehensive field documentation and validation.
    
    The schema ensures type safety and provides clear API contracts
    for agent interactions.
    """
    
    required_field: str = Field(
        ...,
        description="Required string field with specific format requirements. "
                   "Must be non-empty and contain only alphanumeric characters. "
                   "Example: 'user_input_123'"
    )
    
    optional_field: Optional[int] = Field(
        default=None,
        description="Optional integer field for advanced configuration. "
                   "When None, system uses automatic detection. "
                   "Valid range: 1-100. Example: 42"
    )
    
    validated_field: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Float field with validation constraints. "
                   "Represents a percentage value between 0 and 100. "
                   "Used for quality thresholds and confidence scores. "
                   "Example: 75.5"
    )
    
    enum_field: ExampleEnum = Field(
        default=ExampleEnum.DEFAULT,
        description="Enumerated field with predefined valid values. "
                   "Controls the processing mode for the operation. "
                   "See ExampleEnum for available options."
    )
```

## Type Annotations

All code must include comprehensive type annotations:

### Function Signatures

```python
from typing import Dict, List, Optional, Union, Any, Callable, Awaitable

# Simple types
def process_data(data: str) -> int:
    """Process string data and return count."""
    return len(data)

# Complex types
async def fetch_results(
    urls: List[str],
    timeout: Optional[float] = None,
    callback: Optional[Callable[[str], Awaitable[Dict[str, Any]]]] = None
) -> Dict[str, Union[Dict[str, Any], Exception]]:
    """Fetch results from multiple URLs with optional callback."""
    pass

# Generic types
from typing import TypeVar, Generic

T = TypeVar('T')

class Container(Generic[T]):
    """Generic container demonstrating type variable usage."""
    
    def __init__(self, item: T) -> None:
        self.item = item
    
    def get_item(self) -> T:
        return self.item
```

### Class Attributes

```python
class ExampleClass:
    """Example class with typed attributes."""
    
    # Class variables
    DEFAULT_TIMEOUT: float = 30.0
    SUPPORTED_FORMATS: List[str] = ["json", "csv", "xml"]
    
    def __init__(self, config: ExampleConfig) -> None:
        # Instance variables with types
        self.config: ExampleConfig = config
        self.results: List[Dict[str, Any]] = []
        self.status: Optional[str] = None
        self.callbacks: Dict[str, Callable[[], None]] = {}
```

## Inline Comments

Use inline comments to explain complex logic, design decisions, and non-obvious code:

### Good Inline Comments

```python
async def complex_processing(data: List[Dict[str, Any]]) -> ProcessedData:
    """Process complex data with multiple stages."""
    
    # Filter out invalid entries early to reduce processing load
    # Invalid entries are those missing required fields or with null values
    valid_entries = [
        entry for entry in data 
        if entry.get("id") and entry.get("content")
    ]
    
    # Use ThreadPoolExecutor for CPU-intensive processing
    # This prevents blocking the event loop during heavy computation
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Process entries in parallel, maintaining order
        futures = [
            executor.submit(process_single_entry, entry)
            for entry in valid_entries
        ]
        
        # Collect results as they complete
        # Using as_completed to handle failures gracefully
        processed_entries = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=30)
                processed_entries.append(result)
            except Exception as e:
                # Log error but continue processing other entries
                logger.warning(f"Failed to process entry: {e}")
    
    # Sort by processing timestamp to maintain consistent output
    # This is important for reproducible results in testing
    processed_entries.sort(key=lambda x: x.get("processed_at", 0))
    
    return ProcessedData(entries=processed_entries)

def calculate_quality_score(item: Dict[str, Any]) -> float:
    """Calculate quality score using weighted metrics."""
    
    score = 0.0
    
    # Content completeness (40% of total score)
    # Higher weight because content is the primary value
    required_fields = ["title", "content", "url"]
    completeness = sum(1 for field in required_fields if item.get(field)) / len(required_fields)
    score += completeness * 40.0
    
    # Data accuracy (30% of total score)
    # Based on format validation and consistency checks
    accuracy = validate_data_accuracy(item)
    score += accuracy * 30.0
    
    # Freshness (20% of total score)
    # Recent data is more valuable
    freshness = calculate_freshness(item.get("timestamp"))
    score += freshness * 20.0
    
    # Uniqueness (10% of total score)
    # Penalize duplicate or near-duplicate content
    uniqueness = calculate_uniqueness(item)
    score += uniqueness * 10.0
    
    return min(score, 100.0)  # Cap at 100%
```

### Avoid These Comment Patterns

```python
# Bad: Obvious comments
x = x + 1  # Increment x

# Bad: Outdated comments
# TODO: Fix this bug (from 2019)
def working_function():
    pass

# Bad: Commented-out code
# old_implementation()
new_implementation()

# Bad: Vague comments
# Do some stuff
complex_operation()
```

## API Documentation

### Public API Documentation

All public APIs must have comprehensive documentation:

```python
class PublicAPI:
    """
    Public API class for external integrations.
    
    This class provides the main interface for external systems
    to interact with the intelligent web scraper. It demonstrates
    proper API design patterns and error handling.
    """
    
    def __init__(self, config: APIConfig) -> None:
        """
        Initialize the API with configuration.
        
        Args:
            config: API configuration including authentication,
                rate limits, and feature flags.
        """
        self._config = config
        self._validate_config()
    
    async def scrape_website(
        self,
        request: ScrapeRequest
    ) -> ScrapeResponse:
        """
        Scrape a website with intelligent extraction.
        
        This is the main entry point for scraping operations.
        It handles request validation, execution, and response
        formatting according to API standards.
        
        Args:
            request: Scraping request with URL, extraction rules,
                and output preferences.
        
        Returns:
            ScrapeResponse: Structured response with extracted data,
                metadata, and quality metrics.
        
        Raises:
            ValidationError: When request parameters are invalid
            AuthenticationError: When API credentials are invalid
            RateLimitError: When rate limits are exceeded
            ScrapingError: When scraping operation fails
        
        Example:
            ```python
            api = PublicAPI(config)
            
            request = ScrapeRequest(
                url="https://example.com",
                extraction_rules={"title": "h1", "content": "p"},
                max_results=10
            )
            
            response = await api.scrape_website(request)
            print(f"Extracted {len(response.data)} items")
            ```
        """
        # Implementation with detailed error handling
        pass
```

### Internal API Documentation

Internal APIs should also be well-documented:

```python
class _InternalProcessor:
    """
    Internal processor for data transformation.
    
    This class is not part of the public API and may change
    without notice. It handles internal data processing
    operations with specific performance optimizations.
    
    Note:
        This is an internal class. Use PublicAPI for external
        integrations.
    """
    
    def _process_batch(self, batch: List[RawData]) -> List[ProcessedData]:
        """
        Process a batch of raw data items.
        
        Internal method that applies transformation rules
        and quality filters to raw scraped data.
        
        Args:
            batch: List of raw data items from scraper
        
        Returns:
            List of processed and validated data items
        
        Note:
            This method assumes input data has been pre-validated.
            It may raise exceptions for malformed data.
        """
        pass
```

## Examples and Usage

Every public method should include practical examples:

### Basic Examples

```python
def simple_method(input_data: str) -> str:
    """
    Simple method demonstrating basic usage patterns.
    
    Args:
        input_data: Input string to process
    
    Returns:
        Processed string result
    
    Example:
        ```python
        result = simple_method("hello world")
        print(result)  # Output: "HELLO WORLD"
        ```
    """
    return input_data.upper()
```

### Advanced Examples

```python
async def advanced_method(
    self,
    config: AdvancedConfig,
    **options: Any
) -> AdvancedResult:
    """
    Advanced method with comprehensive examples.
    
    Example:
        Basic usage:
        ```python
        config = AdvancedConfig(mode="standard")
        result = await instance.advanced_method(config)
        ```
        
        With custom options:
        ```python
        config = AdvancedConfig(
            mode="advanced",
            quality_threshold=80.0
        )
        
        result = await instance.advanced_method(
            config,
            timeout=60,
            retries=3,
            callback=custom_callback
        )
        ```
        
        Error handling:
        ```python
        try:
            result = await instance.advanced_method(config)
            process_result(result)
        except ValidationError as e:
            logger.error(f"Configuration error: {e}")
            # Handle configuration issues
        except TimeoutError:
            logger.warning("Operation timed out")
            # Handle timeout with fallback
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            # Handle unexpected errors
        ```
        
        Integration with monitoring:
        ```python
        with monitoring.track_operation("advanced_method"):
            result = await instance.advanced_method(config)
            monitoring.record_success(result.quality_score)
        ```
    """
    pass
```

## Architecture Documentation

### Architectural Decision Records (ADRs)

Document major design decisions:

```markdown
# ADR-XXX: Decision Title

## Status
[Proposed | Accepted | Deprecated | Superseded]

## Context
What is the issue that we're seeing that is motivating this decision or change?

## Decision
What is the change that we're proposing or have agreed to implement?

## Consequences
What becomes easier or more difficult to do and any risks introduced by this change?
```

### Design Documentation

```python
"""
Module: intelligent_web_scraper.agents.orchestrator

This module implements the main orchestrator agent that demonstrates
advanced atomic-agents patterns for coordinating complex multi-agent
workflows.

Architecture Overview:
    The orchestrator follows the Coordinator pattern, managing interactions
    between specialized agents and tools:
    
    1. Planning Agent: Analyzes requests and generates strategies
    2. Scraper Tool: Executes scraping operations
    3. Context Providers: Inject dynamic information
    4. Instance Manager: Manages concurrent operations
    5. Export Manager: Handles result formatting
    
    Data flows through the system as follows:
    User Request -> Context Gathering -> Strategy Planning -> 
    Execution Coordination -> Result Processing -> Export

Design Patterns:
    - Coordinator Pattern: Central coordination of distributed operations
    - Strategy Pattern: Pluggable scraping strategies
    - Observer Pattern: Monitoring and event handling
    - Factory Pattern: Tool and instance creation
    - Context Provider Pattern: Dynamic prompt enhancement

Performance Considerations:
    - Async/await throughout for non-blocking operations
    - Thread pools for CPU-intensive tasks
    - Connection pooling for HTTP requests
    - Caching for repeated operations
    - Resource cleanup and lifecycle management

Error Handling Strategy:
    - Graceful degradation with partial results
    - Retry logic with exponential backoff
    - Circuit breaker pattern for external services
    - Comprehensive error categorization and reporting
"""
```

## Documentation Maintenance

### Review Process

1. **Code Review**: All documentation changes must be reviewed
2. **Accuracy Check**: Verify examples work and are up-to-date
3. **Consistency Check**: Ensure consistent style and terminology
4. **Completeness Check**: Verify all public APIs are documented

### Automated Checks

```python
# Example documentation test
def test_example_in_docstring():
    """Test that docstring examples actually work."""
    # Extract and execute examples from docstrings
    # Verify they produce expected results
    pass

def test_docstring_completeness():
    """Test that all public methods have docstrings."""
    # Check that all public methods have comprehensive docstrings
    # Verify required sections are present
    pass
```

### Documentation Metrics

Track documentation quality:
- Docstring coverage percentage
- Example coverage percentage
- Documentation freshness (last updated)
- User feedback on documentation clarity

This comprehensive documentation standard ensures that the Intelligent Web Scraper serves as an excellent educational example while maintaining production-quality code documentation.