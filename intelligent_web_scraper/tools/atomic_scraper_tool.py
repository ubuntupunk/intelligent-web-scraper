"""
Atomic Scraper Tool - Integrated with Intelligent Web Scraper

AI-powered scraping tool adapted for the atomic-agents framework with
proper configuration management, enhanced error handling, and monitoring.
"""

import logging
import time
import os
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import BaseModel, Field, field_validator

from ..config import IntelligentScrapingConfig


logger = logging.getLogger(__name__)


class AtomicScraperToolConfig(BaseModel):
    """Configuration for the Atomic Scraper Tool integrated with intelligent scraping system."""
    
    # Network settings
    base_url: str = Field(..., description="Base URL for scraping")
    request_delay: float = Field(1.0, ge=0.1, le=10.0, description="Delay between requests in seconds")
    timeout: int = Field(30, ge=5, le=300, description="Request timeout in seconds")
    user_agent: str = Field(
        "IntelligentWebScraper/1.0 (+https://github.com/atomic-agents/intelligent-web-scraper)",
        description="User agent string for requests"
    )
    
    # Quality and limits
    min_quality_score: float = Field(50.0, ge=0.0, le=100.0, description="Minimum quality score for results")
    max_pages: int = Field(10, ge=1, le=100, description="Maximum pages to scrape")
    max_results: int = Field(100, ge=1, le=1000, description="Maximum results to return")
    
    # Retry settings
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(2.0, ge=0.1, le=30.0, description="Delay between retries in seconds")
    
    # Compliance settings
    respect_robots_txt: bool = Field(True, description="Whether to respect robots.txt")
    enable_rate_limiting: bool = Field(True, description="Whether to enable rate limiting")
    
    # Monitoring settings
    enable_monitoring: bool = Field(True, description="Whether to enable monitoring")
    instance_id: Optional[str] = Field(None, description="Instance ID for monitoring")
    
    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, v):
        """Validate that base_url is a valid URL."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('base_url must start with http:// or https://')
        return v
    
    @classmethod
    def from_intelligent_config(
        cls, 
        base_url: str, 
        intelligent_config: IntelligentScrapingConfig,
        instance_id: Optional[str] = None,
        **overrides
    ) -> "AtomicScraperToolConfig":
        """Create tool config from intelligent scraping config."""
        config_data = {
            "base_url": base_url,
            "request_delay": intelligent_config.request_delay,
            "min_quality_score": intelligent_config.default_quality_threshold,
            "respect_robots_txt": intelligent_config.respect_robots_txt,
            "enable_rate_limiting": intelligent_config.enable_rate_limiting,
            "enable_monitoring": intelligent_config.enable_monitoring,
            "instance_id": instance_id,
            **overrides
        }
        return cls(**config_data)


class AtomicScraperInputSchema(BaseIOSchema):
    """Input schema for the atomic scraper tool."""
    
    target_url: str = Field(..., description="Website URL to scrape")
    strategy: Dict[str, Any] = Field(..., description="Scraping strategy configuration")
    schema_recipe: Dict[str, Any] = Field(..., description="Schema recipe for data validation")
    max_results: int = Field(10, ge=1, le=1000, description="Maximum results to return")
    quality_threshold: Optional[float] = Field(None, description="Override quality threshold")
    
    @field_validator('target_url')
    @classmethod
    def validate_target_url(cls, v):
        """Validate target URL format."""
        if not v.strip():
            raise ValueError("target_url cannot be empty")
        
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL format: '{v}'")
        
        if parsed.scheme not in ['http', 'https']:
            raise ValueError(f"URL scheme must be http or https, got '{parsed.scheme}'")
        
        return v


class AtomicScraperOutputSchema(BaseIOSchema):
    """Output schema for the atomic scraper tool."""
    
    results: Dict[str, Any] = Field(..., description="Scraping results with extracted data")
    summary: str = Field(..., description="Human-readable summary of results")
    quality_metrics: Dict[str, float] = Field(..., description="Quality metrics for the scraping operation")
    monitoring_data: Optional[Dict[str, Any]] = Field(None, description="Monitoring and performance data")
    errors: List[str] = Field(default_factory=list, description="Errors encountered during scraping")


class ScrapingError(Exception):
    """Base exception for scraping errors."""
    pass


class NetworkError(ScrapingError):
    """Exception for network-related errors."""
    
    def __init__(self, message: str, url: Optional[str] = None):
        super().__init__(message)
        self.url = url


class QualityError(ScrapingError):
    """Exception for quality-related errors."""
    
    def __init__(self, message: str, quality_score: Optional[float] = None):
        super().__init__(message)
        self.quality_score = quality_score


class AtomicScraperTool(BaseTool):
    """
    Atomic Scraper Tool integrated with the Intelligent Web Scraper framework.
    
    This tool provides AI-powered web scraping capabilities with proper configuration
    management, enhanced error handling, and monitoring integration.
    """
    
    input_schema = AtomicScraperInputSchema
    output_schema = AtomicScraperOutputSchema
    
    def __init__(
        self, 
        config: Optional[AtomicScraperToolConfig] = None,
        intelligent_config: Optional[IntelligentScrapingConfig] = None
    ):
        """
        Initialize the atomic scraper tool.
        
        Args:
            config: Tool-specific configuration
            intelligent_config: Intelligent scraping system configuration
        """
        # Initialize intelligent config from environment if not provided
        if intelligent_config is None:
            intelligent_config = IntelligentScrapingConfig.from_env()
        
        self.intelligent_config = intelligent_config
        
        # Initialize tool config
        if config is None:
            config = AtomicScraperToolConfig(
                base_url="https://example.com",
                request_delay=intelligent_config.request_delay,
                min_quality_score=intelligent_config.default_quality_threshold,
                respect_robots_txt=intelligent_config.respect_robots_txt,
                enable_rate_limiting=intelligent_config.enable_rate_limiting,
                enable_monitoring=intelligent_config.enable_monitoring
            )
        
        self.config = config
        
        # Create base tool config for atomic-agents framework
        base_config = BaseToolConfig(
            title="Atomic Scraper Tool",
            description="AI-powered web scraping tool with intelligent strategy execution"
        )
        
        super().__init__(config=base_config)
        
        # Initialize HTTP session with configuration
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.user_agent
        })
        
        # Set request timeout
        self.request_timeout = config.timeout
        
        # Initialize monitoring data
        self.monitoring_data = {
            'requests_made': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0,
            'average_response_time': 0.0,
            'quality_scores': [],
            'errors': []
        }
        
        logger.info(f"AtomicScraperTool initialized with instance_id: {config.instance_id}")
    
    def run(self, input_data: AtomicScraperInputSchema) -> AtomicScraperOutputSchema:
        """
        Execute scraping operation based on strategy and schema.
        
        Args:
            input_data: Scraping parameters and configuration
            
        Returns:
            Scraping results with extracted data and monitoring information
        """
        start_time = time.time()
        operation_id = f"scrape_{int(start_time)}"
        
        try:
            logger.info(f"Starting scraping operation {operation_id} for {input_data.target_url}")
            
            # Update monitoring
            self.monitoring_data['requests_made'] += 1
            
            # Validate inputs
            self._validate_inputs(input_data)
            
            # Override quality threshold if provided
            effective_quality_threshold = (
                input_data.quality_threshold 
                if input_data.quality_threshold is not None 
                else self.config.min_quality_score
            )
            
            # Execute scraping based on strategy
            results = self._execute_scraping_strategy(
                input_data.target_url,
                input_data.strategy,
                input_data.schema_recipe,
                input_data.max_results,
                effective_quality_threshold
            )
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update monitoring data
            self._update_monitoring_data(execution_time, True, results)
            
            # Generate summary and quality metrics
            summary = self._generate_summary(results, execution_time)
            quality_metrics = self._extract_quality_metrics(results, execution_time)
            
            # Prepare monitoring data for output
            monitoring_output = None
            if self.config.enable_monitoring:
                monitoring_output = {
                    'operation_id': operation_id,
                    'instance_id': self.config.instance_id,
                    'execution_time': execution_time,
                    'requests_made': self.monitoring_data['requests_made'],
                    'success_rate': self._calculate_success_rate(),
                    'average_response_time': self.monitoring_data['average_response_time']
                }
            
            logger.info(f"Scraping operation {operation_id} completed successfully in {execution_time:.2f}s")
            
            return AtomicScraperOutputSchema(
                results=results,
                summary=summary,
                quality_metrics=quality_metrics,
                monitoring_data=monitoring_output,
                errors=[]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Scraping operation {operation_id} failed: {str(e)}"
            
            # Update monitoring data for failure
            self._update_monitoring_data(execution_time, False, {})
            self.monitoring_data['errors'].append(error_msg)
            
            logger.error(error_msg, exc_info=True)
            
            # Prepare monitoring data for error output
            monitoring_output = None
            if self.config.enable_monitoring:
                monitoring_output = {
                    'operation_id': operation_id,
                    'instance_id': self.config.instance_id,
                    'execution_time': execution_time,
                    'error': str(e),
                    'success_rate': self._calculate_success_rate()
                }
            
            return AtomicScraperOutputSchema(
                results={
                    'items': [],
                    'total_found': 0,
                    'total_scraped': 0,
                    'strategy_used': input_data.strategy,
                    'errors': [error_msg]
                },
                summary=f"Scraping failed after {execution_time:.2f} seconds: {str(e)}",
                quality_metrics={
                    'average_quality_score': 0.0,
                    'success_rate': 0.0,
                    'total_items_found': 0.0,
                    'total_items_scraped': 0.0,
                    'execution_time': execution_time
                },
                monitoring_data=monitoring_output,
                errors=[error_msg]
            )
    
    def _validate_inputs(self, input_data: AtomicScraperInputSchema) -> None:
        """
        Validate input data and configuration.
        
        Args:
            input_data: Input data to validate
            
        Raises:
            ValueError: If input data is invalid
        """
        # Validate strategy
        if not input_data.strategy:
            raise ValueError("Strategy cannot be empty")
        
        required_strategy_fields = ['scrape_type', 'target_selectors']
        for field in required_strategy_fields:
            if field not in input_data.strategy:
                raise ValueError(f"Strategy missing required field: {field}")
        
        # Validate schema recipe
        if not input_data.schema_recipe:
            raise ValueError("Schema recipe cannot be empty")
        
        if 'fields' not in input_data.schema_recipe:
            raise ValueError("Schema recipe must contain 'fields' definition")
        
        # Validate URL accessibility
        try:
            parsed_url = urlparse(input_data.target_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL format: {input_data.target_url}")
        except Exception as e:
            raise ValueError(f"URL validation failed: {e}")
    
    def _execute_scraping_strategy(
        self,
        target_url: str,
        strategy: Dict[str, Any],
        schema_recipe: Dict[str, Any],
        max_results: int,
        quality_threshold: float
    ) -> Dict[str, Any]:
        """
        Execute the scraping strategy.
        
        Args:
            target_url: URL to scrape
            strategy: Scraping strategy configuration
            schema_recipe: Schema recipe for data extraction
            max_results: Maximum results to return
            quality_threshold: Quality threshold for filtering
            
        Returns:
            Dictionary containing scraping results
        """
        scrape_type = strategy.get('scrape_type', 'list')
        
        if scrape_type == 'list':
            return self._scrape_list_content(target_url, strategy, schema_recipe, max_results, quality_threshold)
        elif scrape_type == 'detail':
            return self._scrape_detail_content(target_url, strategy, schema_recipe, quality_threshold)
        elif scrape_type == 'search':
            return self._scrape_search_results(target_url, strategy, schema_recipe, max_results, quality_threshold)
        else:
            raise ValueError(f"Unsupported scrape_type: {scrape_type}")
    
    def _scrape_list_content(
        self,
        url: str,
        strategy: Dict[str, Any],
        schema_recipe: Dict[str, Any],
        max_results: int,
        quality_threshold: float
    ) -> Dict[str, Any]:
        """
        Scrape list-type content with pagination support.
        
        Args:
            url: Starting URL to scrape
            strategy: Scraping strategy configuration
            schema_recipe: Schema recipe for data extraction
            max_results: Maximum items to return
            quality_threshold: Quality threshold for filtering
            
        Returns:
            Dictionary containing scraping results
        """
        all_items = []
        errors = []
        pages_scraped = 0
        current_url = url
        max_pages = strategy.get('max_pages', self.config.max_pages)
        
        while current_url and pages_scraped < max_pages and len(all_items) < max_results:
            try:
                # Apply rate limiting
                if pages_scraped > 0:
                    self._apply_rate_limiting()
                
                # Fetch page content
                html_content = self._fetch_page_content(current_url)
                
                # Extract items from this page
                page_items = self._extract_items_from_page(
                    html_content, current_url, strategy, schema_recipe
                )
                
                # Filter items that meet quality threshold
                quality_items = []
                for item in page_items:
                    item_quality = self._calculate_item_quality(item, schema_recipe)
                    if item_quality >= quality_threshold:
                        item['quality_score'] = item_quality
                        quality_items.append(item)
                    else:
                        errors.append(f"Item from {current_url} below quality threshold: {item_quality:.1f}%")
                
                all_items.extend(quality_items)
                pages_scraped += 1
                
                logger.info(f"Page {pages_scraped}: Found {len(page_items)} items, {len(quality_items)} passed quality check")
                
                # Check if we have enough results
                if len(all_items) >= max_results:
                    all_items = all_items[:max_results]
                    break
                
                # Find next page URL if pagination is enabled
                pagination_strategy = strategy.get('pagination_strategy')
                if pagination_strategy:
                    current_url = self._find_next_page_url(html_content, current_url, pagination_strategy)
                else:
                    break
                    
            except Exception as e:
                error_msg = f"Error scraping page {current_url}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                break
        
        # Calculate metrics
        total_found = len(all_items) + len([e for e in errors if "below quality threshold" in e])
        avg_quality = sum(item.get('quality_score', 0) for item in all_items) / len(all_items) if all_items else 0.0
        
        return {
            'items': all_items,
            'total_found': total_found,
            'total_scraped': len(all_items),
            'pages_scraped': pages_scraped,
            'average_quality_score': avg_quality,
            'strategy_used': strategy,
            'errors': errors
        }
    
    def _scrape_detail_content(
        self,
        url: str,
        strategy: Dict[str, Any],
        schema_recipe: Dict[str, Any],
        quality_threshold: float
    ) -> Dict[str, Any]:
        """
        Scrape detail page content.
        
        Args:
            url: URL to scrape
            strategy: Scraping strategy configuration
            schema_recipe: Schema recipe for data extraction
            quality_threshold: Quality threshold for filtering
            
        Returns:
            Dictionary containing scraping results
        """
        errors = []
        
        try:
            # Fetch page content
            html_content = self._fetch_page_content(url)
            
            # Extract single item from page
            items = self._extract_items_from_page(html_content, url, strategy, schema_recipe)
            
            if not items:
                errors.append(f"No content found on detail page: {url}")
                return {
                    'items': [],
                    'total_found': 0,
                    'total_scraped': 0,
                    'average_quality_score': 0.0,
                    'strategy_used': strategy,
                    'errors': errors
                }
            
            # Take the first (and typically only) item
            item = items[0]
            item_quality = self._calculate_item_quality(item, schema_recipe)
            
            # Check quality threshold
            if item_quality < quality_threshold:
                errors.append(f"Detail page content below quality threshold: {item_quality:.1f}%")
                return {
                    'items': [],
                    'total_found': 1,
                    'total_scraped': 0,
                    'average_quality_score': item_quality,
                    'strategy_used': strategy,
                    'errors': errors
                }
            
            item['quality_score'] = item_quality
            
            return {
                'items': [item],
                'total_found': 1,
                'total_scraped': 1,
                'average_quality_score': item_quality,
                'strategy_used': strategy,
                'errors': errors
            }
            
        except Exception as e:
            error_msg = f"Error scraping detail page {url}: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            
            return {
                'items': [],
                'total_found': 0,
                'total_scraped': 0,
                'average_quality_score': 0.0,
                'strategy_used': strategy,
                'errors': errors
            }
    
    def _scrape_search_results(
        self,
        url: str,
        strategy: Dict[str, Any],
        schema_recipe: Dict[str, Any],
        max_results: int,
        quality_threshold: float
    ) -> Dict[str, Any]:
        """
        Scrape search results with pagination support.
        
        Args:
            url: Search URL to scrape
            strategy: Scraping strategy configuration
            schema_recipe: Schema recipe for data extraction
            max_results: Maximum items to return
            quality_threshold: Quality threshold for filtering
            
        Returns:
            Dictionary containing scraping results
        """
        # Search results are similar to list content but may have different pagination
        return self._scrape_list_content(url, strategy, schema_recipe, max_results, quality_threshold)
    
    def _fetch_page_content(self, url: str) -> str:
        """
        Fetch HTML content from URL with error handling and retry logic.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content as string
            
        Raises:
            NetworkError: If request fails after retries
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    # Apply retry delay
                    time.sleep(self.config.retry_delay * attempt)
                    logger.info(f"Retrying request to {url} (attempt {attempt + 1})")
                
                response = self.session.get(
                    url, 
                    timeout=self.request_timeout,
                    allow_redirects=True
                )
                response.raise_for_status()
                return response.text
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.warning(f"Request to {url} failed (attempt {attempt + 1}): {e}")
                
                if attempt == self.config.max_retries:
                    break
        
        raise NetworkError(f"Failed to fetch {url} after {self.config.max_retries + 1} attempts: {last_exception}", url=url)
    
    def _extract_items_from_page(
        self,
        html_content: str,
        url: str,
        strategy: Dict[str, Any],
        schema_recipe: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract items from HTML content based on strategy and schema.
        
        Args:
            html_content: HTML content to parse
            url: Source URL for context
            strategy: Scraping strategy configuration
            schema_recipe: Schema recipe for data extraction
            
        Returns:
            List of extracted items
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        items = []
        
        # Get target selectors from strategy
        target_selectors = strategy.get('target_selectors', {})
        container_selector = target_selectors.get('container', 'body')
        item_selector = target_selectors.get('item', 'div')
        
        # Find container and items
        container = soup.select_one(container_selector)
        if not container:
            logger.warning(f"Container not found with selector: {container_selector}")
            return items
        
        item_elements = container.select(item_selector)
        logger.info(f"Found {len(item_elements)} items with selector: {item_selector}")
        
        # Extract data from each item
        for item_element in item_elements:
            item_data = {}
            
            # Extract fields based on schema recipe
            fields = schema_recipe.get('fields', {})
            for field_name, field_config in fields.items():
                try:
                    field_value = self._extract_field_value(item_element, field_config, url)
                    if field_value:
                        item_data[field_name] = field_value
                except Exception as e:
                    logger.warning(f"Failed to extract field {field_name}: {e}")
            
            if item_data:
                items.append(item_data)
        
        return items
    
    def _extract_field_value(self, element, field_config: Dict[str, Any], base_url: str) -> Optional[str]:
        """
        Extract field value from element based on field configuration.
        
        Args:
            element: BeautifulSoup element to extract from
            field_config: Field configuration from schema recipe
            base_url: Base URL for resolving relative URLs
            
        Returns:
            Extracted field value or None
        """
        selector = field_config.get('extraction_selector', '')
        field_type = field_config.get('field_type', 'string')
        
        if not selector:
            return None
        
        # Find target element
        target_element = element.select_one(selector)
        if not target_element:
            return None
        
        # Extract value based on field type
        if field_type == 'url':
            # Extract href attribute and resolve relative URLs
            href = target_element.get('href')
            if href:
                return urljoin(base_url, href)
        elif field_type in ['string', 'text']:
            # Extract text content
            return target_element.get_text(strip=True)
        elif field_type == 'html':
            # Extract HTML content
            return str(target_element)
        else:
            # Default to text extraction
            return target_element.get_text(strip=True)
        
        return None
    
    def _calculate_item_quality(self, item: Dict[str, Any], schema_recipe: Dict[str, Any]) -> float:
        """
        Calculate quality score for an extracted item.
        
        Args:
            item: Extracted item data
            schema_recipe: Schema recipe with quality weights
            
        Returns:
            Quality score as percentage (0-100)
        """
        fields = schema_recipe.get('fields', {})
        total_weight = 0.0
        achieved_weight = 0.0
        
        for field_name, field_config in fields.items():
            field_weight = field_config.get('quality_weight', 1.0)
            total_weight += field_weight
            
            # Check if field is present and has content
            if field_name in item and item[field_name]:
                field_value = str(item[field_name]).strip()
                if field_value:
                    achieved_weight += field_weight
        
        if total_weight == 0:
            return 0.0
        
        return (achieved_weight / total_weight) * 100.0
    
    def _find_next_page_url(
        self, 
        html_content: str, 
        current_url: str, 
        pagination_strategy: Dict[str, Any]
    ) -> Optional[str]:
        """
        Find next page URL based on pagination strategy.
        
        Args:
            html_content: Current page HTML content
            current_url: Current page URL
            pagination_strategy: Pagination configuration
            
        Returns:
            Next page URL or None if no next page
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Get pagination selector
        next_selector = pagination_strategy.get('next_selector', 'a[rel="next"]')
        
        # Find next page link
        next_element = soup.select_one(next_selector)
        if not next_element:
            return None
        
        # Extract href
        next_href = next_element.get('href')
        if not next_href:
            return None
        
        # Resolve relative URL
        return urljoin(current_url, next_href)
    
    def _apply_rate_limiting(self) -> None:
        """Apply rate limiting delay between requests."""
        if self.config.enable_rate_limiting and self.config.request_delay > 0:
            time.sleep(self.config.request_delay)
    
    def _update_monitoring_data(self, execution_time: float, success: bool, results: Dict[str, Any]) -> None:
        """Update monitoring data with operation results."""
        if success:
            self.monitoring_data['successful_requests'] += 1
            
            # Update quality scores
            if 'average_quality_score' in results:
                self.monitoring_data['quality_scores'].append(results['average_quality_score'])
        else:
            self.monitoring_data['failed_requests'] += 1
        
        # Update timing data
        self.monitoring_data['total_processing_time'] += execution_time
        self.monitoring_data['average_response_time'] = (
            self.monitoring_data['total_processing_time'] / self.monitoring_data['requests_made']
        )
    
    def _calculate_success_rate(self) -> float:
        """Calculate current success rate."""
        total_requests = self.monitoring_data['requests_made']
        if total_requests == 0:
            return 0.0
        
        return (self.monitoring_data['successful_requests'] / total_requests) * 100.0
    
    def _generate_summary(self, results: Dict[str, Any], execution_time: float) -> str:
        """
        Generate human-readable summary of scraping results.
        
        Args:
            results: Scraping results
            execution_time: Total execution time
            
        Returns:
            Summary string
        """
        total_items = results.get('total_scraped', 0)
        avg_quality = results.get('average_quality_score', 0)
        pages_scraped = results.get('pages_scraped', 1)
        
        summary = f"Successfully scraped {total_items} items "
        if pages_scraped > 1:
            summary += f"from {pages_scraped} pages "
        summary += f"with average quality score of {avg_quality:.1f}% "
        summary += f"in {execution_time:.2f} seconds."
        
        errors = results.get('errors', [])
        if errors:
            summary += f" Encountered {len(errors)} errors during scraping."
        
        return summary
    
    def _extract_quality_metrics(self, results: Dict[str, Any], execution_time: float) -> Dict[str, float]:
        """
        Extract quality metrics from scraping results.
        
        Args:
            results: Scraping results
            execution_time: Total execution time
            
        Returns:
            Dictionary of quality metrics
        """
        total_found = results.get('total_found', 0)
        total_scraped = results.get('total_scraped', 0)
        
        return {
            'average_quality_score': results.get('average_quality_score', 0.0),
            'success_rate': (total_scraped / total_found * 100.0) if total_found > 0 else 0.0,
            'total_items_found': float(total_found),
            'total_items_scraped': float(total_scraped),
            'execution_time': execution_time,
            'pages_scraped': float(results.get('pages_scraped', 1))
        }
    
    def get_tool_info(self) -> Dict[str, Any]:
        """
        Get information about the tool and its configuration.
        
        Returns:
            Dictionary containing tool information
        """
        return {
            'name': 'AtomicScraperTool',
            'version': '1.0.0',
            'description': 'AI-powered web scraping tool integrated with Intelligent Web Scraper',
            'config': {
                'base_url': self.config.base_url,
                'request_delay': self.config.request_delay,
                'timeout': self.config.timeout,
                'max_pages': self.config.max_pages,
                'max_results': self.config.max_results,
                'min_quality_score': self.config.min_quality_score,
                'respect_robots_txt': self.config.respect_robots_txt,
                'enable_rate_limiting': self.config.enable_rate_limiting,
                'enable_monitoring': self.config.enable_monitoring,
                'instance_id': self.config.instance_id
            },
            'supported_strategies': ['list', 'detail', 'search'],
            'supported_field_types': ['string', 'text', 'url', 'html'],
            'monitoring_enabled': self.config.enable_monitoring
        }
    
    def update_config(self, **config_updates) -> None:
        """
        Update tool configuration dynamically.
        
        Args:
            **config_updates: Configuration parameters to update
        """
        # Update config attributes
        for key, value in config_updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config {key} to {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
        
        # Update session headers if user_agent changed
        if 'user_agent' in config_updates:
            self.session.headers.update({
                'User-Agent': config_updates['user_agent']
            })
        
        # Update timeout if changed
        if 'timeout' in config_updates:
            self.request_timeout = config_updates['timeout']
    
    def get_monitoring_data(self) -> Dict[str, Any]:
        """
        Get current monitoring data.
        
        Returns:
            Dictionary containing monitoring metrics
        """
        return {
            'instance_id': self.config.instance_id,
            'requests_made': self.monitoring_data['requests_made'],
            'successful_requests': self.monitoring_data['successful_requests'],
            'failed_requests': self.monitoring_data['failed_requests'],
            'success_rate': self._calculate_success_rate(),
            'average_response_time': self.monitoring_data['average_response_time'],
            'total_processing_time': self.monitoring_data['total_processing_time'],
            'average_quality_score': (
                sum(self.monitoring_data['quality_scores']) / len(self.monitoring_data['quality_scores'])
                if self.monitoring_data['quality_scores'] else 0.0
            ),
            'error_count': len(self.monitoring_data['errors']),
            'recent_errors': self.monitoring_data['errors'][-5:]  # Last 5 errors
        }
    
    def reset_monitoring_data(self) -> None:
        """Reset monitoring data."""
        self.monitoring_data = {
            'requests_made': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0,
            'average_response_time': 0.0,
            'quality_scores': [],
            'errors': []
        }
        logger.info("Monitoring data reset")