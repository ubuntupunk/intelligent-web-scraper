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

# Import from the actual atomic_scraper_tool package
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from atomic_scraper_tool.tools.atomic_scraper_tool import (
    AtomicScraperTool as BaseAtomicScraperTool,
    AtomicScraperInputSchema as BaseAtomicScraperInputSchema,
    AtomicScraperOutputSchema as BaseAtomicScraperOutputSchema
)
from atomic_scraper_tool.config.scraper_config import AtomicScraperConfig
from atomic_scraper_tool.models.base_models import ScrapingStrategy, ScrapingResult

from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import BaseModel, Field, field_validator

from ..config import IntelligentScrapingConfig


logger = logging.getLogger(__name__)


class AtomicScraperToolConfig(AtomicScraperConfig):
    """Configuration for the Atomic Scraper Tool integrated with intelligent scraping system."""
    
    # Additional monitoring settings for intelligent scraper integration
    enable_monitoring: bool = Field(True, description="Whether to enable monitoring")
    instance_id: Optional[str] = Field(None, description="Instance ID for monitoring")
    
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
            "enable_monitoring": getattr(intelligent_config, 'enable_monitoring', True),
            "instance_id": instance_id,
            **overrides
        }
        return cls(**config_data)


class AtomicScraperInputSchema(BaseAtomicScraperInputSchema):
    """Input schema for the atomic scraper tool with intelligent scraper extensions."""
    
    quality_threshold: Optional[float] = Field(None, description="Override quality threshold")
    enable_monitoring: Optional[bool] = Field(True, description="Enable monitoring for this operation")
    instance_id: Optional[str] = Field(None, description="Instance ID for monitoring")


class AtomicScraperOutputSchema(BaseAtomicScraperOutputSchema):
    """Output schema for the atomic scraper tool with intelligent scraper extensions."""
    
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


class AtomicScraperTool(BaseAtomicScraperTool):
    """
    Atomic Scraper Tool integrated with the Intelligent Web Scraper framework.
    
    This tool extends the base AtomicScraperTool with intelligent scraping capabilities,
    monitoring integration, and enhanced configuration management.
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
                enable_monitoring=getattr(intelligent_config, 'enable_monitoring', True)
            )
        
        # Initialize the base tool with the config
        super().__init__(config=config)
        
        # Store additional config for monitoring
        self.config = config
        
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
            
            # Override quality threshold if provided
            if input_data.quality_threshold is not None:
                original_threshold = self.config.min_quality_score
                self.config.min_quality_score = input_data.quality_threshold
            
            # Call the base implementation
            base_result = super().run(input_data)
            
            # Restore original threshold if it was overridden
            if input_data.quality_threshold is not None:
                self.config.min_quality_score = original_threshold
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update monitoring data
            self._update_monitoring_data(execution_time, True, base_result.results)
            
            # Prepare monitoring data for output
            monitoring_output = None
            if getattr(input_data, 'enable_monitoring', True) and self.config.enable_monitoring:
                monitoring_output = {
                    'operation_id': operation_id,
                    'instance_id': getattr(input_data, 'instance_id', self.config.instance_id),
                    'execution_time': execution_time,
                    'requests_made': self.monitoring_data['requests_made'],
                    'success_rate': self._calculate_success_rate(),
                    'average_response_time': self.monitoring_data['average_response_time']
                }
            
            logger.info(f"Scraping operation {operation_id} completed successfully in {execution_time:.2f}s")
            
            return AtomicScraperOutputSchema(
                target_url=base_result.target_url,
                strategy=base_result.strategy,
                schema_recipe=base_result.schema_recipe,
                max_results=base_result.max_results,
                results=base_result.results,
                summary=base_result.summary,
                quality_metrics=base_result.quality_metrics,
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
            if getattr(input_data, 'enable_monitoring', True) and self.config.enable_monitoring:
                monitoring_output = {
                    'operation_id': operation_id,
                    'instance_id': getattr(input_data, 'instance_id', self.config.instance_id),
                    'execution_time': execution_time,
                    'error': str(e),
                    'success_rate': self._calculate_success_rate()
                }
            
            return AtomicScraperOutputSchema(
                target_url=input_data.target_url,
                strategy=input_data.strategy,
                schema_recipe=input_data.schema_recipe,
                max_results=input_data.max_results,
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
    

    
    def _update_monitoring_data(self, execution_time: float, success: bool, results: Dict[str, Any]) -> None:
        """Update monitoring data with operation results."""
        if success:
            self.monitoring_data['successful_requests'] += 1
            
            # Update quality scores if available
            if isinstance(results, dict) and 'items' in results:
                items = results['items']
                if items:
                    quality_scores = [item.get('quality_score', 0) for item in items if isinstance(item, dict)]
                    self.monitoring_data['quality_scores'].extend(quality_scores)
        else:
            self.monitoring_data['failed_requests'] += 1
        
        # Update timing data
        self.monitoring_data['total_processing_time'] += execution_time
        total_requests = self.monitoring_data['requests_made']
        if total_requests > 0:
            self.monitoring_data['average_response_time'] = (
                self.monitoring_data['total_processing_time'] / total_requests
            )
    
    def _calculate_success_rate(self) -> float:
        """Calculate current success rate."""
        total_requests = self.monitoring_data['requests_made']
        if total_requests == 0:
            return 0.0
        
        successful_requests = self.monitoring_data['successful_requests']
        return (successful_requests / total_requests) * 100.0
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        return {
            'requests_made': self.monitoring_data['requests_made'],
            'successful_requests': self.monitoring_data['successful_requests'],
            'failed_requests': self.monitoring_data['failed_requests'],
            'success_rate': self._calculate_success_rate(),
            'average_response_time': self.monitoring_data['average_response_time'],
            'average_quality_score': (
                sum(self.monitoring_data['quality_scores']) / len(self.monitoring_data['quality_scores'])
                if self.monitoring_data['quality_scores'] else 0.0
            ),
            'total_errors': len(self.monitoring_data['errors'])
        }
    
    def reset_monitoring_stats(self) -> None:
        """Reset monitoring statistics."""
        self.monitoring_data = {
            'requests_made': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0,
            'average_response_time': 0.0,
            'quality_scores': [],
            'errors': []
        }

    
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