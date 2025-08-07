"""
Intelligent Web Scraper Planning Agent.

This module contains the adapted AtomicScraperPlanningAgent that has been
modified for seamless orchestrator integration with enhanced reasoning
and educational explanations.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import (
    SystemPromptGenerator, 
    SystemPromptContextProviderBase
)
from pydantic import Field, field_validator

# Import from atomic_scraper_tool for base functionality
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from atomic_scraper_tool.models.base_models import ScrapingStrategy
from atomic_scraper_tool.models.schema_models import SchemaRecipe, FieldDefinition
from atomic_scraper_tool.analysis.website_analyzer import WebsiteAnalyzer, WebsiteStructureAnalysis
from atomic_scraper_tool.analysis.strategy_generator import StrategyGenerator, StrategyContext
from atomic_scraper_tool.analysis.schema_recipe_generator import SchemaRecipeGenerator, SchemaGenerationContext


class IntelligentPlanningAgentInputSchema(BaseIOSchema):
    """Enhanced input schema for the intelligent planning agent."""
    
    scraping_request: str = Field(
        ..., 
        description="Natural language scraping request describing what to extract"
    )
    target_url: str = Field(
        ..., 
        description="Website URL to analyze and scrape"
    )
    max_results: Optional[int] = Field(
        default=10, 
        ge=1, 
        le=1000, 
        description="Maximum number of items to extract"
    )
    quality_threshold: Optional[float] = Field(
        default=50.0, 
        ge=0.0, 
        le=100.0, 
        description="Minimum quality score for extracted data"
    )
    orchestrator_context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional context provided by the orchestrator"
    )
    
    @field_validator('target_url')
    @classmethod
    def validate_target_url(cls, v):
        """Validate target URL format."""
        from urllib.parse import urlparse
        
        if not v.strip():
            raise ValueError("target_url cannot be empty")
        
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL format: '{v}'")
        
        if parsed.scheme not in ['http', 'https']:
            raise ValueError(f"URL scheme must be http or https, got '{parsed.scheme}'")
        
        return v


class IntelligentPlanningAgentOutputSchema(BaseIOSchema):
    """Enhanced output schema aligned with orchestrator expectations."""
    
    scraping_plan: str = Field(
        ..., 
        description="Detailed human-readable scraping plan with educational explanations"
    )
    strategy: Dict[str, Any] = Field(
        ..., 
        description="Generated scraping strategy compatible with AtomicScraperTool"
    )
    schema_recipe: Dict[str, Any] = Field(
        ..., 
        description="Dynamic schema recipe for data extraction and validation"
    )
    reasoning: str = Field(
        ..., 
        description="Comprehensive reasoning and educational explanations"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score in the generated plan"
    )
    orchestrator_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata for orchestrator coordination"
    )
    educational_insights: Dict[str, Any] = Field(
        default_factory=dict,
        description="Educational insights about scraping patterns and decisions"
    )


class IntelligentScrapingContextProvider(SystemPromptContextProviderBase):
    """Enhanced context provider for intelligent scraping with educational focus."""
    
    def __init__(self):
        """Initialize the intelligent scraping context provider."""
        super().__init__(title="Intelligent Scraping Capabilities & Educational Context")
    
    def get_info(self) -> str:
        """Provide comprehensive context about scraping capabilities with educational focus."""
        return """You are an expert intelligent web scraping planning agent designed for both practical scraping and educational demonstration. Your role is to analyze user requests, understand website structures, and generate optimal scraping strategies while providing clear educational explanations.

### Core Responsibilities:
1. **Strategy Planning**: Analyze websites and generate optimal scraping approaches
2. **Educational Guidance**: Explain scraping concepts and decision-making processes
3. **Orchestrator Integration**: Ensure seamless coordination with the orchestrator agent
4. **Schema Generation**: Create dynamic schemas that match website data structures

### Available Scraping Strategies:
- **list**: Extract multiple items from list pages (product listings, article lists, directories)
  - Best for: E-commerce catalogs, news feeds, event listings, market directories
  - Handles: Pagination, filtering, bulk data extraction
  
- **detail**: Extract comprehensive information from individual pages
  - Best for: Product details, article content, profile information
  - Handles: Deep content extraction, related data, metadata
  
- **search**: Extract results from search functionality
  - Best for: Search result pages, filtered content, query-based extraction
  - Handles: Search parameters, result pagination, relevance scoring
  
- **sitemap**: Extract URLs and metadata from sitemaps
  - Best for: Site discovery, URL collection, content mapping
  - Handles: XML sitemaps, robots.txt, site structure analysis

### Strategy Components & Educational Value:
- **target_selectors**: CSS selectors for content identification
  - Educational: Demonstrates CSS selector specificity and robustness
  - Best Practice: Use semantic selectors with fallbacks
  
- **extraction_rules**: Field-specific extraction with validation
  - Educational: Shows data validation and quality control patterns
  - Best Practice: Include data type validation and cleaning rules
  
- **pagination_strategy**: Multi-page content handling
  - Educational: Demonstrates different pagination patterns on the web
  - Options: next_link, page_numbers, infinite_scroll, load_more
  
- **quality_thresholds**: Data quality requirements
  - Educational: Shows importance of data quality in scraping
  - Metrics: Completeness, accuracy, consistency, relevance

### Schema Recipe Generation Principles:
1. **Semantic Analysis**: Understand content meaning and structure
2. **Field Prioritization**: Identify required vs optional data fields
3. **Validation Rules**: Ensure data integrity and consistency
4. **Quality Weights**: Balance field importance for overall quality scoring
5. **Educational Value**: Explain field choices and validation logic

### Orchestrator Integration Patterns:
- **Input Alignment**: Accept orchestrator context and requirements
- **Output Compatibility**: Generate schemas compatible with AtomicScraperTool
- **Error Handling**: Provide graceful degradation and fallback strategies
- **Monitoring Support**: Include metadata for real-time monitoring

### Educational Focus Areas:
- **Decision Transparency**: Explain why specific strategies are chosen
- **Best Practices**: Demonstrate production-ready scraping patterns
- **Error Prevention**: Show common pitfalls and how to avoid them
- **Scalability**: Explain how strategies scale with different website sizes
- **Ethics & Compliance**: Emphasize respectful scraping practices

### Quality Assurance Principles:
- Always respect robots.txt and rate limiting requirements
- Prefer specific, semantic CSS selectors over generic ones
- Include multiple fallback selectors for robustness
- Set appropriate quality thresholds based on data criticality
- Consider dynamic content loading and JavaScript rendering
- Plan for error handling and graceful degradation"""


class IntelligentWebScraperPlanningAgent(BaseAgent):
    """
    Intelligent Web Scraper Planning Agent.
    
    This agent is an enhanced version of AtomicScraperPlanningAgent specifically
    adapted for orchestrator integration with enhanced reasoning, educational
    explanations, and seamless schema alignment.
    """
    
    def __init__(self, config: BaseAgentConfig):
        """
        Initialize the intelligent planning agent.
        
        Args:
            config: Agent configuration with client and model settings
        """
        # Set up system prompt generator with enhanced context
        context_providers = {
            "scraping_context": IntelligentScrapingContextProvider()
        }
        
        system_prompt_generator = SystemPromptGenerator(
            background=[
                "You are an Intelligent Web Scraper Planning Agent, an advanced AI system designed for both practical web scraping and educational demonstration.",
                "Your primary role is to analyze natural language scraping requests, understand website structures, and generate optimal scraping strategies.",
                "You serve as a key component in the Intelligent Web Scraper orchestration system, demonstrating advanced atomic-agents patterns.",
                "Your outputs must be perfectly aligned with the orchestrator's expectations and the AtomicScraperTool's input requirements.",
                "You provide comprehensive educational explanations to help users understand scraping concepts and best practices."
            ],
            steps=[
                "Parse and analyze the natural language scraping request to understand user intent, data requirements, and quality expectations",
                "Perform comprehensive website structure analysis to identify content patterns, navigation elements, and data organization",
                "Evaluate different scraping strategies (list, detail, search, sitemap) and select the most appropriate approach based on website characteristics and user requirements",
                "Generate specific CSS selectors and extraction rules that are robust, semantic, and maintainable",
                "Create a dynamic schema recipe that accurately represents the expected data structure with proper validation and quality controls",
                "Develop comprehensive reasoning that explains all decisions, demonstrates best practices, and provides educational value",
                "Ensure perfect schema alignment between your output and the AtomicScraperTool's expected input format",
                "Calculate confidence scores based on website analyzability, strategy appropriateness, and expected success rates",
                "Generate orchestrator metadata to support monitoring, error handling, and workflow coordination"
            ],
            output_instructions=[
                "Always provide a detailed, human-readable scraping plan that explains your strategy and approach clearly",
                "Generate strategy configurations that are fully compatible with AtomicScraperTool input requirements",
                "Create schema recipes with proper field definitions, validation rules, and quality weights",
                "Include comprehensive reasoning that demonstrates your decision-making process and provides educational value",
                "Explain scraping concepts, best practices, and potential challenges in an educational manner",
                "Provide confidence scores between 0.0 and 1.0 based on objective assessment criteria",
                "Include orchestrator metadata with operation IDs, timestamps, and coordination information",
                "Generate educational insights that help users understand advanced scraping patterns and concepts",
                "Ensure all outputs follow the defined schema structure with complete and accurate data",
                "Demonstrate respect for robots.txt, rate limiting, and ethical scraping practices in all recommendations"
            ],
            context_providers=context_providers
        )
        
        # Update config with our enhanced schemas and system prompt generator
        config.input_schema = IntelligentPlanningAgentInputSchema
        config.output_schema = IntelligentPlanningAgentOutputSchema
        config.system_prompt_generator = system_prompt_generator
        
        super().__init__(config)
        
        # Initialize analysis components
        self.website_analyzer = WebsiteAnalyzer()
        self.strategy_generator = StrategyGenerator()
        self.schema_generator = SchemaRecipeGenerator()
    
    def run(self, input_data: IntelligentPlanningAgentInputSchema) -> IntelligentPlanningAgentOutputSchema:
        """
        Process scraping request and generate comprehensive plan with orchestrator integration.
        
        Args:
            input_data: Enhanced scraping request with orchestrator context
            
        Returns:
            Comprehensive scraping plan with educational explanations and orchestrator metadata
        """
        operation_start = datetime.utcnow()
        operation_id = str(uuid.uuid4())
        
        try:
            # Step 1: Parse natural language request with enhanced analysis
            parsed_request = self._parse_enhanced_request(input_data.scraping_request)
            
            # Step 2: Analyze target website with comprehensive assessment
            website_analysis = self._analyze_website_comprehensive(input_data.target_url)
            
            # Step 3: Generate optimal scraping strategy with orchestrator alignment
            strategy = self._generate_orchestrator_aligned_strategy(
                website_analysis, 
                parsed_request, 
                input_data
            )
            
            # Step 4: Create dynamic schema recipe with enhanced validation
            schema_recipe = self._create_enhanced_schema_recipe(
                website_analysis, 
                parsed_request, 
                input_data
            )
            
            # Step 5: Generate comprehensive scraping plan with educational value
            scraping_plan = self._generate_educational_scraping_plan(
                strategy, 
                schema_recipe, 
                parsed_request,
                website_analysis
            )
            
            # Step 6: Create detailed reasoning with educational insights
            reasoning = self._generate_comprehensive_reasoning(
                website_analysis, 
                strategy, 
                schema_recipe, 
                parsed_request,
                input_data
            )
            
            # Step 7: Calculate confidence score with detailed assessment
            confidence = self._calculate_enhanced_confidence(
                website_analysis, 
                strategy, 
                schema_recipe,
                parsed_request
            )
            
            # Step 8: Generate orchestrator metadata for coordination
            orchestrator_metadata = self._create_orchestrator_metadata(
                operation_id,
                operation_start,
                input_data,
                strategy,
                schema_recipe
            )
            
            # Step 9: Create educational insights for learning value
            educational_insights = self._generate_educational_insights(
                strategy,
                schema_recipe,
                website_analysis,
                parsed_request
            )
            
            return IntelligentPlanningAgentOutputSchema(
                scraping_plan=scraping_plan,
                strategy=strategy.model_dump(),
                schema_recipe=schema_recipe.model_dump(),
                reasoning=reasoning,
                confidence=confidence,
                orchestrator_metadata=orchestrator_metadata,
                educational_insights=educational_insights
            )
            
        except Exception as e:
            # Handle errors gracefully with educational context
            return self._handle_planning_error(str(e), input_data, operation_id, operation_start)
    
    def _parse_enhanced_request(self, request: str) -> Dict[str, Any]:
        """Parse natural language request with enhanced analysis for educational purposes."""
        request_lower = request.lower()
        
        parsed = {
            'content_type': 'list',  # Default
            'target_data': [],
            'filters': [],
            'keywords': [],
            'temporal_filters': [],
            'location_filters': [],
            'complexity_level': 'medium',
            'educational_focus': [],
            'user_intent': 'data_extraction'
        }
        
        # Enhanced content type detection
        content_indicators = {
            'list': ['list', 'items', 'multiple', 'all', 'collection', 'catalog', 'directory'],
            'detail': ['detail', 'information', 'about', 'specific', 'comprehensive', 'full'],
            'search': ['search', 'find', 'results', 'query', 'lookup', 'discover'],
            'sitemap': ['sitemap', 'structure', 'navigation', 'site map', 'urls']
        }
        
        for content_type, indicators in content_indicators.items():
            if any(indicator in request_lower for indicator in indicators):
                parsed['content_type'] = content_type
                break
        
        # Enhanced data type extraction with educational context
        data_indicators = {
            'markets': {
                'keywords': ['market', 'marketplace', 'bazaar', 'farmers market', 'flea market'],
                'educational_focus': ['location_extraction', 'temporal_data', 'vendor_information']
            },
            'events': {
                'keywords': ['event', 'happening', 'activity', 'festival', 'concert', 'meeting'],
                'educational_focus': ['date_parsing', 'location_extraction', 'event_details']
            },
            'products': {
                'keywords': ['product', 'item', 'goods', 'merchandise', 'catalog'],
                'educational_focus': ['price_extraction', 'product_attributes', 'inventory_data']
            },
            'articles': {
                'keywords': ['article', 'post', 'blog', 'news', 'story', 'content'],
                'educational_focus': ['text_extraction', 'metadata_parsing', 'content_structure']
            },
            'contacts': {
                'keywords': ['contact', 'phone', 'email', 'address', 'directory'],
                'educational_focus': ['contact_validation', 'data_normalization', 'privacy_considerations']
            },
            'prices': {
                'keywords': ['price', 'cost', 'fee', 'rate', 'pricing', 'money'],
                'educational_focus': ['currency_parsing', 'price_comparison', 'numeric_extraction']
            },
            'dates': {
                'keywords': ['date', 'time', 'when', 'schedule', 'calendar'],
                'educational_focus': ['date_parsing', 'timezone_handling', 'temporal_validation']
            },
            'locations': {
                'keywords': ['location', 'place', 'where', 'address', 'venue'],
                'educational_focus': ['address_parsing', 'geocoding', 'location_validation']
            }
        }
        
        for data_type, info in data_indicators.items():
            if any(keyword in request_lower for keyword in info['keywords']):
                parsed['target_data'].append(data_type)
                parsed['educational_focus'].extend(info['educational_focus'])
        
        # Enhanced temporal and location filter extraction
        temporal_patterns = {
            'saturday': ['saturday', 'sat'],
            'sunday': ['sunday', 'sun'],
            'weekend': ['weekend', 'weekends'],
            'today': ['today', 'now'],
            'tomorrow': ['tomorrow'],
            'this_week': ['this week', 'current week'],
            'upcoming': ['upcoming', 'future', 'next']
        }
        
        for pattern, keywords in temporal_patterns.items():
            if any(keyword in request_lower for keyword in keywords):
                parsed['temporal_filters'].append(pattern)
        
        # Location filter extraction with educational context
        location_patterns = {
            'cape_town': ['cape town', 'ct', 'mother city'],
            'johannesburg': ['johannesburg', 'joburg', 'jozi'],
            'durban': ['durban', 'kzn'],
            'local': ['local', 'nearby', 'close'],
            'regional': ['regional', 'province', 'area']
        }
        
        for location, keywords in location_patterns.items():
            if any(keyword in request_lower for keyword in keywords):
                parsed['location_filters'].append(location)
        
        # Determine complexity level for educational purposes
        complexity_indicators = {
            'simple': ['simple', 'basic', 'easy', 'quick'],
            'medium': ['medium', 'standard', 'normal'],
            'complex': ['complex', 'advanced', 'comprehensive', 'detailed', 'thorough']
        }
        
        for level, indicators in complexity_indicators.items():
            if any(indicator in request_lower for indicator in indicators):
                parsed['complexity_level'] = level
                break
        
        # Extract user intent for educational context
        intent_indicators = {
            'learning': ['learn', 'understand', 'example', 'demo', 'tutorial'],
            'production': ['production', 'business', 'commercial', 'scale'],
            'research': ['research', 'analysis', 'study', 'investigate'],
            'monitoring': ['monitor', 'track', 'watch', 'observe']
        }
        
        for intent, indicators in intent_indicators.items():
            if any(indicator in request_lower for indicator in indicators):
                parsed['user_intent'] = intent
                break
        
        # Extract general keywords with stop word filtering
        import re
        words = re.findall(r'\b\w+\b', request_lower)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'scrape', 'get', 'find', 'extract', 'from'
        }
        parsed['keywords'] = [word for word in words if word not in stop_words and len(word) > 2]
        
        return parsed
    
    def _analyze_website_comprehensive(self, url: str) -> WebsiteStructureAnalysis:
        """Perform comprehensive website analysis with enhanced error handling."""
        try:
            # Use the existing website analyzer with enhanced error handling
            import requests
            from bs4 import BeautifulSoup
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Perform comprehensive analysis
            analysis = self.website_analyzer.analyze_website(response.text, url)
            
            # Enhance analysis with educational metadata
            analysis.metadata['analysis_timestamp'] = datetime.utcnow().isoformat()
            analysis.metadata['analysis_method'] = 'comprehensive'
            analysis.metadata['educational_value'] = 'high'
            
            return analysis
            
        except Exception as e:
            # Create enhanced fallback analysis with educational context
            analysis = WebsiteStructureAnalysis(
                url=url,
                title="Website Analysis Failed",
                content_patterns=[],
                metadata={
                    'error': str(e),
                    'fallback_mode': True,
                    'educational_note': 'This demonstrates graceful error handling in web scraping',
                    'analysis_timestamp': datetime.utcnow().isoformat()
                }
            )
            return analysis
    
    def _generate_orchestrator_aligned_strategy(
        self, 
        analysis: WebsiteStructureAnalysis, 
        parsed_request: Dict[str, Any], 
        input_data: IntelligentPlanningAgentInputSchema
    ) -> ScrapingStrategy:
        """Generate scraping strategy aligned with orchestrator expectations."""
        
        # Create enhanced strategy context
        context = StrategyContext(
            user_criteria=input_data.scraping_request,
            target_content_type=parsed_request['content_type'],
            quality_threshold=input_data.quality_threshold,
            max_results=input_data.max_results,
            include_pagination=True,
            extraction_depth=parsed_request['complexity_level']
        )
        
        # Add orchestrator context if available
        if input_data.orchestrator_context:
            context.orchestrator_requirements = input_data.orchestrator_context
        
        # Generate strategy using existing generator
        strategy = self.strategy_generator.generate_strategy(analysis, context)
        
        # Enhance strategy with orchestrator metadata
        strategy.metadata['orchestrator_compatible'] = True
        strategy.metadata['educational_focus'] = parsed_request['educational_focus']
        strategy.metadata['user_intent'] = parsed_request['user_intent']
        strategy.metadata['complexity_level'] = parsed_request['complexity_level']
        
        return strategy
    
    def _create_enhanced_schema_recipe(
        self, 
        analysis: WebsiteStructureAnalysis, 
        parsed_request: Dict[str, Any], 
        input_data: IntelligentPlanningAgentInputSchema
    ) -> SchemaRecipe:
        """Create enhanced schema recipe with educational value and orchestrator alignment."""
        
        # Create enhanced schema generation context
        context = SchemaGenerationContext(
            user_criteria=input_data.scraping_request,
            target_content_type=parsed_request['content_type'],
            sample_html="",  # Would be populated with actual HTML in full implementation
            quality_requirements={
                "completeness": 0.4,
                "accuracy": 0.4,
                "consistency": 0.2
            },
            field_preferences=parsed_request['target_data']
        )
        
        # Generate base schema recipe
        try:
            schema_recipe = self.schema_generator.generate_schema_recipe(analysis, context)
        except Exception:
            # Fallback to creating basic schema recipe
            schema_recipe = self._create_educational_schema_recipe(parsed_request, input_data)
        
        # Enhance with educational metadata
        schema_recipe.metadata['educational_insights'] = {
            'field_selection_rationale': self._explain_field_selection(schema_recipe, parsed_request),
            'validation_strategy': self._explain_validation_strategy(schema_recipe),
            'quality_approach': self._explain_quality_approach(schema_recipe),
            'orchestrator_alignment': 'Schema designed for seamless AtomicScraperTool integration'
        }
        
        return schema_recipe 
   
    def _create_educational_schema_recipe(
        self, 
        parsed_request: Dict[str, Any], 
        input_data: IntelligentPlanningAgentInputSchema
    ) -> SchemaRecipe:
        """Create educational schema recipe with comprehensive field definitions."""
        
        fields = {}
        
        # Always include core fields with educational explanations
        fields['title'] = FieldDefinition(
            field_type='string',
            description='Primary title or name of the item - demonstrates text extraction patterns',
            extraction_selector='h1, h2, h3, .title, .name, [data-title]',
            required=True,
            quality_weight=0.9,
            post_processing=['trim', 'clean'],
            validation_rules=['not_empty', 'max_length:200'],
            educational_note='Title extraction shows semantic HTML usage and fallback strategies'
        )
        
        fields['description'] = FieldDefinition(
            field_type='string',
            description='Detailed description - shows content extraction and cleaning',
            extraction_selector='p, .description, .summary, .content, [data-description]',
            required=False,
            quality_weight=0.7,
            post_processing=['trim', 'clean', 'normalize_whitespace'],
            validation_rules=['max_length:1000'],
            educational_note='Description extraction demonstrates text processing and normalization'
        )
        
        fields['url'] = FieldDefinition(
            field_type='string',
            description='Link URL - demonstrates link extraction and validation',
            extraction_selector='a[href], [data-url], [data-link]',
            required=False,
            quality_weight=0.5,
            post_processing=['trim', 'resolve_relative_urls'],
            validation_rules=['valid_url'],
            educational_note='URL extraction shows attribute handling and link resolution'
        )
        
        # Add specialized fields based on target data with educational context
        if 'markets' in parsed_request['target_data']:
            fields['location'] = FieldDefinition(
                field_type='string',
                description='Market location - demonstrates address extraction patterns',
                extraction_selector='.location, .address, .venue, [data-location]',
                required=False,
                quality_weight=0.8,
                post_processing=['trim', 'clean', 'normalize_address'],
                validation_rules=['not_empty'],
                educational_note='Location extraction shows geographic data handling'
            )
            
            fields['operating_hours'] = FieldDefinition(
                field_type='string',
                description='Operating hours - shows temporal data extraction',
                extraction_selector='.hours, .time, .schedule, [data-hours]',
                required=False,
                quality_weight=0.7,
                post_processing=['trim', 'clean', 'normalize_time'],
                validation_rules=['valid_time_format'],
                educational_note='Time extraction demonstrates temporal data parsing'
            )
        
        if 'prices' in parsed_request['target_data']:
            fields['price'] = FieldDefinition(
                field_type='string',
                description='Price information - demonstrates numeric data extraction',
                extraction_selector='.price, .cost, .fee, .amount, [data-price]',
                required=False,
                quality_weight=0.8,
                post_processing=['trim', 'clean', 'extract_currency', 'normalize_price'],
                validation_rules=['valid_price_format'],
                educational_note='Price extraction shows currency handling and numeric validation'
            )
        
        if 'dates' in parsed_request['target_data']:
            fields['date'] = FieldDefinition(
                field_type='string',
                description='Date information - shows date parsing and validation',
                extraction_selector='.date, .time, time, .published, [datetime]',
                required=False,
                quality_weight=0.7,
                post_processing=['trim', 'clean', 'parse_date'],
                validation_rules=['valid_date_format'],
                educational_note='Date extraction demonstrates temporal data validation'
            )
        
        if 'contacts' in parsed_request['target_data']:
            fields['contact'] = FieldDefinition(
                field_type='string',
                description='Contact information - shows contact data extraction',
                extraction_selector='.contact, .phone, .email, .tel, [data-contact]',
                required=False,
                quality_weight=0.8,
                post_processing=['trim', 'clean', 'validate_contact'],
                validation_rules=['valid_contact_format'],
                educational_note='Contact extraction shows data validation and privacy considerations'
            )
        
        # Create schema name based on request
        keywords = parsed_request['keywords'][:2] if parsed_request['keywords'] else ['intelligent']
        schema_name = '_'.join(keywords) + '_schema'
        
        return SchemaRecipe(
            name=schema_name,
            description=f"Educational schema for {parsed_request['content_type']} data extraction: {input_data.scraping_request}",
            fields=fields,
            validation_rules=[
                'normalize_whitespace', 
                'allow_partial_data', 
                'validate_required_fields',
                'apply_quality_thresholds'
            ],
            quality_weights={
                "completeness": 0.4,
                "accuracy": 0.4,
                "consistency": 0.2
            },
            version="2.0",
            metadata={
                'educational_purpose': True,
                'orchestrator_compatible': True,
                'complexity_level': parsed_request['complexity_level'],
                'target_data_types': parsed_request['target_data']
            }
        )
    
    def _generate_educational_scraping_plan(
        self, 
        strategy: ScrapingStrategy, 
        schema_recipe: SchemaRecipe, 
        parsed_request: Dict[str, Any],
        website_analysis: WebsiteStructureAnalysis
    ) -> str:
        """Generate comprehensive scraping plan with educational explanations."""
        
        plan_parts = []
        
        # Enhanced introduction with educational context
        plan_parts.append(f"# Intelligent Scraping Plan: {parsed_request['content_type'].title()} Data Extraction")
        plan_parts.append("")
        plan_parts.append("## Educational Overview")
        plan_parts.append(f"This plan demonstrates advanced web scraping patterns using the Atomic Agents framework.")
        plan_parts.append(f"**Complexity Level**: {parsed_request['complexity_level'].title()}")
        plan_parts.append(f"**Educational Focus**: {', '.join(parsed_request['educational_focus']) if parsed_request['educational_focus'] else 'General scraping patterns'}")
        plan_parts.append("")
        
        # Strategy overview with educational insights
        plan_parts.append("## Strategy Analysis & Selection")
        plan_parts.append(f"**Selected Strategy**: `{strategy.scrape_type}` - {self._get_strategy_explanation(strategy.scrape_type)}")
        plan_parts.append(f"**Target Selectors**: {', '.join(strategy.target_selectors[:3])}")
        plan_parts.append(f"  - *Educational Note*: These selectors demonstrate progressive specificity and fallback patterns")
        
        if strategy.pagination_strategy:
            plan_parts.append(f"**Pagination Approach**: `{strategy.pagination_strategy}`")
            plan_parts.append(f"  - *Educational Note*: {self._get_pagination_explanation(strategy.pagination_strategy)}")
        
        plan_parts.append(f"**Performance Settings**:")
        plan_parts.append(f"  - Max Pages: {strategy.max_pages} (balances thoroughness with efficiency)")
        plan_parts.append(f"  - Request Delay: {strategy.request_delay}s (demonstrates respectful crawling)")
        plan_parts.append("")
        
        # Website analysis insights
        plan_parts.append("## Website Analysis Results")
        if 'error' not in website_analysis.metadata:
            plan_parts.append(f"**Successfully Analyzed**: {website_analysis.url}")
            plan_parts.append(f"**Page Title**: {getattr(website_analysis, 'title', 'Unknown')}")
            if hasattr(website_analysis, 'content_patterns') and website_analysis.content_patterns:
                plan_parts.append(f"**Content Patterns Identified**: {len(website_analysis.content_patterns)}")
                for i, pattern in enumerate(website_analysis.content_patterns[:3]):
                    plan_parts.append(f"  - Pattern {i+1}: {pattern.get('type', 'unknown')} content")
            plan_parts.append("  - *Educational Note*: Pattern recognition enables targeted extraction strategies")
        else:
            plan_parts.append(f"**Analysis Status**: Fallback mode due to: {website_analysis.metadata.get('error', 'Unknown error')}")
            plan_parts.append("  - *Educational Note*: Demonstrates graceful error handling and fallback strategies")
        plan_parts.append("")
        
        # Data extraction plan with educational context
        plan_parts.append("## Data Extraction Strategy")
        plan_parts.append("### Fields to Extract (with Educational Insights)")
        
        for field_name, field_def in schema_recipe.fields.items():
            required_text = " (Required)" if field_def.required else " (Optional)"
            priority = "High" if field_def.required else "Medium" if field_def.quality_weight > 0.7 else "Low"
            
            plan_parts.append(f"**{field_name.title()}**{required_text} - Priority: {priority}")
            plan_parts.append(f"  - Description: {field_def.description}")
            plan_parts.append(f"  - Selector: `{field_def.extraction_selector}`")
            plan_parts.append(f"  - Quality Weight: {field_def.quality_weight}")
            
            if hasattr(field_def, 'educational_note'):
                plan_parts.append(f"  - *Educational Insight*: {field_def.educational_note}")
            plan_parts.append("")
        
        # Execution workflow with educational explanations
        plan_parts.append("## Execution Workflow")
        plan_parts.append("### Step-by-Step Process (Educational Breakdown)")
        plan_parts.append("1. **Website Navigation & Analysis**")
        plan_parts.append("   - Navigate to target URL with proper headers and user agent")
        plan_parts.append("   - *Educational*: Demonstrates respectful web crawling practices")
        plan_parts.append("")
        plan_parts.append("2. **Content Identification**")
        plan_parts.append(f"   - Identify content using selectors: {', '.join(strategy.target_selectors[:2])}")
        plan_parts.append("   - *Educational*: Shows CSS selector specificity and fallback strategies")
        plan_parts.append("")
        plan_parts.append("3. **Data Extraction & Processing**")
        plan_parts.append("   - Extract data fields using defined selectors and validation rules")
        plan_parts.append("   - Apply post-processing rules for data cleaning and normalization")
        plan_parts.append("   - *Educational*: Demonstrates data quality control and validation patterns")
        plan_parts.append("")
        
        if strategy.pagination_strategy:
            plan_parts.append("4. **Pagination Handling**")
            plan_parts.append(f"   - Handle multi-page content using {strategy.pagination_strategy} strategy")
            plan_parts.append("   - *Educational*: Shows different pagination patterns and handling techniques")
            plan_parts.append("")
        
        plan_parts.append("5. **Quality Assessment & Filtering**")
        plan_parts.append(f"   - Apply quality filtering (minimum score: {strategy.extraction_rules.get('min_quality', 'N/A')})")
        plan_parts.append("   - *Educational*: Demonstrates quality scoring and data validation")
        plan_parts.append("")
        
        # Quality measures with educational context
        plan_parts.append("## Quality Assurance Framework")
        plan_parts.append("### Educational Quality Measures")
        plan_parts.append("- **Data Validation**: Field-level validation with type checking and format verification")
        plan_parts.append("- **Duplicate Detection**: Content-based deduplication using multiple comparison methods")
        plan_parts.append("- **Quality Scoring**: Multi-dimensional scoring based on completeness, accuracy, and consistency")
        plan_parts.append("- **Error Handling**: Graceful degradation with partial data extraction capabilities")
        plan_parts.append("- **Educational Value**: Each step demonstrates production-ready scraping patterns")
        plan_parts.append("")
        
        # Orchestrator integration notes
        plan_parts.append("## Orchestrator Integration")
        plan_parts.append("### Schema Alignment & Coordination")
        plan_parts.append("- **Input Compatibility**: Plan designed for seamless AtomicScraperTool integration")
        plan_parts.append("- **Output Format**: Results structured for orchestrator consumption and monitoring")
        plan_parts.append("- **Error Propagation**: Comprehensive error handling with orchestrator feedback")
        plan_parts.append("- **Monitoring Support**: Built-in metrics and progress tracking for real-time monitoring")
        
        return "\n".join(plan_parts)
    
    def _get_strategy_explanation(self, strategy_type: str) -> str:
        """Get educational explanation for strategy type."""
        explanations = {
            'list': 'Optimized for extracting multiple items from list-based content with pagination support',
            'detail': 'Designed for comprehensive single-item extraction with deep content analysis',
            'search': 'Specialized for search result pages with relevance scoring and filtering',
            'sitemap': 'Efficient for site-wide content discovery and URL collection'
        }
        return explanations.get(strategy_type, 'General-purpose extraction strategy')
    
    def _get_pagination_explanation(self, pagination_type: str) -> str:
        """Get educational explanation for pagination type."""
        explanations = {
            'next_link': 'Follows "Next" links - common in blogs and article listings',
            'page_numbers': 'Navigates numbered pages - typical in search results and catalogs',
            'infinite_scroll': 'Handles dynamic loading - modern social media and e-commerce sites',
            'load_more': 'Clicks "Load More" buttons - hybrid approach for progressive loading'
        }
        return explanations.get(pagination_type, 'Standard pagination handling')
    
    def _explain_field_selection(self, schema_recipe: SchemaRecipe, parsed_request: Dict[str, Any]) -> str:
        """Explain field selection rationale for educational purposes."""
        explanations = []
        
        required_count = sum(1 for field in schema_recipe.fields.values() if field.required)
        total_count = len(schema_recipe.fields)
        
        explanations.append(f"Selected {total_count} fields with {required_count} required for data completeness")
        explanations.append("Field selection based on semantic HTML analysis and user requirements")
        
        if parsed_request['target_data']:
            explanations.append(f"Specialized fields added for: {', '.join(parsed_request['target_data'])}")
        
        explanations.append("Each field includes multiple selector options for robustness")
        explanations.append("Quality weights assigned based on field importance and reliability")
        
        return ". ".join(explanations)
    
    def _explain_validation_strategy(self, schema_recipe: SchemaRecipe) -> str:
        """Explain validation strategy for educational purposes."""
        return ("Multi-layered validation including format checking, data type validation, "
                "and business rule enforcement. Validation rules are applied progressively "
                "to ensure data quality while allowing partial extraction when appropriate.")
    
    def _explain_quality_approach(self, schema_recipe: SchemaRecipe) -> str:
        """Explain quality approach for educational purposes."""
        return ("Quality scoring combines completeness (40%), accuracy (40%), and consistency (20%) "
                "metrics. This balanced approach ensures reliable data while accommodating "
                "real-world website variations and partial data scenarios.") 
   
    def _generate_comprehensive_reasoning(
        self, 
        website_analysis: WebsiteStructureAnalysis, 
        strategy: ScrapingStrategy, 
        schema_recipe: SchemaRecipe, 
        parsed_request: Dict[str, Any],
        input_data: IntelligentPlanningAgentInputSchema
    ) -> str:
        """Generate comprehensive reasoning with enhanced educational explanations."""
        
        reasoning_parts = []
        
        reasoning_parts.append("# Comprehensive Decision Analysis & Educational Insights")
        reasoning_parts.append("")
        
        # Request analysis with educational context
        reasoning_parts.append("## 1. Request Analysis & Intent Understanding")
        reasoning_parts.append("### Natural Language Processing Results")
        reasoning_parts.append(f"**User Intent Identified**: {parsed_request['user_intent']}")
        reasoning_parts.append(f"**Content Type Determined**: '{parsed_request['content_type']}'")
        reasoning_parts.append(f"**Complexity Level**: {parsed_request['complexity_level']}")
        reasoning_parts.append(f"**Target Data Types**: {', '.join(parsed_request['target_data']) if parsed_request['target_data'] else 'general content'}")
        
        if parsed_request['educational_focus']:
            reasoning_parts.append(f"**Educational Focus Areas**: {', '.join(parsed_request['educational_focus'])}")
        
        reasoning_parts.append("")
        reasoning_parts.append("### Educational Insight: Request Parsing")
        reasoning_parts.append("The natural language processing demonstrates how AI agents can interpret "
                             "user intent and translate it into structured scraping requirements. This "
                             "showcases the power of combining NLP with domain-specific knowledge.")
        reasoning_parts.append("")
        
        # Website analysis reasoning with educational value
        reasoning_parts.append("## 2. Website Structure Analysis")
        reasoning_parts.append("### Analysis Results & Methodology")
        
        if 'error' not in website_analysis.metadata:
            reasoning_parts.append(f"**Successfully Analyzed**: {website_analysis.url}")
            reasoning_parts.append(f"**Page Title**: {getattr(website_analysis, 'title', 'Unknown')}")
            
            if hasattr(website_analysis, 'content_patterns') and website_analysis.content_patterns:
                reasoning_parts.append(f"**Content Patterns Detected**: {len(website_analysis.content_patterns)} patterns")
                for i, pattern in enumerate(website_analysis.content_patterns[:3]):
                    reasoning_parts.append(f"  - Pattern {i+1}: {pattern.get('type', 'unknown')} content with {pattern.get('confidence', 0)}% confidence")
            
            reasoning_parts.append("**Analysis Quality**: High - comprehensive structure detection enabled")
            
        else:
            reasoning_parts.append(f"**Analysis Status**: Fallback mode")
            reasoning_parts.append(f"**Reason**: {website_analysis.metadata.get('error', 'Unknown error')}")
            reasoning_parts.append("**Fallback Strategy**: Using common web patterns and best practices")
        
        reasoning_parts.append("")
        reasoning_parts.append("### Educational Insight: Website Analysis")
        reasoning_parts.append("Website structure analysis is crucial for effective scraping. Even when "
                             "analysis fails, having robust fallback strategies ensures the system remains "
                             "functional. This demonstrates resilient system design principles.")
        reasoning_parts.append("")
        
        # Strategy selection reasoning with educational explanations
        reasoning_parts.append("## 3. Strategy Selection & Optimization")
        reasoning_parts.append(f"### Selected Strategy: `{strategy.scrape_type}`")
        
        strategy_reasoning = self._get_detailed_strategy_reasoning(strategy, parsed_request, website_analysis)
        for reason in strategy_reasoning:
            reasoning_parts.append(f"- {reason}")
        
        reasoning_parts.append("")
        reasoning_parts.append("### Configuration Rationale")
        reasoning_parts.append(f"**Max Pages**: {strategy.max_pages}")
        reasoning_parts.append("  - Balances thoroughness with performance and respectful crawling")
        reasoning_parts.append(f"**Request Delay**: {strategy.request_delay}s")
        reasoning_parts.append("  - Ensures respectful server interaction and reduces blocking risk")
        
        if strategy.pagination_strategy:
            reasoning_parts.append(f"**Pagination Strategy**: {strategy.pagination_strategy}")
            reasoning_parts.append(f"  - {self._get_pagination_explanation(strategy.pagination_strategy)}")
        
        reasoning_parts.append("")
        reasoning_parts.append("### Educational Insight: Strategy Selection")
        reasoning_parts.append("Strategy selection combines website characteristics, user requirements, and "
                             "performance considerations. The decision process demonstrates how AI agents "
                             "can make complex trade-offs between competing objectives.")
        reasoning_parts.append("")
        
        # Selector strategy with educational context
        reasoning_parts.append("## 4. CSS Selector Strategy & Best Practices")
        reasoning_parts.append("### Selector Design Philosophy")
        
        selector_reasoning = self._get_enhanced_selector_reasoning(strategy, parsed_request)
        for reason in selector_reasoning:
            reasoning_parts.append(f"- {reason}")
        
        reasoning_parts.append("")
        reasoning_parts.append("### Selector Examples & Explanations")
        for i, selector in enumerate(strategy.target_selectors[:3]):
            reasoning_parts.append(f"**Selector {i+1}**: `{selector}`")
            reasoning_parts.append(f"  - Purpose: {self._explain_selector_purpose(selector)}")
            reasoning_parts.append(f"  - Robustness: {self._assess_selector_robustness(selector)}")
        
        reasoning_parts.append("")
        reasoning_parts.append("### Educational Insight: CSS Selector Strategy")
        reasoning_parts.append("Effective CSS selectors balance specificity with robustness. The progressive "
                             "fallback approach ensures extraction continues even when primary selectors fail, "
                             "demonstrating defensive programming principles.")
        reasoning_parts.append("")
        
        # Schema design reasoning with educational value
        reasoning_parts.append("## 5. Schema Design & Data Modeling")
        reasoning_parts.append("### Field Selection Methodology")
        
        schema_reasoning = self._get_enhanced_schema_reasoning(schema_recipe, parsed_request)
        for reason in schema_reasoning:
            reasoning_parts.append(f"- {reason}")
        
        reasoning_parts.append("")
        reasoning_parts.append("### Field Analysis & Educational Value")
        for field_name, field_def in list(schema_recipe.fields.items())[:5]:
            priority = "High" if field_def.required else "Medium" if field_def.quality_weight > 0.7 else "Low"
            reasoning_parts.append(f"**{field_name.title()}** ({priority} Priority)")
            reasoning_parts.append(f"  - Selector Strategy: `{field_def.extraction_selector}`")
            reasoning_parts.append(f"  - Quality Weight: {field_def.quality_weight} (reflects field importance)")
            reasoning_parts.append(f"  - Validation: {', '.join(field_def.validation_rules) if field_def.validation_rules else 'Basic validation'}")
            
            if hasattr(field_def, 'educational_note'):
                reasoning_parts.append(f"  - Educational Value: {field_def.educational_note}")
            reasoning_parts.append("")
        
        reasoning_parts.append("### Educational Insight: Schema Design")
        reasoning_parts.append("Schema design reflects the balance between data completeness and extraction "
                             "reliability. Quality weights and validation rules demonstrate how to build "
                             "robust data pipelines that handle real-world data variations.")
        reasoning_parts.append("")
        
        # Quality assurance reasoning
        reasoning_parts.append("## 6. Quality Assurance & Validation Framework")
        reasoning_parts.append("### Quality Metrics & Thresholds")
        
        quality_reasoning = self._get_quality_assurance_reasoning(schema_recipe, strategy)
        for reason in quality_reasoning:
            reasoning_parts.append(f"- {reason}")
        
        reasoning_parts.append("")
        reasoning_parts.append("### Educational Insight: Quality Assurance")
        reasoning_parts.append("Quality assurance in web scraping requires multi-dimensional evaluation. "
                             "The framework demonstrates how to balance data quality requirements with "
                             "practical extraction constraints in production systems.")
        reasoning_parts.append("")
        
        # Orchestrator integration reasoning
        reasoning_parts.append("## 7. Orchestrator Integration & Schema Alignment")
        reasoning_parts.append("### Integration Strategy")
        reasoning_parts.append("- **Input Compatibility**: Schema perfectly aligned with AtomicScraperTool expectations")
        reasoning_parts.append("- **Output Format**: Results structured for orchestrator consumption and monitoring")
        reasoning_parts.append("- **Error Handling**: Comprehensive error propagation with actionable feedback")
        reasoning_parts.append("- **Monitoring Support**: Built-in metrics collection for real-time performance tracking")
        reasoning_parts.append("")
        reasoning_parts.append("### Educational Insight: System Integration")
        reasoning_parts.append("Seamless integration between agents requires careful schema alignment and "
                             "error handling design. This demonstrates how to build composable AI systems "
                             "that work together effectively.")
        reasoning_parts.append("")
        
        # Risk assessment with educational context
        reasoning_parts.append("## 8. Risk Assessment & Mitigation Strategies")
        risks = self._assess_comprehensive_risks(website_analysis, strategy, schema_recipe)
        
        if risks:
            reasoning_parts.append("### Identified Risks & Mitigation")
            for risk in risks:
                reasoning_parts.append(f"- {risk}")
        else:
            reasoning_parts.append("### Risk Assessment: Low Risk")
            reasoning_parts.append("- Strategy appears robust with minimal identified risks")
            reasoning_parts.append("- Comprehensive fallback mechanisms in place")
        
        reasoning_parts.append("")
        reasoning_parts.append("### Educational Insight: Risk Management")
        reasoning_parts.append("Proactive risk assessment and mitigation planning are essential for "
                             "production web scraping systems. This demonstrates how to anticipate "
                             "and prepare for common scraping challenges.")
        reasoning_parts.append("")
        
        # Recommendations with educational value
        reasoning_parts.append("## 9. Optimization Recommendations & Best Practices")
        recommendations = self._generate_educational_recommendations(
            website_analysis, strategy, schema_recipe, parsed_request
        )
        
        reasoning_parts.append("### Performance & Reliability Recommendations")
        for rec in recommendations:
            reasoning_parts.append(f"- {rec}")
        
        reasoning_parts.append("")
        reasoning_parts.append("### Educational Insight: Continuous Improvement")
        reasoning_parts.append("Effective scraping systems incorporate feedback loops and optimization "
                             "opportunities. These recommendations demonstrate how to evolve scraping "
                             "strategies based on performance data and changing requirements.")
        
        return "\n".join(reasoning_parts)
    
    def _get_detailed_strategy_reasoning(
        self, 
        strategy: ScrapingStrategy, 
        parsed_request: Dict[str, Any], 
        analysis: WebsiteStructureAnalysis
    ) -> List[str]:
        """Get detailed reasoning for strategy selection with educational context."""
        reasons = []
        
        if strategy.scrape_type == 'list':
            reasons.append("List strategy selected for bulk data extraction capabilities")
            reasons.append("Optimal for handling multiple similar items with consistent structure")
            if parsed_request['content_type'] == 'list':
                reasons.append("User request explicitly indicated list-type content extraction")
            reasons.append("Strategy supports pagination for comprehensive data collection")
            reasons.append("Educational value: Demonstrates scalable data extraction patterns")
        
        elif strategy.scrape_type == 'detail':
            reasons.append("Detail strategy chosen for comprehensive single-item analysis")
            reasons.append("Enables deep content extraction with relationship mapping")
            if parsed_request['content_type'] == 'detail':
                reasons.append("User request focused on detailed information extraction")
            reasons.append("Educational value: Shows in-depth content analysis techniques")
        
        elif strategy.scrape_type == 'search':
            reasons.append("Search strategy selected for query-based content extraction")
            reasons.append("Optimized for handling search result pages and filtering")
            if 'search' in parsed_request['keywords']:
                reasons.append("Search-related terms detected in user request")
            reasons.append("Educational value: Demonstrates search result processing patterns")
        
        elif strategy.scrape_type == 'sitemap':
            reasons.append("Sitemap strategy chosen for comprehensive site discovery")
            reasons.append("Efficient for mapping site structure and content inventory")
            reasons.append("Educational value: Shows systematic site exploration techniques")
        
        # Add analysis-based reasoning
        if 'error' not in analysis.metadata:
            reasons.append("Website analysis successful - strategy optimized for detected patterns")
            if hasattr(analysis, 'content_patterns') and analysis.content_patterns:
                reasons.append(f"Strategy aligned with {len(analysis.content_patterns)} identified content patterns")
        else:
            reasons.append("Fallback strategy selected due to analysis limitations")
            reasons.append("Strategy designed for maximum compatibility with unknown site structures")
        
        return reasons
    
    def _get_enhanced_selector_reasoning(
        self, 
        strategy: ScrapingStrategy, 
        parsed_request: Dict[str, Any]
    ) -> List[str]:
        """Get enhanced reasoning for selector choices with educational context."""
        reasons = []
        
        reasons.append("Selectors designed using progressive specificity principle")
        reasons.append("Semantic HTML elements prioritized for better maintainability")
        reasons.append("Multiple fallback options included for robustness")
        reasons.append("Data attributes and ARIA labels considered for accessibility")
        
        if parsed_request['target_data']:
            reasons.append(f"Selectors optimized for {', '.join(parsed_request['target_data'])} content types")
        
        reasons.append("Educational value: Demonstrates CSS selector best practices")
        reasons.append("Selector strategy balances specificity with flexibility")
        reasons.append("Fallback mechanisms ensure extraction continues despite DOM changes")
        
        return reasons
    
    def _get_enhanced_schema_reasoning(
        self, 
        schema_recipe: SchemaRecipe, 
        parsed_request: Dict[str, Any]
    ) -> List[str]:
        """Get enhanced reasoning for schema design with educational context."""
        reasons = []
        
        required_count = sum(1 for field in schema_recipe.fields.values() if field.required)
        total_count = len(schema_recipe.fields)
        
        reasons.append(f"Schema includes {total_count} fields with {required_count} required for optimal balance")
        reasons.append("Field selection based on semantic analysis and user requirements")
        reasons.append("Quality weights assigned based on field importance and extraction reliability")
        
        if parsed_request['target_data']:
            reasons.append(f"Specialized fields included for: {', '.join(parsed_request['target_data'])}")
        
        reasons.append("Validation rules designed for data integrity and consistency")
        reasons.append("Educational value: Demonstrates data modeling best practices")
        reasons.append("Schema supports partial data extraction for resilient operation")
        
        return reasons
    
    def _explain_selector_purpose(self, selector: str) -> str:
        """Explain the purpose of a specific selector for educational value."""
        if 'h1' in selector or 'h2' in selector or 'h3' in selector:
            return "Targets semantic heading elements for title extraction"
        elif '.title' in selector or '.name' in selector:
            return "Targets common CSS classes used for titles and names"
        elif 'a[href]' in selector:
            return "Extracts links with proper href attributes"
        elif '.price' in selector or '.cost' in selector:
            return "Identifies pricing information using common class patterns"
        elif '.date' in selector or 'time' in selector:
            return "Extracts temporal information from semantic elements"
        else:
            return "General content selector with fallback capability"
    
    def _assess_selector_robustness(self, selector: str) -> str:
        """Assess selector robustness for educational purposes."""
        if ',' in selector:
            return "High - multiple fallback options provided"
        elif '[' in selector and ']' in selector:
            return "Medium-High - attribute-based selection with good specificity"
        elif '.' in selector:
            return "Medium - class-based selection, may be affected by CSS changes"
        else:
            return "High - semantic element selection, very stable"
    
    def _get_quality_assurance_reasoning(
        self, 
        schema_recipe: SchemaRecipe, 
        strategy: ScrapingStrategy
    ) -> List[str]:
        """Get quality assurance reasoning with educational context."""
        reasons = []
        
        reasons.append("Multi-dimensional quality scoring (completeness, accuracy, consistency)")
        reasons.append("Field-level validation with progressive error handling")
        reasons.append("Quality thresholds set based on field importance and reliability")
        reasons.append("Partial data extraction allowed for resilient operation")
        reasons.append("Educational value: Demonstrates production-quality data validation")
        reasons.append("Quality framework supports continuous improvement and monitoring")
        
        return reasons
    
    def _assess_comprehensive_risks(
        self, 
        analysis: WebsiteStructureAnalysis, 
        strategy: ScrapingStrategy, 
        schema_recipe: SchemaRecipe
    ) -> List[str]:
        """Assess comprehensive risks with educational context."""
        risks = []
        
        if 'error' in analysis.metadata:
            risks.append("Website analysis failed - may indicate access restrictions or dynamic content")
            risks.append("Mitigation: Fallback selectors and error handling implemented")
        
        if strategy.max_pages > 10:
            risks.append("High page count may trigger rate limiting or blocking")
            risks.append("Mitigation: Request delays and respectful crawling patterns implemented")
        
        if len(schema_recipe.fields) > 10:
            risks.append("Complex schema may reduce extraction success rate")
            risks.append("Mitigation: Optional fields and partial data extraction supported")
        
        if strategy.request_delay < 1.0:
            risks.append("Short request delay may appear aggressive to servers")
            risks.append("Mitigation: Consider increasing delay for better server relations")
        
        # Educational risks
        risks.append("Educational consideration: Real websites may have anti-scraping measures")
        risks.append("Educational consideration: Content structure may change over time")
        
        return risks
    
    def _generate_educational_recommendations(
        self, 
        analysis: WebsiteStructureAnalysis, 
        strategy: ScrapingStrategy, 
        schema_recipe: SchemaRecipe, 
        parsed_request: Dict[str, Any]
    ) -> List[str]:
        """Generate educational recommendations for optimization."""
        recommendations = []
        
        recommendations.append("Monitor extraction success rates and adjust selectors as needed")
        recommendations.append("Implement caching for repeated website analysis to improve performance")
        recommendations.append("Consider implementing user-agent rotation for large-scale operations")
        recommendations.append("Add structured logging for better debugging and monitoring")
        
        if parsed_request['complexity_level'] == 'simple':
            recommendations.append("Consider adding more fields for comprehensive data collection")
        elif parsed_request['complexity_level'] == 'complex':
            recommendations.append("Monitor performance impact of complex extraction rules")
        
        recommendations.append("Educational: Implement A/B testing for different selector strategies")
        recommendations.append("Educational: Add metrics collection for continuous improvement")
        recommendations.append("Educational: Consider implementing machine learning for selector optimization")
        
        return recommendations 
   
    def _calculate_enhanced_confidence(
        self, 
        analysis: WebsiteStructureAnalysis, 
        strategy: ScrapingStrategy, 
        schema_recipe: SchemaRecipe,
        parsed_request: Dict[str, Any]
    ) -> float:
        """Calculate enhanced confidence score with detailed assessment."""
        
        confidence_factors = []
        
        # Website analysis confidence (30% weight)
        if 'error' not in analysis.metadata:
            if hasattr(analysis, 'content_patterns') and analysis.content_patterns:
                analysis_confidence = min(0.9, len(analysis.content_patterns) * 0.2 + 0.5)
            else:
                analysis_confidence = 0.6  # Basic analysis successful
        else:
            analysis_confidence = 0.3  # Fallback mode
        
        confidence_factors.append(('website_analysis', analysis_confidence, 0.3))
        
        # Strategy appropriateness confidence (25% weight)
        strategy_confidence = 0.8  # Base confidence
        
        if parsed_request['content_type'] == strategy.scrape_type:
            strategy_confidence += 0.1  # Perfect match
        
        if strategy.pagination_strategy and 'multiple' in parsed_request['keywords']:
            strategy_confidence += 0.05  # Pagination needed and provided
        
        strategy_confidence = min(1.0, strategy_confidence)
        confidence_factors.append(('strategy_selection', strategy_confidence, 0.25))
        
        # Schema quality confidence (25% weight)
        required_fields = sum(1 for field in schema_recipe.fields.values() if field.required)
        total_fields = len(schema_recipe.fields)
        
        if total_fields > 0:
            schema_confidence = 0.6 + (required_fields / total_fields) * 0.3
            
            # Bonus for target data alignment
            if parsed_request['target_data']:
                matching_fields = sum(1 for data_type in parsed_request['target_data'] 
                                    if any(data_type in field_name for field_name in schema_recipe.fields.keys()))
                if matching_fields > 0:
                    schema_confidence += 0.1
        else:
            schema_confidence = 0.4
        
        schema_confidence = min(1.0, schema_confidence)
        confidence_factors.append(('schema_quality', schema_confidence, 0.25))
        
        # Technical feasibility confidence (20% weight)
        technical_confidence = 0.7  # Base technical confidence
        
        # Adjust based on complexity
        if parsed_request['complexity_level'] == 'simple':
            technical_confidence += 0.2
        elif parsed_request['complexity_level'] == 'complex':
            technical_confidence -= 0.1
        
        # Adjust based on selector count
        if len(strategy.target_selectors) >= 3:
            technical_confidence += 0.1  # Good fallback options
        
        technical_confidence = min(1.0, max(0.0, technical_confidence))
        confidence_factors.append(('technical_feasibility', technical_confidence, 0.2))
        
        # Calculate weighted confidence
        total_confidence = sum(confidence * weight for _, confidence, weight in confidence_factors)
        
        # Apply educational bonus for learning scenarios
        if parsed_request['user_intent'] == 'learning':
            total_confidence = min(1.0, total_confidence + 0.05)
        
        return round(total_confidence, 3)
    
    def _create_orchestrator_metadata(
        self,
        operation_id: str,
        operation_start: datetime,
        input_data: IntelligentPlanningAgentInputSchema,
        strategy: ScrapingStrategy,
        schema_recipe: SchemaRecipe
    ) -> Dict[str, Any]:
        """Create comprehensive orchestrator metadata for coordination."""
        
        return {
            'operation_id': operation_id,
            'operation_start': operation_start.isoformat(),
            'planning_agent_version': '2.0',
            'orchestrator_compatible': True,
            'input_hash': hash(f"{input_data.scraping_request}{input_data.target_url}"),
            'strategy_type': strategy.scrape_type,
            'schema_name': schema_recipe.name,
            'field_count': len(schema_recipe.fields),
            'required_field_count': sum(1 for field in schema_recipe.fields.values() if field.required),
            'expected_processing_time': self._estimate_processing_time(strategy, schema_recipe),
            'monitoring_enabled': True,
            'educational_mode': input_data.orchestrator_context.get('educational_mode', False),
            'quality_threshold': input_data.quality_threshold,
            'max_results': input_data.max_results,
            'coordination_requirements': {
                'requires_monitoring': True,
                'supports_partial_results': True,
                'error_recovery_enabled': True,
                'progress_reporting': True
            }
        }
    
    def _generate_educational_insights(
        self,
        strategy: ScrapingStrategy,
        schema_recipe: SchemaRecipe,
        website_analysis: WebsiteStructureAnalysis,
        parsed_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate educational insights for learning value."""
        
        insights = {
            'scraping_patterns_demonstrated': [],
            'best_practices_shown': [],
            'common_challenges_addressed': [],
            'advanced_concepts_illustrated': [],
            'production_considerations': [],
            'learning_objectives': []
        }
        
        # Scraping patterns
        insights['scraping_patterns_demonstrated'] = [
            f"{strategy.scrape_type.title()} extraction pattern with {len(strategy.target_selectors)} selector fallbacks",
            f"Multi-field schema with {len(schema_recipe.fields)} data points",
            "Progressive selector specificity for robustness",
            "Quality-weighted field extraction"
        ]
        
        if strategy.pagination_strategy:
            insights['scraping_patterns_demonstrated'].append(f"{strategy.pagination_strategy} pagination handling")
        
        # Best practices
        insights['best_practices_shown'] = [
            "Semantic HTML selector prioritization",
            "Graceful error handling and fallback strategies",
            "Respectful crawling with appropriate delays",
            "Multi-dimensional quality assessment",
            "Schema-driven data validation"
        ]
        
        # Common challenges
        insights['common_challenges_addressed'] = [
            "Dynamic content and JavaScript rendering considerations",
            "CSS selector brittleness mitigation",
            "Data quality and validation challenges",
            "Rate limiting and server politeness",
            "Partial data extraction scenarios"
        ]
        
        # Advanced concepts
        insights['advanced_concepts_illustrated'] = [
            "Agent orchestration and coordination patterns",
            "Schema alignment between system components",
            "Context-aware strategy selection",
            "Educational AI system design",
            "Production-ready error handling"
        ]
        
        # Production considerations
        insights['production_considerations'] = [
            "Monitoring and metrics collection integration",
            "Scalable architecture patterns",
            "Error propagation and recovery mechanisms",
            "Performance optimization strategies",
            "Compliance and ethical scraping practices"
        ]
        
        # Learning objectives
        insights['learning_objectives'] = [
            "Understand intelligent web scraping strategy selection",
            "Learn schema design for robust data extraction",
            "Master CSS selector strategies and fallback patterns",
            "Explore agent coordination in complex systems",
            "Apply production-ready scraping best practices"
        ]
        
        # Add complexity-specific insights
        if parsed_request['complexity_level'] == 'simple':
            insights['learning_objectives'].append("Introduction to basic scraping concepts")
        elif parsed_request['complexity_level'] == 'complex':
            insights['learning_objectives'].append("Advanced scraping patterns and optimization")
        
        # Add domain-specific insights
        if parsed_request['target_data']:
            insights['domain_specific_patterns'] = {
                data_type: f"Specialized extraction patterns for {data_type} data"
                for data_type in parsed_request['target_data']
            }
        
        return insights
    
    def _estimate_processing_time(self, strategy: ScrapingStrategy, schema_recipe: SchemaRecipe) -> float:
        """Estimate processing time for orchestrator planning."""
        
        base_time = 2.0  # Base processing time in seconds
        
        # Add time based on strategy complexity
        strategy_multipliers = {
            'list': 1.5,
            'detail': 1.0,
            'search': 1.3,
            'sitemap': 2.0
        }
        
        base_time *= strategy_multipliers.get(strategy.scrape_type, 1.0)
        
        # Add time based on field count
        field_time = len(schema_recipe.fields) * 0.1
        
        # Add time based on pagination
        if strategy.pagination_strategy:
            base_time += strategy.max_pages * 0.5
        
        # Add request delay time
        base_time += strategy.request_delay * strategy.max_pages
        
        return round(base_time + field_time, 2)
    
    def _handle_planning_error(
        self, 
        error_message: str, 
        input_data: IntelligentPlanningAgentInputSchema,
        operation_id: str,
        operation_start: datetime
    ) -> IntelligentPlanningAgentOutputSchema:
        """Handle planning errors gracefully with educational context."""
        
        # Create fallback strategy
        fallback_strategy = ScrapingStrategy(
            scrape_type='list',
            target_selectors=['article', '.item', '.post', 'div'],
            extraction_rules={'min_quality': input_data.quality_threshold},
            pagination_strategy=None,
            max_pages=1,
            request_delay=2.0,
            metadata={'fallback_mode': True, 'error': error_message}
        )
        
        # Create fallback schema
        fallback_schema = SchemaRecipe(
            name='fallback_schema',
            description='Fallback schema for error recovery',
            fields={
                'title': FieldDefinition(
                    field_type='string',
                    description='Title or heading text',
                    extraction_selector='h1, h2, h3, .title',
                    required=True,
                    quality_weight=0.8,
                    post_processing=['trim', 'clean']
                ),
                'content': FieldDefinition(
                    field_type='string',
                    description='Main content text',
                    extraction_selector='p, .content, .description',
                    required=False,
                    quality_weight=0.6,
                    post_processing=['trim', 'clean']
                )
            },
            validation_rules=['allow_partial_data'],
            quality_weights={"completeness": 0.5, "accuracy": 0.3, "consistency": 0.2},
            version="1.0"
        )
        
        # Create educational error explanation
        error_plan = f"""# Error Recovery Plan
        
## Error Encountered
**Error**: {error_message}

## Fallback Strategy Activated
- **Strategy**: Basic list extraction with minimal requirements
- **Educational Value**: Demonstrates error recovery and graceful degradation
- **Approach**: Use common HTML patterns for content extraction

## Recovery Measures
- Simplified selector strategy with broad compatibility
- Reduced field requirements for higher success probability
- Enhanced error handling and partial data acceptance

## Educational Insight
This error recovery demonstrates how production systems handle unexpected failures
while maintaining functionality and providing educational value about system resilience.
"""
        
        error_reasoning = f"""# Error Analysis & Recovery Strategy

## Error Context
- **Operation ID**: {operation_id}
- **Error Type**: Planning Agent Error
- **Error Message**: {error_message}
- **Recovery Mode**: Activated

## Educational Value of Error Handling
This error scenario demonstrates several important concepts:
1. **Graceful Degradation**: System continues operating with reduced functionality
2. **Fallback Strategies**: Pre-planned alternatives for common failure modes
3. **Error Transparency**: Clear communication about what went wrong and how it's being handled
4. **System Resilience**: Ability to recover and provide value despite failures

## Recovery Strategy Details
The fallback strategy uses the most common HTML patterns and minimal requirements
to maximize the chance of successful extraction while maintaining educational value.
"""
        
        return IntelligentPlanningAgentOutputSchema(
            scraping_plan=error_plan,
            strategy=fallback_strategy.model_dump(),
            schema_recipe=fallback_schema.model_dump(),
            reasoning=error_reasoning,
            confidence=0.3,  # Low confidence due to error
            orchestrator_metadata={
                'operation_id': operation_id,
                'operation_start': operation_start.isoformat(),
                'error_mode': True,
                'error_message': error_message,
                'fallback_strategy': True,
                'educational_value': 'error_handling_demonstration'
            },
            educational_insights={
                'error_handling_demonstrated': [
                    'Graceful degradation patterns',
                    'Fallback strategy implementation',
                    'Error transparency and communication',
                    'System resilience design'
                ],
                'learning_objectives': [
                    'Understand error handling in AI systems',
                    'Learn fallback strategy design',
                    'Explore system resilience patterns'
                ]
            }
        )