"""
Website Analysis Context Provider.

This module provides dynamic website analysis context to agents,
demonstrating context provider patterns in atomic-agents.
"""

from typing import Optional, Dict, Any, List
from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase


class WebsiteAnalysisContextProvider(SystemPromptContextProviderBase):
    """Provides dynamic website analysis context to agents."""
    
    def __init__(self, title: str = "Website Analysis Context"):
        super().__init__(title=title)
        self.analysis_results: Optional[Dict[str, Any]] = None
        self.content_patterns: List[Dict[str, Any]] = []
        self.navigation_info: Optional[Dict[str, Any]] = None
        self.structure_analysis: Optional[Dict[str, Any]] = None
    
    def set_analysis_results(self, analysis_data: Dict[str, Any]) -> None:
        """Set the website analysis results."""
        self.analysis_results = analysis_data
        
        # Extract specific components
        self.content_patterns = analysis_data.get("content_patterns", [])
        self.navigation_info = analysis_data.get("navigation", {})
        self.structure_analysis = analysis_data.get("structure", {})
    
    def get_info(self) -> str:
        """Return formatted website analysis information."""
        if not self.analysis_results:
            return "No website analysis data available."
        
        info_parts = []
        
        # Basic website information
        if self.structure_analysis:
            info_parts.append("Website Structure Analysis:")
            info_parts.append(f"- Page Type: {self.structure_analysis.get('page_type', 'Unknown')}")
            info_parts.append(f"- Content Complexity: {self.structure_analysis.get('complexity', 'Unknown')}")
            info_parts.append(f"- Dynamic Content: {self.structure_analysis.get('has_dynamic_content', False)}")
        
        # Content patterns
        if self.content_patterns:
            info_parts.append("\nIdentified Content Patterns:")
            for pattern in self.content_patterns[:3]:  # Show top 3 patterns
                pattern_type = pattern.get("type", "Unknown")
                confidence = pattern.get("confidence", 0)
                info_parts.append(f"- {pattern_type} (confidence: {confidence:.1%})")
        
        # Navigation information
        if self.navigation_info:
            info_parts.append("\nNavigation Analysis:")
            info_parts.append(f"- Menu Structure: {self.navigation_info.get('menu_type', 'Unknown')}")
            info_parts.append(f"- Pagination: {self.navigation_info.get('has_pagination', False)}")
            info_parts.append(f"- Search Functionality: {self.navigation_info.get('has_search', False)}")
        
        return "\n".join(info_parts) if info_parts else "Website analysis completed but no specific patterns identified."