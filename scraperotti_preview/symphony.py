"""
🎼 Scraperotti Symphony Configuration

This module provides the configuration system for Scraperotti, the maestro of web scraping.
Like a symphony conductor needs to know the tempo, key, and instrumentation,
Scraperotti needs proper configuration to conduct magnificent data extraction performances.
"""

import os
from typing import Optional, Union
from pydantic import BaseModel, Field, field_validator


class SymphonicConfiguration(BaseModel):
    """
    🎭 Configuration for Scraperotti - The Maestro of Web Scraping
    
    This configuration class orchestrates all the settings needed for virtuoso
    web scraping performances. Each setting is carefully tuned to ensure
    every data extraction becomes a standing ovation.
    
    Like a maestro's score, this configuration defines:
    - The AI models that will perform (maestro and virtuoso)
    - The quality standards for the performance
    - The size and tempo of the scraping ensemble
    - The venue where results will be recorded
    - The ethical guidelines for respectful scraping
    
    Attributes:
        maestro_model (str): The LLM model for the main orchestrator.
            Like choosing between different conductors, this determines
            the overall quality and style of the performance.
            Default: "gpt-4o-mini"
            
        virtuoso_model (str): The LLM model for the planning specialist.
            This is your concertmaster - the lead performer who helps
            plan the perfect extraction strategy.
            Default: "gpt-4o-mini"
            
        performance_quality (float): Minimum quality threshold (0-100).
            Sets the bar for what deserves a standing ovation.
            Higher values mean more selective, premium performances.
            Default: 75.0
            
        ensemble_size (int): Maximum concurrent scraping operations.
            Like the size of your orchestra - more musicians can play
            louder and faster, but require more coordination.
            Default: 5
            
        tempo (float): Delay between requests in seconds.
            The rhythm of your performance. Slower tempo is more respectful
            to the target website and less likely to cause disruption.
            Default: 1.0
            
        venue (str): Directory where performance recordings are stored.
            Your concert hall where all the beautiful extracted data
            will be preserved for posterity.
            Default: "./performances"
            
        respect_stage_rules (bool): Whether to honor robots.txt files.
            True maestros always respect the venue's guidelines.
            Default: True
            
        enable_rhythm_control (bool): Whether to enable automatic rate limiting.
            Keeps the performance flowing smoothly without overwhelming
            the audience (target website).
            Default: True
            
        enable_performance_monitoring (bool): Whether to track performance metrics.
            Like having critics review your performance in real-time.
            Default: True
            
        monitoring_interval (float): How often to update performance metrics.
            The frequency of applause measurement during the show.
            Default: 2.0
            
        max_performers (int): Maximum number of scraper instances.
            The total size of your performing company.
            Default: 5
            
        backstage_crew (int): Maximum number of worker threads.
            The technical crew that makes the magic happen behind the scenes.
            Default: 10
            
        concurrent_acts (int): Maximum number of simultaneous async operations.
            How many different acts can perform at the same time.
            Default: 50
    
    Example:
        Create a configuration for an intimate chamber performance:
        
        ```python
        symphony = SymphonicConfiguration(
            maestro_model="gpt-4",
            performance_quality=85.0,
            ensemble_size=3,
            tempo=1.5,
            venue="./intimate_performances"
        )
        ```
        
        Or configure for a grand orchestral performance:
        
        ```python
        symphony = SymphonicConfiguration(
            maestro_model="gpt-4",
            virtuoso_model="gpt-4",
            performance_quality=90.0,
            ensemble_size=12,
            tempo=0.5,
            venue="/concert_hall/grand_performances",
            max_performers=10,
            backstage_crew=20
        )
        ```
    """
    
    # 🎼 The Maestro and Virtuoso Models
    maestro_model: str = Field(
        default="gpt-4o-mini",
        alias="orchestrator_model",  # Backward compatibility
        description="🎭 The LLM model conducting the scraping symphony"
    )
    
    virtuoso_model: str = Field(
        default="gpt-4o-mini",
        alias="planning_agent_model",  # Backward compatibility  
        description="🎹 The LLM model serving as the planning virtuoso"
    )
    
    # 🎯 Performance Standards
    performance_quality: float = Field(
        default=75.0,
        alias="default_quality_threshold",  # Backward compatibility
        description="🏆 Minimum quality threshold for a standing ovation (0-100)"
    )
    
    # 🎪 Orchestra Configuration
    ensemble_size: int = Field(
        default=5,
        alias="max_concurrent_requests",  # Backward compatibility
        description="🎻 Size of the concurrent scraping ensemble"
    )
    
    tempo: float = Field(
        default=1.0,
        alias="request_delay",  # Backward compatibility
        description="🎵 Tempo between scraping movements (seconds)"
    )
    
    # 🏛️ Venue and Storage
    venue: str = Field(
        default="./performances",
        alias="results_directory",  # Backward compatibility
        description="🎪 Where the performance recordings are stored"
    )
    
    default_program: str = Field(
        default="json",
        alias="default_export_format",  # Backward compatibility
        description="📜 Default format for the performance program"
    )
    
    # 🎭 Ethical Performance Guidelines
    respect_stage_rules: bool = Field(
        default=True,
        alias="respect_robots_txt",  # Backward compatibility
        description="🎩 Whether to respect the venue's rules (robots.txt)"
    )
    
    enable_rhythm_control: bool = Field(
        default=True,
        alias="enable_rate_limiting",  # Backward compatibility
        description="🎼 Enable automatic rhythm control (rate limiting)"
    )
    
    # 📊 Performance Monitoring
    enable_performance_monitoring: bool = Field(
        default=True,
        alias="enable_monitoring",  # Backward compatibility
        description="👏 Enable real-time performance monitoring"
    )
    
    monitoring_interval: float = Field(
        default=2.0,
        description="⏱️ Interval between performance metric updates (seconds)"
    )
    
    # 🎪 Company Configuration
    max_performers: int = Field(
        default=5,
        alias="max_instances",  # Backward compatibility
        description="🎭 Maximum number of performer instances"
    )
    
    backstage_crew: int = Field(
        default=10,
        alias="max_workers",  # Backward compatibility
        description="🔧 Maximum number of backstage crew (worker threads)"
    )
    
    concurrent_acts: int = Field(
        default=50,
        alias="max_async_tasks",  # Backward compatibility
        description="🎪 Maximum number of concurrent acts (async tasks)"
    )
    
    # 🎨 Performance Personality
    theatrical_mode: bool = Field(
        default=True,
        description="🎭 Enable theatrical interface and messaging"
    )
    
    celebration_level: str = Field(
        default="enthusiastic",
        description="🎉 Level of celebration for successful performances"
    )
    
    @field_validator('performance_quality')
    @classmethod
    def validate_performance_quality(cls, v: float) -> float:
        """🎯 Ensure performance quality is worthy of the stage."""
        if not 0 <= v <= 100:
            raise ValueError('🎭 Performance quality must be between 0 and 100 - even understudies deserve a chance!')
        return v
    
    @field_validator('ensemble_size')
    @classmethod
    def validate_ensemble_size(cls, v: int) -> int:
        """🎻 Ensure we have enough musicians for a proper performance."""
        if v <= 0:
            raise ValueError('🎼 Every performance needs at least one musician!')
        if v > 50:
            raise ValueError('🎪 Even the grandest orchestra rarely exceeds 50 concurrent performers!')
        return v
    
    @field_validator('tempo')
    @classmethod
    def validate_tempo(cls, v: float) -> float:
        """🎵 Ensure the tempo allows for a graceful performance."""
        if v < 0:
            raise ValueError('🎼 Tempo cannot be negative - we\'re not playing backwards!')
        if v > 60:
            raise ValueError('⏰ A tempo over 60 seconds might put the audience to sleep!')
        return v
    
    @field_validator('default_program')
    @classmethod
    def validate_program_format(cls, v: str) -> str:
        """📜 Ensure the program format is suitable for the audience."""
        valid_formats = {'json', 'csv', 'markdown', 'excel'}
        if v not in valid_formats:
            raise ValueError(f'🎪 Program format must be one of: {valid_formats}')
        return v
    
    @field_validator('celebration_level')
    @classmethod
    def validate_celebration_level(cls, v: str) -> str:
        """🎉 Ensure appropriate celebration for the venue."""
        valid_levels = {'subdued', 'polite', 'enthusiastic', 'exuberant', 'operatic'}
        if v not in valid_levels:
            raise ValueError(f'🎭 Celebration level must be one of: {valid_levels}')
        return v
    
    @classmethod
    def from_environment(cls) -> "SymphonicConfiguration":
        """
        🌟 Create configuration from environment variables.
        
        This method allows the maestro to read the performance requirements
        from the environment, making it perfect for different venues
        (development, staging, production).
        
        Environment Variables:
            MAESTRO_MODEL: The conducting AI model
            VIRTUOSO_MODEL: The planning specialist model  
            PERFORMANCE_QUALITY: Quality threshold (0-100)
            ENSEMBLE_SIZE: Number of concurrent performers
            TEMPO: Delay between movements (seconds)
            VENUE: Where to store performance recordings
            THEATRICAL_MODE: Enable theatrical interface (true/false)
            
        Returns:
            SymphonicConfiguration: A perfectly tuned configuration
            ready for magnificent performances.
        
        Example:
            ```bash
            export MAESTRO_MODEL="gpt-4"
            export PERFORMANCE_QUALITY="85.0"
            export ENSEMBLE_SIZE="8"
            export THEATRICAL_MODE="true"
            ```
            
            ```python
            symphony = SymphonicConfiguration.from_environment()
            print(f"🎭 Ready for {symphony.maestro_model} performance!")
            ```
        """
        return cls(
            maestro_model=os.getenv("MAESTRO_MODEL", "gpt-4o-mini"),
            virtuoso_model=os.getenv("VIRTUOSO_MODEL", "gpt-4o-mini"),
            performance_quality=float(os.getenv("PERFORMANCE_QUALITY", "75.0")),
            ensemble_size=int(os.getenv("ENSEMBLE_SIZE", "5")),
            tempo=float(os.getenv("TEMPO", "1.0")),
            venue=os.getenv("VENUE", "./performances"),
            default_program=os.getenv("DEFAULT_PROGRAM", "json"),
            respect_stage_rules=os.getenv("RESPECT_STAGE_RULES", "true").lower() in ("true", "1", "yes", "on"),
            enable_rhythm_control=os.getenv("ENABLE_RHYTHM_CONTROL", "true").lower() in ("true", "1", "yes", "on"),
            enable_performance_monitoring=os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() in ("true", "1", "yes", "on"),
            monitoring_interval=float(os.getenv("MONITORING_INTERVAL", "2.0")),
            max_performers=int(os.getenv("MAX_PERFORMERS", "5")),
            backstage_crew=int(os.getenv("BACKSTAGE_CREW", "10")),
            concurrent_acts=int(os.getenv("CONCURRENT_ACTS", "50")),
            theatrical_mode=os.getenv("THEATRICAL_MODE", "true").lower() in ("true", "1", "yes", "on"),
            celebration_level=os.getenv("CELEBRATION_LEVEL", "enthusiastic"),
        )
    
    def get_performance_summary(self) -> str:
        """
        🎪 Get a beautiful summary of the performance configuration.
        
        Returns:
            str: A theatrical description of the current configuration,
            perfect for displaying to users before the show begins.
        """
        summary = f"""
🎭 Scraperotti Performance Configuration 🎭

🎼 The Orchestra:
   Maestro: {self.maestro_model}
   Virtuoso: {self.virtuoso_model}
   Ensemble Size: {self.ensemble_size} performers
   Backstage Crew: {self.backstage_crew} technicians

🎯 Performance Standards:
   Quality Threshold: {self.performance_quality}% (for standing ovation)
   Tempo: {self.tempo}s between movements
   Celebration Style: {self.celebration_level}

🏛️ Venue Details:
   Performance Hall: {self.venue}
   Program Format: {self.default_program.upper()}
   Stage Rules Respected: {'Yes' if self.respect_stage_rules else 'No'}
   Rhythm Control: {'Enabled' if self.enable_rhythm_control else 'Disabled'}

📊 Monitoring:
   Performance Tracking: {'Enabled' if self.enable_performance_monitoring else 'Disabled'}
   Update Interval: {self.monitoring_interval}s
   
🎪 Tonight's Performance Promises to be Magnificent! 🎪
        """
        return summary.strip()


# 🎭 Backward Compatibility Aliases
# These ensure existing code continues to work while users migrate to the new theatrical names

IntelligentScrapingConfig = SymphonicConfiguration

# Legacy environment loading method
def from_env() -> SymphonicConfiguration:
    """Legacy method for backward compatibility."""
    return SymphonicConfiguration.from_environment()

# Attach the legacy method to the old class name for complete compatibility
IntelligentScrapingConfig.from_env = staticmethod(from_env)


# 🎼 Configuration Presets for Different Performance Types

class PerformancePresets:
    """🎪 Pre-configured settings for different types of performances."""
    
    @staticmethod
    def intimate_chamber() -> SymphonicConfiguration:
        """🎻 Configuration for intimate, high-quality performances."""
        return SymphonicConfiguration(
            maestro_model="gpt-4",
            virtuoso_model="gpt-4",
            performance_quality=90.0,
            ensemble_size=2,
            tempo=2.0,
            celebration_level="polite",
            venue="./chamber_performances"
        )
    
    @staticmethod
    def grand_orchestra() -> SymphonicConfiguration:
        """🎪 Configuration for large-scale, high-throughput performances."""
        return SymphonicConfiguration(
            maestro_model="gpt-4",
            virtuoso_model="gpt-4o-mini",
            performance_quality=80.0,
            ensemble_size=12,
            tempo=0.5,
            max_performers=8,
            backstage_crew=16,
            concurrent_acts=80,
            celebration_level="exuberant",
            venue="./grand_performances"
        )
    
    @staticmethod
    def budget_recital() -> SymphonicConfiguration:
        """🎵 Configuration for cost-effective performances."""
        return SymphonicConfiguration(
            maestro_model="gpt-4o-mini",
            virtuoso_model="gpt-4o-mini",
            performance_quality=65.0,
            ensemble_size=3,
            tempo=1.5,
            celebration_level="subdued",
            venue="./budget_performances"
        )
    
    @staticmethod
    def speed_performance() -> SymphonicConfiguration:
        """⚡ Configuration optimized for rapid data extraction."""
        return SymphonicConfiguration(
            maestro_model="gpt-4o-mini",
            virtuoso_model="gpt-4o-mini",
            performance_quality=60.0,
            ensemble_size=10,
            tempo=0.2,
            max_performers=8,
            concurrent_acts=100,
            celebration_level="enthusiastic",
            venue="./speed_performances"
        )


# 🎭 Export the main configuration class and presets
__all__ = [
    "SymphonicConfiguration",
    "IntelligentScrapingConfig",  # Backward compatibility
    "PerformancePresets",
    "from_env",  # Legacy function
]