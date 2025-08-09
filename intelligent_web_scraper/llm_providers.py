"""
LLM Provider Management for Intelligent Web Scraper.

This module provides unified access to different LLM providers including
OpenAI, Gemini, DeepSeek, OpenRouter, and Anthropic. It abstracts the
provider-specific implementations behind a common interface.
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class LLMProviderNotAvailableError(LLMProviderError):
    """Raised when a requested LLM provider is not available."""
    pass


class LLMProviderConfigError(LLMProviderError):
    """Raised when LLM provider configuration is invalid."""
    pass


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the provider-specific client."""
        pass
    
    @abstractmethod
    def get_client(self):
        """Get the initialized client for this provider."""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the provider configuration."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""
    
    def _initialize_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            api_key = self.config.get("api_key")
            base_url = self.config.get("base_url", "https://api.openai.com/v1")
            
            if not api_key:
                raise LLMProviderConfigError("OpenAI API key is required")
            
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            
            logger.info(f"OpenAI client initialized with base URL: {base_url}")
            
        except ImportError:
            raise LLMProviderNotAvailableError("OpenAI package not installed. Install with: pip install openai")
        except Exception as e:
            raise LLMProviderConfigError(f"Failed to initialize OpenAI client: {e}")
    
    def get_client(self):
        """Get the OpenAI client."""
        return self.client
    
    def validate_config(self) -> bool:
        """Validate OpenAI configuration."""
        api_key = self.config.get("api_key")
        return bool(api_key)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""
    
    def _initialize_client(self) -> None:
        """Initialize Gemini client."""
        try:
            import google.generativeai as genai
            
            api_key = self.config.get("api_key")
            
            if not api_key:
                raise LLMProviderConfigError("Gemini API key is required")
            
            genai.configure(api_key=api_key)
            self.client = genai
            
            logger.info("Gemini client initialized")
            
        except ImportError:
            raise LLMProviderNotAvailableError("Google Generative AI package not installed. Install with: pip install google-generativeai")
        except Exception as e:
            raise LLMProviderConfigError(f"Failed to initialize Gemini client: {e}")
    
    def get_client(self):
        """Get the Gemini client."""
        return self.client
    
    def validate_config(self) -> bool:
        """Validate Gemini configuration."""
        api_key = self.config.get("api_key")
        return bool(api_key)


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek LLM provider (OpenAI-compatible API)."""
    
    def _initialize_client(self) -> None:
        """Initialize DeepSeek client."""
        try:
            from openai import OpenAI
            
            api_key = self.config.get("api_key")
            base_url = self.config.get("base_url", "https://api.deepseek.com/v1")
            
            if not api_key:
                raise LLMProviderConfigError("DeepSeek API key is required")
            
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            
            logger.info(f"DeepSeek client initialized with base URL: {base_url}")
            
        except ImportError:
            raise LLMProviderNotAvailableError("OpenAI package not installed. Install with: pip install openai")
        except Exception as e:
            raise LLMProviderConfigError(f"Failed to initialize DeepSeek client: {e}")
    
    def get_client(self):
        """Get the DeepSeek client."""
        return self.client
    
    def validate_config(self) -> bool:
        """Validate DeepSeek configuration."""
        api_key = self.config.get("api_key")
        return bool(api_key)


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter LLM provider (OpenAI-compatible API)."""
    
    def _initialize_client(self) -> None:
        """Initialize OpenRouter client."""
        try:
            from openai import OpenAI
            
            api_key = self.config.get("api_key")
            base_url = self.config.get("base_url", "https://openrouter.ai/api/v1")
            
            if not api_key:
                raise LLMProviderConfigError("OpenRouter API key is required")
            
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            
            logger.info(f"OpenRouter client initialized with base URL: {base_url}")
            
        except ImportError:
            raise LLMProviderNotAvailableError("OpenAI package not installed. Install with: pip install openai")
        except Exception as e:
            raise LLMProviderConfigError(f"Failed to initialize OpenRouter client: {e}")
    
    def get_client(self):
        """Get the OpenRouter client."""
        return self.client
    
    def validate_config(self) -> bool:
        """Validate OpenRouter configuration."""
        api_key = self.config.get("api_key")
        return bool(api_key)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM provider."""
    
    def _initialize_client(self) -> None:
        """Initialize Anthropic client."""
        try:
            from anthropic import Anthropic
            
            api_key = self.config.get("api_key")
            
            if not api_key:
                raise LLMProviderConfigError("Anthropic API key is required")
            
            self.client = Anthropic(api_key=api_key)
            
            logger.info("Anthropic client initialized")
            
        except ImportError:
            raise LLMProviderNotAvailableError("Anthropic package not installed. Install with: pip install anthropic")
        except Exception as e:
            raise LLMProviderConfigError(f"Failed to initialize Anthropic client: {e}")
    
    def get_client(self):
        """Get the Anthropic client."""
        return self.client
    
    def validate_config(self) -> bool:
        """Validate Anthropic configuration."""
        api_key = self.config.get("api_key")
        return bool(api_key)


class LLMProviderManager:
    """Manager for different LLM providers."""
    
    PROVIDERS = {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "deepseek": DeepSeekProvider,
        "openrouter": OpenRouterProvider,
        "anthropic": AnthropicProvider,
    }
    
    def __init__(self):
        self._providers: Dict[str, BaseLLMProvider] = {}
    
    def get_provider(self, provider_name: str, config: Dict[str, Any]) -> BaseLLMProvider:
        """Get or create a provider instance."""
        if provider_name not in self.PROVIDERS:
            available_providers = list(self.PROVIDERS.keys())
            raise LLMProviderError(f"Unknown provider '{provider_name}'. Available providers: {available_providers}")
        
        # Create a cache key based on provider name and config
        cache_key = f"{provider_name}_{hash(str(sorted(config.items())))}"
        
        if cache_key not in self._providers:
            provider_class = self.PROVIDERS[provider_name]
            self._providers[cache_key] = provider_class(config)
        
        return self._providers[cache_key]
    
    def validate_provider_config(self, provider_name: str, config: Dict[str, Any]) -> bool:
        """Validate configuration for a specific provider."""
        try:
            provider = self.get_provider(provider_name, config)
            return provider.validate_config()
        except Exception as e:
            logger.error(f"Provider validation failed for {provider_name}: {e}")
            return False
    
    def get_available_providers(self) -> Dict[str, bool]:
        """Get a list of available providers and their availability status."""
        availability = {}
        
        for provider_name, provider_class in self.PROVIDERS.items():
            try:
                # Try to create a dummy instance to check if dependencies are available
                dummy_config = {"api_key": "dummy"}
                provider_class(dummy_config)
                availability[provider_name] = True
            except LLMProviderNotAvailableError:
                availability[provider_name] = False
            except Exception:
                # Other errors (like config errors) mean the provider is available but misconfigured
                availability[provider_name] = True
        
        return availability
    
    def get_client_for_config(self, config_dict: Dict[str, Any]):
        """Get an LLM client based on configuration dictionary."""
        provider_name = config_dict.get("provider", "openai")
        
        # Extract provider-specific config
        provider_config = {
            "api_key": config_dict.get("api_key"),
            "base_url": config_dict.get("base_url")
        }
        
        # Remove None values
        provider_config = {k: v for k, v in provider_config.items() if v is not None}
        
        provider = self.get_provider(provider_name, provider_config)
        return provider.get_client()


# Global provider manager instance
provider_manager = LLMProviderManager()


def get_llm_client(config_dict: Dict[str, Any]):
    """Convenience function to get an LLM client."""
    return provider_manager.get_client_for_config(config_dict)


def validate_llm_config(provider_name: str, config: Dict[str, Any]) -> bool:
    """Convenience function to validate LLM configuration."""
    return provider_manager.validate_provider_config(provider_name, config)


def get_available_providers() -> Dict[str, bool]:
    """Convenience function to get available providers."""
    return provider_manager.get_available_providers()


def test_provider_connection(provider_name: str, config: Dict[str, Any]) -> bool:
    """Test connection to a specific provider."""
    try:
        provider = provider_manager.get_provider(provider_name, config)
        client = provider.get_client()
        
        # Perform a simple test based on provider type
        if provider_name == "openai" or provider_name in ["deepseek", "openrouter"]:
            # Test with a simple completion
            response = client.chat.completions.create(
                model=config.get("model", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return bool(response.choices)
            
        elif provider_name == "gemini":
            # Test with Gemini
            model = client.GenerativeModel(config.get("model", "gemini-1.5-flash"))
            response = model.generate_content("Hello")
            return bool(response.text)
            
        elif provider_name == "anthropic":
            # Test with Anthropic
            response = client.messages.create(
                model=config.get("model", "claude-3-haiku-20240307"),
                max_tokens=5,
                messages=[{"role": "user", "content": "Hello"}]
            )
            return bool(response.content)
        
        return True
        
    except Exception as e:
        logger.error(f"Provider connection test failed for {provider_name}: {e}")
        return False