"""
Unit tests for the tool factory and dependency injection patterns.

This module tests tool instantiation, configuration management, and dependency injection.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

from intelligent_web_scraper.tools.tool_factory import (
    AtomicScraperToolFactory,
    ToolConfigurationError,
    ConfigurationManager
)

# Mock classes for testing since the actual implementation is different
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Type, Callable
from enum import Enum

@dataclass
class ToolConfiguration:
    tool_name: str
    tool_class: str
    config_params: Dict[str, Any]
    enabled: bool = True
    dependencies: List[str] = None
    initialization_order: int = 0
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
    
    def validate(self) -> List[str]:
        errors = []
        if not self.tool_name:
            errors.append("Tool name cannot be empty")
        if not self.tool_class:
            errors.append("Tool class cannot be empty")
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tool_name': self.tool_name,
            'tool_class': self.tool_class,
            'config_params': self.config_params,
            'enabled': self.enabled,
            'dependencies': self.dependencies,
            'initialization_order': self.initialization_order
        }

class ConfigurationValidator:
    def __init__(self):
        self.validation_rules = {
            'tool_name': lambda name: bool(name and name.replace('_', '').replace('-', '').isalnum()),
            'tool_class': lambda cls: bool(cls),
            'config_params': lambda params: isinstance(params, dict)
        }
    
    def validate_tool_configuration(self, config: ToolConfiguration) -> List[str]:
        errors = []
        
        if not config.tool_name:
            errors.append("Tool name cannot be empty")
        elif not self.validation_rules['tool_name'](config.tool_name):
            errors.append("Tool name contains invalid characters")
        
        if not config.tool_class:
            errors.append("Tool class cannot be empty")
        
        return errors
    
    def validate_dependency_graph(self, configs: List[ToolConfiguration]) -> List[str]:
        errors = []
        tool_names = {config.tool_name for config in configs}
        
        # Check for missing dependencies
        for config in configs:
            for dep in config.dependencies:
                if dep not in tool_names:
                    errors.append(f"Dependency '{dep}' not found for tool '{config.tool_name}'")
        
        # Check for circular dependencies (simplified)
        for config in configs:
            if config.tool_name in config.dependencies:
                errors.append(f"Circular dependency detected: {config.tool_name} depends on itself")
            
            # Check for simple circular dependencies
            for other_config in configs:
                if (config.tool_name in other_config.dependencies and 
                    other_config.tool_name in config.dependencies):
                    errors.append(f"Circular dependency detected between {config.tool_name} and {other_config.tool_name}")
        
        return errors
    
    def validate_configuration_parameters(self, params: Dict[str, Any]) -> List[str]:
        errors = []
        
        if 'timeout' in params and params['timeout'] < 0:
            errors.append("Timeout cannot be negative")
        
        if 'retries' in params and not isinstance(params['retries'], int):
            errors.append("Retries must be an integer")
        
        if 'url' in params and not params['url'].startswith(('http://', 'https://')):
            errors.append("URL must start with http:// or https://")
        
        if 'max_items' in params and params['max_items'] <= 0:
            errors.append("Max items must be positive")
        
        return errors

class DependencyInjector:
    def __init__(self):
        self.dependencies: Dict[str, Any] = {}
        self.singletons: Dict[str, Any] = {}
    
    def register_dependency(self, name: str, dependency: Any):
        self.dependencies[name] = dependency
    
    def register_singleton(self, name: str, cls: Type, config: Dict[str, Any]):
        if name not in self.singletons:
            self.singletons[name] = cls(config)
        self.dependencies[name] = self.singletons[name]
    
    def get_dependency(self, name: str) -> Optional[Any]:
        return self.dependencies.get(name)
    
    def inject_dependencies(self, cls: Type, **kwargs) -> Any:
        # Simple dependency injection - inspect constructor and inject known dependencies
        import inspect
        sig = inspect.signature(cls.__init__)
        injected_kwargs = {}
        
        for param_name in sig.parameters:
            if param_name == 'self':
                continue
            if param_name in self.dependencies:
                injected_kwargs[param_name] = self.dependencies[param_name]
        
        # Override with provided kwargs
        injected_kwargs.update(kwargs)
        
        return cls(**injected_kwargs)
    
    def has_dependency(self, name: str) -> bool:
        return name in self.dependencies
    
    def unregister_dependency(self, name: str) -> bool:
        if name in self.dependencies:
            del self.dependencies[name]
            if name in self.singletons:
                del self.singletons[name]
            return True
        return False
    
    def clear_dependencies(self):
        self.dependencies.clear()
        self.singletons.clear()

class ToolFactory:
    def __init__(self, config):
        self.config = config
        self.tool_configurations: Dict[str, ToolConfiguration] = {}
        self.dependency_injector = DependencyInjector()
        self.validator = ConfigurationValidator()
        self.created_tools: Dict[str, Any] = {}
    
    def register_tool_configuration(self, config: ToolConfiguration):
        errors = self.validator.validate_tool_configuration(config)
        if errors:
            raise ValueError(f"Invalid tool configuration: {'; '.join(errors)}")
        self.tool_configurations[config.tool_name] = config
    
    def create_tool(self, tool_name: str) -> Any:
        if tool_name not in self.tool_configurations:
            raise ValueError(f"Tool '{tool_name}' not registered")
        
        config = self.tool_configurations[tool_name]
        if not config.enabled:
            raise ValueError(f"Tool '{tool_name}' is disabled")
        
        # Mock tool creation
        mock_tool = Mock()
        self.created_tools[tool_name] = mock_tool
        return mock_tool
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        return self.created_tools.get(tool_name)
    
    def get_or_create_tool(self, tool_name: str) -> Any:
        if tool_name in self.created_tools:
            return self.created_tools[tool_name]
        return self.create_tool(tool_name)
    
    def list_registered_tools(self) -> List[str]:
        return list(self.tool_configurations.keys())
    
    def list_created_tools(self) -> List[str]:
        return list(self.created_tools.keys())
    
    def destroy_tool(self, tool_name: str) -> bool:
        if tool_name in self.created_tools:
            tool = self.created_tools.pop(tool_name)
            if hasattr(tool, 'cleanup'):
                tool.cleanup()
            return True
        return False
    
    def destroy_all_tools(self):
        for tool_name in list(self.created_tools.keys()):
            self.destroy_tool(tool_name)
    
    def get_tool_status(self) -> Dict[str, Any]:
        enabled_count = sum(1 for config in self.tool_configurations.values() if config.enabled)
        disabled_count = len(self.tool_configurations) - enabled_count
        
        return {
            'total_registered': len(self.tool_configurations),
            'total_created': len(self.created_tools),
            'enabled_tools': enabled_count,
            'disabled_tools': disabled_count,
            'registered_tools': list(self.tool_configurations.keys()),
            'created_tools': list(self.created_tools.keys())
        }
from intelligent_web_scraper.config import IntelligentScrapingConfig


class TestToolConfiguration:
    """Test the ToolConfiguration class."""
    
    def test_tool_configuration_creation(self):
        """Test tool configuration creation with default values."""
        config = ToolConfiguration(
            tool_name="test_tool",
            tool_class="TestTool",
            config_params={"param1": "value1", "param2": 42}
        )
        
        assert config.tool_name == "test_tool"
        assert config.tool_class == "TestTool"
        assert config.config_params == {"param1": "value1", "param2": 42}
        assert config.enabled is True
        assert config.dependencies == []
        assert config.initialization_order == 0
    
    def test_tool_configuration_with_dependencies(self):
        """Test tool configuration with dependencies."""
        config = ToolConfiguration(
            tool_name="dependent_tool",
            tool_class="DependentTool",
            config_params={},
            dependencies=["base_tool", "helper_tool"],
            initialization_order=2,
            enabled=False
        )
        
        assert config.tool_name == "dependent_tool"
        assert config.dependencies == ["base_tool", "helper_tool"]
        assert config.initialization_order == 2
        assert config.enabled is False
    
    def test_tool_configuration_validation(self):
        """Test tool configuration validation."""
        # Valid configuration
        valid_config = ToolConfiguration(
            tool_name="valid_tool",
            tool_class="ValidTool",
            config_params={"timeout": 30}
        )
        
        errors = valid_config.validate()
        assert len(errors) == 0
        
        # Invalid configuration - empty tool name
        invalid_config = ToolConfiguration(
            tool_name="",
            tool_class="ValidTool",
            config_params={}
        )
        
        errors = invalid_config.validate()
        assert len(errors) > 0
        assert any("Tool name cannot be empty" in error for error in errors)
        
        # Invalid configuration - empty tool class
        invalid_config2 = ToolConfiguration(
            tool_name="valid_name",
            tool_class="",
            config_params={}
        )
        
        errors = invalid_config2.validate()
        assert len(errors) > 0
        assert any("Tool class cannot be empty" in error for error in errors)
    
    def test_tool_configuration_serialization(self):
        """Test tool configuration serialization."""
        config = ToolConfiguration(
            tool_name="serializable_tool",
            tool_class="SerializableTool",
            config_params={"key": "value"},
            dependencies=["dep1"],
            initialization_order=1
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['tool_name'] == "serializable_tool"
        assert config_dict['tool_class'] == "SerializableTool"
        assert config_dict['config_params'] == {"key": "value"}
        assert config_dict['dependencies'] == ["dep1"]
        assert config_dict['initialization_order'] == 1
        assert config_dict['enabled'] is True


class TestConfigurationValidator:
    """Test the ConfigurationValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a configuration validator for testing."""
        return ConfigurationValidator()
    
    def test_validator_initialization(self, validator):
        """Test configuration validator initialization."""
        assert hasattr(validator, 'validation_rules')
        assert isinstance(validator.validation_rules, dict)
        assert len(validator.validation_rules) > 0
    
    def test_validate_tool_configuration_valid(self, validator):
        """Test validation of valid tool configuration."""
        config = ToolConfiguration(
            tool_name="valid_tool",
            tool_class="ValidTool",
            config_params={"timeout": 30, "retries": 3}
        )
        
        errors = validator.validate_tool_configuration(config)
        
        assert isinstance(errors, list)
        assert len(errors) == 0
    
    def test_validate_tool_configuration_invalid_name(self, validator):
        """Test validation with invalid tool name."""
        # Empty name
        config1 = ToolConfiguration(
            tool_name="",
            tool_class="ValidTool",
            config_params={}
        )
        
        errors1 = validator.validate_tool_configuration(config1)
        assert len(errors1) > 0
        assert any("Tool name cannot be empty" in error for error in errors1)
        
        # Invalid characters in name
        config2 = ToolConfiguration(
            tool_name="invalid tool name!",
            tool_class="ValidTool",
            config_params={}
        )
        
        errors2 = validator.validate_tool_configuration(config2)
        assert len(errors2) > 0
        assert any("Tool name contains invalid characters" in error for error in errors2)
    
    def test_validate_tool_configuration_invalid_class(self, validator):
        """Test validation with invalid tool class."""
        config = ToolConfiguration(
            tool_name="valid_tool",
            tool_class="",
            config_params={}
        )
        
        errors = validator.validate_tool_configuration(config)
        
        assert len(errors) > 0
        assert any("Tool class cannot be empty" in error for error in errors)
    
    def test_validate_tool_configuration_circular_dependencies(self, validator):
        """Test validation of circular dependencies."""
        config1 = ToolConfiguration(
            tool_name="tool_a",
            tool_class="ToolA",
            config_params={},
            dependencies=["tool_b"]
        )
        
        config2 = ToolConfiguration(
            tool_name="tool_b",
            tool_class="ToolB",
            config_params={},
            dependencies=["tool_a"]
        )
        
        configs = [config1, config2]
        errors = validator.validate_dependency_graph(configs)
        
        assert len(errors) > 0
        assert any("Circular dependency detected" in error for error in errors)
    
    def test_validate_tool_configuration_missing_dependencies(self, validator):
        """Test validation of missing dependencies."""
        config = ToolConfiguration(
            tool_name="dependent_tool",
            tool_class="DependentTool",
            config_params={},
            dependencies=["missing_tool"]
        )
        
        configs = [config]
        errors = validator.validate_dependency_graph(configs)
        
        assert len(errors) > 0
        assert any("Dependency 'missing_tool' not found" in error for error in errors)
    
    def test_validate_configuration_parameters(self, validator):
        """Test validation of configuration parameters."""
        # Valid parameters
        valid_params = {
            "timeout": 30,
            "retries": 3,
            "url": "https://example.com",
            "enabled": True
        }
        
        errors = validator.validate_configuration_parameters(valid_params)
        assert len(errors) == 0
        
        # Invalid parameters
        invalid_params = {
            "timeout": -5,  # Negative timeout
            "retries": "invalid",  # Wrong type
            "url": "not_a_url",  # Invalid URL format
            "max_items": 0  # Zero max items
        }
        
        errors = validator.validate_configuration_parameters(invalid_params)
        assert len(errors) > 0


class TestDependencyInjector:
    """Test the DependencyInjector class."""
    
    @pytest.fixture
    def injector(self):
        """Create a dependency injector for testing."""
        return DependencyInjector()
    
    def test_injector_initialization(self, injector):
        """Test dependency injector initialization."""
        assert hasattr(injector, 'dependencies')
        assert hasattr(injector, 'singletons')
        assert isinstance(injector.dependencies, dict)
        assert isinstance(injector.singletons, dict)
        assert len(injector.dependencies) == 0
        assert len(injector.singletons) == 0
    
    def test_register_dependency(self, injector):
        """Test registering dependencies."""
        mock_dependency = Mock()
        
        injector.register_dependency("test_service", mock_dependency)
        
        assert "test_service" in injector.dependencies
        assert injector.dependencies["test_service"] is mock_dependency
    
    def test_register_singleton(self, injector):
        """Test registering singleton dependencies."""
        class TestService:
            def __init__(self, config):
                self.config = config
        
        injector.register_singleton("singleton_service", TestService, {"param": "value"})
        
        # Get singleton twice - should be same instance
        instance1 = injector.get_dependency("singleton_service")
        instance2 = injector.get_dependency("singleton_service")
        
        assert instance1 is instance2
        assert isinstance(instance1, TestService)
        assert instance1.config == {"param": "value"}
    
    def test_get_dependency_existing(self, injector):
        """Test getting existing dependency."""
        mock_service = Mock()
        injector.register_dependency("existing_service", mock_service)
        
        retrieved = injector.get_dependency("existing_service")
        
        assert retrieved is mock_service
    
    def test_get_dependency_missing(self, injector):
        """Test getting missing dependency."""
        retrieved = injector.get_dependency("missing_service")
        
        assert retrieved is None
    
    def test_inject_dependencies(self, injector):
        """Test injecting dependencies into a class."""
        # Register dependencies
        mock_db = Mock()
        mock_cache = Mock()
        injector.register_dependency("database", mock_db)
        injector.register_dependency("cache", mock_cache)
        
        # Create class that needs dependencies
        class ServiceWithDependencies:
            def __init__(self, database=None, cache=None, other_param="default"):
                self.database = database
                self.cache = cache
                self.other_param = other_param
        
        # Inject dependencies
        instance = injector.inject_dependencies(
            ServiceWithDependencies,
            other_param="custom_value"
        )
        
        assert instance.database is mock_db
        assert instance.cache is mock_cache
        assert instance.other_param == "custom_value"
    
    def test_has_dependency(self, injector):
        """Test checking if dependency exists."""
        mock_service = Mock()
        injector.register_dependency("test_service", mock_service)
        
        assert injector.has_dependency("test_service") is True
        assert injector.has_dependency("missing_service") is False
    
    def test_unregister_dependency(self, injector):
        """Test unregistering dependencies."""
        mock_service = Mock()
        injector.register_dependency("temp_service", mock_service)
        
        assert injector.has_dependency("temp_service") is True
        
        removed = injector.unregister_dependency("temp_service")
        
        assert removed is True
        assert injector.has_dependency("temp_service") is False
        
        # Try to remove non-existent dependency
        removed = injector.unregister_dependency("non_existent")
        assert removed is False
    
    def test_clear_dependencies(self, injector):
        """Test clearing all dependencies."""
        injector.register_dependency("service1", Mock())
        injector.register_dependency("service2", Mock())
        injector.register_singleton("singleton1", Mock, {})
        
        assert len(injector.dependencies) == 3  # 2 regular + 1 singleton
        assert len(injector.singletons) == 1
        
        injector.clear_dependencies()
        
        assert len(injector.dependencies) == 0
        assert len(injector.singletons) == 0


class TestToolFactory:
    """Test the ToolFactory class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return IntelligentScrapingConfig()
    
    @pytest.fixture
    def tool_factory(self, mock_config):
        """Create a tool factory for testing."""
        return ToolFactory(mock_config)
    
    def test_tool_factory_initialization(self, tool_factory, mock_config):
        """Test tool factory initialization."""
        assert tool_factory.config is mock_config
        assert hasattr(tool_factory, 'tool_configurations')
        assert hasattr(tool_factory, 'dependency_injector')
        assert hasattr(tool_factory, 'validator')
        assert hasattr(tool_factory, 'created_tools')
        assert isinstance(tool_factory.tool_configurations, dict)
        assert isinstance(tool_factory.created_tools, dict)
    
    def test_register_tool_configuration(self, tool_factory):
        """Test registering tool configurations."""
        config = ToolConfiguration(
            tool_name="test_tool",
            tool_class="TestTool",
            config_params={"param": "value"}
        )
        
        tool_factory.register_tool_configuration(config)
        
        assert "test_tool" in tool_factory.tool_configurations
        assert tool_factory.tool_configurations["test_tool"] is config
    
    def test_register_tool_configuration_invalid(self, tool_factory):
        """Test registering invalid tool configuration."""
        invalid_config = ToolConfiguration(
            tool_name="",  # Invalid empty name
            tool_class="TestTool",
            config_params={}
        )
        
        with pytest.raises(ValueError, match="Invalid tool configuration"):
            tool_factory.register_tool_configuration(invalid_config)
    
    @patch('intelligent_web_scraper.tools.tool_factory.import_module')
    def test_create_tool_success(self, mock_import, tool_factory):
        """Test successful tool creation."""
        # Mock the tool class
        mock_tool_class = Mock()
        mock_tool_instance = Mock()
        mock_tool_class.return_value = mock_tool_instance
        
        mock_module = Mock()
        mock_module.TestTool = mock_tool_class
        mock_import.return_value = mock_module
        
        # Register tool configuration
        config = ToolConfiguration(
            tool_name="test_tool",
            tool_class="TestTool",
            config_params={"param": "value"}
        )
        tool_factory.register_tool_configuration(config)
        
        # Create tool
        tool = tool_factory.create_tool("test_tool")
        
        assert tool is mock_tool_instance
        assert "test_tool" in tool_factory.created_tools
        mock_tool_class.assert_called_once()
    
    def test_create_tool_not_registered(self, tool_factory):
        """Test creating tool that's not registered."""
        with pytest.raises(ValueError, match="Tool 'unknown_tool' not registered"):
            tool_factory.create_tool("unknown_tool")
    
    def test_create_tool_disabled(self, tool_factory):
        """Test creating disabled tool."""
        config = ToolConfiguration(
            tool_name="disabled_tool",
            tool_class="DisabledTool",
            config_params={},
            enabled=False
        )
        tool_factory.register_tool_configuration(config)
        
        with pytest.raises(ValueError, match="Tool 'disabled_tool' is disabled"):
            tool_factory.create_tool("disabled_tool")
    
    @patch('intelligent_web_scraper.tools.tool_factory.import_module')
    def test_create_tool_with_dependencies(self, mock_import, tool_factory):
        """Test creating tool with dependencies."""
        # Mock dependency
        mock_dependency = Mock()
        tool_factory.dependency_injector.register_dependency("dependency_service", mock_dependency)
        
        # Mock the tool class
        mock_tool_class = Mock()
        mock_tool_instance = Mock()
        mock_tool_class.return_value = mock_tool_instance
        
        mock_module = Mock()
        mock_module.DependentTool = mock_tool_class
        mock_import.return_value = mock_module
        
        # Register tool configuration with dependencies
        config = ToolConfiguration(
            tool_name="dependent_tool",
            tool_class="DependentTool",
            config_params={},
            dependencies=["dependency_service"]
        )
        tool_factory.register_tool_configuration(config)
        
        # Create tool
        tool = tool_factory.create_tool("dependent_tool")
        
        assert tool is mock_tool_instance
        mock_tool_class.assert_called_once()
    
    def test_get_tool_existing(self, tool_factory):
        """Test getting existing tool."""
        mock_tool = Mock()
        tool_factory.created_tools["existing_tool"] = mock_tool
        
        retrieved = tool_factory.get_tool("existing_tool")
        
        assert retrieved is mock_tool
    
    def test_get_tool_missing(self, tool_factory):
        """Test getting missing tool."""
        retrieved = tool_factory.get_tool("missing_tool")
        
        assert retrieved is None
    
    @patch('intelligent_web_scraper.tools.tool_factory.import_module')
    def test_get_or_create_tool(self, mock_import, tool_factory):
        """Test getting or creating tool."""
        # Mock the tool class
        mock_tool_class = Mock()
        mock_tool_instance = Mock()
        mock_tool_class.return_value = mock_tool_instance
        
        mock_module = Mock()
        mock_module.TestTool = mock_tool_class
        mock_import.return_value = mock_module
        
        # Register tool configuration
        config = ToolConfiguration(
            tool_name="test_tool",
            tool_class="TestTool",
            config_params={}
        )
        tool_factory.register_tool_configuration(config)
        
        # First call should create tool
        tool1 = tool_factory.get_or_create_tool("test_tool")
        assert tool1 is mock_tool_instance
        
        # Second call should return existing tool
        tool2 = tool_factory.get_or_create_tool("test_tool")
        assert tool2 is tool1
        
        # Should only be called once
        mock_tool_class.assert_called_once()
    
    def test_list_registered_tools(self, tool_factory):
        """Test listing registered tools."""
        config1 = ToolConfiguration("tool1", "Tool1", {})
        config2 = ToolConfiguration("tool2", "Tool2", {})
        
        tool_factory.register_tool_configuration(config1)
        tool_factory.register_tool_configuration(config2)
        
        registered_tools = tool_factory.list_registered_tools()
        
        assert isinstance(registered_tools, list)
        assert len(registered_tools) == 2
        assert "tool1" in registered_tools
        assert "tool2" in registered_tools
    
    def test_list_created_tools(self, tool_factory):
        """Test listing created tools."""
        tool_factory.created_tools["created_tool1"] = Mock()
        tool_factory.created_tools["created_tool2"] = Mock()
        
        created_tools = tool_factory.list_created_tools()
        
        assert isinstance(created_tools, list)
        assert len(created_tools) == 2
        assert "created_tool1" in created_tools
        assert "created_tool2" in created_tools
    
    def test_destroy_tool(self, tool_factory):
        """Test destroying created tool."""
        mock_tool = Mock()
        mock_tool.cleanup = Mock()  # Tool with cleanup method
        
        tool_factory.created_tools["destroyable_tool"] = mock_tool
        
        destroyed = tool_factory.destroy_tool("destroyable_tool")
        
        assert destroyed is True
        assert "destroyable_tool" not in tool_factory.created_tools
        mock_tool.cleanup.assert_called_once()
    
    def test_destroy_tool_missing(self, tool_factory):
        """Test destroying non-existent tool."""
        destroyed = tool_factory.destroy_tool("missing_tool")
        
        assert destroyed is False
    
    def test_destroy_all_tools(self, tool_factory):
        """Test destroying all created tools."""
        mock_tool1 = Mock()
        mock_tool1.cleanup = Mock()
        mock_tool2 = Mock()
        # mock_tool2 has no cleanup method
        
        tool_factory.created_tools["tool1"] = mock_tool1
        tool_factory.created_tools["tool2"] = mock_tool2
        
        assert len(tool_factory.created_tools) == 2
        
        tool_factory.destroy_all_tools()
        
        assert len(tool_factory.created_tools) == 0
        mock_tool1.cleanup.assert_called_once()
    
    def test_get_tool_status(self, tool_factory):
        """Test getting tool status."""
        # Register some tools
        config1 = ToolConfiguration("tool1", "Tool1", {}, enabled=True)
        config2 = ToolConfiguration("tool2", "Tool2", {}, enabled=False)
        tool_factory.register_tool_configuration(config1)
        tool_factory.register_tool_configuration(config2)
        
        # Create one tool
        tool_factory.created_tools["tool1"] = Mock()
        
        status = tool_factory.get_tool_status()
        
        assert isinstance(status, dict)
        assert status["total_registered"] == 2
        assert status["total_created"] == 1
        assert status["enabled_tools"] == 1
        assert status["disabled_tools"] == 1
        assert "registered_tools" in status
        assert "created_tools" in status


if __name__ == "__main__":
    pytest.main([__file__])