#!/usr/bin/env python3
"""
Health check script for Intelligent Web Scraper.

This script performs comprehensive health checks to validate that the
application is running correctly and all dependencies are available.
"""

import sys
import os
import asyncio
import json
import time
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from intelligent_web_scraper.config import IntelligentScrapingConfig
    from intelligent_web_scraper.agents.orchestrator import IntelligentScrapingOrchestrator
    from intelligent_web_scraper import validate_ecosystem_compatibility
except ImportError as e:
    print(f"CRITICAL: Failed to import application modules: {e}")
    sys.exit(1)


class HealthChecker:
    """Comprehensive health checker for the Intelligent Web Scraper."""
    
    def __init__(self, verbose: bool = False, timeout: int = 30):
        self.verbose = verbose
        self.timeout = timeout
        self.results: Dict[str, Any] = {}
        self.start_time = time.time()
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with timestamp."""
        if self.verbose or level in ["ERROR", "CRITICAL"]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        self.log("Checking Python version...")
        
        try:
            version = sys.version_info
            if version.major != 3 or version.minor < 11:
                self.log(f"Python {version.major}.{version.minor} detected, but 3.11+ required", "ERROR")
                return False
            
            self.log(f"Python {version.major}.{version.minor}.{version.micro} - OK")
            return True
            
        except Exception as e:
            self.log(f"Failed to check Python version: {e}", "ERROR")
            return False
    
    def check_environment_variables(self) -> bool:
        """Check required environment variables."""
        self.log("Checking environment variables...")
        
        try:
            required_vars = []  # No strictly required vars, but we can warn
            recommended_vars = ["OPENAI_API_KEY"]
            
            missing_required = []
            missing_recommended = []
            
            for var in required_vars:
                if not os.getenv(var):
                    missing_required.append(var)
            
            for var in recommended_vars:
                if not os.getenv(var):
                    missing_recommended.append(var)
            
            if missing_required:
                self.log(f"Missing required environment variables: {missing_required}", "ERROR")
                return False
            
            if missing_recommended:
                self.log(f"Missing recommended environment variables: {missing_recommended}", "WARNING")
            
            self.log("Environment variables - OK")
            return True
            
        except Exception as e:
            self.log(f"Failed to check environment variables: {e}", "ERROR")
            return False
    
    def check_configuration_loading(self) -> bool:
        """Check configuration loading."""
        self.log("Checking configuration loading...")
        
        try:
            config = IntelligentScrapingConfig.from_env()
            
            # Validate configuration values
            if not (0 <= config.default_quality_threshold <= 100):
                self.log("Invalid quality threshold in configuration", "ERROR")
                return False
            
            if config.max_concurrent_requests <= 0:
                self.log("Invalid max concurrent requests in configuration", "ERROR")
                return False
            
            self.log("Configuration loading - OK")
            return True
            
        except Exception as e:
            self.log(f"Failed to load configuration: {e}", "ERROR")
            return False
    
    def check_module_imports(self) -> bool:
        """Check that all required modules can be imported."""
        self.log("Checking module imports...")
        
        required_modules = [
            "atomic_agents",
            "atomic_agents.agents.base_agent",
            "atomic_agents.lib.base.base_io_schema",
            "atomic_agents.lib.components.system_prompt_generator",
            "requests",
            "bs4",  # beautifulsoup4 imports as bs4
            "rich",
            "aiohttp",
            "pydantic",
        ]
        
        try:
            for module in required_modules:
                try:
                    __import__(module)
                    self.log(f"  {module} - OK")
                except ImportError as e:
                    self.log(f"  {module} - FAILED: {e}", "ERROR")
                    return False
            
            self.log("Module imports - OK")
            return True
            
        except Exception as e:
            self.log(f"Failed to check module imports: {e}", "ERROR")
            return False
    
    def check_ecosystem_compatibility(self) -> bool:
        """Check atomic-agents ecosystem compatibility."""
        self.log("Checking ecosystem compatibility...")
        
        try:
            compatibility = validate_ecosystem_compatibility()
            
            failed_checks = [k for k, v in compatibility.items() if not v]
            
            if failed_checks:
                self.log(f"Failed compatibility checks: {failed_checks}", "ERROR")
                return False
            
            self.log("Ecosystem compatibility - OK")
            return True
            
        except Exception as e:
            self.log(f"Failed to check ecosystem compatibility: {e}", "ERROR")
            return False
    
    def check_orchestrator_initialization(self) -> bool:
        """Check orchestrator agent initialization."""
        self.log("Checking orchestrator initialization...")
        
        try:
            config = IntelligentScrapingConfig.from_env()
            orchestrator = IntelligentScrapingOrchestrator(config=config)
            
            # Verify required attributes
            if not hasattr(orchestrator, 'input_schema'):
                self.log("Orchestrator missing input_schema", "ERROR")
                return False
            
            if not hasattr(orchestrator, 'output_schema'):
                self.log("Orchestrator missing output_schema", "ERROR")
                return False
            
            if not hasattr(orchestrator, 'system_prompt_generator'):
                self.log("Orchestrator missing system_prompt_generator", "ERROR")
                return False
            
            self.log("Orchestrator initialization - OK")
            return True
            
        except Exception as e:
            self.log(f"Failed to initialize orchestrator: {e}", "ERROR")
            return False
    
    def check_file_system_permissions(self) -> bool:
        """Check file system permissions."""
        self.log("Checking file system permissions...")
        
        try:
            config = IntelligentScrapingConfig.from_env()
            results_dir = Path(config.results_directory)
            
            # Check if results directory exists or can be created
            if not results_dir.exists():
                try:
                    results_dir.mkdir(parents=True, exist_ok=True)
                    self.log(f"Created results directory: {results_dir}")
                except PermissionError:
                    self.log(f"Cannot create results directory: {results_dir}", "ERROR")
                    return False
            
            # Test write permissions
            test_file = results_dir / "health_check_test.txt"
            try:
                test_file.write_text("health check test")
                test_file.unlink()
                self.log("Write permissions - OK")
            except PermissionError:
                self.log(f"No write permission to results directory: {results_dir}", "ERROR")
                return False
            
            self.log("File system permissions - OK")
            return True
            
        except Exception as e:
            self.log(f"Failed to check file system permissions: {e}", "ERROR")
            return False
    
    def check_network_connectivity(self) -> bool:
        """Check network connectivity."""
        self.log("Checking network connectivity...")
        
        try:
            import requests
            
            # Test basic HTTP connectivity
            test_urls = [
                "https://httpbin.org/status/200",
                "https://api.openai.com/v1/models" if os.getenv("OPENAI_API_KEY") else None
            ]
            
            for url in test_urls:
                if url is None:
                    continue
                
                try:
                    headers = {}
                    if "api.openai.com" in url and os.getenv("OPENAI_API_KEY"):
                        headers["Authorization"] = f"Bearer {os.getenv('OPENAI_API_KEY')}"
                    
                    response = requests.get(url, headers=headers, timeout=10)
                    if response.status_code in [200, 401]:  # 401 is OK for API key test
                        self.log(f"  {url} - OK")
                    else:
                        self.log(f"  {url} - HTTP {response.status_code}", "WARNING")
                        
                except requests.RequestException as e:
                    self.log(f"  {url} - FAILED: {e}", "WARNING")
            
            self.log("Network connectivity - OK")
            return True
            
        except Exception as e:
            self.log(f"Failed to check network connectivity: {e}", "ERROR")
            return False
    
    async def check_async_functionality(self) -> bool:
        """Check async functionality."""
        self.log("Checking async functionality...")
        
        try:
            # Test basic async operation
            await asyncio.sleep(0.1)
            
            # Test async HTTP client
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("https://httpbin.org/status/200", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        self.log("Async HTTP client - OK")
                    else:
                        self.log(f"Async HTTP client - HTTP {response.status}", "WARNING")
            
            self.log("Async functionality - OK")
            return True
            
        except Exception as e:
            self.log(f"Failed to check async functionality: {e}", "ERROR")
            return False
    
    def check_memory_usage(self) -> bool:
        """Check memory usage."""
        self.log("Checking memory usage...")
        
        try:
            import psutil
            
            # Get current process memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            self.log(f"Current memory usage: {memory_mb:.1f} MB")
            
            # Get system memory
            system_memory = psutil.virtual_memory()
            available_mb = system_memory.available / 1024 / 1024
            
            self.log(f"Available system memory: {available_mb:.1f} MB")
            
            if available_mb < 512:
                self.log("Low system memory detected", "WARNING")
            
            self.log("Memory usage - OK")
            return True
            
        except ImportError:
            self.log("psutil not available, skipping memory check", "WARNING")
            return True
        except Exception as e:
            self.log(f"Failed to check memory usage: {e}", "ERROR")
            return False
    
    async def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        self.log("Starting comprehensive health check...")
        
        checks = [
            ("python_version", self.check_python_version),
            ("environment_variables", self.check_environment_variables),
            ("configuration_loading", self.check_configuration_loading),
            ("module_imports", self.check_module_imports),
            ("ecosystem_compatibility", self.check_ecosystem_compatibility),
            ("orchestrator_initialization", self.check_orchestrator_initialization),
            ("file_system_permissions", self.check_file_system_permissions),
            ("network_connectivity", self.check_network_connectivity),
            ("async_functionality", self.check_async_functionality),
            ("memory_usage", self.check_memory_usage),
        ]
        
        results = {}
        passed = 0
        failed = 0
        
        for check_name, check_func in checks:
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                results[check_name] = {
                    "status": "PASS" if result else "FAIL",
                    "passed": result
                }
                
                if result:
                    passed += 1
                else:
                    failed += 1
                    
            except Exception as e:
                self.log(f"Exception in {check_name}: {e}", "ERROR")
                results[check_name] = {
                    "status": "ERROR",
                    "passed": False,
                    "error": str(e)
                }
                failed += 1
        
        # Calculate overall health
        total_checks = len(checks)
        health_percentage = (passed / total_checks) * 100
        
        overall_status = "HEALTHY" if failed == 0 else "DEGRADED" if passed > failed else "UNHEALTHY"
        
        end_time = time.time()
        duration = end_time - self.start_time
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "health_percentage": health_percentage,
            "total_checks": total_checks,
            "passed_checks": passed,
            "failed_checks": failed,
            "duration_seconds": round(duration, 2),
            "checks": results
        }
        
        self.log(f"Health check completed: {overall_status} ({health_percentage:.1f}%)")
        self.log(f"Passed: {passed}, Failed: {failed}, Duration: {duration:.2f}s")
        
        return summary


async def main():
    """Main entry point for health check."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent Web Scraper Health Check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--timeout", "-t", type=int, default=30, help="Timeout in seconds")
    parser.add_argument("--json", "-j", action="store_true", help="Output JSON format")
    parser.add_argument("--exit-code", "-e", action="store_true", help="Exit with non-zero code on failure")
    
    args = parser.parse_args()
    
    # Create health checker
    checker = HealthChecker(verbose=args.verbose, timeout=args.timeout)
    
    try:
        # Run health check with timeout
        results = await asyncio.wait_for(
            checker.run_comprehensive_health_check(),
            timeout=args.timeout
        )
        
        # Output results
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\nHealth Check Summary:")
            print(f"Status: {results['overall_status']}")
            print(f"Health: {results['health_percentage']:.1f}%")
            print(f"Passed: {results['passed_checks']}/{results['total_checks']}")
            print(f"Duration: {results['duration_seconds']}s")
            
            if results['failed_checks'] > 0:
                print(f"\nFailed Checks:")
                for check_name, check_result in results['checks'].items():
                    if not check_result['passed']:
                        status = check_result['status']
                        error = check_result.get('error', '')
                        print(f"  - {check_name}: {status} {error}")
        
        # Exit with appropriate code
        if args.exit_code and results['overall_status'] != "HEALTHY":
            sys.exit(1)
        else:
            sys.exit(0)
            
    except asyncio.TimeoutError:
        print(f"Health check timed out after {args.timeout} seconds")
        sys.exit(1)
    except Exception as e:
        print(f"Health check failed with exception: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())