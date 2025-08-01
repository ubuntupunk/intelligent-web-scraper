#!/usr/bin/env python3
"""Simple test to debug the hanging issue."""

print("Starting test...")

try:
    print("Testing basic imports...")
    import os
    import threading
    import time
    from datetime import datetime
    from enum import Enum
    print("Basic imports successful")
    
    print("Testing pydantic...")
    from pydantic import BaseModel, Field
    print("Pydantic import successful")
    
    print("Testing psutil...")
    import psutil
    print("Psutil import successful")
    
    print("Testing config...")
    from intelligent_web_scraper.config import IntelligentScrapingConfig
    config = IntelligentScrapingConfig()
    print("Config creation successful")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()