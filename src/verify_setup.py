#!/usr/bin/env python3
"""
verify_setup.py - Verifies the setup for the Climate Stance Detection project.
"""

import sys
import os
import warnings
import platform
from pathlib import Path
import pandas as pd
import numpy as np
from importlib.metadata import version, PackageNotFoundError

# Get project root once, at module level
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def check_python_version():
    """Check if Python version meets requirements."""
    required_version = (3, 12, 4)
    current_version = sys.version_info[:3]
    
    print(f"Checking Python version...")
    print(f"Current version: {'.'.join(map(str, current_version))}")
    print(f"Required version: {'.'.join(map(str, required_version))}")
    
    if current_version >= required_version:
        print("✓ Python version OK")
        return True
    else:
        print("✗ Python version too old")
        return False

def check_memory():
    """Check if system has enough memory."""
    import psutil
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024 ** 3)
    available_gb = memory.available / (1024 ** 3)
    
    print(f"\nChecking system memory...")
    print(f"Total memory: {total_gb:.1f} GB")
    print(f"Available memory: {available_gb:.1f} GB")
    
    if total_gb >= 8:
        print("✓ System has enough total memory")
        sufficient_memory = True
    else:
        print("✗ System needs at least 8GB RAM")
        sufficient_memory = False
        
    if available_gb >= 4:
        print("✓ Sufficient available memory")
        return sufficient_memory and True
    else:
        print("⚠ Warning: Less than 4GB available memory")
        return sufficient_memory and False

def check_dependencies():
    """Check if all required packages are installed with correct versions."""
    requirements_file = PROJECT_ROOT / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"\n✗ requirements.txt not found at {requirements_file}")
        return False
    
    print("\nChecking package dependencies...")
    with open(requirements_file) as f:
        requirements = f.read().splitlines()
    
    all_satisfied = True
    for requirement in requirements:
        if requirement.strip() and not requirement.startswith('#'):
            try:
                pkg_name = requirement.split('~=')[0].strip()
                required_version = requirement.split('~=')[1].strip()
                
                try:
                    installed_version = version(pkg_name)
                    print(f"Checking {pkg_name}: required={required_version}, installed={installed_version}")
                    
                    if installed_version >= required_version:
                        print(f"✓ {pkg_name} OK")
                    else:
                        print(f"✗ {pkg_name} version mismatch")
                        all_satisfied = False
                        
                except PackageNotFoundError:
                    print(f"✗ {pkg_name} not installed")
                    all_satisfied = False
                    
            except Exception as e:
                print(f"⚠ Error checking {requirement}: {str(e)}")
                all_satisfied = False
    
    return all_satisfied

def check_directory_structure():
    """Check if the required directory structure exists."""
    print("\nChecking directory structure...")
    required_dirs = [
        PROJECT_ROOT / 'data/raw',
        PROJECT_ROOT / 'data/processed',
        PROJECT_ROOT / 'notebooks/01_data_exploration.ipynb',
        PROJECT_ROOT / 'notebooks/02_preprocessing.ipynb',
        PROJECT_ROOT / 'notebooks/03_model_development.ipynb',
        PROJECT_ROOT / 'notebooks/04_cross_lingual_analysis.ipynb',
        PROJECT_ROOT / 'src/collect_data.py',
        PROJECT_ROOT / 'src/verify_setup.py',
    ]
    
    all_exist = True
    for directory in required_dirs:
        if directory.exists():
            print(f"✓ {directory.relative_to(PROJECT_ROOT)} exists")
        else:
            print(f"✗ {directory.relative_to(PROJECT_ROOT)} missing")
            all_exist = False
            try:
                directory.mkdir(parents=True)
                print(f"  Created {directory.relative_to(PROJECT_ROOT)}")
            except Exception as e:
                print(f"  Error creating {directory.relative_to(PROJECT_ROOT)}: {str(e)}")
    
    return all_exist

def check_api_config():
    """Check if Reddit API configuration exists."""
    print("\nChecking API configuration...")
    config_file = PROJECT_ROOT / 'config.ini'
    
    if not config_file.exists():
        print("✗ config.ini missing")
        print("Please create config.ini with Reddit API credentials")
        return False
    
    import configparser
    config = configparser.ConfigParser()
    try:
        config.read(config_file)
        required_fields = ['client_id', 'client_secret', 'user_agent']
        
        if 'Reddit' not in config:
            print("✗ [Reddit] section missing in config.ini")
            return False
            
        for field in required_fields:
            if field not in config['Reddit']:
                print(f"✗ {field} missing in config.ini")
                return False
            if not config['Reddit'][field]:
                print(f"✗ {field} is empty in config.ini")
                return False
        
        print("✓ API configuration OK")
        return True
        
    except Exception as e:
        print(f"✗ Error reading config.ini: {str(e)}")
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Climate Change Stance Detection - Setup Verification")
    print(f"Project Root: {PROJECT_ROOT}")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_memory(),
        check_dependencies(),
        check_directory_structure(),
        check_api_config()
    ]
    
    print("\n" + "=" * 60)
    if all(checks):
        print("\n✅ All checks passed! Setup is complete.")
        sys.exit(0)
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
