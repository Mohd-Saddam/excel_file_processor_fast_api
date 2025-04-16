"""
Pytest configuration file.

This file is automatically loaded by pytest and provides global test configuration.
It ensures the project's source directory is added to the Python path so that
modules can be imported properly during test execution.
"""
import os
import sys

# Add the current directory to the Python path
# This ensures imports work correctly when running tests
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))