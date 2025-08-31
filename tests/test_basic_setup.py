"""
Basic test to verify the testing setup is working correctly.
This test should run without requiring the full application setup.
"""

import pytest
import sys
import os

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_pytest_is_working():
    """Basic test to verify pytest is functioning."""
    assert True


def test_python_version():
    """Verify we're running on Python 3.11+ as expected."""
    assert sys.version_info >= (3, 11)


def test_imports_work():
    """Test that basic imports work."""
    import io
    import pytest
    assert hasattr(pytest, 'fixture')


@pytest.mark.unit
def test_unit_marker():
    """Test that unit test marker works."""
    assert True


@pytest.mark.integration  
def test_integration_marker():
    """Test that integration test marker works."""
    assert True


@pytest.mark.performance
def test_performance_marker():
    """Test that performance test marker works."""
    assert True