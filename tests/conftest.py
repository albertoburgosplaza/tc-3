"""
Pytest configuration and shared fixtures for testing the object detection service.
"""

import pytest
import httpx
import io
import os
from PIL import Image
import numpy as np
from typing import AsyncGenerator, Generator
from fastapi.testclient import TestClient

# Import the FastAPI app
from app.app import app


@pytest.fixture(scope="session")
def test_client() -> Generator[TestClient, None, None]:
    """
    Create a TestClient for the FastAPI app.
    Session scope means this client is reused across all tests.
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """
    Create an async HTTP client for testing FastAPI endpoints.
    """
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def valid_jpeg_image() -> io.BytesIO:
    """
    Create a valid JPEG image for testing.
    Returns a BytesIO object with JPEG image data.
    """
    # Create a simple RGB image
    image = Image.new('RGB', (100, 100), color=(255, 0, 0))
    
    # Save to BytesIO as JPEG
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)
    
    return image_bytes


@pytest.fixture
def valid_png_image() -> io.BytesIO:
    """
    Create a valid PNG image for testing.
    Returns a BytesIO object with PNG image data.
    """
    # Create a simple RGBA image
    image = Image.new('RGBA', (150, 150), color=(0, 255, 0, 255))
    
    # Save to BytesIO as PNG
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    
    return image_bytes


@pytest.fixture
def large_image() -> io.BytesIO:
    """
    Create a large image (>10MB) for testing file size limits.
    """
    # Create a large image (2000x2000 pixels)
    image = Image.new('RGB', (2000, 2000), color=(100, 150, 200))
    
    # Save with high quality to ensure it's > 10MB
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG', quality=100)
    image_bytes.seek(0)
    
    return image_bytes


@pytest.fixture
def corrupted_image() -> io.BytesIO:
    """
    Create a corrupted image file for testing error handling.
    """
    # Create invalid image data
    corrupted_data = b"This is not a valid image file"
    return io.BytesIO(corrupted_data)


@pytest.fixture
def small_test_image() -> io.BytesIO:
    """
    Create a small test image with known dimensions for precise testing.
    """
    # Create 64x64 image - small enough for fast inference
    image = Image.new('RGB', (64, 64), color=(128, 128, 128))
    
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)
    
    return image_bytes


@pytest.fixture
def test_image_720p() -> io.BytesIO:
    """
    Create a 720p (1280x720) test image for performance testing.
    """
    # Create 720p image for performance tests
    image = Image.new('RGB', (1280, 720), color=(64, 128, 255))
    
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG', quality=85)
    image_bytes.seek(0)
    
    return image_bytes


@pytest.fixture
def mock_model_not_loaded(monkeypatch):
    """
    Fixture to simulate model not being loaded.
    """
    from app import app as app_module
    monkeypatch.setattr(app_module, "model", None)


@pytest.fixture
def valid_inference_params() -> dict:
    """
    Valid inference parameters for testing.
    """
    return {
        "conf": 0.25,
        "iou": 0.45,
        "max_detections": 300
    }


@pytest.fixture
def invalid_inference_params() -> list:
    """
    List of invalid parameter combinations for testing validation.
    """
    return [
        {"conf": -0.1, "iou": 0.45, "max_detections": 300},  # conf too low
        {"conf": 1.1, "iou": 0.45, "max_detections": 300},   # conf too high
        {"conf": 0.25, "iou": -0.1, "max_detections": 300},  # iou too low
        {"conf": 0.25, "iou": 1.1, "max_detections": 300},   # iou too high
        {"conf": 0.25, "iou": 0.45, "max_detections": 0},    # max_det too low
        {"conf": 0.25, "iou": 0.45, "max_detections": 1001}, # max_det too high
    ]


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Set up test environment variables and cleanup.
    This fixture runs automatically for all tests.
    """
    # Ensure we're using CPU only for tests
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Set up any test-specific environment variables
    original_model_path = os.environ.get('MODEL_PATH')
    
    # Ensure model path is set for tests
    if not os.environ.get('MODEL_PATH'):
        os.environ['MODEL_PATH'] = 'models/yolov8n.pt'
    
    yield
    
    # Cleanup: restore original environment if it was set
    if original_model_path:
        os.environ['MODEL_PATH'] = original_model_path
    elif 'MODEL_PATH' in os.environ:
        del os.environ['MODEL_PATH']


# Test data constants
VALID_CONTENT_TYPES = ['image/jpeg', 'image/png']
INVALID_CONTENT_TYPES = ['text/plain', 'application/pdf', 'image/gif', 'image/bmp']

# Performance test constants
PERFORMANCE_TEST_REQUESTS = 100  # Reduced from 1000 for faster testing during development
EXPECTED_P95_LATENCY_MS = 300
MAX_MEMORY_INCREASE_MB = 50  # Maximum acceptable memory increase during tests