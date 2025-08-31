#!/bin/bash

# Test runner script for the object detection service
# This script builds the test container and runs all tests

set -e

echo "🧪 Starting test suite for Object Detection Service"
echo "================================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed or not in PATH"
    exit 1
fi

echo "🏗️  Building test container..."
docker-compose -f docker-compose.test.yml build object-detector-test

echo "📦 Creating test images..."
# Run the test image creation script inside the container first
docker-compose -f docker-compose.test.yml run --rm object-detector-test python tests/data/create_test_images.py

echo "🧪 Running unit tests..."
docker-compose -f docker-compose.test.yml run --rm object-detector-test pytest tests/unit/ -v -m unit

echo "🔗 Running integration tests..."
docker-compose -f docker-compose.test.yml run --rm object-detector-test pytest tests/integration/ -v -m integration

echo "⚡ Running performance tests..."
docker-compose -f docker-compose.test.yml run --rm object-detector-test pytest tests/performance/ -v -m performance

echo "📊 Running comprehensive test suite with Python runner..."
docker-compose -f docker-compose.test.yml run --rm object-detector-test python run_tests_complete.py

echo "✅ All tests completed!"
echo "📋 Coverage report available in htmlcov/index.html"

# Optional: Run quick mode for development
if [[ "$1" == "--quick" ]]; then
    echo "⚡ Running quick test suite (no performance tests)..."
    docker-compose -f docker-compose.test.yml run --rm object-detector-test python run_tests_complete.py --quick
fi