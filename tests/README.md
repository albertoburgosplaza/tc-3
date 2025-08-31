# Test Suite Documentation

## Overview

This directory contains a comprehensive test suite for the Object Detection Service, including unit tests, integration tests, and performance tests with coverage reporting.

## Test Structure

```
tests/
├── conftest.py                 # Pytest configuration and shared fixtures
├── data/                      # Test data and image generation
│   └── create_test_images.py
├── unit/                      # Unit tests (fast, isolated)
│   ├── test_parameter_validation.py
│   ├── test_error_handling.py
│   └── test_detection_parsing.py
├── integration/               # Integration tests (component interaction)
│   ├── test_inference_golden.py
│   └── test_map_validation.py
├── performance/               # Performance tests (latency, memory)
│   ├── test_latency_performance.py
│   └── test_memory_leaks.py
└── README.md                  # This file
```

## Test Categories

### Unit Tests (`-m unit`)
- **Purpose**: Test individual functions and methods in isolation
- **Speed**: Fast (< 1 second each)
- **Scope**: Parameter validation, error handling, output parsing
- **Files**: `tests/unit/`

### Integration Tests (`-m integration`)
- **Purpose**: Test component interaction and end-to-end workflows
- **Speed**: Medium (1-10 seconds each)
- **Scope**: API endpoints, golden image comparisons, mAP validation
- **Files**: `tests/integration/`

### Performance Tests (`-m performance`)
- **Purpose**: Test performance characteristics (latency, memory usage)
- **Speed**: Slow (30+ seconds each)
- **Scope**: P95 latency ≤ 300ms, memory leak detection
- **Files**: `tests/performance/`

## Running Tests

### Docker (Recommended)

```bash
# Full test suite with coverage
./run_tests.sh

# Quick mode (skip performance tests)
./run_tests.sh --quick

# Individual test categories
docker-compose -f docker-compose.test.yml run --rm object-detector-test pytest tests/unit/ -v -m unit
docker-compose -f docker-compose.test.yml run --rm object-detector-test pytest tests/integration/ -v -m integration
docker-compose -f docker-compose.test.yml run --rm object-detector-test pytest tests/performance/ -v -m performance
```

### Local (Python Environment)

```bash
# Complete test suite
python run_tests_complete.py

# Quick mode (no performance tests)
python run_tests_complete.py --quick

# Coverage only (unit + integration)
python run_tests_complete.py --coverage-only

# Individual categories
pytest tests/unit/ -v -m unit
pytest tests/integration/ -v -m integration
pytest tests/performance/ -v -m performance

# With coverage
pytest tests/ --cov=app --cov-report=html --cov-report=term-missing
```

## Test Configuration

### pytest.ini
- Configured for 80% minimum coverage requirement
- Generates HTML and XML coverage reports
- Includes markers for test categorization
- Async mode enabled for FastAPI tests

### Fixtures (conftest.py)
- `test_client`: TestClient for FastAPI app
- `async_client`: Async HTTP client
- `valid_jpeg_image`: Sample JPEG image
- `test_image_720p`: 720p image for performance tests
- `valid_inference_params`: Valid API parameters
- `setup_test_environment`: Automatic test environment setup

## Coverage Requirements

The test suite enforces **80% code coverage** minimum on the `app/` directory:

```
tests/
├── Unit tests:        Cover parameter validation, error handling
├── Integration tests: Cover API endpoints, model inference
├── Performance tests: Cover sustained load scenarios
└── Total coverage:    Must exceed 80% (--cov-fail-under=80)
```

## Performance Targets

### Latency Tests
- **Target**: P95 latency ≤ 300ms on CPU
- **Test**: 1000 requests with 720p images
- **Concurrent**: 10 connections × 10 requests each

### Memory Tests  
- **Target**: Memory increase ≤ 50MB during sustained load
- **Test**: 500 requests with memory monitoring
- **Detection**: Uses psutil to track memory usage patterns

## Test Reports

After running tests, reports are generated in:

```
htmlcov/index.html          # HTML coverage report (interactive)
coverage.xml                # XML coverage report (CI/CD)
test-results/junit.xml      # JUnit test results (CI/CD)
```

## Development Workflow

1. **Write tests first** (TDD approach recommended)
2. **Run unit tests** during development (`pytest tests/unit/ -v`)
3. **Run integration tests** before commits (`pytest tests/integration/ -v`)
4. **Run full suite** before deployment (`./run_tests.sh`)
5. **Check coverage** in HTML report (`open htmlcov/index.html`)

## Debugging Failed Tests

### Unit Test Failures
```bash
# Run with detailed output
pytest tests/unit/test_parameter_validation.py::test_invalid_confidence -v --tb=long

# Run single test with prints
pytest tests/unit/test_parameter_validation.py::test_invalid_confidence -v -s
```

### Integration Test Failures  
```bash
# Check FastAPI app logs
pytest tests/integration/ -v --log-cli-level=DEBUG

# Run with test client debugging
pytest tests/integration/test_inference_golden.py -v --tb=long
```

### Performance Test Failures
```bash
# Run individual performance test with output
pytest tests/performance/test_latency_performance.py::TestLatencyPerformance::test_concurrent_latency_performance -v -s

# Memory debugging
pytest tests/performance/test_memory_leaks.py -v -s --tb=long
```

## Continuous Integration

The test suite is designed for CI/CD pipelines:

```yaml
# Example GitHub Actions/GitLab CI
- name: Run Test Suite
  run: |
    docker-compose -f docker-compose.test.yml build
    ./run_tests.sh
    
- name: Upload Coverage
  uses: codecov/codecov-action@v1
  with:
    file: ./coverage.xml
```

## Test Data

### Golden Images
- Generated by `tests/data/create_test_images.py`
- Contains known objects for mAP validation
- Automatically created during test setup

### Test Fixtures
- Controlled image sizes (64x64, 720p, 2000x2000)
- Various formats (JPEG, PNG, corrupted)
- Realistic test parameters

## Best Practices

1. **Keep unit tests fast** (< 1 second each)
2. **Use appropriate markers** (`@pytest.mark.unit`, `@pytest.mark.integration`)
3. **Mock external dependencies** in unit tests
4. **Use realistic test data** in integration tests
5. **Monitor performance trends** in performance tests
6. **Maintain high coverage** (aim for > 90%)
7. **Write descriptive test names** and docstrings
8. **Test error conditions** as well as happy paths