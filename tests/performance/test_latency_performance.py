"""
Performance tests for latency measurements on object detection service.
Tests P95 latency requirements with 1000 requests using 720p images.
"""

import pytest
import time
import statistics
import asyncio
from typing import List
import httpx
from tests.conftest import PERFORMANCE_TEST_REQUESTS, EXPECTED_P95_LATENCY_MS


@pytest.mark.performance
class TestLatencyPerformance:
    """Test latency performance metrics for the detection service."""

    @pytest.mark.asyncio
    async def test_latency_p95_performance(self, test_image_720p, async_client: httpx.AsyncClient):
        """
        Test P95 latency with 1000 requests using 720p images.
        Validates that P95 latency is ≤ 300ms on CPU.
        """
        latencies: List[float] = []
        requests_count = 1000  # Full test with 1000 requests as specified
        
        # Warm up the model with a few requests
        warmup_requests = 5
        for _ in range(warmup_requests):
            test_image_720p.seek(0)
            files = {"file": ("test_image.jpg", test_image_720p, "image/jpeg")}
            await async_client.post("/detect", files=files)
        
        print(f"Starting latency test with {requests_count} requests...")
        
        # Perform the actual test requests
        for i in range(requests_count):
            test_image_720p.seek(0)
            files = {"file": ("test_image.jpg", test_image_720p, "image/jpeg")}
            
            start_time = time.perf_counter()
            response = await async_client.post("/detect", files=files)
            end_time = time.perf_counter()
            
            # Ensure request was successful
            assert response.status_code == 200, f"Request {i+1} failed with status {response.status_code}"
            
            # Calculate latency in milliseconds
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Progress indicator every 100 requests
            if (i + 1) % 100 == 0:
                current_p95 = self._calculate_percentile(latencies, 95)
                print(f"Completed {i+1}/{requests_count} requests. Current P95: {current_p95:.2f}ms")
        
        # Calculate statistics
        p50_latency = self._calculate_percentile(latencies, 50)
        p95_latency = self._calculate_percentile(latencies, 95)
        p99_latency = self._calculate_percentile(latencies, 99)
        mean_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        # Print comprehensive results
        print(f"\n=== Latency Performance Results ===")
        print(f"Total requests: {requests_count}")
        print(f"Min latency: {min_latency:.2f}ms")
        print(f"Mean latency: {mean_latency:.2f}ms")
        print(f"P50 latency: {p50_latency:.2f}ms")
        print(f"P95 latency: {p95_latency:.2f}ms")
        print(f"P99 latency: {p99_latency:.2f}ms")
        print(f"Max latency: {max_latency:.2f}ms")
        print(f"Expected P95 threshold: {EXPECTED_P95_LATENCY_MS}ms")
        
        # Assert P95 latency meets requirements
        assert p95_latency <= EXPECTED_P95_LATENCY_MS, (
            f"P95 latency {p95_latency:.2f}ms exceeds threshold of {EXPECTED_P95_LATENCY_MS}ms"
        )
    
    @pytest.mark.asyncio
    async def test_concurrent_latency_performance(self, test_image_720p, async_client: httpx.AsyncClient):
        """
        Test latency under concurrent load to simulate real-world scenarios.
        Uses 10 concurrent connections with 10 requests each.
        """
        concurrent_connections = 10
        requests_per_connection = 10
        
        async def single_connection_test():
            """Execute requests in a single connection and return latencies."""
            connection_latencies = []
            
            for _ in range(requests_per_connection):
                test_image_720p.seek(0)
                files = {"file": ("test_image.jpg", test_image_720p, "image/jpeg")}
                
                start_time = time.perf_counter()
                response = await async_client.post("/detect", files=files)
                end_time = time.perf_counter()
                
                assert response.status_code == 200
                latency_ms = (end_time - start_time) * 1000
                connection_latencies.append(latency_ms)
            
            return connection_latencies
        
        print(f"Starting concurrent latency test with {concurrent_connections} connections...")
        
        # Run concurrent connections
        tasks = [single_connection_test() for _ in range(concurrent_connections)]
        results = await asyncio.gather(*tasks)
        
        # Flatten all latencies
        all_latencies = [latency for connection_latencies in results for latency in connection_latencies]
        
        # Calculate statistics
        p95_latency = self._calculate_percentile(all_latencies, 95)
        mean_latency = statistics.mean(all_latencies)
        
        print(f"\n=== Concurrent Latency Results ===")
        print(f"Concurrent connections: {concurrent_connections}")
        print(f"Requests per connection: {requests_per_connection}")
        print(f"Total requests: {len(all_latencies)}")
        print(f"Mean latency: {mean_latency:.2f}ms")
        print(f"P95 latency: {p95_latency:.2f}ms")
        
        # Under concurrent load, we allow slightly higher P95 latency
        concurrent_threshold = EXPECTED_P95_LATENCY_MS * 1.5  # 50% tolerance for concurrent load
        assert p95_latency <= concurrent_threshold, (
            f"Concurrent P95 latency {p95_latency:.2f}ms exceeds threshold of {concurrent_threshold}ms"
        )
    
    def _calculate_percentile(self, data: List[float], percentile: int) -> float:
        """Calculate the specified percentile from a list of values."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index
            
            return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight