"""
Memory leak detection tests for the object detection service.
Uses psutil to monitor memory usage during sustained load and detect anomalous increases.
"""

import pytest
import time
import psutil
import gc
import os
from typing import List, Tuple
import httpx
from tests.conftest import MAX_MEMORY_INCREASE_MB


@pytest.mark.performance
class TestMemoryLeaks:
    """Test memory usage patterns to detect potential memory leaks."""

    @pytest.mark.asyncio
    async def test_memory_leak_detection_sustained_load(self, test_image_720p, async_client: httpx.AsyncClient):
        """
        Test for memory leaks during sustained load.
        Monitors memory usage during 500 requests and detects anomalous increases.
        """
        # Get current process
        current_process = psutil.Process(os.getpid())
        
        # Warm up the model and garbage collection
        warmup_requests = 10
        for _ in range(warmup_requests):
            test_image_720p.seek(0)
            files = {"file": ("test_image.jpg", test_image_720p, "image/jpeg")}
            await async_client.post("/detect", files=files)
        
        # Force garbage collection before baseline measurement
        gc.collect()
        time.sleep(1)  # Allow memory to stabilize
        
        # Baseline memory measurement
        baseline_memory_mb = current_process.memory_info().rss / (1024 * 1024)
        print(f"Baseline memory usage: {baseline_memory_mb:.2f} MB")
        
        # Execute sustained load test
        sustained_requests = 500
        memory_samples: List[float] = []
        sample_interval = 50  # Sample memory every 50 requests
        
        print(f"Starting memory leak test with {sustained_requests} requests...")
        
        for i in range(sustained_requests):
            test_image_720p.seek(0)
            files = {"file": ("test_image.jpg", test_image_720p, "image/jpeg")}
            
            response = await async_client.post("/detect", files=files)
            assert response.status_code == 200, f"Request {i+1} failed"
            
            # Sample memory usage at intervals
            if (i + 1) % sample_interval == 0:
                current_memory_mb = current_process.memory_info().rss / (1024 * 1024)
                memory_samples.append(current_memory_mb)
                
                print(f"Completed {i+1}/{sustained_requests} requests. "
                      f"Memory: {current_memory_mb:.2f} MB "
                      f"(+{current_memory_mb - baseline_memory_mb:.2f} MB)")
        
        # Final memory measurement with garbage collection
        gc.collect()
        time.sleep(1)
        final_memory_mb = current_process.memory_info().rss / (1024 * 1024)
        
        memory_increase = final_memory_mb - baseline_memory_mb
        
        print(f"\n=== Memory Leak Detection Results ===")
        print(f"Baseline memory: {baseline_memory_mb:.2f} MB")
        print(f"Final memory: {final_memory_mb:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")
        print(f"Max allowed increase: {MAX_MEMORY_INCREASE_MB} MB")
        print(f"Memory samples: {len(memory_samples)}")
        
        if memory_samples:
            max_memory = max(memory_samples)
            min_memory = min(memory_samples)
            print(f"Peak memory during test: {max_memory:.2f} MB")
            print(f"Memory range: {min_memory:.2f} - {max_memory:.2f} MB")
        
        # Assert no significant memory increase
        assert memory_increase <= MAX_MEMORY_INCREASE_MB, (
            f"Memory increased by {memory_increase:.2f} MB, "
            f"exceeding threshold of {MAX_MEMORY_INCREASE_MB} MB. Possible memory leak detected."
        )
    
    @pytest.mark.asyncio
    async def test_memory_stability_batch_processing(self, test_image_720p, async_client: httpx.AsyncClient):
        """
        Test memory stability during batch processing scenarios.
        Processes multiple batches and ensures memory returns to baseline between batches.
        """
        current_process = psutil.Process(os.getpid())
        
        # Initial cleanup and baseline
        gc.collect()
        time.sleep(1)
        initial_baseline_mb = current_process.memory_info().rss / (1024 * 1024)
        
        batch_size = 100
        num_batches = 5
        batch_results: List[Tuple[float, float]] = []  # (peak_memory, post_batch_memory)
        
        print(f"Starting batch memory stability test: {num_batches} batches of {batch_size} requests each")
        print(f"Initial baseline: {initial_baseline_mb:.2f} MB")
        
        for batch_num in range(num_batches):
            # Pre-batch memory
            gc.collect()
            pre_batch_memory_mb = current_process.memory_info().rss / (1024 * 1024)
            
            print(f"\nBatch {batch_num + 1}/{num_batches} - Pre-batch memory: {pre_batch_memory_mb:.2f} MB")
            
            peak_memory_mb = pre_batch_memory_mb
            
            # Process batch
            for i in range(batch_size):
                test_image_720p.seek(0)
                files = {"file": ("test_image.jpg", test_image_720p, "image/jpeg")}
                
                response = await async_client.post("/detect", files=files)
                assert response.status_code == 200
                
                # Track peak memory during batch
                if (i + 1) % 25 == 0:  # Sample every 25 requests
                    current_memory_mb = current_process.memory_info().rss / (1024 * 1024)
                    peak_memory_mb = max(peak_memory_mb, current_memory_mb)
            
            # Post-batch cleanup and measurement
            gc.collect()
            time.sleep(0.5)
            post_batch_memory_mb = current_process.memory_info().rss / (1024 * 1024)
            
            batch_results.append((peak_memory_mb, post_batch_memory_mb))
            
            print(f"Batch {batch_num + 1} - Peak: {peak_memory_mb:.2f} MB, "
                  f"Post-batch: {post_batch_memory_mb:.2f} MB")
            
            # Check that memory doesn't accumulate excessively between batches
            memory_accumulation = post_batch_memory_mb - initial_baseline_mb
            batch_threshold = MAX_MEMORY_INCREASE_MB * 0.7  # 70% of max for inter-batch accumulation
            
            assert memory_accumulation <= batch_threshold, (
                f"Memory accumulation after batch {batch_num + 1}: {memory_accumulation:.2f} MB "
                f"exceeds inter-batch threshold of {batch_threshold} MB"
            )
        
        # Final analysis
        final_memory_mb = current_process.memory_info().rss / (1024 * 1024)
        total_accumulation = final_memory_mb - initial_baseline_mb
        
        peak_memories = [peak for peak, _ in batch_results]
        post_batch_memories = [post for _, post in batch_results]
        
        print(f"\n=== Batch Memory Stability Results ===")
        print(f"Initial baseline: {initial_baseline_mb:.2f} MB")
        print(f"Final memory: {final_memory_mb:.2f} MB")
        print(f"Total accumulation: {total_accumulation:.2f} MB")
        print(f"Peak memory across all batches: {max(peak_memories):.2f} MB")
        print(f"Memory range post-batch: {min(post_batch_memories):.2f} - {max(post_batch_memories):.2f} MB")
        
        # Assert overall memory stability
        assert total_accumulation <= MAX_MEMORY_INCREASE_MB, (
            f"Total memory accumulation {total_accumulation:.2f} MB "
            f"exceeds threshold of {MAX_MEMORY_INCREASE_MB} MB"
        )
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_after_errors(self, test_image_720p, corrupted_image, async_client: httpx.AsyncClient):
        """
        Test that memory is properly cleaned up after error conditions.
        Ensures error handling doesn't cause memory leaks.
        """
        current_process = psutil.Process(os.getpid())
        
        # Baseline measurement
        gc.collect()
        time.sleep(1)
        baseline_memory_mb = current_process.memory_info().rss / (1024 * 1024)
        
        # Mix of successful and error requests
        total_requests = 200
        error_rate = 0.3  # 30% error requests
        
        print(f"Testing memory cleanup with {total_requests} requests ({error_rate*100:.0f}% error rate)")
        print(f"Baseline memory: {baseline_memory_mb:.2f} MB")
        
        error_count = 0
        success_count = 0
        
        for i in range(total_requests):
            # Decide whether to send valid or corrupted image
            if (i % 10) < (error_rate * 10):  # Generate errors based on error_rate
                # Send corrupted image (should cause 400 error)
                corrupted_image.seek(0)
                files = {"file": ("corrupted.jpg", corrupted_image, "image/jpeg")}
                response = await async_client.post("/detect", files=files)
                
                # Expect error response
                assert response.status_code in [400, 422], f"Expected error status, got {response.status_code}"
                error_count += 1
            else:
                # Send valid image
                test_image_720p.seek(0)
                files = {"file": ("test_image.jpg", test_image_720p, "image/jpeg")}
                response = await async_client.post("/detect", files=files)
                
                assert response.status_code == 200
                success_count += 1
            
            # Sample memory periodically
            if (i + 1) % 50 == 0:
                current_memory_mb = current_process.memory_info().rss / (1024 * 1024)
                print(f"Progress: {i+1}/{total_requests}, "
                      f"Memory: {current_memory_mb:.2f} MB, "
                      f"Errors: {error_count}, Success: {success_count}")
        
        # Final memory measurement
        gc.collect()
        time.sleep(1)
        final_memory_mb = current_process.memory_info().rss / (1024 * 1024)
        memory_increase = final_memory_mb - baseline_memory_mb
        
        print(f"\n=== Error Handling Memory Results ===")
        print(f"Total requests: {total_requests}")
        print(f"Successful requests: {success_count}")
        print(f"Error requests: {error_count}")
        print(f"Actual error rate: {(error_count/total_requests)*100:.1f}%")
        print(f"Baseline memory: {baseline_memory_mb:.2f} MB")
        print(f"Final memory: {final_memory_mb:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")
        
        # Memory should not accumulate significantly even with errors
        assert memory_increase <= MAX_MEMORY_INCREASE_MB, (
            f"Memory increased by {memory_increase:.2f} MB after error handling test, "
            f"exceeding threshold of {MAX_MEMORY_INCREASE_MB} MB"
        )
        
        # Verify we actually tested error conditions
        assert error_count > 0, "No error conditions were tested"
        assert success_count > 0, "No successful requests were processed"