"""
Integration tests for the inference endpoint with golden set images.
These tests validate the complete inference pipeline with real images.
"""

import pytest
from fastapi.testclient import TestClient
from app.app import app
import os
from pathlib import Path


class TestInferenceGoldenSet:
    """Integration tests using known test images (golden set)"""
    
    def test_inference_with_person_image(self, test_client: TestClient):
        """Test inference with an image containing a person"""
        # Use the existing test_person.jpg image
        test_image_path = Path("test_person.jpg")
        
        if test_image_path.exists():
            with open(test_image_path, "rb") as f:
                response = test_client.post(
                    "/infer",
                    files={"image": ("test_person.jpg", f, "image/jpeg")},
                    params={"conf": 0.25}
                )
                
            assert response.status_code == 200
            result = response.json()
            
            # Validate response structure
            assert "model" in result
            assert "time_ms" in result
            assert "image" in result
            assert "detections" in result
            
            # Image should be 640x480 based on the test file
            assert result["image"]["width"] == 640
            assert result["image"]["height"] == 480
            
            # Should detect at least one person
            detections = result["detections"]
            person_detections = [d for d in detections if d["class_name"] == "person"]
            assert len(person_detections) >= 1, "Should detect at least one person"
            
            # Validate detection structure
            for detection in person_detections:
                assert 0.0 <= detection["confidence"] <= 1.0
                assert detection["class_id"] == 0  # Person class ID
                assert len(detection["bbox_xyxy"]) == 4
                assert len(detection["bbox_xywh"]) == 4
                assert len(detection["bbox_norm_xyxy"]) == 4
                
                # Validate normalized coordinates
                for coord in detection["bbox_norm_xyxy"]:
                    assert 0.0 <= coord <= 1.0
    
    def test_inference_with_multiple_parameters(self, test_client: TestClient):
        """Test inference with various confidence thresholds"""
        test_image_path = Path("test_person.jpg")
        
        if test_image_path.exists():
            # Test with different confidence thresholds
            conf_levels = [0.1, 0.25, 0.5, 0.75]
            previous_count = float('inf')
            
            for conf in conf_levels:
                with open(test_image_path, "rb") as f:
                    response = test_client.post(
                        "/infer",
                        files={"image": ("test_person.jpg", f, "image/jpeg")},
                        params={"conf": conf}
                    )
                
                assert response.status_code == 200
                result = response.json()
                current_count = len(result["detections"])
                
                # Higher confidence should result in fewer or equal detections
                assert current_count <= previous_count, \
                    f"Higher conf ({conf}) should have fewer detections"
                previous_count = current_count
    
    def test_inference_with_empty_image(self, test_client: TestClient, valid_jpeg_image):
        """Test inference with an image that has no objects"""
        # Use the small solid color image from fixtures
        response = test_client.post(
            "/infer",
            files={"image": ("empty.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
            params={"conf": 0.5}
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Should return valid structure even with no detections
        assert "detections" in result
        assert isinstance(result["detections"], list)
        # Solid color image should have few or no detections at high confidence
        assert len(result["detections"]) <= 2  # Allow for potential false positives
    
    def test_inference_consistency(self, test_client: TestClient):
        """Test that same image with same parameters gives consistent results"""
        test_image_path = Path("test_person.jpg")
        
        if test_image_path.exists():
            results = []
            
            # Run inference multiple times
            for _ in range(3):
                with open(test_image_path, "rb") as f:
                    response = test_client.post(
                        "/infer",
                        files={"image": ("test_person.jpg", f, "image/jpeg")},
                        params={"conf": 0.25, "iou": 0.45}
                    )
                
                assert response.status_code == 200
                results.append(response.json())
            
            # Check consistency of detection counts
            detection_counts = [len(r["detections"]) for r in results]
            assert len(set(detection_counts)) == 1, \
                "Same image should produce same number of detections"
            
            # Check consistency of class detections
            for i in range(1, len(results)):
                classes_0 = {d["class_name"] for d in results[0]["detections"]}
                classes_i = {d["class_name"] for d in results[i]["detections"]}
                assert classes_0 == classes_i, \
                    "Same image should detect same classes"
    
    def test_inference_performance_metrics(self, test_client: TestClient):
        """Test that inference returns reasonable performance metrics"""
        test_image_path = Path("test_person.jpg")
        
        if test_image_path.exists():
            with open(test_image_path, "rb") as f:
                response = test_client.post(
                    "/infer",
                    files={"image": ("test_person.jpg", f, "image/jpeg")}
                )
            
            assert response.status_code == 200
            result = response.json()
            
            # Check time_ms is reasonable (between 1ms and 5000ms)
            assert 1 <= result["time_ms"] <= 5000, \
                f"Inference time {result['time_ms']}ms seems unreasonable"
            
            # Time should be a positive number
            assert result["time_ms"] > 0
    
    def test_inference_with_different_image_sizes(self, test_client: TestClient,
                                                   small_test_image, test_image_720p):
        """Test inference with images of different sizes"""
        test_cases = [
            ("small.jpg", small_test_image, 64, 64),
            ("720p.jpg", test_image_720p, 1280, 720)
        ]
        
        for filename, image_fixture, expected_width, expected_height in test_cases:
            response = test_client.post(
                "/infer",
                files={"image": (filename, image_fixture.getvalue(), "image/jpeg")},
                params={"conf": 0.25}
            )
            
            assert response.status_code == 200
            result = response.json()
            
            # Check image dimensions are correctly reported
            assert result["image"]["width"] == expected_width
            assert result["image"]["height"] == expected_height
            
            # All detections should be within image bounds
            for detection in result["detections"]:
                x1, y1, x2, y2 = detection["bbox_xyxy"]
                assert 0 <= x1 <= expected_width
                assert 0 <= x2 <= expected_width
                assert 0 <= y1 <= expected_height
                assert 0 <= y2 <= expected_height
    
    def test_max_detections_limit_in_practice(self, test_client: TestClient):
        """Test that max_detections parameter actually limits detections"""
        test_image_path = Path("test_person.jpg")
        
        if test_image_path.exists():
            # First, get baseline with low confidence and high max_detections
            with open(test_image_path, "rb") as f:
                baseline_response = test_client.post(
                    "/infer",
                    files={"image": ("test_person.jpg", f, "image/jpeg")},
                    params={"conf": 0.01, "max_detections": 100}
                )
            
            baseline_count = len(baseline_response.json()["detections"])
            
            # If we have detections, test limiting works
            if baseline_count > 1:
                limit = min(1, baseline_count - 1)
                with open(test_image_path, "rb") as f:
                    limited_response = test_client.post(
                        "/infer",
                        files={"image": ("test_person.jpg", f, "image/jpeg")},
                        params={"conf": 0.01, "max_detections": limit}
                    )
                
                limited_count = len(limited_response.json()["detections"])
                assert limited_count <= limit, \
                    f"Should have at most {limit} detections, got {limited_count}"
    
    def test_class_filtering(self, test_client: TestClient):
        """Test that only person and car classes are detected"""
        test_image_path = Path("test_person.jpg")
        
        if test_image_path.exists():
            with open(test_image_path, "rb") as f:
                response = test_client.post(
                    "/infer",
                    files={"image": ("test_person.jpg", f, "image/jpeg")},
                    params={"conf": 0.01}  # Low confidence to get more detections
                )
            
            assert response.status_code == 200
            result = response.json()
            
            # All detections should be either person or car
            valid_classes = {"person", "car"}
            valid_class_ids = {0, 2}
            
            for detection in result["detections"]:
                assert detection["class_name"] in valid_classes, \
                    f"Unexpected class: {detection['class_name']}"
                assert detection["class_id"] in valid_class_ids, \
                    f"Unexpected class_id: {detection['class_id']}"
    
    def test_bbox_format_consistency(self, test_client: TestClient):
        """Test that different bbox formats are consistent"""
        test_image_path = Path("test_person.jpg")
        
        if test_image_path.exists():
            with open(test_image_path, "rb") as f:
                response = test_client.post(
                    "/infer",
                    files={"image": ("test_person.jpg", f, "image/jpeg")},
                    params={"conf": 0.25}
                )
            
            assert response.status_code == 200
            result = response.json()
            
            image_width = result["image"]["width"]
            image_height = result["image"]["height"]
            
            for detection in result["detections"]:
                # Extract bbox formats
                x1, y1, x2, y2 = detection["bbox_xyxy"]
                x_center, y_center, width, height = detection["bbox_xywh"]
                nx1, ny1, nx2, ny2 = detection["bbox_norm_xyxy"]
                
                # Verify xywh is consistent with xyxy
                assert abs(x_center - (x1 + x2) / 2) < 1.0, "x_center mismatch"
                assert abs(y_center - (y1 + y2) / 2) < 1.0, "y_center mismatch"
                assert abs(width - (x2 - x1)) < 1.0, "width mismatch"
                assert abs(height - (y2 - y1)) < 1.0, "height mismatch"
                
                # Verify normalized coords are correct
                assert abs(nx1 - x1 / image_width) < 0.01, "nx1 mismatch"
                assert abs(ny1 - y1 / image_height) < 0.01, "ny1 mismatch"
                assert abs(nx2 - x2 / image_width) < 0.01, "nx2 mismatch"
                assert abs(ny2 - y2 / image_height) < 0.01, "ny2 mismatch"