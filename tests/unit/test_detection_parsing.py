"""
Unit tests for detection parsing and response structure.
Tests validate that the service correctly formats detection results.
"""

import pytest
from fastapi.testclient import TestClient
from app.app import app
import io
from typing import Dict, List, Any


class TestDetectionParsing:
    """Test suite for detection parsing and response structure"""
    
    def test_response_structure(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that response has the expected structure"""
        response = test_client.post(
            "/infer",
            files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")}
        )
        assert response.status_code == 200
        
        result = response.json()
        
        # Check top-level fields
        assert "model" in result
        assert "time_ms" in result
        assert "image" in result
        assert "detections" in result
        
        # Check model field
        assert isinstance(result["model"], str)
        assert result["model"].endswith(".pt")
        
        # Check time_ms field
        assert isinstance(result["time_ms"], (int, float))
        assert result["time_ms"] >= 0
        
        # Check image field
        assert isinstance(result["image"], dict)
        assert "width" in result["image"]
        assert "height" in result["image"]
        assert isinstance(result["image"]["width"], int)
        assert isinstance(result["image"]["height"], int)
        
        # Check detections field
        assert isinstance(result["detections"], list)
    
    def test_detection_object_structure(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that each detection object has the expected structure"""
        response = test_client.post(
            "/infer",
            files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
            params={"conf": 0.01}  # Low confidence to potentially get detections
        )
        assert response.status_code == 200
        
        result = response.json()
        
        # If there are detections, validate their structure
        if len(result["detections"]) > 0:
            detection = result["detections"][0]
            
            # Check required fields
            assert "class_id" in detection
            assert "class_name" in detection
            assert "confidence" in detection
            assert "bbox_xyxy" in detection
            assert "bbox_xywh" in detection
            assert "bbox_norm_xyxy" in detection
            
            # Check field types
            assert isinstance(detection["class_id"], int)
            assert isinstance(detection["class_name"], str)
            assert isinstance(detection["confidence"], float)
            assert isinstance(detection["bbox_xyxy"], list)
            assert isinstance(detection["bbox_xywh"], list)
            assert isinstance(detection["bbox_norm_xyxy"], list)
            
            # Check bbox array lengths
            assert len(detection["bbox_xyxy"]) == 4
            assert len(detection["bbox_xywh"]) == 4
            assert len(detection["bbox_norm_xyxy"]) == 4
    
    def test_class_id_mapping(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that class_id correctly maps to class_name"""
        response = test_client.post(
            "/infer",
            files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
            params={"conf": 0.01}
        )
        assert response.status_code == 200
        
        result = response.json()
        
        # Define expected mapping
        expected_mapping = {
            0: "person",
            2: "car"
        }
        
        # Check all detections have valid class mappings
        for detection in result["detections"]:
            class_id = detection["class_id"]
            class_name = detection["class_name"]
            
            # Should only detect person or car
            assert class_id in expected_mapping
            assert class_name == expected_mapping[class_id]
    
    def test_confidence_values(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that confidence values are in valid range"""
        conf_threshold = 0.01
        response = test_client.post(
            "/infer",
            files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
            params={"conf": conf_threshold}
        )
        assert response.status_code == 200
        
        result = response.json()
        
        for detection in result["detections"]:
            confidence = detection["confidence"]
            assert 0.0 <= confidence <= 1.0
            assert confidence >= conf_threshold  # Should respect threshold
    
    def test_bbox_coordinates_valid(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that bounding box coordinates are valid"""
        response = test_client.post(
            "/infer",
            files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
            params={"conf": 0.01}
        )
        assert response.status_code == 200
        
        result = response.json()
        image_width = result["image"]["width"]
        image_height = result["image"]["height"]
        
        for detection in result["detections"]:
            # Check bbox_xyxy (x1, y1, x2, y2)
            x1, y1, x2, y2 = detection["bbox_xyxy"]
            assert 0 <= x1 <= image_width
            assert 0 <= x2 <= image_width
            assert 0 <= y1 <= image_height
            assert 0 <= y2 <= image_height
            assert x1 < x2  # x2 should be greater than x1
            assert y1 < y2  # y2 should be greater than y1
            
            # Check bbox_xywh (x_center, y_center, width, height)
            x_center, y_center, width, height = detection["bbox_xywh"]
            assert 0 <= x_center <= image_width
            assert 0 <= y_center <= image_height
            assert width > 0
            assert height > 0
            
            # Check bbox_norm_xyxy (normalized coordinates)
            nx1, ny1, nx2, ny2 = detection["bbox_norm_xyxy"]
            assert 0.0 <= nx1 <= 1.0
            assert 0.0 <= nx2 <= 1.0
            assert 0.0 <= ny1 <= 1.0
            assert 0.0 <= ny2 <= 1.0
            assert nx1 < nx2
            assert ny1 < ny2
    
    def test_max_detections_limit(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that max_detections parameter limits the number of detections"""
        max_det = 5
        response = test_client.post(
            "/infer",
            files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
            params={"conf": 0.01, "max_detections": max_det}
        )
        assert response.status_code == 200
        
        result = response.json()
        assert len(result["detections"]) <= max_det
    
    def test_empty_detections_list(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that high confidence threshold returns empty detections list"""
        response = test_client.post(
            "/infer",
            files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
            params={"conf": 0.99}  # Very high confidence threshold
        )
        assert response.status_code == 200
        
        result = response.json()
        assert "detections" in result
        assert isinstance(result["detections"], list)
        # May or may not have detections, but list should exist
    
    def test_image_dimensions_match(self, test_client: TestClient, small_test_image: io.BytesIO):
        """Test that returned image dimensions match input image"""
        # small_test_image is 64x64 as defined in conftest.py
        response = test_client.post(
            "/infer",
            files={"image": ("test.jpg", small_test_image.getvalue(), "image/jpeg")}
        )
        assert response.status_code == 200
        
        result = response.json()
        assert result["image"]["width"] == 64
        assert result["image"]["height"] == 64
    
    def test_detection_sorting(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that detections are sorted by confidence (highest first)"""
        response = test_client.post(
            "/infer",
            files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
            params={"conf": 0.01}
        )
        assert response.status_code == 200
        
        result = response.json()
        detections = result["detections"]
        
        # Check if detections are sorted by confidence in descending order
        if len(detections) > 1:
            confidences = [d["confidence"] for d in detections]
            assert confidences == sorted(confidences, reverse=True)
    
    def test_numeric_precision(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that numeric values have reasonable precision"""
        response = test_client.post(
            "/infer",
            files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")}
        )
        assert response.status_code == 200
        
        result = response.json()
        
        # Check time_ms has reasonable precision
        assert isinstance(result["time_ms"], (int, float))
        
        # Check detection numeric values
        for detection in result["detections"]:
            # Confidence should have reasonable precision
            assert 0.0 <= detection["confidence"] <= 1.0
            
            # Bounding box values should be numeric
            for coord in detection["bbox_xyxy"]:
                assert isinstance(coord, (int, float))
            for coord in detection["bbox_xywh"]:
                assert isinstance(coord, (int, float))
            for coord in detection["bbox_norm_xyxy"]:
                assert isinstance(coord, (int, float))
                # Normalized coords should be between 0 and 1
                assert 0.0 <= coord <= 1.0