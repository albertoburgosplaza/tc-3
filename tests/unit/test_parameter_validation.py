"""
Unit tests for parameter validation in the object detection service.
Tests validate that the service correctly handles various parameter inputs.
"""

import pytest
from fastapi.testclient import TestClient
from app.app import app
import io


class TestParameterValidation:
    """Test suite for validating request parameters"""
    
    def test_valid_conf_parameter(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that valid confidence values (0.0-1.0) are accepted"""
        valid_conf_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        
        for conf in valid_conf_values:
            response = test_client.post(
                "/infer",
                files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
                params={"conf": conf}
            )
            assert response.status_code == 200, f"Failed for conf={conf}"
            assert "detections" in response.json()
    
    def test_invalid_conf_parameter(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that invalid confidence values are rejected with 422 status"""
        invalid_conf_values = [-0.1, -1.0, 1.1, 2.0, 999]
        
        for conf in invalid_conf_values:
            response = test_client.post(
                "/infer",
                files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
                params={"conf": conf}
            )
            assert response.status_code == 422, f"Should reject conf={conf}"
            assert "conf" in response.json()["detail"].lower()
    
    def test_valid_iou_parameter(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that valid IoU values (0.0-1.0) are accepted"""
        valid_iou_values = [0.0, 0.1, 0.45, 0.5, 0.75, 0.9, 1.0]
        
        for iou in valid_iou_values:
            response = test_client.post(
                "/infer",
                files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
                params={"iou": iou}
            )
            assert response.status_code == 200, f"Failed for iou={iou}"
            assert "detections" in response.json()
    
    def test_invalid_iou_parameter(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that invalid IoU values are rejected with 422 status"""
        invalid_iou_values = [-0.1, -1.0, 1.1, 2.0, 100]
        
        for iou in invalid_iou_values:
            response = test_client.post(
                "/infer",
                files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
                params={"iou": iou}
            )
            assert response.status_code == 422, f"Should reject iou={iou}"
            assert "iou" in response.json()["detail"].lower()
    
    def test_valid_max_detections_parameter(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that valid max_detections values (1-1000) are accepted"""
        valid_max_det_values = [1, 10, 100, 300, 500, 999, 1000]
        
        for max_det in valid_max_det_values:
            response = test_client.post(
                "/infer",
                files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
                params={"max_detections": max_det}
            )
            assert response.status_code == 200, f"Failed for max_detections={max_det}"
            assert "detections" in response.json()
    
    def test_invalid_max_detections_parameter(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that invalid max_detections values are rejected with 422 status"""
        invalid_max_det_values = [0, -1, -100, 1001, 10000, 999999]
        
        for max_det in invalid_max_det_values:
            response = test_client.post(
                "/infer",
                files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
                params={"max_detections": max_det}
            )
            assert response.status_code == 422, f"Should reject max_detections={max_det}"
            assert "max_detections" in response.json()["detail"].lower()
    
    def test_default_parameters(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that default parameters are used when not specified"""
        response = test_client.post(
            "/infer",
            files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")}
        )
        assert response.status_code == 200
        result = response.json()
        assert "detections" in result
        assert "model" in result
        assert "time_ms" in result
        assert "image" in result
    
    def test_combined_parameters(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that multiple valid parameters work together"""
        response = test_client.post(
            "/infer",
            files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
            params={
                "conf": 0.3,
                "iou": 0.5,
                "max_detections": 50
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert "detections" in result
        assert len(result["detections"]) <= 50
    
    def test_string_to_float_conversion(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that string parameters are correctly converted to floats"""
        response = test_client.post(
            "/infer",
            files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
            params={
                "conf": "0.5",
                "iou": "0.4"
            }
        )
        assert response.status_code == 200
    
    def test_invalid_parameter_types(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that non-numeric parameter values are rejected"""
        invalid_params = [
            {"conf": "abc"},
            {"iou": "not_a_number"},
            {"max_detections": "many"}
        ]
        
        for params in invalid_params:
            response = test_client.post(
                "/infer",
                files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
                params=params
            )
            assert response.status_code == 422
    
    def test_boundary_values(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test boundary values for all parameters"""
        boundary_tests = [
            {"conf": 0.0, "iou": 0.0, "max_detections": 1},      # All minimum
            {"conf": 1.0, "iou": 1.0, "max_detections": 1000},   # All maximum
            {"conf": 0.0, "iou": 1.0, "max_detections": 500},    # Mixed
        ]
        
        for params in boundary_tests:
            response = test_client.post(
                "/infer",
                files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
                params=params
            )
            assert response.status_code == 200, f"Failed for params={params}"