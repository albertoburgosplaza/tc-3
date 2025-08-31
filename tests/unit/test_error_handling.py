"""
Unit tests for error handling in the object detection service.
Tests validate that the service returns appropriate HTTP error codes for various failure scenarios.
"""

import pytest
from fastapi.testclient import TestClient
from app.app import app
import io
from PIL import Image


class TestErrorHandling:
    """Test suite for error handling scenarios"""
    
    def test_unsupported_image_format(self, test_client: TestClient):
        """Test that unsupported image formats return 400 error"""
        unsupported_formats = [
            ("test.gif", b"GIF89a", "image/gif"),
            ("test.bmp", b"BM", "image/bmp"),
            ("test.txt", b"plain text", "text/plain"),
            ("test.pdf", b"%PDF-1.4", "application/pdf"),
            ("test.svg", b"<svg>", "image/svg+xml"),
        ]
        
        for filename, content, content_type in unsupported_formats:
            response = test_client.post(
                "/infer",
                files={"image": (filename, content, content_type)}
            )
            assert response.status_code == 400, f"Should reject {content_type}"
            assert "formato" in response.json()["detail"].lower()
    
    def test_corrupted_image_file(self, test_client: TestClient, corrupted_image: io.BytesIO):
        """Test that corrupted image files return 400 error"""
        response = test_client.post(
            "/infer",
            files={"image": ("corrupted.jpg", corrupted_image.getvalue(), "image/jpeg")}
        )
        assert response.status_code == 400
        assert "corrupta" in response.json()["detail"].lower()
    
    def test_file_size_limit(self, test_client: TestClient):
        """Test that files larger than 10MB return 413 error"""
        # Create a file larger than 10MB
        large_data = b"x" * (11 * 1024 * 1024)  # 11MB of data
        
        response = test_client.post(
            "/infer",
            files={"image": ("large.jpg", large_data, "image/jpeg")}
        )
        assert response.status_code == 413
        assert "demasiado grande" in response.json()["detail"].lower()
    
    def test_missing_image_file(self, test_client: TestClient):
        """Test that missing image file returns 422 error"""
        response = test_client.post("/infer")
        assert response.status_code == 422
        assert "field required" in str(response.json()["detail"]).lower()
    
    def test_empty_image_file(self, test_client: TestClient):
        """Test that empty image file returns 400 error"""
        response = test_client.post(
            "/infer",
            files={"image": ("empty.jpg", b"", "image/jpeg")}
        )
        assert response.status_code == 400
    
    def test_invalid_jpeg_header(self, test_client: TestClient):
        """Test that invalid JPEG headers are detected"""
        # Create invalid JPEG data (wrong header)
        invalid_jpeg = b"NOT_A_JPEG_HEADER" + b"\x00" * 100
        
        response = test_client.post(
            "/infer",
            files={"image": ("invalid.jpg", invalid_jpeg, "image/jpeg")}
        )
        assert response.status_code == 400
        assert "corrupta" in response.json()["detail"].lower() or "inválida" in response.json()["detail"].lower()
    
    def test_invalid_png_header(self, test_client: TestClient):
        """Test that invalid PNG headers are detected"""
        # Create invalid PNG data (wrong header)
        invalid_png = b"NOT_A_PNG_HEADER" + b"\x00" * 100
        
        response = test_client.post(
            "/infer",
            files={"image": ("invalid.png", invalid_png, "image/png")}
        )
        assert response.status_code == 400
        assert "corrupta" in response.json()["detail"].lower() or "inválida" in response.json()["detail"].lower()
    
    def test_model_not_loaded_error(self, test_client: TestClient, valid_jpeg_image: io.BytesIO, monkeypatch):
        """Test that 500 error is returned when model is not loaded"""
        # Simulate model not being loaded
        import app.app as app_module
        monkeypatch.setattr(app_module, "model", None)
        
        response = test_client.post(
            "/infer",
            files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")}
        )
        assert response.status_code == 500
        assert "modelo no cargado" in response.json()["detail"].lower()
    
    def test_truncated_image_file(self, test_client: TestClient):
        """Test that truncated image files are handled properly"""
        # Create a valid JPEG header but truncated data
        image = Image.new('RGB', (100, 100), color=(255, 0, 0))
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        truncated_data = image_bytes.getvalue()[:50]  # Take only first 50 bytes
        
        response = test_client.post(
            "/infer",
            files={"image": ("truncated.jpg", truncated_data, "image/jpeg")}
        )
        assert response.status_code == 400
    
    def test_multiple_parameter_errors(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that the first parameter error is returned when multiple are invalid"""
        response = test_client.post(
            "/infer",
            files={"image": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")},
            params={
                "conf": -1.0,    # Invalid
                "iou": 2.0,      # Invalid
                "max_detections": 0  # Invalid
            }
        )
        assert response.status_code == 422
        # Should catch at least one of the parameter errors
        detail = response.json()["detail"].lower()
        assert "conf" in detail or "iou" in detail or "max_detections" in detail
    
    def test_wrong_field_name(self, test_client: TestClient, valid_jpeg_image: io.BytesIO):
        """Test that wrong field names in multipart form are handled"""
        response = test_client.post(
            "/infer",
            files={"wrong_field": ("test.jpg", valid_jpeg_image.getvalue(), "image/jpeg")}
        )
        assert response.status_code == 422
    
    def test_concurrent_error_conditions(self, test_client: TestClient):
        """Test handling of multiple simultaneous error conditions"""
        # Corrupted file + invalid parameters
        response = test_client.post(
            "/infer",
            files={"image": ("bad.jpg", b"not an image", "image/jpeg")},
            params={"conf": -1.0}
        )
        # Should catch parameter error first (422) or image error (400)
        assert response.status_code in [400, 422]
    
    def test_error_response_format(self, test_client: TestClient):
        """Test that error responses have the expected format"""
        response = test_client.post(
            "/infer",
            files={"image": ("test.txt", b"text content", "text/plain")}
        )
        assert response.status_code == 400
        
        error_response = response.json()
        assert "detail" in error_response
        assert isinstance(error_response["detail"], str)
        assert len(error_response["detail"]) > 0