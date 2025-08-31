"""
Object Detection Service - Main Entry Point

This service provides a local HTTP API for object detection using YOLOv8.
It detects persons and cars in uploaded images.

Usage:
    python main.py

The service will start on http://localhost:8000
- Health check: GET /health
- Object detection: POST /infer
"""

import uvicorn
import logging
from app.config import DEFAULT_CONF, DEFAULT_IOU

def main():
    """Start the FastAPI application with uvicorn server"""
    uvicorn.run(
        "app.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )

if __name__ == "__main__":
    # Configure main logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Object Detection Service...")
    logger.info(f"Default configuration: CONF={DEFAULT_CONF}, IOU={DEFAULT_IOU}")
    logger.info("Service will be available at: http://localhost:8000")
    logger.info("Health check endpoint: http://localhost:8000/health")
    main()