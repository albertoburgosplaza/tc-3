from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
import io
import time
import os
import torch
import logging
from typing import Optional, List, Dict, Any

from ultralytics import YOLO
from .config import (
    CLASSES_KEEP, 
    MODEL_PATH, 
    DEFAULT_CONF, 
    DEFAULT_IOU, 
    DEFAULT_MAX_DETECTIONS
)

# Configure logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Force CPU-only usage
torch.set_default_tensor_type('torch.FloatTensor')
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA devices

app = FastAPI(
    title="Local Object Detector",
    description="Local containerized service for object detection (person and car) using YOLOv8",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global model instance - will be loaded on startup
model = None

@app.on_event("startup")
async def startup_event():
    """Load YOLO model on application startup and ensure CPU usage"""
    global model
    startup_start_time = time.time()
    
    logger.info(f"Starting application startup - Loading YOLO model: {MODEL_PATH}")
    logger.info("Forcing CPU-only execution...")
    
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            raise RuntimeError(f"Model file not found: {MODEL_PATH}")
        
        # Log model file info
        model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # Size in MB
        logger.info(f"Model file found: {MODEL_PATH} ({model_size:.1f} MB)")
        
        # Fix PyTorch 2.6+ weights_only issue for YOLO model loading
        original_torch_load = torch.load
        def patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
            # Force weights_only=False for model loading to avoid pickle restrictions
            kwargs['weights_only'] = False
            return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
        
        # Temporarily patch torch.load
        torch.load = patched_torch_load
        
        # Load model and explicitly move to CPU
        logger.info("Loading YOLO model...")
        model = YOLO(MODEL_PATH)
        model.to('cpu')  # Ensure model runs on CPU
        
        # Test model with a small dummy prediction to verify it's working
        logger.info("Testing model functionality...")
        import numpy as np
        from PIL import Image
        dummy_image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        _ = model.predict(dummy_image, verbose=False)
        
        logger.info("YOLO model loaded successfully on CPU")
        logger.info(f"Available detection classes: {list(CLASSES_KEEP.keys())}")
        
        # Log startup performance metrics
        startup_end_time = time.time()
        startup_time_ms = (startup_end_time - startup_start_time) * 1000
        logger.info(
            f"STARTUP_METRICS - "
            f"model_path='{MODEL_PATH}' "
            f"model_size_mb={model_size:.1f} "
            f"startup_time_ms={startup_time_ms:.2f} "
            f"available_classes={list(CLASSES_KEEP.keys())}"
        )
        logger.info(f"Application startup completed in {startup_time_ms:.2f}ms")
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise RuntimeError(f"Cannot start application: Model file not found at {MODEL_PATH}")
    
    except MemoryError as e:
        logger.error(f"Insufficient memory to load model: {e}")
        raise RuntimeError("Cannot start application: Insufficient memory to load YOLO model")
    
    except RuntimeError as e:
        if "JIT" in str(e) or "compilation" in str(e):
            logger.error(f"PyTorch JIT compilation error: {e}")
            raise RuntimeError(f"Cannot start application: PyTorch model compilation error - {e}")
        else:
            # Re-raise other RuntimeErrors
            raise
    
    except Exception as e:
        logger.error(f"Unexpected error during model loading: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        raise RuntimeError(f"Cannot start application: Failed to load YOLO model - {e}")


# Pydantic models for API responses
class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify service is running"""
    logger.info("Health check endpoint accessed")
    return HealthResponse(
        status="healthy",
        service="Local Object Detector", 
        version="1.0.0"
    )


@app.post("/infer")
async def infer_objects(
    image: UploadFile = File(...),
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    max_detections: int = DEFAULT_MAX_DETECTIONS
):
    """
    Endpoint for object inference on uploaded images
    
    Args:
        image: UploadFile - The image file to process (JPEG/PNG)
        conf: float - Confidence threshold for detections (default from config)
        iou: float - IoU threshold for Non-Maximum Suppression (default from config)  
        max_detections: int - Maximum number of detections to return (default from config)
    
    Returns:
        Object detection results for person and car classes
    """
    # Start timing the entire request
    request_start_time = time.time()
    
    logger.info(f"New inference request - File: {image.filename}, Content-Type: {image.content_type}, "
                f"Size: {image.size} bytes, Parameters: conf={conf}, iou={iou}, max_detections={max_detections}")
    
    # Check if model is loaded (500 - Internal Server Error)
    if model is None:
        logger.error("Model not loaded - cannot perform inference")
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor: Modelo no cargado"
        )
    
    # Validate inference parameters (422 - Unprocessable Entity)
    if not (0.0 <= conf <= 1.0):
        logger.warning(f"Invalid confidence parameter: {conf} (must be between 0.0 and 1.0)")
        raise HTTPException(
            status_code=422,
            detail="Parámetro 'conf' inválido. Debe estar entre 0.0 y 1.0"
        )
    
    if not (0.0 <= iou <= 1.0):
        logger.warning(f"Invalid IoU parameter: {iou} (must be between 0.0 and 1.0)")
        raise HTTPException(
            status_code=422,
            detail="Parámetro 'iou' inválido. Debe estar entre 0.0 y 1.0"
        )
    
    if max_detections < 1 or max_detections > 1000:
        logger.warning(f"Invalid max_detections parameter: {max_detections} (must be between 1 and 1000)")
        raise HTTPException(
            status_code=422,
            detail="Parámetro 'max_detections' inválido. Debe estar entre 1 y 1000"
        )
    
    # Validate content_type - only accept JPEG and PNG (400 - Bad Request)
    allowed_content_types = {'image/jpeg', 'image/png'}
    if image.content_type not in allowed_content_types:
        logger.warning(f"Unsupported image format: {image.content_type}")
        raise HTTPException(
            status_code=400, 
            detail=f"Formato de imagen no soportado. Formatos permitidos: {', '.join(allowed_content_types)}"
        )
    
    # Validate file size (413 - Payload Too Large)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
    if image.size > MAX_FILE_SIZE:
        logger.warning(f"File too large: {image.size} bytes (max: {MAX_FILE_SIZE} bytes)")
        raise HTTPException(
            status_code=413,
            detail="Archivo demasiado grande. Máximo 10MB"
        )
    
    # Validate image can be opened with PIL (400 - Bad Request for invalid images)
    try:
        # Read image content
        image_content = await image.read()
        
        # Try to open the image with PIL
        pil_image = Image.open(io.BytesIO(image_content))
        pil_image.verify()  # Verify it's a valid image
        
        # Reset file pointer after verify() (which consumes the file)
        pil_image = Image.open(io.BytesIO(image_content))
        
    except (UnidentifiedImageError, OSError, IOError) as e:
        logger.warning(f"Invalid or corrupted image file: {image.filename}, Error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail="Imagen corrupta o inválida"
        )
    except Exception as e:
        logger.error(f"Unexpected error during image validation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor al procesar la imagen"
        )
    
    # Task 6.1 - Measure inference timing
    logger.info(f"Starting YOLO inference on image: {pil_image.size[0]}x{pil_image.size[1]} pixels")
    inference_start_time = time.time()
    
    try:
        # Task 6.2 - Perform model inference with class filtering
        # Filter to only detect person (class 0) and car (class 2)
        results = model.predict(
            pil_image, 
            conf=conf, 
            iou=iou, 
            max_det=max_detections, 
            classes=[0, 2],  # Only person and car classes
            verbose=False
        )
        
        inference_end_time = time.time()
        inference_time_ms = (inference_end_time - inference_start_time) * 1000
        logger.info(f"YOLO inference completed in {inference_time_ms:.2f}ms")
        
    except Exception as e:
        logger.error(f"Model inference failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor durante la inferencia"
        )
    
    try:
        # Task 6.3 - Extract and convert bounding boxes to multiple formats
        # Task 6.4 - Create class ID to name mapping
        id_to_name = {v: k for k, v in CLASSES_KEEP.items()}  # {0: 'person', 2: 'car'}
        
        detections = []
        
        if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            image_height, image_width = pil_image.size[1], pil_image.size[0]  # PIL returns (width, height)
            
            for box in boxes:
                # Extract bounding box in different formats
                bbox_xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                bbox_xywh = box.xywh[0].tolist()  # [x_center, y_center, width, height]
                
                # Normalize xyxy coordinates (values between 0 and 1)
                bbox_norm_xyxy = [
                    bbox_xyxy[0] / image_width,   # x1_normalized
                    bbox_xyxy[1] / image_height,  # y1_normalized
                    bbox_xyxy[2] / image_width,   # x2_normalized
                    bbox_xyxy[3] / image_height   # y2_normalized
                ]
                
                # Extract class_id and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Task 6.4 - Map class_id to class name
                class_name = id_to_name.get(class_id, 'unknown')
                
                # Task 6.5 - Create final detection structure
                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox_xyxy": bbox_xyxy,
                    "bbox_xywh": bbox_xywh, 
                    "bbox_norm_xyxy": bbox_norm_xyxy
                }
                detections.append(detection)
        
        # Task 6.5 - Limit detections to max_detections if needed
        if len(detections) > max_detections:
            detections = detections[:max_detections]
        
        # Log detection results by class
        detection_counts = {}
        for detection in detections:
            class_name = detection["class_name"]
            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
        
        total_detections = len(detections)
        
        # Calculate total request processing time
        request_end_time = time.time()
        total_request_time_ms = (request_end_time - request_start_time) * 1000
        
        # Enhanced performance metrics logging
        logger.info(
            f"PERFORMANCE_METRICS - "
            f"file='{image.filename}' "
            f"image_size={pil_image.size[0]}x{pil_image.size[1]} "
            f"file_size_bytes={image.size} "
            f"parameters=conf:{conf},iou:{iou},max_det:{max_detections} "
            f"inference_time_ms={inference_time_ms:.2f} "
            f"total_time_ms={total_request_time_ms:.2f} "
            f"processing_overhead_ms={total_request_time_ms - inference_time_ms:.2f} "
            f"detections_total={total_detections} "
            f"detections_by_class={detection_counts}"
        )
        
        # Summary log for easy monitoring
        logger.info(f"Request completed successfully: {total_detections} objects detected in {total_request_time_ms:.2f}ms")
        
        # Task 7.1 - Create base JSON response structure
        return {
            "model": os.path.basename(MODEL_PATH),
            "time_ms": round(inference_time_ms, 2),
            "image": {
                "width": pil_image.size[0],  # PIL returns (width, height)
                "height": pil_image.size[1]
            },
            "detections": detections
        }
        
    except Exception as e:
        logger.error(f"Error processing inference results: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor al procesar los resultados"
        )