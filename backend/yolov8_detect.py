from ultralytics import YOLO
import cv2
import numpy as np
import psutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = YOLO('model/best.pt')

def preprocess_image(image, target_size=640):
    """Smart preprocessing that preserves detection quality"""
    h, w = image.shape[:2]
    
    # Only resize if image is significantly larger than target
    if max(h, w) > target_size * 1.5:  # 1.5x threshold prevents over-downscaling
        scale = target_size / max(h, w)
        image = cv2.resize(image, (0, 0), 
                          fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    
    # Optional: Contrast boost (helps with low-light images)
    # image = cv2.convertScaleAbs(image, alpha=1.2, beta=0)
    
    return image

def detect_objects(image_path, use_preprocessing=True, debug=False):
    try:
        # Memory check
        mem = psutil.virtual_memory()
        if mem.available < 200 * 1024 * 1024:  # 200MB threshold
            logger.warning(f"Low memory: {mem.available/1024/1024:.1f}MB available")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return []

        # Debug: Save original
        if debug:
            cv2.imwrite("debug_original.jpg", image)
        
        # Preprocess
        if use_preprocessing:
            image = preprocess_image(image)
            if debug:
                cv2.imwrite("debug_processed.jpg", image)
        
        # Detection
        results = model(image)
        
        # Extract classes
        detected = [model.names[int(box.cls.item())] for box in results[0].boxes]
        
        # Debug info
        if debug:
            logger.info(f"Detected: {detected}")
            logger.info(f"Image size: {image.shape}")
            logger.info(f"Memory used: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB")
        
        return detected if detected else []

    except Exception as e:
        logger.error(f"Detection failed: {str(e)}", exc_info=True)
        return []