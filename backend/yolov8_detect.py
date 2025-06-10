from ultralytics import YOLO
import cv2
import logging
import os
import gc
import warnings
import torch

# Disable GPU globally for Render compatibility
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # <-- Explicit GPU avoidance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress torchvision warnings
warnings.filterwarnings('ignore', category=UserWarning)

_model = None

def get_model():
    global _model
    if _model is None:
        try:
            logger.info("Initializing YOLO model...")
            
            # Force CPU and verify
            if torch.cuda.is_available():
                logger.warning("CUDA is available but forcing CPU usage")
                
            _model = YOLO('model/best.pt').to('cpu')  # <-- Explicit CPU loading
            _model.fuse()
            
            # Verify device
            device = next(_model.model.parameters()).device
            logger.info(f"Model loaded successfully on device: {device}")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}", exc_info=True)
            raise
    return _model

def detect_objects(image_path):
    try:
        # Verify image exists
        if not os.path.exists(image_path):
            logger.error(f"Image not found at {image_path}")
            return []

        # Load image in reduced quality
        img = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_2)
        if img is None:
            logger.error("Failed to decode image")
            return []

        # Convert color space (BGR→RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # <-- More efficient for YOLO
        
        # Let YOLO handle resizing internally
        model = get_model()
        
        # Memory cleanup before inference
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        try:
            # Inference with explicit CPU
            results = model(img, device='cpu', verbose=False)  # <-- No manual resizing
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                logger.error("CUDA OOM error despite CPU usage", exc_info=True)
            raise

        # Process results
        detected = set()
        for result in results:
            for box in result.boxes:
                detected.add(model.names[int(box.cls.item())])
        
        logger.info(f"Detected: {detected}")
        return list(detected)
        
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}", exc_info=True)
        return []
    
    finally:
        # Cleanup
        if 'img' in locals():
            del img
        gc.collect()