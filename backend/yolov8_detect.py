from ultralytics import YOLO
import cv2
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Singleton pattern with memory management
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            logger.info("Loading YOLO model...")
            _model = YOLO('model/best.pt')  # Load only once
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    return _model

def detect_objects(image_path):
    try:
        # Verify image exists
        if not os.path.exists(image_path):
            logger.error(f"Image not found at {image_path}")
            return []

        # Read and resize image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image at {image_path}")
            return []

        # Further reduced size for memory efficiency
        image = cv2.resize(image, (256, 192))  # Reduced from 320x240
        
        # Get model and predict
        model = get_model()
        results = model(image, verbose=False, imgsz=256)  # Smaller input size
        
        # Process results
        detected_objects = set()
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = model.names[class_id]
                detected_objects.add(class_name)
        
        logger.info(f"Detected objects: {detected_objects}")
        return list(detected_objects)
    
    except Exception as e:
        logger.error(f"Detection error: {str(e)}", exc_info=True)
        return []