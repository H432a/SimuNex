from ultralytics import YOLO
import cv2
import logging
import os
import gc
import warnings
import torch
from typing import List, Optional

# Hard-disable GPU and optimize memory usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
torch.backends.quantized.engine = 'qnnpack'  # Use quantized backend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

_model = None

def get_model() -> YOLO:
    global _model
    if _model is None:
        try:
            logger.info("Loading optimized YOLO model...")
            
            # Force CPU with memory-efficient settings
            torch.set_flush_denormal(True)  # Reduce memory overhead
            _model = YOLO('model/best.pt', task='detect')
            
            # Optimize model for inference
            _model.fuse()
            _model = _model.to('cpu')
            _model.model.eval()
            _model.model.float()  # Ensure FP32
            
            # Verify configuration
            device = next(_model.model.parameters()).device
            mem_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            logger.info(f"Model loaded on {device} | Mem usage: {mem_usage/1e6:.1f}MB")
            
        except Exception as e:
            logger.critical(f"Model init failed: {str(e)}", exc_info=True)
            raise MemoryError("Failed to load model - insufficient resources")
            
    return _model

def detect_objects(image_path: str, max_detections: int = 3) -> List[str]:
    """Optimized detection with strict memory controls"""
    try:
        # Validate input
        if not os.path.exists(image_path):
            return []
            
        # Ultra-low-memory image loading
        img = cv2.imread(image_path, cv2.IMREAD_REDUCED_GRAYSCALE_4)
        if img is None:
            return []
            
        # Minimal color conversion
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (160, 120))  # Fixed small size
        
        # Force memory cleanup
        gc.collect()
        if 'torch' in globals():
            torch.cuda.empty_cache()
        
        # Strict inference parameters
        model = get_model()
        results = model(
            img,
            device='cpu',
            imgsz=160,
            verbose=False,
            max_det=max_detections,  # Critical limit
            conf=0.5,  # Higher confidence threshold
            iou=0.45  # Standard NMS
        )
        
        # Efficient results processing
        detected = set()
        for r in results:
            for box in r.boxes[:max_detections]:  # Hard limit
                detected.add(model.names[int(box.cls.item())])
        
        return list(detected)
        
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA OOM despite CPU-only mode")
        return []
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            logger.error("System out of memory")
        return []
    finally:
        # Aggressive cleanup
        if 'img' in locals():
            del img
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None