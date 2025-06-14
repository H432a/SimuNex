from ultralytics import YOLO
import cv2

model = YOLO('model/best.pt')

def detect_objects(image_path):
    try:
        # Efficient image loading
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to read image")
            
        results = model(img)
        return list({model.names[int(box.cls[0])] for box in results[0].boxes})
    except Exception as e:
        print(f"Detection error: {e}")
        return []

    results = model(image) 

    detected_objects = set()
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item()) 
            class_name = model.names[class_id]
            detected_objects.add(class_name)
    
    return list(detected_objects)