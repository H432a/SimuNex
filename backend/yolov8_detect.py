from ultralytics import YOLO
import cv2

# Load YOLO model (Make sure the path is correct)
model = YOLO('model/best.pt')

def detect_objects(image_path):
    image = cv2.imread(image_path)

    # Ensure the image is loaded properly
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return []

    results = model(image)  # Alternative to model.predict(source=image)

    detected_objects = set()
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())  # .item() ensures conversion to int
            class_name = model.names[class_id]
            detected_objects.add(class_name)
    
    return list(detected_objects)
