import cv2
from ultralytics import YOLO

# 1. Load the model
# modelpath = r"F:\pothole\manhole\best (13).pt"  # <-- Replace with your model file name
modelpath = r"F:\pothole\manhole\hole.pt"  # <-- Replace with your model file name
model = YOLO(modelpath)

# 2. Read the image with OpenCV
image_path = r"F:\pothole\manhole\data\image2.png"  # <-- Replace with your image file name
img = cv2.imread(image_path)

# 3. Run prediction
results = model(img)[0]

# 4. Extract boxes, names, and scores
for box in results.boxes:
    # Get bounding box coordinates [xmin, ymin, xmax, ymax]
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    
    # Get class ID and match it to its string name
    class_id = int(box.cls[0].item())
    class_name = model.names[class_id]
    
    # Get confidence score
    confidence = box.conf[0].item()

    # 5. Draw the bounding box (Green rectangle, thickness of 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 6. Prepare label text (e.g., "person 85%")
    label = f"{class_name} {confidence:.0%}"
    
    # 7. Draw a background filled rectangle for text readability
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - 20), (x1 + text_w, y1), (0, 255, 0), -1)
    
    # 8. Put the text label over the bounding box
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# 9. Display the image
# shaped_imz = cv2.resize(img, (400, 200))  # Resize for better visibility
cv2.imshow("Custom Box & Class Labels", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
