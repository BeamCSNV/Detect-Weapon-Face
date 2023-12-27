from IPython.display import display, Image
import cv2
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Load detection model
detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8',
                                                     model_path='best_collab.pt',
                                                     confidence_threshold=0.6,
                                                     device='cpu')

# Open the webcam
videocapture = cv2.VideoCapture(0)  # 0 indicates the default camera

while videocapture.isOpened():
    success, frame = videocapture.read()
    if not success:
        break

    # Perform object detection
    results = get_sliced_prediction(frame,
                                    detection_model,
                                    slice_height=600,
                                    slice_width=600,
                                    overlap_height_ratio=0.2,
                                    overlap_width_ratio=0.2)
    object_prediction_list = results.object_prediction_list

    boxes_list = []
    clss_list = []

    # Extract bounding boxes and classes
    for ind, _ in enumerate(object_prediction_list):
        boxes = object_prediction_list[ind].bbox.minx, object_prediction_list[ind].bbox.miny, \
            object_prediction_list[ind].bbox.maxx, object_prediction_list[ind].bbox.maxy
        clss = object_prediction_list[ind].category.name
        boxes_list.append(boxes)
        clss_list.append(clss)

    # Draw bounding boxes on the frame
    for box, cls in zip(boxes_list, clss_list):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2)
        label = str(cls)
        t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
        cv2.rectangle(frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), (56, 56, 255),
                      -1)
        cv2.putText(frame,
                    label, (int(x1), int(y1) - 2),
                    0,
                    0.6, [255, 255, 255],
                    thickness=1,
                    lineType=cv2.LINE_AA)

    # Convert OpenCV image (BGR) to PIL image (RGB)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Display the image in the notebook
    display(pil_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam when finished
videocapture.release()
cv2.destroyAllWindows()
