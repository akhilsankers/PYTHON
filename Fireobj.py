from ultralytics import YOLO
import cv2
import math
import time
import firebase_admin
from firebase_admin import db, credentials

# Load a model
model = YOLO('yolov8s.pt')  # load an official model

# Authenticate to Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://fir123-b4740-default-rtdb.firebaseio.com"})

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Define object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Variable to store the previous label
prev_label = None

# Variable to store the timestamp of the last label
last_label_time = time.time()

# Function to update database connection status
def update_connection_status(status):
    try:
        db.reference("/").update({"conn": status})
        print("Connection status updated to:", status)
    except firebase_admin.exceptions.FirebaseError as firebase_error:
        print(f"Firebase error: {firebase_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Update connection status to True when camera is connected
update_connection_status(True)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Predict with the model on the current frame
    results = model(frame, show=True, conf=0.4, verbose=False, save=False)

    # Process detected objects
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name}'

            # Check if the label is different from the previous label
            if label != prev_label:
                # Check if 3 seconds have passed since the last label
                if time.time() - last_label_time >= 2:
                    # Render the label on the frame
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    print(label)
                    # Update the previous label and timestamp
                    prev_label = label
                    try:
                        # Creating a reference to the root node
                        ref = db.reference("/")

                        # Initializing title_count before the transaction
                        db.reference("/title_count").set(0)

                        # Retrieving data from the root node
                        data = ref.get()
                        if data:
                            print(data)
                        else:
                            print("No data found.")

                        # Update operation (add new key 'name' with value 'python')
                        db.reference("/").update({"name": label})
                        print(ref.get())

                        # Get value from a specific key ('name' in this case)
                        name_value = db.reference("/name").get()
                        if name_value is not None:
                            print(name_value)
                        else:
                            print("No value found for key 'name'.")

                    except firebase_admin.exceptions.FirebaseError as firebase_error:
                        print(f"Firebase error: {firebase_error}")
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")
                    last_label_time = time.time()

    # Display the annotated frame
    cv2.imshow("Object Detection", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Update connection status to False when camera is disconnected
update_connection_status(False)

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
