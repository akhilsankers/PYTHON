import cv2
import urllib.request
import numpy as np
from ultralytics import YOLO
import math
import time
import firebase_admin
from firebase_admin import db, credentials

# Load a model
model = YOLO('yolov8s.pt')  # load an official model
cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://thirdeye-7a6d6-default-rtdb.firebaseio.com"
})

# Reference to the Firebase database node containing the 'conn' value
conn_ref = db.reference("/conn")

# Variable to store the previous label
prev_label = None

# Variable to store the timestamp of the last label
last_label_time = time.time()

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

# Pixel-to-Centimeter Conversion Factor (hypothetical, adjust as needed)
pixel_to_cm = 0.1


def calculate_distance(point1, point2):
    # Calculate distance in pixels
    distance_pixels = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    # Convert distance to centimeters
    distance_cm = distance_pixels * pixel_to_cm
    return distance_cm


# Load a model
model = YOLO('yolov8s.pt')

# Replace the URL with the IP camera's stream URL
url = 'http://192.168.252.82/cam-mid.jpg'

cv2.namedWindow("ESP32 Camera Stream", cv2.WINDOW_NORMAL)

while True:
    try:
        # Read a frame from the video stream
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)

        # Decode the image
        frame = cv2.imdecode(imgnp, -1)

        # Check if decoding was successful
        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            # Display the frame
            cv2.imshow('ESP32 Camera Stream', frame)

            # Set 'conn' to True if the video is successfully captured
            conn_ref.set(True)

            results = model(frame, show=True, conf=0.4, verbose=False, save=False)

            # Process detected objects
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Calculate distance
                    distance = calculate_distance((x1, y1), (x2, y2))

                    if distance >= 6.0 and conf > 0.75:  # Adjust the threshold as needed
                        cls = int(box.cls[0])
                        class_name = classNames[cls]
                        label = f'{class_name} '
                        label2 = f'{class_name} - {conf:.2f}, Distance: {distance:.2f} cm'

                        # Check if the label is different from the previous label
                        if label != prev_label:
                            # Check if 3 seconds have passed since the last label
                            if time.time() - last_label_time >= 2:
                                # Render the label on the frame
                                cv2.putText(frame, label2, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                            2)
                                print(label)
                                print(label2)

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

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Display the frame
            cv2.imshow('Object Detection', frame)
        else:
            print("Invalid frame dimensions. Skipping.")

        # Break the loop if 'q' key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")

# Set 'conn' to False before exiting the loop
conn_ref.set(False)

# Release resources
cv2.destroyAllWindows
