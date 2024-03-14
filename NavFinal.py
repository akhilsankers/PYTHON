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

# Variable to store the timestamp of the last label
last_label_time = time.time()
# Define the Region of Interest (ROI) coordinates for three regions
roi_coordinates1 = (10, 50, 250, 500)  # Red ROI
roi_coordinates2 = (275, 50, 250, 500)  # Green ROI
roi_coordinates3 = (540, 50, 250, 500)  # Blue ROI
# Check if objects are detected in each ROI and change the color accordingly
color1 = (0, 0, 255)  # Red
color2 = (0, 255, 0)  # Green
color3 = (255, 0, 0)  # Blue
# Load a model
model = YOLO('yolov8s.pt')
dir = ""
# Replace the URL with the IP camera's stream URL
url = 'http://192.168.88.82/cam-mid.jpg'

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
            # Crop the frame to the specified ROIs
            x1, y1, w1, h1 = roi_coordinates1
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), color1, 2)
            roi_frame1 = frame[y1:y1 + h1, x1:x1 + w1]

            x2, y2, w2, h2 = roi_coordinates2
            cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), color2, 2)
            roi_frame2 = frame[y2:y2 + h2, x2:x2 + w2]

            x3, y3, w3, h3 = roi_coordinates3
            cv2.rectangle(frame, (x3, y3), (x3 + w3, y3 + h3), color3, 2)
            roi_frame3 = frame[y3:y3 + h3, x3:x3 + w3]

            # Predict with the model using the cropped ROI frames
            results1 = model(source=roi_frame1, show=True, conf=0.4, verbose=False, save=False)
            results2 = model(source=roi_frame2, show=True, conf=0.4, verbose=False, save=False)
            results3 = model(source=roi_frame3, show=True, conf=0.4, verbose=False, save=False)

            # calculate distance
            def calculate_distance(point1, point2):
                # Calculate distance in pixels
                distance_pixels = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

                # Convert distance to centimeters
                pixel_to_cm = 0.1
                distance_cm = distance_pixels * pixel_to_cm
                return distance_cm


            red_detected = False
            for r in results1:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # Calculate distance
                    distance = calculate_distance((x1, y1), (x2, y2))
                    if distance >= 6.0 and conf > 0.4:
                        red_detected = True
                        break

            # Process detected objects in ROI 2
            green_detected = False
            for r in results2:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # Calculate distance
                    distance = calculate_distance((x1, y1), (x2, y2))
                    if distance >= 6.0 and conf > 0.4:
                        green_detected = True
                        break

            # Process detected objects in ROI 3
            blue_detected = False
            for r in results3:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # Calculate distance
                    distance = calculate_distance((x1, y1), (x2, y2))
                    if distance >= 6.0 and conf > 0.4:
                        blue_detected = True
                        break

            # Take actions based on detected objects in each region
            if red_detected and not green_detected and not blue_detected:
                dir  = "Move to the left r"
            elif green_detected and not red_detected and not blue_detected:
                dir  = "Move to the right g"
            elif blue_detected and not red_detected and not green_detected:
                dir  = "Move to the left b"
            elif red_detected and green_detected and not blue_detected:
                dir  = "Move to the left"
            elif green_detected and blue_detected and not red_detected:
                dir  = "Move to the right"
            elif red_detected and blue_detected and not green_detected:
                dir  = "Go straight"
            elif red_detected and green_detected and blue_detected:
                dir  = "Stop"
            cv2.imshow('Original Frame with Colored ROIs', frame)
        else:
            print("Invalid frame dimensions. Skipping.")
        if time.time() - last_label_time >= 2:
            if dir is not None:
                print(dir)
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
                    db.reference("/").update({"dir": dir})
                    print(ref.get())

                    # Get value from a specific key ('name' in this case)
                    name_value = db.reference("/dir").get()
                    if name_value is not None:
                        print(name_value)
                    else:
                        print("No value found for key 'name'.")
                except firebase_admin.exceptions.FirebaseError as firebase_error:
                    print(f"Firebase error: {firebase_error}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                last_label_time = time.time()  # Reset the timestamp

        # Break the loop if 'q' key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")

# Release resources
cv2.destroyAllWindows()