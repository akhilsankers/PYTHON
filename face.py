import cv2
import time
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a video capture object (0 for default camera)
cap = cv2.VideoCapture(0)

# Define ROI parameters
roi_width = 300
roi_height = 600
roi_y = 150

# Define the region of interest (ROI) rectangles in the middle of the frame
rois = [
    {"x": 50, "color": (0, 0, 250)},  # Red ROI
    {"x": 360, "color": (0, 255, 0)},  # Green ROI
    {"x": 670, "color": (0, 255, 255)}  # Yellow ROI
]

# Define a delay and a flag to control the frequency of print statements
print_delay = 2  # seconds
last_print_time = time.time()

while True:
    # Read a frame from the video source
    ret, frame = cap.read()

    # Resize the frame (you can adjust the dimensions as needed)
    frame = cv2.resize(frame, (1000, 800))

    # Flags to track face detection in each ROI
    red_detected = False
    green_detected = False
    yellow_detected = False

    # Iterate through each ROI
    for roi_info in rois:
        roi_x = roi_info["x"]
        roi_color = roi_info["color"]

        # Extract the ROI
        roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # Convert the ROI to grayscale (face detection works on grayscale images)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Run the face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Update the flags based on face detection in each ROI
        if roi_color == (0, 0, 250):  # Red ROI
            red_detected = len(faces) > 0
        elif roi_color == (0, 255, 0):  # Green ROI
            green_detected = len(faces) > 0
        elif roi_color == (0, 255, 255):  # Yellow ROI
            yellow_detected = len(faces) > 0

        # Draw bounding boxes around detected faces in the ROI
        for (x, y, w, h) in faces:
            if time.time() - last_print_time > print_delay:
                if roi_color == (0, 0, 250):
                    print("Move to right")
                    message = "Move to right"
                elif roi_color == (0, 255, 0):
                    # Determine the action based on face detection in different ROIs
                    if green_detected and red_detected:
                        print("Move to right")
                        message = "Move to right"
                    elif green_detected and yellow_detected:
                        print("Move to left")
                        message = "Move to left"
                    elif red_detected and green_detected and yellow_detected:
                        message = "Stop"
                    elif green_detected:
                        print("Move to right")
                        message = "Move to right"
                elif roi_color == (0, 255, 255):
                    print("Move to left")
                    message = "Move to left"

                # Use text-to-speech to vocalize the message
                engine.say(message)
                engine.runAndWait()

                last_print_time = time.time()

            cv2.rectangle(frame, (x + roi_x, y + roi_y), (x + w + roi_x, y + h + roi_y), roi_color, 2)

        # Draw a colored rectangle to mark the specific part of the frame
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), roi_color, 2)

    # Display the result
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
