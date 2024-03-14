import cv2
import pytesseract
import urllib.request
import numpy as np

# Replace the URL with the IP camera's stream URL
url = 'http://192.168.88.82/cam-mid.jpg'
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
cv2.namedWindow("ESP32 Camera Stream", cv2.WINDOW_NORMAL)

def extract_and_print_unique_text_from_webcam():
    try:
        while True:
            # Read a frame from the webcam
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(imgnp, -1)

            if frame is not None:
                # Convert the frame to grayscale for better OCR performance
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Extract text using Tesseract OCR
                text = pytesseract.image_to_string(gray_frame)
                text2 = text.strip().lower()

                # Compare text with the previous frame
                if 'previous_text' not in locals():
                    previous_text = ""
                if text2 != previous_text:
                    # Print the text if it is different
                    print(f'Text: {text2}')

                # Update previous_text for the next iteration
                previous_text = text2

                # Display the frame
                cv2.imshow("ESP32 Camera Stream", frame)

            # Check for the 'q' key to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break

        # Release the webcam and close all windows
        cv2.destroyAllWindows()

    except Exception as e:
        print(f'Error: {e}')

# Call the function to start webcam processing
extract_and_print_unique_text_from_webcam()
