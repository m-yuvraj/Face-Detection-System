import cv2  # OpenCV for image processing and video capture
import tkinter as tk  # Tkinter for creating the GUI
from tkinter import Label, Button, messagebox  # Tkinter widgets for displaying content
from PIL import Image, ImageTk  # PIL for handling images compatible with Tkinter
import threading  # Threading to run detection in parallel to the GUI
import time  # For controlling the frame rate

# Initialize the Tkinter window
root = tk.Tk()
root.title("Face, Eye & Gender Detection System")
root.geometry("800x600")  # Set window size

# Load the pre-trained Haar Cascade models for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load pre-trained model for gender detection (OpenCV DNN model)
gender_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Labels for gender prediction
GENDER_LIST = ['Male', 'Female']

# Capture video from the default webcam
cap = cv2.VideoCapture(0)

# Create a Tkinter Label widget to display the video frames
label = Label(root)
label.pack()  # Place the label in the window

# A flag to control the detection loop
running = True

# Function that detects faces, eyes, and gender in the video stream
def detect_faces_eyes_and_gender():
    while running:  # Loop while the 'running' flag is True
        ret, frame = cap.read()  # Capture a frame from the webcam
        if not ret:  # If frame capture fails, exit the loop
            break
        
        # Resize the frame to 50% of its original size to speed up processing
        frame = cv2.resize(frame, None, fx=0.6, fy=0.6)

        # Convert the frame to grayscale (required for Haar Cascade detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop over the detected faces
        for (x, y, w, h) in faces:
            # Draw a blue rectangle around each detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Define regions of interest (ROIs) for the face (in both grayscale and color)
            roi_gray = gray[y:y+h, x:x+w]  # Grayscale ROI for eye detection
            roi_color = frame[y:y+h, x:x+w]  # Color ROI for drawing rectangles

            # Detect eyes within the face ROI
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                # Draw a green rectangle around each detected eye
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Gender Detection
            face_blob = cv2.dnn.blobFromImage(frame[y:y+h, x:x+w], 1.0, (227, 227), (104, 177, 123), swapRB=False)
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]
            
            # Display gender label on the frame
            cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Convert the frame to RGB (Tkinter requires RGB format, OpenCV uses BGR)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the OpenCV frame to a PIL image
        img = Image.fromarray(img)

        # Convert the PIL image to an ImageTk format for displaying in Tkinter
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the label with the new image
        label.imgtk = imgtk  # Store reference to the image
        label.configure(image=imgtk)  # Display the image

        # Control the frame rate to approximately 30 FPS
        time.sleep(0.03)

# Function to start face, eye, and gender detection in a separate thread
def start_detection():
    # Start the detect_faces_eyes_and_gender function in a new thread to prevent freezing the GUI
    threading.Thread(target=detect_faces_eyes_and_gender, daemon=True).start()

# Function to stop the video capture and close the window gracefully
def on_closing():
    global running  # Use the global 'running' flag
    running = False  # Stop the detection loop
    cap.release()  # Release the video capture object
    root.quit()  # Close the Tkinter window

# Create a Start Detection button that triggers the start of face, eye, and gender detection
start_button = Button(root, text="Start Detection", command=start_detection, font=("Helvetica", 14), bg="green", fg="white")
start_button.pack(pady=20)  # Place the button in the window with padding

# Bind the window close event to the on_closing function to stop video capture properly
root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the Tkinter main loop to keep the window open and responsive
root.mainloop()
