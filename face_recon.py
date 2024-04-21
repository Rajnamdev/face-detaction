import cv2
import os
import numpy as np
import face_recognition

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load face data directory
data_dir = 'face_data'
if not os.path.exists(data_dir):
    print("No face data directory found.")
    exit()

# Load names of persons
names_file = os.path.join(data_dir, 'names.txt')
if not os.path.exists(names_file):
    print("No names file found.")
    exit()

with open(names_file, 'r') as f:
    person_names = f.read().splitlines()

# Load face encodings and corresponding labels
face_encodings = []
labels = []
for name in person_names:
    person_dir = os.path.join(data_dir, name)
    for filename in os.listdir(person_dir):
        img_path = os.path.join(person_dir, filename)
        img = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(img)[0]
        face_encodings.append(encoding)
        labels.append(name)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Convert the frame to RGB (face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find all face locations and encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings_frame = face_recognition.face_encodings(rgb_frame, face_locations)
    
    # Draw rectangles around the faces and label them with names
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings_frame):
        matches = face_recognition.compare_faces(face_encodings, face_encoding)
        name = "Unknown"
        
        if True in matches:
            first_match_index = matches.index(True)
            name = labels[first_match_index]
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Face Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
