import cv2
import os

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video from webcam (0 for default webcam)
video_capture = cv2.VideoCapture(1)

# Create directory to store face data
data_dir = 'face_data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Initialize variables
face_data = []
person_names = []
count = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the faces and collect face data
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_roi = gray[y:y+h, x:x+w]
        face_data.append(face_roi)
        count += 1
        # Save face data every 10 frames
        if count % 100 == 0:
            person_name = input("Enter the name of the person: ")
            person_names.append(person_name)
            print("Saving face ", count//10)
            cv2.imwrite(os.path.join(data_dir, f'{person_name}_{count//10}.jpg'), face_roi)

    # Display the resulting frame
    cv2.imshow('Face Capture', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()

# Write the names of persons to a file
with open(os.path.join(data_dir, 'names.txt'), 'w') as f:
    for name in person_names:
        f.write(name + '\n')
