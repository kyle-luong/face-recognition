#!/usr/bin/env python3
"""
enroll.py

Enrolls a face via webcam to the embeddings folder.
"""

import os
import torch
import cv2
import numpy as np

from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

########################################

# Check for GPU availability
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Initialize Face Recognition Model (InceptionResnetV1) and Face Detection Model (MTCNN)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN()

# Function to capture and enroll a face from the webcam
def enroll_face(name):
    cap = cv2.VideoCapture(2, cv2.CAP_AVFOUNDATION) # Webcam setup (MacOS)

    print("Press 's' to capture a photo for enrollment or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        cv2.imshow("Enrollment - Press 's' to save", frame) # Display the frame
        key = cv2.waitKey(1) & 0xFF # Check for key presses
        
        if key == ord('s'):  # Save the current frame when 's' is pressed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            face = mtcnn(img) # Detect and align face
            if face is None:  # Handle detection failure
                print("No face detected. Please try again.")
                continue

            # Generate and save embedding
            embedding = model(face.unsqueeze(0)).detach().numpy()
            os.makedirs('embeddings', exist_ok=True)
            np.save(f'embeddings/{name}.npy', embedding)
            print(f"Face enrolled for {name}.")
            break

        elif key == ord('q'):  # Exit without saving when 'q' is pressed
            print("Enrollment canceled.")
            break

     # Release webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Direct enroll_face() usage
if __name__ == "__main__":
    person_name = input("Enter the name of the person to enroll: ")
    enroll_face(person_name)