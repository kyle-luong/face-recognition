import torch
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import os

# --- Enrollment (Adding Approved Faces) ---

# Initialize model and MTCNN
model = InceptionResnetV1(pretrained='casia-webface').eval()
mtcnn = MTCNN()

# Function to capture and enroll a face from the webcam
def enroll_face_from_camera(name):
    # Initialize the webcam
    cap = cv2.VideoCapture(2, cv2.CAP_AVFOUNDATION) # Change based on device

    print("Press 's' to capture a photo for enrollment or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Display the frame
        cv2.imshow("Enrollment - Press 's' to save", frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Press 's' to save the current frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Detect face and generate embedding
            face = mtcnn(img)
            if face is not None:
                embedding = model(face.unsqueeze(0)).detach().numpy()
                
                # Create the 'embeddings' directory if it doesn't exist
                if not os.path.exists('embeddings'):
                    os.makedirs('embeddings')
                
                # Generate a unique filename for each enrollment
                filename = f'embeddings/{name}.npy'
                file_index = 1
                while os.path.exists(filename):
                    filename = f'embeddings/{name}_{file_index}.npy'
                    file_index += 1
                
                # Save the embedding with the unique filename
                np.save(filename, embedding)
                print(f"Face enrolled for {name}. Saved as {filename}")
            else:
                print("No face detected. Please try again.")
            break  # Exit after saving the face

        elif key == ord('q'):  # Press 'q' to quit without saving
            print("Enrollment canceled.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Usage: Enroll a face using the webcam
enroll_face_from_camera('Kyle')
