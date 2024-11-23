import torch
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import os

# --- FaceID Verification ---

# Initialize Face Recognition Model and MTCNN
model = InceptionResnetV1(pretrained='vggface2').eval()  # Load InceptionResnetV1 model
mtcnn = MTCNN()  # Initialize MTCNN for face detection

# Load approved face embeddings (modify for your database)
approved_embeddings = {}
import os
for filename in os.listdir('embeddings'):
    if filename.endswith('.npy'):
        name = filename[:-4]  # Remove '.npy' extension
        embedding = np.load(os.path.join('embeddings', filename))
        approved_embeddings[name] = embedding

# Initialize webcam
cap = cv2.VideoCapture(2, cv2.CAP_AVFOUNDATION) # Change based on device
threshold = 1.0  # Adjust threshold as needed
count = 0
face_recognized = False  # Flag to track if a face was recognized

while count < 5 and not face_recognized:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from camera.")
        break  # Exit the loop if frame is not read

    # Convert frame to RGB for face detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        # Detect faces in the frame
        boxes, _ = mtcnn.detect(Image.fromarray(frame_rgb))
        
        if boxes is not None:
            for box in boxes:
                # Crop and preprocess the face
                x1, y1, x2, y2 = map(int, box)
                face = frame_rgb[y1:y2, x1:x2]
                face = Image.fromarray(face)

                # Get embedding for the detected face
                face_embedding = model(mtcnn(face).unsqueeze(0)).detach().numpy()

                # Compare face embedding with each approved embedding
                for name, approved_embedding in approved_embeddings.items():
                    distance = np.linalg.norm(face_embedding - approved_embedding)
                    if distance < threshold:
                        print(f"Face matched: {name} (Distance: {distance:.2f})")
                        face_recognized = True
                        count = 0  # Reset count on successful recognition
                        break  # Exit the embedding loop on match

                if face_recognized:
                    break  # Exit the boxes loop if a match is found

                # Draw bounding box around detected face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Increment count if no faces were recognized
            if not face_recognized:
                print("Face not recognized.")
                count += 1

    except Exception as e:
        print(f"Error: {e}")

    # Display the video feed
    cv2.imshow('FaceID', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Final message based on recognition status
if face_recognized:
    print("Access granted: Face recognized.")
elif count >= 5:
    print("Access denied: Too many unrecognized attempts.")