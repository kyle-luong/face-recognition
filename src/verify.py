import torch
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import os
import time

# --- Face Verification ---

# Initialize Face Recognition Model (InceptionResnetV1) and Face Detection Model (MTCNN)
model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN()

# Load approved face embeddings
def load_embeddings(embeddings_path="embeddings"):
    embeddings = {}
    for filename in os.listdir(embeddings_path):
        if filename.endswith('.npy'):
            name = filename[:-4]  # Remove '.npy' extension
            embedding = np.load(os.path.join(embeddings_path, filename))
            embeddings[name] = embedding
    return embeddings

# Recognize faces in a given frame
def recognize_face(frame, model, mtcnn, approved_embeddings, threshold=1):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    boxes, _ = mtcnn.detect(Image.fromarray(frame_rgb))  # Detect faces
    
    results = []  # Store results for drawing
    recognized = False  # Flag to indicate recognition success
    
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinate
            face = frame_rgb[y1:y2, x1:x2]  # Crop face
            face = Image.fromarray(face)

            face = mtcnn(face) # Detect and align face
            if face is None:  # Handle detection failure
                print("No face detected. Please try again.")
                continue
            
            # Generate the face embedding
            face_embedding = model(face.unsqueeze(0)).detach().numpy()

            # Compare with approved embeddings
            matched_name = None
            for name, approved_embedding in approved_embeddings.items():
                distance = np.linalg.norm(face_embedding - approved_embedding)
                if distance < threshold:
                    print(f"Face matched: {name} (Distance: {distance:.2f})")
                    matched_name = name
                    recognized = True
                    break
                
             # Determine result
            if recognized:
                box_color = (0, 255, 0)  # Green for recognized
                label = f"Recognized: {matched_name}"
            else:
                box_color = (0, 0, 255)  # Red for not recognized
                label = "Not Recognized"
                
            results.append((x1, y1, x2, y2, label, box_color))
            
    return results, recognized

# Verify faces using webcam feed
def verify_face():
    approved_embeddings = load_embeddings() # Load embeddings
    cap = cv2.VideoCapture(2, cv2.CAP_AVFOUNDATION) # Initialize webcam (works for MacOS)
    
    start_time = time.time()
    timeout = 5  # Maximum duration in seconds
    face_recognized = False  # Flag to track recognition status

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break  # Exit loop if no frame is read

        # Detect and recognize faces in the current frame
        results, recognized = recognize_face(frame, model, mtcnn, approved_embeddings)

        # Draw bounding boxes and labels on the frame
        for (x1, y1, x2, y2, label, box_color) in results:
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Update recognition status
        if recognized:
            face_recognized = True

        # Check for timeout
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            print("Timeout reached. Exiting...")
            break

        # Display the video feed
        cv2.imshow('FaceID', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
            print("Manual exit triggered.")
            break

    # Release webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Final message based on recognition status
    if face_recognized:
        print("Access granted: Face recognized.")
    else:
        print("Access denied: Face not recognized.")
            
# Direct verify_face() usage
if __name__ == "__main__":
    verify_face()