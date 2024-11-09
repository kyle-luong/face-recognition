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

# Function to enroll a new face
def enroll_face(image_path, name):
    img = Image.open(image_path)
    face = mtcnn(img)
    if face is not None:
        embedding = model(face.unsqueeze(0)).detach().numpy()
        # Create the 'embeddings' directory if it doesn't exist
        if not os.path.exists('embeddings'):
            os.makedirs('embeddings')
        # Save embedding with name (you can modify this for your database)
        np.save(f'embeddings/{name}.npy', embedding)
        print(f"Face enrolled for {name}")
    else:
        print("No face detected in the image.")

# Usage: Enroll a face
enroll_face('IMG_4683.jpeg', 'Kyle')