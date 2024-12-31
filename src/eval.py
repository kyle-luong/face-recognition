#!/usr/bin/env python3
"""
eval.py

Evaluates face recognition performance on the LFW (Labeled Faces in the Wild) dataset
using facenet-pytorch (MTCNN + InceptionResnetV1). Compares multiple distance metrics
(Euclidean, Cosine, Manhattan) across different thresholds to determine the best match accuracy.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.datasets import fetch_lfw_people
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

########################################
# Configuration
########################################

# Check for GPU availability
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Initialize Face Detection Model (MTCNN) and Face Recognition Model (InceptionResnetV1)
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Distance metrics to compare
DISTANCE_METRICS = ["euclidean", "cosine", "manhattan"]

########################################
# Load LFW Dataset
########################################]
# Fetch images per person >= 20 to reduce noise
lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=True)

X = lfw_people.images   # face images with shape: (n_samples, height, width, 3)
y = lfw_people.target   # identity labels
names = lfw_people.target_names

print(f"Loaded LFW dataset: {X.shape[0]} images, each size {X.shape[1:4]}")

########################################
# Preprocessing & Embedding Extraction
########################################

def get_embedding(image_np):
    """
    Convert NumPy image to PIL, detect face with MTCNN, and compute embedding
    using InceptionResnetV1. Returns None if face detection fails.
    """
    # Normalize to [0, 255] if in [0, 1]
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)

    # Detect and align face
    face = Image.fromarray(image_np, 'RGB')  # NumPy to PIL
    face = mtcnn(face)
    if face is None: 
        return None
    
    # Compute face embedding
    face_embeddings = face.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(face_embeddings).cpu().numpy().flatten()  # generated embedding numpy shape: (512,)
    return embedding

embeddings = []
labels = []

for img_np, label_id in tqdm((zip(X, y)), total=len(X), desc="Generating embeddings"):
    emb = get_embedding(img_np)
    if emb is not None:
        embeddings.append(emb)
        labels.append(label_id)

embeddings = np.array(embeddings)  # shape: (num_valid_samples, 512)
labels = np.array(labels)

print(f"Created {embeddings.shape[0]} embeddings.")

########################################
# Create Positive & Negative Pairs
########################################

def create_pairs(embs, lbls, n_pairs=2000):
    """
    Create (embedding1, embedding2, label) pairs.
    label=1 if same person, else 0.
    """
    rng = np.random.default_rng(seed=42)
    pairs = []
    unique_labels = np.unique(lbls)

    # Group embeddings by label
    label_to_indices = {}
    for idx, lab in enumerate(lbls):
        label_to_indices.setdefault(lab, []).append(idx)

    # Sample half positive pairs, half negative pairs
    half = n_pairs // 2

    # Positive pairs
    for _ in range(half):
        lab = rng.choice(unique_labels)
        indices = label_to_indices[lab]
        if len(indices) < 2:
            continue
        i1, i2 = rng.choice(indices, 2, replace=False)
        pairs.append((embs[i1], embs[i2], 1))

    # Negative pairs
    for _ in range(half):
        lab1, lab2 = rng.choice(unique_labels, 2, replace=False)
        i1 = rng.choice(label_to_indices[lab1])
        i2 = rng.choice(label_to_indices[lab2])
        pairs.append((embs[i1], embs[i2], 0))

    rng.shuffle(pairs)
    return pairs

pairs = create_pairs(embeddings, labels, n_pairs=2000)
print(f"Generated {len(pairs)} pairs for evaluation.")

########################################
# Define Distance Metrics
########################################

def compute_distance(emb1, emb2, metric="euclidean"):
    """
    Computes distance between two embeddings using the given metric.
    """
    if metric == "euclidean":
        return np.linalg.norm(emb1 - emb2)

    elif metric == "cosine":
        # Convert cosine similarity to a distance: 0 means identical; 2 means opposite.
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        cos_sim = dot / (norm1 * norm2 + 1e-8)
        return 1.0 - cos_sim

    elif metric == "manhattan":
        return np.sum(np.abs(emb1 - emb2))

    else:
        raise ValueError(f"Unknown metric: {metric}")

########################################
# Evaluate Distance Metrics & Thresholds
########################################

def evaluate_metric(pairs, metric, thresholds):
    """
    For each metric, compute distance between embeddings in pairs
    and evaluate match accuracy over a range of thresholds.
    Returns best accuracy, best threshold, and (threshold, accuracy) pairs.
    """
    distances = []
    labels_ = []
    for emb1, emb2, label in pairs:
        dist = compute_distance(emb1, emb2, metric=metric)
        distances.append(dist)
        labels_.append(label)

    labels_ = np.array(labels_)

    best_acc = 0.0
    best_thresh = 0.0
    results = []

    for thr in thresholds:
        # Predict same person if distance < threshold.
        preds = [1 if d < thr else 0 for d in distances]
        acc = np.mean(labels_ == preds)
        results.append((thr, acc))
        if acc > best_acc:
            best_acc = acc
            best_thresh = thr

    return best_acc, best_thresh, np.array(results)

thresholds = np.arange(0.0, 2.0, 0.02)  # (start, stop, step)

print("Evaluating metrics:")
for metric in DISTANCE_METRICS:
    best_acc, best_thr, curve = evaluate_metric(pairs, metric, thresholds)
    print(f"Metric={metric:10s}  Best Acc={best_acc:.3f} at thr={best_thr:.2f}")

########################################
# Plot Accuracy vs. Threshold
########################################
plt.figure(figsize=(8,6))

for metric in DISTANCE_METRICS:
    _, _, curve = evaluate_metric(pairs, metric, thresholds)
    plt.plot(curve[:,0], curve[:,1], label=metric)

plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Threshold for Distance Metrics on InceptionResnetV1 Embeddings")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Done.")
