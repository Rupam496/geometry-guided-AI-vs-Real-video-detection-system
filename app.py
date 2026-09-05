import os
import tempfile

import cv2
import numpy as np
import torch

from fastapi import FastAPI, UploadFile, File
from transformers import AutoImageProcessor, AutoModel


app = FastAPI()


# ============================================================
# Device
# ============================================================

device = torch.device("cpu")

print(f"Using device: {device}")


# ============================================================
# Load DINOv2 ONCE
# ============================================================

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)

model = AutoModel.from_pretrained(
    "facebook/dinov2-base"
)

model = model.to(device)
model.eval()


# ============================================================
# Extract frames
# ============================================================

def frames_extractor(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame is None or frame.size == 0:
            continue

        # BGR → RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(frame)

    cap.release()

    return frames


# ============================================================
# Generate embeddings
# ============================================================

def embeddings_extractor(frames):

    embeddings = []

    with torch.no_grad():

        for frame in frames:

            inputs = processor(
                images=frame,
                return_tensors="pt"
            )

            # CPU
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            outputs = model(**inputs)

            # CLS token
            embedding = outputs.last_hidden_state[:, 0, :]

            embedding = (
                embedding
                .squeeze(0)
                .cpu()
                .numpy()
            )

            embeddings.append(embedding)

    return embeddings


# ============================================================
# API endpoint
# ============================================================

@app.post("/embeddings")
def get_embeddings(
    video: UploadFile = File(...)
):

    # Save uploaded video
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(video.filename)[1]
    ) as temp:

        temp.write(video.file.read())
        video_path = temp.name

    try:

        # Video → frames
        frames = frames_extractor(video_path)

        # Frames → embeddings
        embeddings = embeddings_extractor(frames)

        # NumPy → JSON-compatible lists
        embeddings = [
            embedding.tolist()
            for embedding in embeddings
        ]

        return {
            "filename": video.filename,
            "num_frames": len(frames),
            "embedding_dimension": (
                len(embeddings[0])
                if embeddings
                else 0
            ),
            "embeddings": embeddings
        }

    finally:

        os.remove(video_path)