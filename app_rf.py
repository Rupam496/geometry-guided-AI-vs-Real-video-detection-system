import os
import tempfile
import requests
import joblib
import numpy as np

from fastapi import FastAPI, UploadFile, File


# ============================================================
# Configuration
# ============================================================

APP_URL = "http://127.0.0.1:8000/embeddings"
MODEL_PATH = "model.pkl"


# ============================================================
# Load model
# ============================================================

model = joblib.load(MODEL_PATH)

app = FastAPI()


# ============================================================
# Feature functions
# ============================================================

def angles_extractor(embeddings):

    angles = []

    for t in range(1, len(embeddings) - 1):

        v1 = embeddings[t] - embeddings[t - 1]
        v2 = embeddings[t + 1] - embeddings[t]

        cos_theta = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)
        )

        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        theta = np.arccos(cos_theta) * (180.0 / np.pi)

        angles.append(theta)

    return angles


def compute_distance_stats(embeddings):

    if embeddings.shape[0] < 2:
        return 0, 0, 0, 0, np.zeros(1)

    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    deltas = embeddings[1:] - embeddings[:-1]

    # Distance between consecutive embeddings
    d = np.linalg.norm(deltas, axis=1)

    mu_d = np.mean(d)
    min_d = np.min(d)
    max_d = np.max(d)
    var_d = np.var(d)

    return mu_d, min_d, max_d, var_d, d


def compute_tsis_from_embeddings(embeddings, eps=1e-8):

    z = np.array(embeddings)

    if len(z) < 3:
        return 0.0

    v = z[1:] - z[:-1]

    a = v[1:] - v[:-1]

    v_norm = np.linalg.norm(v, axis=1)
    a_norm = np.linalg.norm(a, axis=1)

    return np.var(a_norm) / (
        np.mean(v_norm) + eps
    )


def count_sign_changes(embeddings, eps=1e-8):

    v = embeddings[1:] - embeddings[:-1]

    dots = np.sum(
        v[:-1] * v[1:],
        axis=1
    )

    s = np.sign(dots)

    s[np.abs(dots) < eps] = 0

    sign_changes = (
        s[:-1] != s[1:]
    ).astype(int)

    return int(np.sum(sign_changes))


# ============================================================
# Create 250 features
# ============================================================

def create_features(embeddings):

    embeddings = np.asarray(embeddings)

    # Angles
    angles = angles_extractor(
        list(embeddings)
    )

    if len(angles) == 0:
        angles = [0.0]

    # TSIS
    tsis = compute_tsis_from_embeddings(
        embeddings
    )

    # Distance statistics
    mu_d, min_d, max_d, var_d, distances = (
        compute_distance_stats(embeddings)
    )

    distances = list(distances.flatten())

    # Sign changes
    sign_changes = count_sign_changes(
        embeddings
    )

    features = []

    # X0 - X119 : distances
    for i in range(120):

        if i < len(distances):
            features.append(distances[i])
        else:
            features.append(0)

    # X120 - X239 : angles
    for i in range(120):

        if i < len(angles):
            features.append(angles[i])
        else:
            features.append(0)

    # X240 - X243 : angle statistics
    features.append(np.nanmean(angles))
    features.append(np.nanvar(angles))
    features.append(np.nanmin(angles))
    features.append(np.nanmax(angles))

    # X244 - X247 : distance statistics
    features.append(mu_d)
    features.append(var_d)
    features.append(min_d)
    features.append(max_d)

    # X248 : sign changes
    features.append(sign_changes)

    # X249 : TSIS
    features.append(tsis)

    return np.array(features)


# ============================================================
# Send video to app.py
# ============================================================

def get_embeddings(video_path):

    with open(video_path, "rb") as f:

        files = {
            "video": f
        }

        response = requests.post(
            APP_URL,
            files=files
        )

    response.raise_for_status()

    data = response.json()

    return np.array(data["embeddings"])


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ) as temp:

        temp.write(await file.read())

        video_path = temp.name

    try:

        # ----------------------------------------------------
        # 1. Get embeddings from app.py
        # ----------------------------------------------------

        embeddings = get_embeddings(video_path)

        # ----------------------------------------------------
        # 2. Create 250 features
        # ----------------------------------------------------

        features = create_features(
            embeddings
        )

        # ----------------------------------------------------
        # 3. Random Forest prediction
        # ----------------------------------------------------

        prediction = model.predict(
            features.reshape(1, -1)
        )[0]

        # 0 = Real
        # 1 = AI

        if prediction == 0:
            result = "Real"
        else:
            result = "AI"

        return {
            "prediction": result
        }

    finally:

        os.remove(video_path)
