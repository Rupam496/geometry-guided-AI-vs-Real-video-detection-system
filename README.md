# AI vs Real Video Classifier

A video classification system that determines whether a video is **real** or **AI-generated** by analyzing the **temporal dynamics of visual embeddings**.

Instead of directly using raw pixel values, the system extracts semantic visual representations from video frames using **DINOv2** and then analyzes how these representations evolve over time. Temporal trajectory features are extracted from the embeddings and used to train a machine-learning classifier.

---

## Overview

AI-generated videos can look visually convincing at individual frames, making frame-level classification difficult.

This project approaches the problem from a **temporal perspective**.

The core idea is:

> Real and AI-generated videos may exhibit different temporal patterns in how their visual representations change from frame to frame.

For every input video, the system:

1. Extracts video frames.
2. Generates DINOv2 embeddings for each frame.
3. Arranges the embeddings in temporal order.
4. Computes trajectory-based temporal features.
5. Produces a fixed-length **250-dimensional feature vector**.
6. Uses a trained Random Forest classifier to predict:

   * `Real`
   * `AI`

## The deployed inference pipeline is implemented using **FastAPI** for embedding extraction and **Gradio** for the user-facing classification interface.

# System Architecture

```text
                    Input Video
                         │
                         ▼
                 ┌───────────────┐
                 │ Frame         │
                 │ Extraction    │
                 └───────┬───────┘
                         │
                         ▼
                Video Frames
                         │
                         ▼
                 ┌───────────────┐
                 │   DINOv2      │
                 │ ViT-B/14      │
                 └───────┬───────┘
                         │
                         ▼
              Frame-level Embeddings
                         │
                         ▼
                 ┌───────────────┐
                 │ Temporal      │
                 │ Feature       │
                 │ Extraction    │
                 └───────┬───────┘
                         │
                         ▼
                  250 Features
                         │
                         ▼
                 ┌───────────────┐
                 │ Random Forest │
                 │  Classifier    │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Prediction     │
                 │ Real / AI      │
                 └───────────────┘
```

---

# Key Idea

Let the embedding of frame \(t\) be:

$$
z_t \in \mathbb{R}^D
$$

where \(D\) is the DINOv2 embedding dimension.

The ordered embeddings form a trajectory:

$$
z_1,z_2,z_3,\ldots,z_T
$$

Instead of classifying individual frames, the project studies the **movement of this trajectory over time**.

## Embedding Velocity

The change between consecutive frame embeddings is:

$$
v_t=z_{t+1}-z_t
$$

This represents the movement of the embedding trajectory between two consecutive frames.

## Embedding Acceleration

The change in the movement is:

$$
a_t=v_{t+1}-v_t
$$

This captures how the temporal dynamics themselves change.

The project then derives several trajectory-based statistics from these quantities.

---

# Feature Engineering

The final representation contains **250 temporal features**.

The feature vector contains:

### 1. Distance Features

Distances between consecutive embeddings are calculated and incorporated into the feature representation.

Statistical properties of these distances are also computed:

* Mean
* Variance
* Minimum
* Maximum

---

### 2. Angular Features

Angular changes between consecutive embedding movements are calculated.

The feature representation contains:

* Up to 120 individual angle features
* Mean angle
* Variance of angles
* Minimum angle
* Maximum angle

The implementation pads the sequence when fewer than 120 values are available so that every sample has the same dimensionality.

---

### 3. Directional Sign Changes

The system counts changes in the direction/sign of the temporal movement.

This provides an additional measure of how frequently the embedding trajectory changes direction.

---

### 4. Temporal Smoothness Instability Score (TSIS)

The project also uses a temporal instability measure based on acceleration magnitude.

Conceptually:

$$
TSIS =
\frac{\operatorname{Var}(\|a_t\|)}
{\operatorname{Mean}(\|v_t\|)+\epsilon}
$$

where:

* \(v_t\) = embedding movement between consecutive frames
* \(a_t\) = change in embedding movement
* \(\|v_t\|\) = magnitude of embedding movement
* \(\|a_t\|\) = magnitude of embedding acceleration
* \(\epsilon\) = small constant for numerical stability

TSIS therefore measures the **variability of acceleration magnitude relative to the average embedding movement**.

A lower value indicates more consistent temporal dynamics, while a higher value indicates greater irregularity in the temporal trajectory.

---

# Why DINOv2?

Raw pixels are not ideal for this task.

Flattening an image into a pixel vector makes the representation highly sensitive to changes such as:

* Lighting
* Noise
* Small movements
* Pixel-level changes

DINOv2 instead produces a semantic visual representation that captures higher-level properties such as:

* Shapes
* Textures
* Visual structure
* Semantic information

Therefore, visually similar frames can have similar embeddings even when their individual pixels are not identical.

This is particularly useful because this project is interested in **how visual representations change over time**, rather than simply comparing raw pixels.

---

# Dataset

A custom dataset was created using:

* Real human videos
* AI-generated synthetic videos

Long-duration videos were divided into fixed-length **5-second clips**.

The final dataset contained:

| Property           |  Value |
| ------------------ | -----: |
| Total samples      | 14,979 |
| Real samples       |  8,018 |
| AI samples         |  6,961 |
| Feature dimension  |    250 |
| Training samples   | 10,185 |
| Validation samples |  1,798 |
| Test samples       |  2,996 |

The dataset was constructed by extracting frames from the clips, generating embeddings, computing temporal features, handling invalid values, and storing the resulting fixed-length feature vectors with their labels.

---

# Machine Learning Models

Several machine-learning models were evaluated:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* Multi-Layer Perceptron (MLP)
* XGBoost

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

The Random Forest model achieved:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.8828 |
| Precision | 0.8668 |
| Recall    | 0.8836 |
| F1-score  | 0.8751 |
| ROC-AUC   | 0.9449 |

The project report also evaluates XGBoost and other baseline models for comparison.

---

# Inference Pipeline

For a new video, the deployed classifier follows this pipeline:

```text
Video
  │
  ├──► Extract Frames
  │
  ├──► DINOv2 Embeddings
  │
  ├──► Temporal Feature Extraction
  │
  ├──► 250-D Feature Vector
  │
  ├──► Random Forest
  │
  └──► Real / AI
```

The current inference implementation loads DINOv2 once, extracts frames using OpenCV, generates embeddings using the pretrained `facebook/dinov2-base` model, and puts the embeddings into the temporal feature-generation pipeline.

---

# Technologies Used

### Deep Learning

* PyTorch
* Hugging Face Transformers
* DINOv2

### Computer Vision

* OpenCV

### Machine Learning

* Scikit-learn
* Random Forest
* Logistic Regression
* SVM
* MLP
* XGBoost

### Backend

* FastAPI


---



# FastAPI Embedding Service

The project also contains a FastAPI service that accepts a video and returns its frame-level DINOv2 embeddings.

Start the server with:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API endpoint is:

```text
POST /embeddings
```

The request expects a video file.

The response contains:

```json
{
    "filename": "example.mp4",
    "num_frames": 100,
    "embedding_dimension": 768,
    "embeddings": [...]
}
```

The current implementation uses the pretrained:

```text
facebook/dinov2-base
```

model and extracts the CLS token from the final hidden state as the frame representation.

---

# API Workflow

```text
Client
  │
  │ POST /embeddings
  │
  ▼
FastAPI
  │
  ▼
Save uploaded video
  │
  ▼
OpenCV
  │
  ▼
Extract frames
  │
  ▼
DINOv2
  │
  ▼
Frame embeddings
  │
  ▼
JSON response
```

---

# Example Prediction

After uploading a video:

```text
Input:
example_video.mp4

Processing:
Number of frames: 120

Embedding shape:
(120, 768)

Feature shape:
(250,)

Prediction:
Real
```

The classifier converts class `0` to `Real` and class `1` to `AI`.








---
