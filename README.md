# AI vs Real Video Detection System

A video classification system for detecting whether a video is **AI-generated** or **real** using visual feature representations from **DINOv2 ViT-B/14** and temporal geometry-based features classified using a **Random Forest** model.

The system analyzes the temporal behavior of video-frame embeddings in the DINOv2 feature space rather than relying only on individual-frame appearance.

---

## Overview

Recent advances in generative AI have made it increasingly difficult to distinguish AI-generated videos from real videos.

This project explores a geometry-guided approach to video detection. Each video frame is converted into a high-dimensional visual representation using **DINOv2 ViT-B/14**. The sequence of embeddings is then analyzed in feature space to capture temporal characteristics of the video.

The resulting temporal features are used by a trained Random Forest classifier to predict whether the input video is:

- **Real**
- **AI-generated**

### Pipeline

```text
                    Input Video
                         │
                         ▼
                  Video Frame Extraction
                         │
                         ▼
                  DINOv2 ViT-B/14
                         │
                         ▼
                Frame-level Embeddings
                         │
                         ▼
              Temporal Geometry Analysis
                         │
            ┌────────────┼────────────┐
            │            │            │
            ▼            ▼            ▼
        Distances      Angles       TSIS
            │            │            │
            └────────────┼────────────┘
                         │
                         ▼
                 250-D Feature Vector
                         │
                         ▼
                  Random Forest
                         │
                         ▼
                    AI / Real
