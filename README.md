# Geometry-Guided AI vs Real Video Detection System

A machine learning-based video classification system that detects whether a given video is **AI-generated** or **Real** using DINOv2 ViT-B/14 visual embeddings, temporal geometric features, and a Random Forest classifier.

---

## Project Description

The **Geometry-Guided AI vs Real Video Detection System** analyzes the temporal behavior of video frames in a learned visual feature space.

Instead of classifying each frame independently, the system extracts a visual embedding from every video frame using **DINOv2 ViT-B/14**.

The sequence of frame embeddings is then treated as a trajectory in the embedding space. Temporal and geometric properties of this trajectory are extracted to construct a **250-dimensional feature vector**.

Finally, a trained **Random Forest classifier** uses these features to determine whether the input video is:

- `AI`
- `Real`

---

## Objective

The objective of this project is to investigate whether the **temporal geometry of learned visual embeddings** can be used as a signal for distinguishing AI-generated videos from real videos.

The system focuses on how the representation of a video changes over time rather than relying only on spatial artifacts present in individual frames.

---

# Overall Pipeline

```text
                         Input Video
                              |
                              v
                    +-------------------+
                    | Frame Extraction  |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | DINOv2 ViT-B/14   |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | Frame Embeddings  |
                    +---------+---------+
                              |
                              v
              +-------------------------------+
              | Temporal / Geometric Analysis |
              +---------------+---------------+
                              |
                              v
                    +-------------------+
                    | 250-D Feature     |
                    | Vector            |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | Random Forest     |
                    | Classifier        |
                    +---------+---------+
                              |
                              v
                         AI / Real
