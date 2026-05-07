# Project Guide: FoldNet

## 1. Project Overview
**FoldNet** is a unified deep learning framework designed to predict two critical structural features of proteins directly from their amino acid sequences:
1.  **Secondary Structure (1D)**: Predicting whether each residue is part of a Helix (H), Sheet (E), or Coil (C).
2.  **Contact Map (2D)**: Predicting which pairs of amino acids are spatially close (within 8Å) in 3D space.

By predicting these 2D features, FoldNet bridges the gap between a linear sequence and a full 3D structure (like AlphaFold), but in a much more lightweight and efficient way.

---

## 2. Team Responsibilities

### **Shubham: Data & Feature Engineering**
*   **Goal**: The "Data Provider".
*   **Tasks**: Download CB513 (Secondary Structure) and PDB (Contact Maps). Parse them into clean tensors.
*   **Key Deliverable**: A `ProteinDataset` that provides sequences, labels, and **ESM-2 Embeddings** (1280-dimensional vectors that capture evolutionary information).

### **Shivam: Machine Learning Models (Our Focus)**
*   **Goal**: The "Brain Builder".
*   **Tasks**: Implement the encoders (CNN, BiLSTM, Transformer) and the prediction heads.
*   **Key Deliverable**: `foldnet_model.py` and the multi-task training loop.

### **Tanishka: Evaluation & Visualization**
*   **Goal**: The "Judge".
*   **Tasks**: Implement metrics like Q3 Accuracy and Precision@L. Create beautiful heatmaps for contact maps.
*   **Key Deliverable**: `evaluate.py` and benchmark reports comparing different architectures.

### **Vaibhav: Integration & Tracking**
*   **Goal**: The "Glue".
*   **Tasks**: Write the master `run_experiment.py` script. Set up experiment tracking (WandB).
*   **Key Deliverable**: A fully runnable pipeline and deployment scripts (FastAPI/ONNX).

---

## 3. Shivam's Work Plan (Detailed)
We will focus on these components sequentially:

1.  **Base Encoders**:
    *   **1D CNN**: Fast local pattern recognition.
    *   **BiLSTM**: Captures sequential context.
    *   **Transformer**: Global attention for long-range interactions.
2.  **Multi-Task Heads**:
    *   **SS Head**: Simple classifier for each residue.
    *   **Contact Head**: This is the hardest part. It converts 1D sequences into a 2D matrix using pairwise concatenation and refines it with a 2D Residual CNN.
3.  **Multi-Task Loss**:
    *   Balancing $\mathcal{L}_{SS}$ and $\mathcal{L}_{contact}$ so the model learns both tasks equally well.
4.  **Training Loop**:
    *   Implementing AdamW with learning rate warm-up to ensure stable training on your GPU.

---

## 4. Hardware Compatibility Report
**Device**: HP Omen 16 (RTX 4050 6GB, Ryzen 7000 HS)

### **Can it run? YES.**
However, 6GB of VRAM requires specific configurations to avoid "Out of Memory" (OOM) errors:

| Component | VRAM Impact | Optimization Strategy |
| :--- | :--- | :--- |
| **ESM-2 (650M)** | High (~5GB) | **Pre-compute offline**: We won't load the ESM-2 model during training; we only load the small embedding tensors. |
| **2D ResNet** | Medium | **Mixed Precision (FP16)**: Uses 16-bit floats instead of 32-bit, saving 50% memory. |
| **Batch Size** | High | **Small Batches + Accumulation**: We will use a batch size of 2-4 and accumulate gradients to simulate a batch of 32. |
| **Transformer** | Medium | **Layer Reduction**: We will use 4 layers instead of 12 for the encoder variant. |

### **Recommendation**
Your **Ryzen 7000 HS** is excellent for the preprocessing stage (Shubham's work). For our training (Shivam's work), we will prioritize **Efficiency** over brute-force model size.
