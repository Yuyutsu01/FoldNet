# FoldNet: Comprehensive Research Defensibility & Technical Viva Handbook

This handbook serves as the ultimate study guide and documentation for FoldNet. It bridges theory, biology, mathematical derivations, software engineering, and line-by-line PyTorch implementation to prepare you for any research committee, technical interview, or viva presentation.

---

## Chapter 1: Vision, Motivation, and Core Biology

### 1.1 Why FoldNet Exists
A protein's biological function is determined by its three-dimensional tertiary structure ($\text{3D}$ shape). However, while determining the amino acid sequence of a protein is cheap and fast (via DNA sequencing), determining its experimental $3\text{D}$ structure (via X-ray crystallography, NMR spectroscopy, or Cryo-EM) is extremely expensive, time-consuming (often taking months), and labor-intensive. 

This gap between known sequences and known structures is the **Protein Folding Problem**. FoldNet solves a critical subset of this problem: predicting **Secondary Structure** (local 1D conformations) and **Contact Maps** (2D pairwise spatial relationships) directly from amino acid sequences without relying on expensive physical experiments.

### 1.2 Limitations of Alternative Methods
*   **AlphaFold / AlphaFold2**: While highly accurate, AlphaFold is computationally heavy. It relies heavily on constructing Multiple Sequence Alignments (MSAs) from massive sequence databases (like BFD, UniRef), which can take minutes to hours per query, making real-time interactive inference impossible.
*   **ESMFold / OmegaFold**: Although they bypass MSAs using protein language models (PLMs) to predict coordinates, their fully-autoregressive or frame-matching structure modules have a massive memory footprint, scaling quadratically $O(L^2)$ or cubically $O(L^3)$ in memory and time.
*   **FoldNet Solution**: By framing structure prediction as a multi-task learning task (1D secondary classification and 2D contact mapping) using frozen PLM embeddings, FoldNet provides instantaneous inference in under 30ms, making it ideal for web-based dashboard visualizers and high-throughput virtual screening.

### 1.3 Why 1D Secondary Structure + 2D Contact Maps?
Instead of predicting raw 3D Cartesian coordinates (which requires dealing with translation and rotation invariance, and is highly sensitive to coordinate frame alignment), FoldNet predicts **2D contact maps**. 
*   **Rotational and Translational Invariance**: A 2D contact matrix is invariant to global rotations and translations of the protein.
*   **Constraint Satisfaction**: 2D contact maps represent physical distance constraints. These constraints can be directly fed into classical distance geometry algorithms (like CNS or Rosetta) to reconstruct the physical 3D backbone.
*   **Multi-Task Synergy**: Training secondary structure classification and contact mapping in parallel forces the hidden representations to capture both local folding patterns (helices/strands) and global topology (contacts).

### 1.4 Biological Hierarchy Checklist
1.  **Primary Structure**: The linear sequence of amino acids linked by covalent peptide bonds.
2.  **Secondary Structure**: Local structural conformations stabilized by hydrogen bonds between backbone amine ($\text{N-H}$) and carbonyl ($\text{C=O}$) groups.
    *   **Alpha-Helix ($\text{H}$)**: A right-handed spiral where hydrogen bonds form every $i \rightarrow i+4$ residues.
    *   **Beta-Sheet ($\text{E}$)**: Parallel or anti-parallel sheets of peptide strands stabilized by lateral hydrogen bonds.
    *   **Coil ($\text{C}$)**: Irregular, flexible loops connecting helices and sheets.
3.  **Tertiary Structure**: The overall $3\text{D}$ folding of a single polypeptide chain, driven by hydrophobic interactions, disulfide bonds, and electrostatic attractions.
4.  **Quaternary Structure**: The spatial arrangement of multiple polypeptide subunits.
5.  **C$\alpha$ and C$\beta$ Atoms**: The central carbon of each amino acid backbone is the alpha carbon ($\text{C}_\alpha$). The side chain branches from the beta carbon ($\text{C}_\beta$, except for Glycine which has only a hydrogen atom). Distance measurements for contact maps are typically taken between $\text{C}_\beta$ (or $\text{C}_\alpha$ for Glycine) atoms to represent side-chain orientation.
6.  **RMSD (Root-Mean-Square Deviation)**: The measure of the average distance between the atoms of two superimposed structures:
    $$\text{RMSD} = \sqrt{\frac{1}{N}\sum_{i=1}^N \|x_i - y_i\|^2}$$
7.  **TM-score (Template Modeling Score)**: A length-independent metric for measuring structural similarity. Ranging from 0 to 1, a TM-score $> 0.5$ generally indicates that the proteins share the same fold.
8.  **DSSP (Define Secondary Structure of Proteins)**: The gold-standard algorithm that parses atomic $3\text{D}$ coordinates from PDB files and assigns secondary structure labels based on hydrogen-bonding energy heuristics:
    $$E = q_1 q_2 \left( \frac{1}{r_{ON}} + \frac{1}{r_{CH}} - \frac{1}{r_{OH}} - \frac{1}{r_{CN}} \right) \times 332 \text{ kcal/mol}$$
    A hydrogen bond is assigned if $E < -0.5 \text{ kcal/mol}$.

---

## Chapter 2: The Dataset & Data Preprocessing

### 2.1 Dataset Sources & Properties
*   **CullPDB**: Used for model training. It is a high-resolution, non-redundant subset of the Protein Data Bank (PDB) curated by the PISCES server.
*   **CB513**: The gold-standard benchmark test set containing 514 non-redundant proteins, used to measure final generalization performance.
*   **Sequence Redundancy Control**: To prevent data leakage, training sequences in CullPDB are filtered to have **less than 25% sequence identity** with any sequence in the CB513 test set. This ensures that the model cannot achieve high scores through memorization of homologous sequences.

| Metric / Property | CullPDB Dataset | CB513 Dataset |
| :--- | :--- | :--- |
| **Total Proteins** | ~5,926 | 514 |
| **Min / Max Length** | 30 / 700 | 30 / 700 |
| **Average Sequence Length** | ~200 residues | ~190 residues |
| **Sequence Identity Limit** | < 25% | Test set baseline |

### 2.2 Data Preprocessing Pipeline

The step-by-step pipeline transforms raw PDB structural data into static features for training:

```
[Raw PDB Files] 
       │
       ▼ (Bio.PDB Parsing)
[Extract Atom Coordinates] ──► Generate DSSP Labels ──► Classify 3-Class Q3 (H, E, C)
       │
       ▼ (Pairwise L2 Distances)
[Distance Matrix] ──► Threshold at <= 8.0 Å ──► Mask sequence separation < 6 residues ──► [2D Contact Map]
       │
       ▼ (Meta ESM-2 Inference)
[Per-Residue 1280-dim Embeddings] ──► Save as NumPy arrays (.npy)
       │
       ▼
[DataLoader Collation] ──► Pad variable lengths to max batch length ──► Apply Loss Masking.
```

#### Preprocessing Details:
1.  **Coordinate Extraction**: Coordinates for all standard amino acids are extracted. We target the $\text{C}_\beta$ atom (or $\text{C}_\alpha$ for Glycine) to represent side-chain centers.
2.  **Contact Map Generation**: Pairwise Euclidean distances are computed. Cells with distance $\le 8.0 \text{ \AA}$ are labeled `1.0` (contacts), others `0.0`.
3.  **Local Contact Masking**: Residues close in primary sequence sequence-wise ($|i - j| < 6$) are forced to `0.0`. These local contacts represent trivial alpha-helical loops; masking them forces the model to focus on non-trivial tertiary folding patterns.
4.  **Embedding Extraction**: The sequences are passed through the frozen **ESM-2** model to output $(L, 1280)$ embeddings.

---

## Chapter 3: Deep Learning Architecture & Mathematical Formulations

FoldNet supports three interchangeable sequence encoders: a 1D Residual CNN, a Bidirectional LSTM, and a Multi-Head Transformer.

```
Input Features (batch, L, 1280)
           │
     ┌─────┴────────────────────────┐
     ▼                              ▼
Linear Projection (hidden_dim)    Conv1D (hidden_dim)
     │                              │
Positional Encoding                 │
     │                              │
TransformerEncoderLayer × 4      ResidualBlock1D × 5
     │                              │
     └─────┬────────────────────────┘
           │
           ├─► SecondaryStructureHead ──► Linear ──► SS Logits (batch, L, 3)
           │
           └─► ContactMapHead ──► Outer Concatenation ──► Conv2D ──► Contact Logits (batch, L, L)
```

### 3.1 1D Residual CNN Encoder
*   **Convolution Operations**:
    For an input tensor $X \in \mathbb{R}^{B \times d \times L}$ and kernel $W \in \mathbb{R}^{d \times d \times k}$, the 1D convolution at residue index $t$ is:
    $$\text{Conv1D}(X)_t = \sum_{j=1}^{d} \sum_{m=-k/2}^{k/2} X_{j, t+m} W_{j, m} + b$$
*   **Residual Blocks**:
    Protects gradients from vanishing. The block maps:
    $$Y = \text{ReLU}\left( \text{BN}\left( \text{Conv1D}\left( \text{ReLU}\left( \text{BN}\left( \text{Conv1D}(X) \right) \right) \right) \right) + X \right)$$

### 3.2 Bidirectional LSTM Encoder
*   **Recurrence Equations**:
    For each residue $t$, cell state $c_t$ and hidden state $h_t$ are updated using gates:
    $$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(Forget Gate)}$$
    $$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(Input Gate)}$$
    $$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \quad \text{(Candidate Cell State)}$$
    $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(State Update)}$$
    $$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(Output Gate)}$$
    $$h_t = o_t \odot \tanh(c_t) \quad \text{(Hidden Output)}$$
*   The bidirectional LSTM concatenates forward and backward passes: $h_t^{\text{bi}} = [\vec{h}_t \,\|\, \overleftarrow{h}_t]$.

### 3.3 Transformer Encoder
*   **Sinusoidal Positional Encodings**:
    Added to input embeddings to preserve residue order:
    $$\text{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad \text{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d}}\right)$$
*   **Self-Attention**:
    Queries ($Q$), Keys ($K$), and Values ($V$) are projected from sequence states:
    $$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$
    The scale factor $\sqrt{d_k}$ prevents dot products from growing too large in high dimensions, which would cause the softmax function to enter regions with extremely small gradients.

### 3.4 Contact Map Head (Outer Concatenation)
To translate 1D residue-level sequence embeddings into 2D pairwise interaction maps, FoldNet uses **Outer Concatenation**:
1.  Let $H \in \mathbb{R}^{B \times L \times d}$ be the latent state sequence.
2.  Expand $H$ column-wise to form $H_{\text{row}} \in \mathbb{R}^{B \times L \times L \times d}$ and row-wise to form $H_{\text{col}} \in \mathbb{R}^{B \times L \times L \times d}$.
3.  Concatenate them along the channel dimension:
    $$\mathbf{P}_{i,j} = [h_i \,\|\, h_j] \in \mathbb{R}^{B \times L \times L \times 2d}$$
4.  Permute to shape $(B, 2d, L, L)$ and apply 2D Residual convolutions to predict pairwise logits.

---

## Chapter 4: Multi-Task Losses and Optimization

### 4.1 Loss Functions & Masking
FoldNet is optimized using a weighted multi-task loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SS}} + \lambda_{\text{contact}} \mathcal{L}_{\text{contact}}$$

#### 1D Cross-Entropy Loss ($\mathcal{L}_{\text{SS}}$)
Evaluates secondary structure predictions. Padded residues are assigned a target label of `-1` and skipped using PyTorch's `ignore_index=-1` option:
$$\mathcal{L}_{\text{SS}} = - \frac{1}{N_{\text{valid}}} \sum_{i \in \text{valid}} \log \left( \frac{e^{z_{i, y_i}}}{\sum_{j=1}^3 e^{z_{i, j}}} \right)$$

*   **Mathematical Gradient Derivation**:
    Let $p_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$ be the softmax probability.
    *   If $j = y$ (the true target class):
        $$\frac{\partial \mathcal{L}_{\text{SS}}}{\partial z_y} = \frac{\partial}{\partial z_y} \left( -z_y + \log \sum_k e^{z_k} \right) = -1 + \frac{e^{z_y}}{\sum_k e^{z_k}} = p_y - 1$$
    *   If $j \neq y$ (non-target classes):
        $$\frac{\partial \mathcal{L}_{\text{SS}}}{\partial z_j} = \frac{\partial}{\partial z_j} \left( \log \sum_k e^{z_k} \right) = \frac{e^{z_j}}{\sum_k e^{z_k}} = p_j$$
    *   Thus, the gradient simplifies to $\mathbf{p} - \mathbf{y}$ (predicted probability vector minus the one-hot target vector).

#### 2D Masked Binary Cross-Entropy Loss ($\mathcal{L}_{\text{contact}}$)
To prevent padding regions in batched tensors from corrupting the contact map gradients, we compute a 2D Boolean mask:
$$\text{Mask}_{2\text{D}} = (1 - \text{Mask}_{\text{pad}}) \otimes (1 - \text{Mask}_{\text{pad}})^T$$
$$\text{where } \text{Mask}_{\text{pad}} \in \{0, 1\}^L \text{ flags padded residues.}$$
$$\mathcal{L}_{\text{contact}} = - \frac{1}{N_{\text{pairs}}} \sum_{i,j \in \text{Mask}_{2\text{D}}} \left[ y_{i,j} \log \sigma(z_{i,j}) + (1-y_{i,j}) \log(1 - \sigma(z_{i,j})) \right]$$

### 4.2 Optimization Details
*   **AdamW Weight Decay**: Weight decay is decoupled from the gradient updates, regularizing weights directly:
    $$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda_{\text{wd}} \theta_t \right)$$
*   **Warmup and Cosine Scheduler**:
    A 5-epoch linear warmup linearly increases the learning rate to prevent early training instability, followed by cosine annealing:
    $$\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\text{max}}}\pi\right)\right)$$
*   **Gradient Accumulation**: To fit effective large batches (size = 16) into limited VRAM, we accumulate gradients over 4 backward steps before calling `optimizer.step()`.

---

## Chapter 5: Complete Code Flow and Tensor Shape Tracing

### 5.1 Training Pipeline Code Flow
The standard execution pipeline follows:

```
run.py (Entry Point)
   │
   ▼ (YAML Parse)
configs/baseline_cnn.yaml
   │
   ▼ (Instantiate Dataloaders)
get_dataloaders() ──► ProteinDataset ──► collate_fn (Dynamic Padding)
   │
   ▼ (Init LightningModule)
FoldNet (LightningModule)
   │
   ▼ (Loop Epochs)
Trainer.fit() 
   │   │
   │   ├──► training_step() ──► Forward Pass ──► Loss Evaluation ──► backward() 
   │   │                                                                 │ (Accumulate 4 batches)
   │   │                                                                 ▼
   │   │                                                          optimizer.step() & scheduler.step()
   │   │
   │   └──► validation_step() ──► Accumulate outputs ──► on_validation_epoch_end()
   │                                                                 │
   │                                                                 ▼
   │                                                         Compute Q3, Precision@L
   ▼ (Callback Checkpoint)
Save results/checkpoints/best.ckpt ──► Log to TensorBoard/Wandb
```

### 5.2 Step-by-Step Tensor Shapes Trace
The table below traces the dimensional changes of a batch ($B=4$, maximum sequence length $L=512$, and hidden projection size $d=512$):

| Stage / Layer | Tensor Name / Variable | Dimensions | Explanation |
| :--- | :--- | :--- | :--- |
| **Input Sequence** | `sequences` | List of $B$ strings | Raw amino acid strings |
| **ESM Tokenizer** | `tokens` | $(B, L)$ | Integer indices mapped to dictionary |
| **ESM-2 Embedding** | `features` | $(B, L, 1280)$ | Hidden representations from ESM-2 |
| **Encoder Projection**| `x` | $(B, d, L)$ | Project 1280 to 512 channels, transpose for 1D Conv |
| **1D ResNet Encoder** | `embeddings` | $(B, L, d)$ | Latent state representation |
| **SS Head Classifier**| `ss_logits` | $(B, L, 3)$ | Output logits for 3 secondary structure classes |
| **Outer Concat Row**  | `x_i` | $(B, L, L, d)$ | Duplicated horizontally |
| **Outer Concat Col**  | `x_j` | $(B, L, L, d)$ | Duplicated vertically |
| **Concat Pairwise**   | `pairwise_features` | $(B, L, L, 2d)$ | Combined features for each coordinate pair $(i, j)$ |
| **Permuted 2D Conv**  | `pairwise_features` | $(B, 2d, L, L)$ | Permuted to match PyTorch 2D Conv format |
| **2D Projection**     | `out` | $(B, 64, L, L)$ | Reduced from 1024 to 64 channels |
| **2D ResNet Blocks**  | `out` | $(B, 64, L, L)$ | Spatial convolution feature processing |
| **Output Projection** | `contact_logits` | $(B, L, L)$ | Squeezed logits (sigmoided in loss layer) |

---

## Chapter 6: Python Files Architecture

### 6.1 Core Directory Layout
*   [`foldnet/data/dataset.py`](file:///c:/Users/shiva/OneDrive/Desktop/projects/FoldNet-1/foldnet/data/dataset.py): Defines the `ProteinDataset` and `collate_fn`. Loads pre-extracted NumPy ESM-2 embeddings and processes 2D contact matrix `.npz` files.
*   [`foldnet/models/encoders.py`](file:///c:/Users/shiva/OneDrive/Desktop/projects/FoldNet-1/foldnet/models/encoders.py): Implements sequence encoder backbones (`CNNEncoder` with residual 1D blocks, `BiLSTMEncoder` using bidirectionality, and `TransformerEncoder` using multi-head self-attention).
*   [`foldnet/models/heads.py`](file:///c:/Users/shiva/OneDrive/Desktop/projects/FoldNet-1/foldnet/models/heads.py): Contains the prediction heads (`SecondaryStructureHead` projecting to 3 classes, and `ContactMapHead` implementing Outer Concatenation followed by a 2D ResNet).
*   [`foldnet/models/foldnet.py`](file:///c:/Users/shiva/OneDrive/Desktop/projects/FoldNet-1/foldnet/models/foldnet.py): The master PyTorch Lightning wrapper coordinating training steps, multi-task losses, validation loops, metric evaluation, and scheduler steps.
*   [`viewer/app.py`](file:///c:/Users/shiva/OneDrive/Desktop/projects/FoldNet-1/viewer/app.py): FastAPI backend providing prediction API routes (`/api/predict` and `/api/test_protein/{id}`) and static directory routing for the visual dashboard.

---

## Chapter 7: Evaluation Metrics & Research Benchmarks

### 7.1 Secondary Structure Evaluation
*   **Q3 Accuracy**: Fraction of correctly predicted states (Helix, Strand, Coil) over total residues:
    $$\text{Q3} = \frac{\sum_{i=1}^3 \text{TP}_i}{L}$$
*   **Precision and Recall**: Evaluated per class to trace directional failures (e.g., sheets misclassified as coils).

### 7.2 Contact Map Evaluation
Because contact matrices are highly sparse, standard accuracy is not meaningful. Instead, we use **Precision@L**:
*   **Precision@L / Precision@L/5**: The percentage of true contacts within the top-$L$ (or top-$L/5$) predicted contacts, sorted by model confidence.
*   **Sequence Separation Ranges**: Evaluated across ranges:
    *   **Short-range**: $6 \le |i - j| < 12$ residues.
    *   **Medium-range**: $12 \le |i - j| < 24$ residues.
    *   **Long-range**: $|i - j| \ge 24$ residues. (Long-range precision is the key indicator of model quality, as these contacts define the tertiary fold topology).

### 7.3 Experimental Performance Benchmarks (CB513 Test Set)
Ablation studies comparing the three backbones on the CB513 test set:

| Architecture Backbone | Q3 Accuracy (%) | MCC (Macro) | Precision@L (All) | Long-Range Precision |
| :--- | :--- | :--- | :--- | :--- |
| **Residual CNN** | **83.66%** | **0.7448** | 0.0258 | 0.0120 |
| **BiLSTM Fusion** | 82.55% | 0.7270 | **0.1022** | **0.0845** |
| **Transformer-Hybrid** | 82.93% | 0.7326 | 0.0226 | 0.0105 |

> [!NOTE]
> *Insight*: The **Residual CNN** performs best on local 1D secondary structure classification (Q3 = 83.66%), but the **BiLSTM Fusion** model exhibits significantly higher precision on 2D contact maps (Long-Range Precision = 0.0845). This occurs because LSTMs track long-range sequences more effectively, feeding better global contexts into the outer concatenation layer.

---

## Chapter 8: Visualization and Web Interface System

### 8.1 3Dmol.js WebGL Integration
*   **Concept**: 3Dmol.js is a hardware-accelerated WebGL viewer that runs entirely on the client side.
*   **Mechanism**: The dashboard pulls the reference PDB coordinate file via `/api/pdb/{id}`. The JavaScript rendering engine colors the structure dynamically:
    *   **Native Mode**: Colors the ribbon by the ground-truth secondary structure.
    *   **Prediction Mode**: Colors the ribbon by the FoldNet predicted secondary structure.
    *   **Difference Mode**: Highlights errors (Green for correct prediction, Red for misclassified residues).

### 8.2 Synchronized Plotly Hover Events
The 2D Plotly contact map and the 3D WebGL viewer are bound bidirectionally:
1.  **Plotly Hover**: Hovering over cell $(i, j)$ in the contact map triggers a JavaScript listener.
2.  **3D Rendering Update**: The listener calls 3Dmol.js coordinates API:
    ```javascript
    // Add temporary sphere billboards at the hovered Cα residue positions
    viewer.addSphere({center: {x: posI.x, y: posI.y, z: posI.z}, radius: 2.0, color: 'blue'});
    viewer.addSphere({center: {x: posJ.x, y: posJ.y, z: posJ.z}, radius: 2.0, color: 'blue'});
    viewer.render();
    ```
3.  This establishes a clear link between 2D predictions and physical 3D spaces.

---

## Chapter 9: System Architecture & Deployment

### 9.1 Unified System Flow Diagram

The complete system architecture, showing data flow from user interface to prediction backend, is mapped below:

```mermaid
graph TD
    User([User in Browser]) -->│Sequence / Target ID│ UI[React/HTML5 UI]
    UI -->│HTTP POST /api/predict│ API[FastAPI Web Server]
    
    subgraph FastAPI Backend
        API -->│Check Cache│ Cache{Inference Cache}
        Cache -->│Miss│ ModelLoader[Model & Weights Manager]
        Cache -->│Hit│ UI
        ModelLoader -->│Run ESM-2 Extract│ ESM[ESM-2 650M Transformer]
        ESM -->│Embeddings: L x 1280│ FoldNetModel[FoldNet Core Model]
        FoldNetModel -->│Forward Pass│ Predictions[Probabilities & Contact Matrix]
        Predictions -->│JSON Payload│ API
    end
    
    API -->│PDB Coordinates & Predictions│ UI
    UI -->│Render WebGL Viewport│ Mol[3Dmol.js Canvas]
    UI -->│Render Matrix Grid│ Plotly[Plotly Heatmap]
```

### 9.2 Deployment Specifications
*   **Base OS**: Ubuntu 20.04+ / Windows 10+
*   **Python**: 3.9 - 3.11
*   **PyTorch / Lightning**: PyTorch >= 2.0.0, PyTorch Lightning >= 2.0.0
*   **VRAM Footprint**:
    *   *Inference*: ~3.5GB VRAM (due to ESM-2 650M weights).
    *   *Training (Batch size 4, length 512)*: ~5.8GB VRAM (fits on standard consumer GPUs).

---

## Chapter 10: Defensibility checklist & Design Alternatives

### 10.1 Key Design Decisions & Alternatives
*   **Why ESM-2 over ProtT5?**
    *   *Decision*: ESM-2 was selected because its embeddings are more structural-aware due to training on UniRef databases with evolutionary masking. ProtT5 was trained on Google T5 span denoising, which captures functional text semantics better but is less performant on residue-residue physical distances.
*   **Why a sequence separation threshold of 6 residues?**
    *   *Decision*: Alpha-helices have a natural pitch period of 3.6 residues. Residues spaced within 5 positions physically touch as a result of simple helix spirals. Masking sequence separations $<6$ ensures that the model learns global folding topologies instead of trivial local helices.
*   **Why frozen embeddings?**
    *   *Decision*: Fine-tuning ESM-2 (650M parameters) alongside FoldNet (5-10M parameters) on small structural datasets leads to model collapse and severe overfitting. Freezing ESM-2 acts as a powerful regularizer, retaining general biological representation spaces.

### 10.2 Whiteboard equations for the Viva
Be prepared to write these equations on a whiteboard during your viva:

1.  **Multi-Task Loss Combination**:
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SS}} + \lambda_{\text{contact}} \mathcal{L}_{\text{contact}}$$
2.  **Outer Concatenation Operations**:
    $$\mathbf{P}_{i,j} = [h_i \,\|\, h_j] \quad \text{where } h \in \mathbb{R}^{d}, \mathbf{P}_{i,j} \in \mathbb{R}^{2d}$$
3.  **Secondary Structure Q3 Accuracy Metric**:
    $$\text{Q3} = \frac{\text{Correct Helix} + \text{Correct Strand} + \text{Correct Coil}}{\text{Total Sequence Length } L} \times 100\%$$
4.  **Local Masking Rule**:
    $$\text{Target}_{i, j} = 0 \quad \text{for all } |i - j| < 6$$
