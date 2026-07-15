# FoldNet Team Collaboration and Responsibility Distribution Plan

This document establishes the official project management framework, division of labor, git branching workflow, and research paper contribution matrix for the FoldNet core development team: **Shivam, Shubham, Tanishka, and Vaibhav**.

---

## 1. Professional Collaboration Concept

In high-performing deep learning research labs, projects succeed when team members possess **specialized ownership** over specific subsystems while maintaining **active collaboration** across overlapping boundaries. To prevent siloed development and ensure a unified codebase:
1.  **Specialization**: Each member leads a core component matching their role:
    *   **Shivam**: Deep Learning & Model Architecture
    *   **Shubham**: Data Engineering & Preprocessing
    *   **Vaibhav**: System Engineering, Metrics Engine, & Automated Testing
    *   **Tanishka**: User Interface (UI), WebGL/Plotly Visualizations, Multi-Task Losses, & Scientific Communication
2.  **Cross-Review**: Members act as reviewers for neighboring modules (e.g., the System Engineer reviews the Model Lead's forward-pass interfaces; the Data Engineer reviews the Model Lead's dataloader logic).
3.  **Unified Documentation**: Every code implementation is accompanied by joint documentation updates, ensuring the research paper matches the active repository.

---

## 2. Deliverables & Frameworks

### 2.1 Responsibility Matrix (RACI Format)
*   **R (Responsible)**: The person who performs the work.
*   **A (Accountable)**: The person with final decision-making power and veto authority over the module.
*   **C (Consulted)**: Subject matter experts whose input is sought before or during the work.
*   **I (Informed)**: Stakeholders who are updated upon completion.

| Deliverable / Module | Shivam (Model Lead) | Shubham (Data Lead) | Vaibhav (Systems & Metrics) | Tanishka (UI & Research) |
| :--- | :---: | :---: | :---: | :---: |
| **ESM Embeddings Pipeline** | **A** / **R** | **C** | **I** | **I** |
| **Encoders & heads.py** | **A** / **R** | **I** | **C** | **C** |
| **Multi-Task Loss Design (loss.py)**| **C** | **I** | **I** | **A** / **R** |
| **Dataloader & Padding Mask** | **C** | **A** / **R** | **I** | **I** |
| **DSSP Coordinate Parsing** | **I** | **A** / **R** | **C** | **I** |
| **FastAPI Backend /app.py** | **C** | **I** | **A** / **R** | **C** (UI feeds) |
| **WebGL & Plotly Frontends** | **I** | **I** | **C** | **A** / **R** |
| **Web Interface (HTML/CSS)** | **I** | **I** | **I** | **A** / **R** |
| **Validation Metrics Calculation**| **I** | **C** | **A** / **R** | **I** |
| **PyTest Testing Harnesses** | **C** | **C** | **A** / **R** | **I** |
| **System Architecture Schema**| **I** | **I** | **A** / **R** | **R** |
| **Research Paper Writing** | **R** (Method) | **R** (Results) | **R** (Metrics) | **A** / **R** (Lead) |
| **Documentation Manuals** | **R** (Training) | **R** (Data) | **R** (Deploy/Metrics) | **A** / **R** (Lead) |

---

### 2.2 Weekly Project Timeline (6-Week Schedule)

```
[Week 1] ──► Preprocessing setup (Shubham) & Literature review & CSS frameworks (Tanishka)
[Week 2] ──► Coordinate extraction (Shubham) & Encoder architecture prototypes (Shivam)
[Week 3] ──► Multi-task training loops (Shivam) & Loss function implementation (Tanishka)
[Week 4] ──► FastAPI backend routing (Vaibhav) & Metric calculation engine (Vaibhav)
[Week 5] ──► WebGL 3Dmol.js rendering & Plotly UI sync (Tanishka) & Automated PyTest suites (Vaibhav)
[Week 6] ──► Paper assembly (Tanishka) & Refactoring, review, and final packaging (All)
```

*   **Week 1: Foundations & Architecture Setup**
    *   *Shivam*: Prototypes basic input projections and config parser schemas.
    *   *Shubham*: Downloads PDB raw files and sets up Bio.PDB parses.
    *   *Vaibhav*: Configures basic FastAPI skeleton, logging systems, and metric schemas.
    *   *Tanishka*: Collects papers on ESM-2, sets up research LaTeX templates, and constructs mock CSS layouts for the dashboard.
*   **Week 2: Data Engineering & Prototyping**
    *   *Shivam*: Implements `CNNEncoder`, `BiLSTMEncoder`, and `TransformerEncoder`.
    *   *Shubham*: Preprocesses secondary structure classifications (DSSP8 to Q3 conversion).
    *   *Vaibhav*: Formulates structures for testing metric validations (Q3 and macro MCC frameworks).
    *   *Tanishka*: Designs CSS glassmorphic variables, colors, and responsive control panels.
*   **Week 3: Model Core Development & Loss Configuration**
    *   *Shivam*: Codes multi-task training steps, optimizer warmup, and gradient accumulation.
    *   *Shubham*: Computes L2 residue distances and masks out local helical contacts ($|i-j| < 6$).
    *   *Vaibhav*: Implements baseline validation metric evaluation callbacks in the Lightning loop.
    *   *Tanishka*: Codes the custom loss functions in [loss.py](file:///c:/Users/shiva/OneDrive/Desktop/projects/FoldNet-1/foldnet/models/loss.py) (weighted multi-task losses and Focal Loss options).
*   **Week 4: System Integration & Web API Development**
    *   *Shivam*: Exports trained checkpoints and validates inference pipelines.
    *   *Shubham*: Runs validation evaluation and verifies splits data.
    *   *Vaibhav*: Connects model predictions with the FastAPI router and implements Precision@L, L/2, L/5 contact map metrics.
    *   *Tanishka*: Integrates static templates and coordinates page routing structures on the client side.
*   **Week 5: Visualization, Benchmarks & UI Polish**
    *   *Shivam*: Conducts hyperparameter tuning and assists with ablation runs.
    *   *Shubham*: Generates Q3 and contact benchmark reports.
    *   *Vaibhav*: Writes automated test cases using pytest (in `tests/test_model.py` and `tests/test_data.py`) to verify tensor shapes and boundary conditions.
    *   *Tanishka*: Integrates Plotly 2D heatmap rendering and 3Dmol.js WebGL canvas views with hover triggers.
*   **Week 6: Paper Packaging & Final Presentation Prep**
    *   *Shivam*: Generates training curves and assists with the final code review.
    *   *Shubham*: Analyzes validation failures and profiles edge-case error logs.
    *   *Vaibhav*: Dockerizes the complete system and compiles hardware performance tests.
    *   *Tanishka*: Assembles final paper drafts and compiles the team presentation slides.

---

### 2.3 Work Breakdown Structure (WBS)
*   **1.0 Data Engineering & Preprocessing (Lead: Shubham)**
    *   1.1 Coordinate Parsing: Build PDB parser logic using Bio.PDB modules.
    *   1.2 Secondary Structure: Map DSSP 8-state classifications to 3-state Q3 vectors.
    *   1.3 Contact Maps: Compute distance matrices and implement local contact sequence-separation masking.
    *   1.4 Loader Pipeline: Develop `ProteinDataset` and collate-masking structures.
*   **2.0 Model Development & Optimization (Lead: Shivam)**
    *   2.1 Encoder Modules: Implement 1D CNN, BiLSTM, and Multi-Head Transformer backbones.
    *   2.2 Task Heads: Construct linear Q3 classifiers and Outer Concatenation 2D ResNets.
    *   2.3 Schedulers: Set up sequentially combined warmup and cosine annealing decay.
*   **3.0 Systems Engineering, Metrics & Testing (Lead: Vaibhav)**
    *   3.1 Backend Server: Configure FastAPI server routes, PDB delivery endpoints, and cache.
    *   3.2 Metrics Engine: Code the metric evaluation modules (Q3, MCC, Precision@L, and Long-Range Precision).
    *   3.3 Quality Assurance: Implement pytest suites to verify tensor shapes, model interfaces, and processing rules.
    *   3.4 Deploy: Dockerize app with appropriate CUDA drivers and package dependencies.
*   **4.0 UI Development & Scientific Writing (Lead: Tanishka)**
    *   4.1 Web Interface: Construct responsive CSS glassmorphic templates, HTML skeletons, and control forms.
    *   4.2 Interactive Rendering: Program Plotly 2D heatmaps and 3Dmol.js WebGL canvas integrations.
    *   4.3 Loss Engineering: Implement custom multi-task losses, weighted BCEs, and Focal Losses in `loss.py`.
    *   4.4 Paper Outline & Writing: Establish LaTeX layouts, write related works, methodology, results, and discussion.

---

### 2.4 Module Ownership and File Mappings

This diagram shows how code files map to team ownership:

```
                  ┌──────────────────────────────┐
                  │          FoldNet App         │
                  └──────────────┬───────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Shivam      │     │     Shubham     │     │     Vaibhav     │
│   (Modeling)    │     │     (Data)      │     │ (Sys & Metrics) │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ├─► encoders.py         ├─► dataset.py          ├─► app.py
         ├─► heads.py            ├─► preprocess_ss.py    ├─► metrics_ss.py
         └─► foldnet.py          └─► preprocess_con.py   ├─► metrics_contacts.py
                                                         └─► tests/
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │       Tanishka        │
                     │  (UI & Research Lead) │
                     └───────────┬───────────┘
                                 │
                                 ├─► loss.py
                                 ├─► viewer/static/ (index.html, app.js, comparison.js)
                                 ├─► foldnet_viva_handbook.md
                                 └─► evaluation_and_workflow.md
```

---

### 2.5 Collaboration Workflow
Information and assets flow across roles during development:

```
[Shubham: Dataset Processing]
             │
             ▼ (Numpy arrays & splits.json)
[Shivam: Model Development] ◄────────► [Tanishka: Loss Function Design]
             │
             ▼ (Lightning checkpoints)
[Vaibhav: API Server Integration & Metrics Engine]
             │
             ▼ (API endpoints & data feeds)
[Tanishka: Web UI & Scientific Writing]
```

*   **Shubham $\rightarrow$ Shivam**: Shubham generates preprocessed NumPy embedding files and split configurations, which Shivam uses for model training.
*   **Tanishka $\rightarrow$ Shivam**: Tanishka develops loss functions in [loss.py](file:///c:/Users/shiva/OneDrive/Desktop/projects/FoldNet-1/foldnet/models/loss.py), which Shivam integrates into the model's training loops.
*   **Shivam $\rightarrow$ Vaibhav**: Shivam trains model configurations and delivers checkpoints (`.ckpt`) to Vaibhav for backend inference.
*   **Vaibhav $\rightarrow$ Tanishka**: Vaibhav implements metric calculations and delivers endpoints (including predicted labels, coordinates, and contact values) which Tanishka consumes in the frontend UI.
*   **Tanishka $\rightarrow$ All**: Tanishka delivers responsive UI designs and compiles literature reviews, aligning system features with the scientific presentation.

---

### 2.6 Research Paper Authorship Mapping

| Paper Section | Primary Owner | Secondary Contributor(s) | Section Responsibilities |
| :--- | :--- | :--- | :--- |
| **Abstract** | **Tanishka** | All | Summarizes overall architecture, findings, and benchmark results. |
| **Introduction** | **Tanishka** | Shivam | Establishes the sequence-to-structure gap and motivation. |
| **Related Work** | **Tanishka** | Vaibhav | Evaluates AlphaFold, ESMFold, and older prediction baselines. |
| **Methodology** | **Shivam** | Tanishka | Documents structural models and mathematical formulations. |
| **Dataset** | **Shubham** | Tanishka | Highlights CullPDB and CB513 preprocessing rules. |
| **Model Arch** | **Shivam** | Vaibhav | Maps internal 1D encoders, Outer-Cat, and 2D ResNet layers. |
| **Experimental Setup** | **Shubham** | Shivam | Outlines hyperparameters, learning rate schedules, and cross-validation. |
| **Results** | **Vaibhav** | Shubham / Tanishka | Formulates Q3 accuracy charts and Precision@L tables. |
| **Discussion** | **Tanishka** | All | Analyzes failure modes, biological contexts, and edge-case limits. |
| **Conclusion** | **Tanishka** | All | Wraps up overall research insights and lists future expansions. |

---

### 2.7 Documentation Ownership Matrix

*   **Installation Guide**: Owned by **Vaibhav**. Details environment setups, CUDA compatibility, and PyTorch configurations.
*   **Data Pipeline Guide**: Owned by **Shubham**. Documents raw PDB fetching, DSSP sequence mapping, and 2D matrix masking.
*   **Model Training Guide**: Owned by **Shivam**. Outlines multi-task optimization parameters, learning rate combined scheduling, and gradient accumulation.
*   **Inference & Deployment Guide**: Owned by **Vaibhav**. Details FastAPI endpoints, CPU concurrency, and Docker containerization.
*   **Loss & Metrics Documentation**: Owned by **Tanishka / Vaibhav**. Tanishka explains loss formulations and backpropagation rules; Vaibhav documents metrics calculations and evaluation structures.
*   **User Interface Manual**: Owned by **Tanishka**. Explains dashboard controls, interactive Plotly hovers, and WebGL ribbon features.

---

## 3. Git and Code Review Workflow

### 3.1 Git Branching Strategy
To maintain repository integrity, the team uses a structured branching pipeline:

*   **`main`**: Production-ready code. Vaibhav manages and deploys this branch. Only merged from `develop` after testing.
*   **`develop`**: Master integration branch. All feature branches merge here after code review.
*   **Feature Branches**:
    *   `feat/model-arch-shivam`: Shivam's workspace for model architectures and training routines.
    *   `feat/data-pipeline-shubham`: Shubham's workspace for PDB parsing and dataset classes.
    *   `feat/systems-metrics-vaibhav`: Vaibhav's workspace for FastAPI, test scripts, and metrics code.
    *   `feat/ui-research-tanishka`: Tanishka's workspace for HTML/CSS layouts, visualizers, loss classes, and papers.

### 3.2 Pull Request and Code Review Responsibilities
All merges into the `develop` branch require a Pull Request (PR) and approval from at least one reviewer:

*   **Shivam's PRs** $\rightarrow$ Reviewed by **Vaibhav**: Ensures model configuration loaders, forward passes, and checkpoints interface correctly with backend requirements.
*   **Shubham's PRs** $\rightarrow$ Reviewed by **Shivam**: Verifies dataset loader mappings feed the correct dimensions to training.
*   **Vaibhav's PRs** $\rightarrow$ Reviewed by **Tanishka**: Checks that metrics values match what is expected in the frontend UI rendering.
*   **Tanishka's PRs** $\rightarrow$ Reviewed by **Vaibhav**: Verifies that custom loss code and UI script calls interface correctly with server routes.

---

## 4. Final Percentage Contributions

*   **Shivam (Modeling Lead - 25%)**: Built the core model backbones, project loaders, and training steps.
*   **Shubham (Data Lead - 25%)**: Coded the coordinate extraction pipeline, distance contact mapping, and dataset loaders.
*   **Vaibhav (Systems & Metrics Lead - 25%)**: Engineered the FastAPI backend, mathematical metrics calculation engine (Q3, MCC, Precision@L), and automated test suites.
*   **Tanishka (UI & Research Lead - 25%)**: Programmed the frontend layout, responsive CSS glassmorphic elements, 3Dmol.js WebGL canvas views, Plotly interactive charts, custom training losses, and drafted the research paper.
