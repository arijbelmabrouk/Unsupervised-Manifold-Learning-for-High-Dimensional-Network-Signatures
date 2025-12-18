# Unsupervised Manifold Learning for High-Dimensional Network Signatures

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview
This repository implements a **Topology-Preserving Manifold Discovery** framework for high-dimensional, sparse telecom signal spaces. By leveraging **Self-Organizing Maps (SOM)**, the project projects sparse DPI (Deep Packet Inspection) bitmaps and behavioral metrics into a low-dimensional latent space to identify structural invariants and perform non-convex quantization error minimization.

The objective is to map complex subscriber traffic signatures to optimal **Resource Profiles** (Service Classes) while maintaining the topological relationships of the input manifold.

## Mathematical Formulation

### 1. Manifold Projection (SOM Dynamics)
The framework utilizes a competitive learning process to minimize the quantization error between the input vector $x$ and the weight vectors $w$:

* **Best Matching Unit (BMU) Selection:** $$c(t) = \arg\min_j \| x(t) - w_j(t) \|_2$$
* **Weight Adaptation:** $$w_j(t + 1) = w_j(t) + \alpha(t) \cdot h_{c,j}(r(t)) \cdot ( x(t) - w_j(t) )$$
* **Neighborhood Decay (Gaussian):** $$h_{c,j}(r) = \exp\left( -\frac{r^2}{2\sigma(t)^2} \right)$$

### 2. Feature Sparsification Pipeline
To handle high-dimensional categorical metadata (DPI policies, content-types), we implement a **Bitmap Encoding** pipeline that transforms discrete logs into sparse binary vectors, preserving feature cardinality while enabling Euclidean distance metrics in the latent space.

## Comparison of Dimensionality Reduction Techniques

| Method | Complexity | Topology Preservation | Scalability |
| :--- | :--- | :--- | :--- |
| **SOM (O(n · grid))** | **Non-linear Manifolds** | **High (Local & Global)** | **Batch-Friendly** |
| PCA | Linear Projection | Low (Linear Only) | O(d²n) |
| t-SNE | Non-linear | High (Local Only) | O(n²) - Memory Intensive |

## Project Structure

├── models2/ # Serialized Model Artifacts (Joblib)
├── results/ # U-Matrix and Quantization Error Diagnostics
├── Dockerfile # Reproducible Research Environment
├── notebook.ipynb # Derivations, Training, and Fidelity Analysis
├── main_sub_multi.py # Batch Manifold Projection Pipeline
└── requirements.txt # System Dependencies

**Implementation Highlights**

- Scalable Pipeline: Designed to process 106+ high-dimensional records via batch-processing with periodic SQLite commits.  
- Structural Fidelity: Includes automated computation of Quantization Error and Topographic Error to validate manifold stability.  
- Reproducibility: Fully Dockerized environment to ensure consistent execution across heterogeneous computational nodes.  

**Usage**

Build the research environment:

docker build -t net-manifold-som .

Run batch inference on network signatures:

python main_sub_multi.py


**Key Research Artifacts**

- U-Matrix Visualization: Reveals clusters of behavior based on topological distance.  
- Quantization Decay: Monitoring convergence stability during competitive training.  
- ![U-Matrix](results/u_matrix.png)

---

### 4. The "Missing Piece" (Your Homework)
You have a folder called `results/`. **You must upload a U-Matrix plot (image) into that folder.** If Shihada clicks this repo and sees the math in the README but no **Visual Proof**, he will think you didn't actually run it.  

* **Find the U-Matrix image** generated in your notebook.  
* **Name it `u_matrix.png`.**  
* **Upload it to GitHub**.  
* **Add this line to the README under "Key Research Artifacts":**  
  `![U-Matrix](results/u_matrix.png)`

**Does this new README look like something you'd see in a networking lab, or are you still worried it's too technical?** (Hint: In Basem's lab, there is no such thing as "too technical.")

