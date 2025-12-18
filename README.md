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

Model Artifacts & Manifold Geometry

The system relies on pre-trained topological weights and scalar parameters stored in the models2/ directory. These artifacts define the manifold space used to represent and analyze network signatures.

Required Artifacts
File Name	Description
som_model.joblib	Trained MiniSom object containing the learned topological grid
som_weights1.npy	Raw NumPy weight matrix for high-speed, vectorized manifold projection
centroid_feature_map.joblib	Mapping of SOM nodes to latent regime assignments
scaler.joblib	StandardScaler parameters used for signal normalization
medians.joblib	Feature-wise medians for robust imputation of missing telemetry
policy_to_bit.joblib	Encoding dictionary for bitmap transformation of DpiPolicy
numeric_cols.joblib	Definitive feature ordering to ensure consistent manifold projection
**Key Research Artifacts**

## Key Research Artifacts: Manifold Interpretation

The following visualizations illustrate the topological preservation and feature-space correlation discovered by the SOM:

### 1. Global Weight Distribution
The latent manifold reveals three distinct regime clusters (1, 2, 3) representing different classes of network traffic signatures.

![Global Weight Manifold](results/global_weight_manifold.png)

### 2. Feature Component Planes (C-Planes)
By decomposing the manifold into individual feature planes (IpProtocol, Bytes from Client, etc.), we observe the non-linear correlations that drive cluster formation. This is critical for understanding how specific system constraints (e.g., bandwidth vs. protocol type) influence the latent representation.

![Component Planes](results/feature_component_planes.png)

### 3. Multivariate Latent Projections
Three-feature combination visualizations showing the interaction between high-dimensional features across the 2D lattice, confirming the stability of the topology-preserving mapping.

![Multivariate Projections](results/multivariate_latent_projections.png)
