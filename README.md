[![CI/CD Pipeline](https://github.com/Sikorski06/GeoStyle-Advisor/blob/main/.github/workflows/ci-cd.yaml/badge.svg)](https://https://github.com/Sikorski06/GeoStyle-Advisor/actions)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Status](https://img.shields.io/badge/Status-Completed-success)

# GeoStyle-Advisor: Biometric Face Analysis Engine

Privacy-First Face Analysis: A local biometric engine that processes data within an isolated cluster, eliminating the need to transmit images to external cloud APIs.

## Project Overview & Problem Statement
Modern face analysis systems often rely on external SaaS providers, raising privacy concerns and generating significant network latency.

GeoStyle-Advisor is a comprehensive **End-to-End solution** that classifies facial geometric features in real-time. The system utilizes a mathematical 4D Euclidean distance model to match face shapes against predefined biometric profiles and suggest optimal styling.

**This project exceeds standard Python scripts – it is a fully containerized microservice application managed by Kubernetes and automated via CI/CD pipelines.**

## Project Architecture
The system is designed following the Infrastructure as Code (IaC) paradigm, decoupling computational logic from environmental configuration.

1. **Core Engine:** Utilizes `MediaPipe` and `OpenCV` to extract 468 characteristic landmarks (Face Mesh).

2. **Geometry 4D Logic:** A proprietary algorithm calculating proportion vectors (Height/Width, Forehead/Jaw).

3. **Serving Layer:** `Streamlit` frontend optimized for performance.

4. **Orchestration:** `Kubernetes` cluster featuring Horizontal Pod Autoscaling (HPA) and Sticky Sessions for stream stability.

5. **CI/CD Pipeline:** `GitHub Actions` automatically builds images, validates manifests using Kubeconform, and pushes artifacts to Docker Hub.

## 🛠 Key Engineering Highlights

* **Zero-Cloud Inference:** All computations occur within the container. No image frames leave your infrastructure.

* **High-Performance Video Rendering:** Implemented direct image byte injection into the UI to eliminate 404 errors in multi-node clusters.

* **Production-Ready K8s:** Utilizes ConfigMap for face profile management, allowing database updates without rebuilding the Docker image.

* **Automated Validation:** The CI/CD pipeline rigorously verifies Kubernetes manifest syntax using Kubeconform before publication.

## Tech Stack

Machine Learning / Vision: `Python`, `OpenCV`, `MediaPipe`, `NumPy`

Infrastructure: `Docker`, `Kubernetes (K8s)`, `Helm-ready YAMLs`

CI/CD & DevOps: `GitHub Actions`, `Docker Hub`, `Kubeconform`

Frontend: `Streamlit`

## Tests Performed

### Entry Page
![Entry Page](tests/Entry%20Page.png)

### Scanning Page
![Scan Page](tests/Scan%20Page.png.png)

### Results Page
![Result Page](tests/Results%20Page.png)

## Getting Started (Local Development)

### Full containerization ensures no local Python environment setup is required.

**Step 1: Clone the repository**
```bash
git clone [https://github.com/Sikorski06/GeoStyle-Advisor.git](https://github.com/Sikorski06/GeoStyle-Advisor.git)
cd GeoStyle-Advisor
```

**Step 2: Build the Docker Image**
```bash
docker build -t geostyle-app .
```

**Step 3: Deploy to Local Kubernetes Cluster**
```bash
kubectl apply -f k8s/
```

**Step 4: Access the Application*
Open your browser and navigate to:
`http://localhost`

*Note: The system defaults to port 80 (HTTP). Ensure no other service is occupying this port.*