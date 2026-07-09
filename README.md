# Data science development environment

[![Sync release](https://github.com/gperdrizet/datascience-devcontainer/actions/workflows/sync-release.yml/badge.svg)](https://github.com/gperdrizet/datascience-devcontainer/actions/workflows/sync-release.yml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-189fdd)](https://xgboost.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-latest-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Docker Pulls datascience-nvidia](https://img.shields.io/docker/pulls/gperdrizet/datascience-nvidia?label=datascience-nvidia&logo=docker)](https://hub.docker.com/r/gperdrizet/datascience-nvidia)
[![Docker Pulls datascience-cpu](https://img.shields.io/docker/pulls/gperdrizet/datascience-cpu?label=datascience-cpu&logo=docker)](https://hub.docker.com/r/gperdrizet/datascience-cpu)
[![Docker Pulls datascience-mac](https://img.shields.io/docker/pulls/gperdrizet/datascience-mac?label=datascience-mac&logo=docker)](https://hub.docker.com/r/gperdrizet/datascience-mac)

A ready-to-use data science environment for VS Code, designed for data science and ML bootcamp students. Covers data visualization, data cleaning, feature engineering, and traditional machine learning.

## Requirements

**All users**
- [Docker Desktop](https://docs.docker.com/desktop/) (Windows / Mac) or [Docker Engine](https://docs.docker.com/engine/install/) (Linux)
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

**NVIDIA GPU users** (also required)
- NVIDIA driver ≥570 ([download](https://www.nvidia.com/Download/index.aspx))
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) *(Linux only, not needed on Windows)*

> **Mac users:** GPU acceleration (Metal/MPS) does not pass through to Docker containers. The Mac configuration uses native ARM64 CPU, no extra setup needed beyond Docker Desktop.

## Quick start

1. **Fork** this repository (click **Fork** at the top of this page)

2. **Clone** your fork:
   ```bash
   git clone https://github.com/<your-username>/datascience-devcontainer.git
   ```

3. **Open the folder in VS Code**, then open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and run **Dev Containers: Open Folder in Container**

   > VS Code will ask which configuration to use, pick the one that matches your machine (see table below).

4. **Verify** your setup by running `notebooks/environment_test.ipynb`

## Which configuration should I use?

| If you have... | Choose this |
|----------------|-------------|
| NVIDIA GPU (GTX 10xx / RTX / Quadro / Tesla) | **DataScience NVIDIA** |
| Windows or Linux machine, no NVIDIA GPU | **DataScience CPU** |
| Apple Silicon Mac (M1 / M2 / M3 / M4) | **DataScience Mac** |

Not sure if your GPU is compatible? Check: [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus) (need compute capability ≥6.0).

## Using as a template for new projects

Fork this repo once, then use it as a GitHub template to spin up new projects instantly.

### One-time setup

1. Go to your fork on GitHub
2. Click **Settings** → scroll to **Template repository** → enable it

### Creating a new project

1. Go to your fork and click **Use this template** → **Create a new repository**
2. Name your new repo and click **Create repository**
3. Clone it and start working:
   ```bash
   git clone https://github.com/<your-username>/my-new-project.git
   ```

4. **Clean it up** - remove anything that doesn't belong to your project:
   - Update `README.md` to describe your project
   - Delete unused devcontainer configs (e.g. if you only use CPU, remove `nvidia/` and `mac/`)
   - Remove or replace `notebooks/environment_test.ipynb` with your own notebooks
   - Delete test data from `data/`
   ```bash
   git add -A && git commit -m "Initial project setup" && git push
   ```

## Adding Python packages

### Temporary (lost on container rebuild)

```bash
pip install <package-name>
```

### Permanent (recommended)

1. Create a `requirements.txt` in the repository root:
   ```
   lightgbm
   shap
   ```

2. Add a `postCreateCommand` to the relevant `.devcontainer/*/devcontainer.json`:
   ```json
   "postCreateCommand": "pip install -r requirements.txt"
   ```

3. Rebuild the container (`Ctrl+Shift+P` → **Dev Containers: Rebuild Container**)

## Keeping your fork updated

```bash
# Add upstream once
git remote add upstream https://github.com/gperdrizet/datascience-devcontainer.git

# Pull in updates
git fetch upstream && git merge upstream/main
```

## What's included

| Package | Purpose |
|---------|---------|
| numpy, pandas, scipy | Core data science stack |
| scikit-learn, xgboost, statsmodels | Machine learning and statistics |
| matplotlib, seaborn, plotly | Visualization |
| optuna | Hyperparameter optimization |
| jupyterlab | Interactive notebooks |
| cupy-cuda12x | GPU-accelerated arrays (NVIDIA only) |
| python-dotenv | Environment variable management |

## GPU compatibility (NVIDIA)

Requires compute capability ≥6.0 (Pascal / GTX 10xx or newer):

| Architecture | Example GPUs | Compute Capability |
|--------------|--------------|-------------------|
| Pascal | GTX 1050–1080, Tesla P100 | 6.0–6.1 |
| Volta | Tesla V100, Titan V | 7.0 |
| Turing | RTX 2060–2080, GTX 1660 | 7.5 |
| Ampere | RTX 3060–3090, A100 | 8.0–8.6 |
| Ada Lovelace | RTX 4060–4090 | 8.9 |
| Hopper | H100, H200 | 9.0 |
| Blackwell | RTX 5070–5090, B100, B200 | 10.0 |

## Project structure

```
datascience-devcontainer/
├── .devcontainer/
│   ├── nvidia/
│   │   └── devcontainer.json   # NVIDIA GPU configuration
│   ├── cpu/
│   │   └── devcontainer.json   # CPU configuration
│   └── mac/
│       └── devcontainer.json   # Mac (ARM64) configuration
├── data/                       # Store datasets here
├── notebooks/
│   └── environment_test.ipynb  # Verify your setup
├── .gitignore
├── LICENSE
└── README.md
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Docker won't start | Enable virtualization in BIOS / enable WSL2 on Windows |
| Permission denied (Linux) | Add your user to the docker group, then log out and back in |
| GPU not detected | Update NVIDIA drivers (≥570); Linux: install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| Container build fails | Check your internet connection |
| Module not found | Add the package to `requirements.txt` and rebuild the container |
