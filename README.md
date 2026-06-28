# Cluster CLI 🚀

A lightweight, personal cluster orchestrator designed to streamline High-Performance Computing (HPC) workflows. Automate project scaffolding, container image generation (Apptainer/Singularity), SSH data transfer, SLURM job submission, status monitoring, and log fetching directly from your terminal.

---

## ✨ Features

- **📂 Project Scaffolding (`cluster init`)**: Automatically generates a standardized project structure with pre-configured YAML settings, stub scripts, and directories.
- **🐳 Dynamic Apptainer Builds (`cluster build`)**: Auto-detects Python dependencies using `pipreqs`, generates custom `apptainer.def` definition files, and builds container images (`.sif`).
- **⚡ Remote Execution & SLURM Integration (`cluster run`)**: Syncs container images and source code over SSH using `Fabric`, generates automated SLURM submission scripts (`submit.sh`), and tracks job execution state.
- **📊 Real-Time Job Monitoring (`cluster status`)**: Queries `scontrol` and `sacct` over SSH to display rich, color-coded status tables of your running jobs.
- **📜 Instant Log Access (`cluster logs`)**: Fetches and formats remote stdout and stderr log outputs neatly in your terminal.

---

## 📦 Installation

Ensure you have Python 3.10+ installed. Clone the repository and install it in editable mode:

```bash
git clone https://github.com/arxisme/cluster_cli.git
cd cluster-cli
pip install -e .
```

### System Prerequisites
To fully leverage local container building and remote execution, ensure the following tools are installed on your local environment:
- **Apptainer / Singularity**: Required locally for running `cluster build`.
- **pipreqs**: Required for automatic Python dependency detection (`pip install pipreqs`).

---

## 🛠️ Usage Workflow

### 1. Initialize a Project
Create a new project directory with standard HPC subfolders (`src/`, `datasets/`, `models/`, `results/`) and a configuration file:
```bash
cluster init my_experiment
cd my_experiment
```

Inspect and customize `cluster.yaml` with your cluster host, username, and job specifications:
```yaml
project_name: my_experiment

cluster:
  host: 172.16.112.202
  user: your_username

stages:
  prepare:
    - python3 scripts/download.py
  build:
    base_image: pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime
    auto_detect_dependencies: true
  run:
    partition: gpu-P100
    gpus: 1
    cpus: 8
    memory: 32G
    time: "04:00:00"
    command: python3 src/train.py --output ./results/
```

### 2. Build Container Image
Auto-detect dependencies from your code and build the `.sif` Apptainer container:
```bash
cluster build
```

### 3. Run Pipeline on Cluster
Execute preparation steps, upload assets via SSH, and submit the SLURM job:
```bash
cluster run
# Prompt for SSH password if key-based authentication is not configured:
cluster run --ask-pass
```

### 4. Monitor Status
Check the real-time status of your active SLURM job:
```bash
cluster status
```

### 5. View Logs
Fetch standard output and error logs directly from the cluster:
```bash
cluster logs
# View only error logs:
cluster logs --error
```

---

## 📄 License

MIT License.
