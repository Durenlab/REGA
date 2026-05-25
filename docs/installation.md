# Installation

## Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| Python | ≥ 3.9 | conda strongly recommended |
| PyTorch | ≥ 1.13 | GPU build recommended; install separately |
| CUDA-capable GPU | — | Strongly recommended (CPU is 10–100× slower) |
| bedtools | ≥ 2.30 | For RE-TG pair building |
| HOMER | any recent | For motif scanning; genome package required |

---

## 1. Clone and install REGA

```bash
# 1. Clone the repository
git clone https://github.com/Lixin017/REGA.git
cd REGA

# 2. Create conda environment
conda env create -f environment.yml          # GPU/CPU-agnostic base environment
# or for a CPU-only named environment:
# conda env create -f environment-cpu.yml

# 3. Activate environment
conda activate rega

# 4. Install PyTorch — choose the build that matches your hardware
# Visit https://pytorch.org/get-started/locally/ for the exact command.
# Examples:
#   GPU (CUDA 12.1):  pip install torch --index-url https://download.pytorch.org/whl/cu121
#   GPU (CUDA 12.4):  pip install torch --index-url https://download.pytorch.org/whl/cu124
#   CPU-only:         pip install torch --index-url https://download.pytorch.org/whl/cpu

# 5. Install REGA
pip install .
```

> **HPC users**: if your home directory has a tight quota, install the conda
> environment in project storage:
> ```bash
> conda env create -f environment.yml --prefix /path/to/project/envs/rega
> conda activate /path/to/project/envs/rega
> ```

---

## 2. PyTorch — install separately

PyTorch is **not included** in the conda environment files because the correct
build (CPU, CUDA 11.x, CUDA 12.x, ROCm) depends on your hardware and driver
versions. You must install it manually after activating the environment.

**Step 1** — check your CUDA version (GPU users):

```bash
nvidia-smi    # CUDA version shown in top-right corner
```

**Step 2** — go to https://pytorch.org/get-started/locally/ and select your
OS / package manager / CUDA version to get the exact install command.

Common examples:

```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Step 3** — verify:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
# GPU build expected: CUDA available: True
```

> **Note**: CPU training is 10–100× slower than GPU. Use GPU whenever possible.

---

## 3. bedtools

bedtools is used by `rega retg` for genomic interval intersection.
It is included in both `environment.yml` and `environment-cpu.yml` via bioconda,
so no separate install is needed when using the provided conda environments.

If you manage your own environment:

```bash
conda install -c bioconda bedtools   # conda
sudo apt install bedtools             # Ubuntu/Debian
module load bedtools                  # HPC module system
```

Verify:

```bash
bedtools --version
# bedtools v2.31.x
```

---

## 4. HOMER (motif scanning)

HOMER is used by `rega motif` to scan peaks for transcription factor
binding motifs.

**Install HOMER**:

Follow the official guide at http://homer.ucsd.edu/homer/introduction/install.html

```bash
mkdir homer && cd homer
wget http://homer.ucsd.edu/homer/configureHomer.pl
perl configureHomer.pl -install
```

**Install the hg38 genome package** (required for human data):

```bash
perl /path/to/homer/configureHomer.pl -install hg38
```

**Add HOMER to PATH**:

```bash
export PATH=$PATH:/path/to/homer/bin
```

Add this line to your `~/.bashrc` or `~/.bash_profile` for persistence.

Verify:

```bash
findMotifsGenome.pl    # should print HOMER usage
```

> **HPC users**: HOMER may be available as a module:
> ```bash
> module load HOMER
> ```
> Check with `module avail homer` or ask your sysadmin.

---

## 5. Verify the full installation

```bash
# REGA CLI
rega --version
rega --help

# PyTorch + GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# bedtools (via pybedtools)
python -c "import pybedtools; pybedtools.BedTool('a', from_string=True); print('pybedtools OK')"

# HOMER
findMotifsGenome.pl 2>&1 | head -2
```

---

## 6. HPC-specific tips

### Long-running jobs

Model training (`rega train`) can take hours on GPU.
Use `tmux` or `screen` to keep sessions alive:

```bash
tmux new -s rega_run
# ... run your commands ...
# Ctrl+B, D  to detach without killing the session
tmux attach -t rega_run  # reconnect later
```

### GPU node allocation (SLURM)

Allocate an interactive GPU session before training:

```bash
srun -p gpu --gres=gpu:1 --mem=32G --time=8:00:00 --pty bash
conda activate rega
rega train input.h5ad --output-dir model/
```

Or submit as a batch job:

```bash
#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=rega_train

conda activate /path/to/envs/rega
rega train naiveCD4T_rega_input.h5ad \
    --output-dir model/ \
    --K 20
```
