# MRPC DistilBERT: Containerized Training

This repository turns the Project 1 notebook into a reproducible CLI + Docker workflow. It trains DistilBERT on GLUE MRPC and logs to Weights & Biases (optional).

## Folder structure
```
.
├─ src/                # data and model modules
├─ main.py             # CLI entry point
├─ requirements-*.txt  # dependency pins
├─ Dockerfile.gpu      # GPU build (NVIDIA)
├─ Dockerfile.cpu      # CPU-only build (small)
├─ models/             # created at runtime
└─ README.md
```

## Quickstart (local, no Docker)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt           # CPU-only torch
# or: pip install -r requirements-gpu.txt     # if you already have CUDA torch
export WANDB_API_KEY=YOUR_KEY                 # optional
python main.py --checkpoint_dir models --epochs 3 --seed 42 \
  --train_batch_size 32 --eval_batch_size 64 \
  --optimizer_name adamw --lr 1e-4 --lr_scheduler_type linear \
  --warmup_ratio 0.11 --weight_decay 0.015
```

## One-command training
```bash
python main.py --checkpoint_dir models --lr 1e-4
```
All hyperparameters are configurable via flags. Artifacts are saved under `models/<auto-run-name>/`.

## Docker (GPU)
Requires the NVIDIA Container Toolkit.
```bash
docker build -t mrpc-gpu -f Dockerfile.gpu .
# optional: pass WANDB credentials at runtime
docker run --gpus all --rm -e WANDB_API_KEY=$WANDB_API_KEY -v $PWD/models:/app/models mrpc-gpu
```

## Docker (CPU)
Suitable for constrained environments.
```bash
docker build -t mrpc-cpu -f Dockerfile.cpu .
docker run --rm -v $PWD/models:/app/models mrpc-cpu
```

## GitHub Codespaces
1. Create a Codespace on this repo (4‑core+ recommended).
2. In the terminal:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements-cpu.txt
   export WANDB_API_KEY=YOUR_KEY   # optional
   python main.py --checkpoint_dir models --epochs 3 --seed 42 \
     --train_batch_size 32 --eval_batch_size 64 \
     --optimizer_name adamw --lr 1e-4 --lr_scheduler_type linear \
     --warmup_ratio 0.11 --weight_decay 0.015
   ```
   Or build and run the CPU Docker image:
   ```bash
   docker build -t mrpc-cpu -f Dockerfile.cpu .
   docker run --rm -v $PWD/models:/app/models mrpc-cpu
   ```

## Docker Playground
Playground VMs are small. Use the CPU image:
```bash
git clone <this-repo> && cd <repo>
docker build -t mrpc-cpu -f Dockerfile.cpu .
docker run --rm -v $PWD/models:/app/models mrpc-cpu
```
If memory is tight, remove the pre-download step in `Dockerfile.cpu` and rely on runtime downloads.

## Reusing the trained model
Outputs are saved in `models/<run>/` as a standard Hugging Face folder. Load with:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
m = AutoModelForSequenceClassification.from_pretrained("models/<run>")
tok = AutoTokenizer.from_pretrained("models/<run>")
```

## Notes
- Training config defaults to the best setup reported in Project 1: epochs=3, seed=42, batch train/eval=32/64, AdamW, linear schedule, lr=1e-4, warmup_ratio=0.11, weight_decay=0.015.
- Logging uses W&B if `WANDB_API_KEY` is present; disable via `--no_wandb`.
- `accelerator="auto"` lets Lightning use GPU when available.
