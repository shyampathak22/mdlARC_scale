# mdlARC_scale

A PyTorch reproduction and scaling study of [mdlARC](https://github.com/mvakde/mdlARC), applying Minimum Description Length (MDL) principles to solve [ARC-AGI](https://arcprize.org/) tasks with transformer models.

This repository extends the original mdlARC approach with systematic ablations, multiple model scales, and infrastructure for scaling experiments.

## Acknowledgments

This work is based on **mdlARC** by [Mithil Vadke](https://mvakde.github.io/) ([@evilmathkid](https://x.com/evilmathkid)). The original mdlARC achieved 27.5% on ARC-AGI-1 for just $2 in training costs, demonstrating that MDL principles combined with simple transformer architectures can achieve strong results without complex reasoning pipelines.

See the original repository: [https://github.com/mvakde/mdlARC](https://github.com/mvakde/mdlARC)


## Installation

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mdlARC_scale.git
cd mdlARC_scale

# Install dependencies
uv sync

# (Optional) Install dev dependencies for testing
uv sync --group dev
```

## Quick Start

### Training

```bash
# Train small model (recommended starting point)
uv run train --config small

# Train with custom settings
uv run train --config small --epochs 50 --lr 1e-4 --batch-size 16

# Resume from checkpoint
uv run train --config small --resume checkpoints/small/best.pt
```

Training automatically downloads ARC-1 and ConceptARC datasets on first run.

### Evaluation

```bash
# Evaluate on ARC-AGI evaluation set
uv run evaluate --checkpoint checkpoints/small/best.pt

# With test-time adaptation
uv run evaluate --checkpoint checkpoints/small/best.pt \
    --tta-steps 10 --tta-lr 0.1 --num-color-perms 20

# Generate submission file
uv run evaluate --checkpoint checkpoints/small/best.pt \
    --output submission.json
```

### Ablation Studies

```bash
# Run all ablations with tiny model
uv run ablations --size tiny --epochs 20

# Run specific ablation groups
uv run ablations --size tiny --ablations rope,aug
```

## Model Configurations

| Config | Parameters | d_model | n_heads | n_layers | d_ff |
|--------|-----------|---------|---------|----------|------|
| tiny   | ~3.6M     | 256     | 4       | 4        | 704  |
| small  | ~29M      | 768     | 12      | 4        | 2048 |
| medium | ~103M     | 1024    | 16      | 8        | 2752 |
| large  | ~237M     | 1280    | 20      | 12       | 3392 |

Configuration files are in `configs/`. Each config specifies model architecture, training hyperparameters, augmentation settings, and logging options.

## Architecture

```
src/
├── data.py          # Tokenization, augmentation, dataset pipeline
├── transformer.py   # TinyTransformer with 3D RoPE
├── train.py         # Training loop with DDP support
├── eval.py          # Evaluation and submission generation
├── inference.py     # Batched generation with KV cache
└── voting.py        # AAIVR voting aggregation

scripts/
├── train_wrapper.py # Multi-GPU training launcher
├── eval_wrapper.py  # Evaluation wrapper
├── ablations.py     # Ablation study runner
└── sanitize_data.py # Data leakage prevention

configs/
├── tiny.toml        # ~3.6M params
├── small.toml       # ~29M params
├── medium.toml      # ~103M params
└── large.toml       # ~237M params
```

### Tokenization

Grids are tokenized as sequences:
- Colors 0-9 map to token IDs 0-9
- Special tokens: `START` (10), `NEWLINE` (11), `IO_SEP` (12), `END` (13)
- Format: `[START] input_row1 [NEWLINE] input_row2 ... [IO_SEP] output_row1 [NEWLINE] ... [END]`

### 3D Position Encoding

Each token receives a 3D position (x, y, z):
- **x**: Column position (0-30)
- **y**: Row position (0-29)
- **z**: Semantic layer (0=START, 1=input, 2=IO_SEP, 3=output, 4=END)

The transformer uses rotary position embeddings (RoPE) extended to 3D by splitting head dimensions across the three axes.

## Ablation Results

Ablations on the tiny model (5 epochs) demonstrate the importance of each component:

| Ablation | Val Loss | Val Accuracy | Delta |
|----------|----------|--------------|-------|
| **Baseline** | 0.364 | 60.5% | - |
| LayerNorm (vs RMSNorm) | 0.365 | 60.4% | -0.1% |
| GELU (vs SwiGLU) | 0.385 | 60.3% | -0.2% |
| 1D RoPE (vs 3D) | 0.465 | 62.7% | +2.2%* |
| No RoPE | 0.736 | 74.4% | +13.9%* |
| No Color Aug | 0.347 | 60.4% | -0.1% |
| No Dihedral Aug | 0.597 | 62.5% | +2.0%* |
| No Augmentation | 0.581 | 61.4% | +0.9%* |
| Output-Only Loss | 0.237 | 60.3% | -0.2% |
| No ConceptARC | 0.385 | 60.5% | +0.0% |
| Train Split Only | 0.451 | 60.5% | +0.0% |

*Higher accuracy with higher loss indicates overfitting to training patterns without learning generalizable transformations.

Key findings:
- **3D RoPE is critical**: Removing or reducing to 1D significantly hurts generalization
- **Dihedral augmentation matters**: Geometric invariance is essential
- **Color augmentation has less impact**: Primarily a regularizer
- **Uniform loss weighting works**: Output-only loss shows similar performance

## Training Details

### Data Pipeline

1. Load ARC-1 (400 training tasks) + ConceptARC (560 tasks)
2. Expand with test pairs (using known solutions for training tasks)
3. Apply dihedral augmentation (8x)
4. Apply color permutation augmentation (100 permutations, cycled per epoch)
5. Length-bucketed batching to minimize padding

### Optimization

- AdamW with selective weight decay (exclude biases, norms, embeddings)
- Linear warmup (5% of steps) + cosine decay
- Gradient clipping (max norm 1.0)
- Mixed precision training (BF16 on CUDA)
- Gradient accumulation for large effective batch sizes

### Data Leakage Prevention

The codebase includes safeguards against accidentally training on evaluation data:
- Separate loading paths for training vs evaluation solutions
- `sanitize_data.py` script to clean data directories
- Warnings when solution files contain unexpected tasks

## Multi-GPU Training

Distributed training via PyTorch DDP:

```bash
# 2 GPUs
uv run train --config small --gpus 2

# 4 GPUs with custom settings
uv run train --config medium --gpus 4 --batch-size 4 --effective-batch-size 64
```

## Configuration Reference

Key configuration options (see `configs/*.toml` for full examples):

```toml
[model]
size = "small"              # tiny, small, medium, large

[training]
batch_size = 8              # Per-GPU batch size
effective_batch_size = 32   # Total batch size (via gradient accumulation)
epochs = 100
lr = 3e-4
weight_decay = 0.01
warmup_fraction = 0.05

[augmentation]
apply_dihedral = true       # 8x geometric transforms
num_color_perms = 100       # Color permutations per epoch

[loss]
uniform_loss_weight = true  # Equal weight for all tokens (original mdlARC)
```

## CLI Reference

### Training

```
uv run train --config CONFIG [OPTIONS]

Options:
  --config PATH          Config file or preset name (tiny/small/medium/large)
  --epochs N             Number of training epochs
  --batch-size N         Per-GPU batch size
  --effective-batch-size N  Total batch size (via gradient accumulation)
  --lr FLOAT             Learning rate
  --save-dir PATH        Checkpoint save directory
  --resume PATH          Resume from checkpoint
  --no-wandb             Disable Weights & Biases logging
  --gpus N               Number of GPUs for DDP training
```

### Evaluation

```
uv run evaluate --checkpoint PATH [OPTIONS]

Options:
  --checkpoint PATH      Model checkpoint to evaluate
  --challenges PATH      Path to challenges JSON
  --solutions PATH       Path to solutions JSON (for scoring)
  --output PATH          Output submission JSON path
  --tta-steps N          Test-time adaptation steps (0 to disable)
  --tta-lr FLOAT         TTA learning rate
  --num-color-perms N    Color permutations for voting
  --temperature FLOAT    Sampling temperature (0 = greedy)
```

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test module
uv run pytest tests/test_transformer.py -v
```

## License

MIT License. See LICENSE file for details.

## Citation

If you use this code in your research, please cite the original mdlARC work:

```
@misc{vadke2024mdlarc,
  author = {Vadke, Mithil},
  title = {mdlARC: MDL Principles for ARC-AGI},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/mvakde/mdlARC}
}
```

## References

- [ARC-AGI](https://arcprize.org/) - The Abstraction and Reasoning Corpus
- [Original mdlARC](https://github.com/mvakde/mdlARC) - Mithil Vadke's implementation
- [ARC-AGI Dataset](https://github.com/fchollet/ARC-AGI) - Francois Chollet's dataset repository
- [ConceptARC](https://github.com/victorvikram/ConceptARC) - Extended ARC dataset

