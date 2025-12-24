"""
ARC Data Pipeline - Tokenization, Augmentation, and Dataset.

Carefully designed to prevent any data leakage:
- Test solutions are NEVER loaded during training
- Augmentations are applied consistently
- All preprocessing is deterministic and reproducible

Supports multiple dataset variants:
- ARC-1 (original ARC dataset)
- ARC-2 / ARC-AGI (extended ARC dataset)
- ConceptARC (concept-based ARC tasks)

Use download_dataset() and build_dataset() for multi-source training.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

# =============================================================================
# TOKENIZATION
# =============================================================================

# Token vocabulary: 0-9 for colors, 10-13 for special tokens
VOCAB_SIZE = 14

# Special token IDs
START_TOKEN_ID = 10
NEWLINE_TOKEN_ID = 11
IO_SEP_TOKEN_ID = 12
END_TOKEN_ID = 13
PAD_TOKEN_ID = END_TOKEN_ID  # Use END token for padding (matches upstream)


def grid_to_tokens(grid: list[list[int]]) -> list[int]:
    """
    Convert a 2D grid to a flat token sequence.
    Inserts NEWLINE token after each row.

    Example: [[1,2],[3,4]] -> [1, 2, NEWLINE, 3, 4, NEWLINE]
    """
    tokens = []
    for row in grid:
        tokens.extend(row)
        tokens.append(NEWLINE_TOKEN_ID)
    return tokens


def tokens_to_grid(tokens: list[int]) -> list[list[int]]:
    """
    Convert a token sequence back to a 2D grid.
    Splits on NEWLINE tokens.
    """
    grid = []
    current_row = []
    for tok in tokens:
        if tok == NEWLINE_TOKEN_ID:
            if current_row:  # Don't add empty rows
                grid.append(current_row)
                current_row = []
        elif tok in (START_TOKEN_ID, IO_SEP_TOKEN_ID, END_TOKEN_ID):
            continue  # Skip special tokens
        else:
            current_row.append(tok)
    if current_row:
        grid.append(current_row)
    return grid


def encode_example(
    input_grid: list[list[int]],
    output_grid: list[list[int]] | None = None,
) -> list[int]:
    """
    Encode an input-output pair as a token sequence.

    Format: <start> input_tokens <io_sep> output_tokens <end>
    If output_grid is None, only encode input: <start> input_tokens <io_sep>
    """
    tokens = [START_TOKEN_ID]
    tokens.extend(grid_to_tokens(input_grid))
    tokens.append(IO_SEP_TOKEN_ID)

    if output_grid is not None:
        tokens.extend(grid_to_tokens(output_grid))
        tokens.append(END_TOKEN_ID)

    return tokens


def decode_output_grid(tokens: list[int]) -> list[list[int]] | None:
    """
    Extract the output grid from a token sequence.
    Returns None if no valid output found.
    """
    # Find IO separator
    try:
        sep_idx = tokens.index(IO_SEP_TOKEN_ID)
    except ValueError:
        return None

    # Extract output tokens (after separator, before end)
    output_tokens = []
    for tok in tokens[sep_idx + 1:]:
        if tok == END_TOKEN_ID:
            break
        output_tokens.append(tok)

    if not output_tokens:
        return None

    return tokens_to_grid(output_tokens)


# =============================================================================
# 3D POSITION COMPUTATION
# =============================================================================

def compute_positions_3d(
    tokens: list[int],
    max_x: int = 30,
    max_y: int = 29,
) -> np.ndarray:
    """
    Compute 3D positions for each token.

    Position encoding (matches original mdlARC):
    - x: column position (0 to max_x, clamped)
    - y: row position (0 to max_y, clamped)
    - z: semantic layer
        - 0: START token and special boundaries
        - 1: input grid tokens (colors)
        - 2: IO separator
        - 3: output grid tokens (colors)
        - 4: END token

    Returns: [seq_len, 3] array of (x, y, z) positions
    """
    positions = np.zeros((len(tokens), 3), dtype=np.int64)

    x, y = 0, 0
    in_output = False

    for i, tok in enumerate(tokens):
        if tok == START_TOKEN_ID:
            # START token at z=0
            positions[i] = [0, 0, 0]
            x, y = 0, 0
            in_output = False
        elif tok == NEWLINE_TOKEN_ID:
            # NEWLINE tokens: z=1 for input, z=3 for output
            z = 3 if in_output else 1
            positions[i] = [0, y, z]
            x = 0
            y = min(y + 1, max_y)
        elif tok == IO_SEP_TOKEN_ID:
            # IO separator at z=2
            positions[i] = [0, 0, 2]
            x, y = 0, 0
            in_output = True
        elif tok == END_TOKEN_ID:
            # END token at z=4
            positions[i] = [0, 0, 4]
        else:
            # Grid tokens (colors 0-9): z=1 for input, z=3 for output
            z = 3 if in_output else 1
            positions[i] = [min(x, max_x), min(y, max_y), z]
            x += 1

    return positions


# =============================================================================
# DIHEDRAL AUGMENTATION (8 geometric transforms)
# =============================================================================

def identity(grid: list[list[int]]) -> list[list[int]]:
    """No transformation."""
    return [row[:] for row in grid]


def rot90(grid: list[list[int]]) -> list[list[int]]:
    """Rotate 90° clockwise."""
    return [list(row) for row in zip(*grid[::-1], strict=False)]


def rot180(grid: list[list[int]]) -> list[list[int]]:
    """Rotate 180°."""
    return [row[::-1] for row in grid[::-1]]


def rot270(grid: list[list[int]]) -> list[list[int]]:
    """Rotate 270° clockwise (90° counter-clockwise)."""
    return [list(row) for row in zip(*grid, strict=False)][::-1]


def flip_h(grid: list[list[int]]) -> list[list[int]]:
    """Flip horizontally."""
    return [row[::-1] for row in grid]


def flip_v(grid: list[list[int]]) -> list[list[int]]:
    """Flip vertically."""
    return grid[::-1]


def flip_diag(grid: list[list[int]]) -> list[list[int]]:
    """Flip along main diagonal (transpose)."""
    return [list(row) for row in zip(*grid, strict=False)]


def flip_anti_diag(grid: list[list[int]]) -> list[list[int]]:
    """Flip along anti-diagonal (rot180 of transpose)."""
    transposed = [list(row) for row in zip(*grid, strict=False)]
    return [row[::-1] for row in transposed[::-1]]


# All 8 dihedral transformations
DIHEDRAL_TRANSFORMS = [
    identity,
    rot90,
    rot180,
    rot270,
    flip_h,
    flip_v,
    flip_diag,
    flip_anti_diag,
]


def apply_dihedral_transform(
    input_grid: list[list[int]],
    output_grid: list[list[int]],
    transform_idx: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Apply the same dihedral transform to both input and output grids."""
    transform = DIHEDRAL_TRANSFORMS[transform_idx]
    return transform(input_grid), transform(output_grid)


def inverse_dihedral(grid: list[list[int]], transform_idx: int) -> list[list[int]]:
    """Apply inverse of a dihedral transform to recover original orientation."""
    # Inverse mappings: [identity, rot270, rot180, rot90, flip_h, flip_v, flip_diag, flip_anti_diag]
    inverse_idx = [0, 3, 2, 1, 4, 5, 6, 7][transform_idx]
    return DIHEDRAL_TRANSFORMS[inverse_idx](grid)


# =============================================================================
# COLOR AUGMENTATION
# =============================================================================

def generate_color_permutations(n_perms: int, seed: int = 42) -> np.ndarray:
    """
    Generate n color permutations.

    Colors 1-9 are permuted; color 0 (background) stays fixed.
    First permutation is always identity.

    Returns: [n_perms, 10] array where each row maps old_color -> new_color
    """
    rng = np.random.default_rng(seed)

    perms = np.zeros((n_perms, 10), dtype=np.int64)
    perms[:, 0] = 0  # Color 0 always maps to 0

    # First perm is identity
    perms[0, 1:] = np.arange(1, 10)

    # Generate random permutations for rest
    for i in range(1, n_perms):
        perms[i, 1:] = rng.permutation(np.arange(1, 10))

    return perms


def apply_color_perm_to_tokens(
    tokens: torch.Tensor,
    perm: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a color permutation to token sequences.

    Args:
        tokens: [batch, seq] token IDs
        perm: [10] mapping old_color -> new_color

    Returns: Permuted tokens (special tokens unchanged)
    """
    result = tokens.clone()
    # Only permute color tokens (0-9)
    mask = tokens < 10
    result[mask] = perm[tokens[mask]]
    return result


def inverse_color_perm(perm: np.ndarray) -> np.ndarray:
    """Compute inverse of a color permutation."""
    inv = np.zeros_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


class ColorAugmentor:
    """Deterministic per-epoch color augmentation scheduler."""

    def __init__(
        self,
        perms: torch.Tensor,
        apply_to_test_split: bool = False,
        seed: int = 42,
    ) -> None:
        self.perms = perms
        self.apply_to_test_split = apply_to_test_split
        self.seed = seed
        self._epoch = 0
        self._cached_index = 0
        self._compute_index()

    @property
    def num_permutations(self) -> int:
        return int(self.perms.size(0)) if self.perms is not None else 0

    @property
    def current_index(self) -> int:
        return self._cached_index

    def set_index(self, epoch: int) -> None:
        if self.num_permutations == 0:
            return
        self._epoch = max(0, int(epoch))
        self._compute_index()

    def _compute_index(self) -> None:
        n = self.num_permutations
        if n == 0:
            self._cached_index = 0
            return

        cycle = self._epoch // n
        step = self._epoch % n

        # First step of any cycle is identity.
        if step == 0 or n <= 1:
            self._cached_index = 0
            return

        g = torch.Generator()
        g.manual_seed(self.seed + cycle)

        # Permute indices [1...n-1]
        perm = torch.randperm(n - 1, generator=g)
        random_offset = perm[step - 1].item()
        self._cached_index = random_offset + 1

    def mapping_for_split(self, split: str) -> torch.Tensor | None:
        if self.num_permutations == 0:
            return None
        if split == "test" and not self.apply_to_test_split:
            return None
        return self.perms[self._cached_index]

    def get_perm(self) -> torch.Tensor:
        """Get the current permutation tensor."""
        if self.num_permutations == 0:
            raise ValueError("ColorAugmentor has no permutations configured.")
        return self.perms[self._cached_index]

    def apply(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply the current permutation to tokens."""
        if self.num_permutations <= 1:
            return tokens
        perm = self.get_perm()
        return apply_color_perm_to_tokens(tokens, perm)


# =============================================================================
# DATASET
# =============================================================================

@dataclass
class ARCExample:
    """A single ARC example (one input-output pair from a task)."""
    task_id: str
    example_id: int  # Integer task identifier (all pairs from same task share this)
    pair_idx: int
    split: str  # "train" or "test"
    tokens: list[int]
    positions: np.ndarray  # [seq_len, 3]
    has_output: bool
    dihedral_idx: int = 0  # Which of 8 transforms was applied


class ARCDataset(Dataset):
    """
    ARC Dataset with careful train/test separation.

    IMPORTANT: Test solutions are NEVER loaded during training.
    They can only be loaded separately for evaluation scoring.
    """

    def __init__(
        self,
        data_path: str | Path,
        splits: tuple[str, ...] = ("train",),
        include_test_outputs: bool = False,
        solutions_path: str | Path | None = None,
        apply_dihedral: bool = False,
        max_seq_len: int = 2048,
        drop_long: bool = True,
    ):
        """
        Args:
            data_path: Path to challenges JSON file
            splits: Which splits to include ("train", "test", or both)
            include_test_outputs: If True and solutions_path provided, include test outputs
            solutions_path: Path to solutions JSON (ONLY for evaluation)
            apply_dihedral: If True, expand dataset 8x with dihedral augmentations
            max_seq_len: Maximum sequence length
            drop_long: If True, drop sequences longer than max_seq_len
        """
        self.max_seq_len = max_seq_len
        self.drop_long = drop_long
        self.apply_dihedral = apply_dihedral

        # Load challenges
        with open(data_path) as f:
            challenges = json.load(f)

        # Load solutions ONLY if explicitly requested for eval
        solutions = {}
        if include_test_outputs and solutions_path is not None:
            with open(solutions_path) as f:
                solutions = json.load(f)

            # LEAK DETECTION: Warn if solutions contain tasks not in challenges
            # This could indicate eval solutions leaking into training data
            extra_tasks = set(solutions.keys()) - set(challenges.keys())
            if extra_tasks:
                import warnings

                warnings.warn(
                    f"DATA LEAK WARNING: solutions_path contains {len(extra_tasks)} task(s) "
                    f"not in challenges file. These may be eval solutions that should not be "
                    f"used for training. Extra tasks: {list(extra_tasks)[:5]}..."
                    if len(extra_tasks) > 5
                    else f"DATA LEAK WARNING: solutions_path contains {len(extra_tasks)} task(s) "
                    f"not in challenges file: {list(extra_tasks)}",
                    UserWarning,
                    stacklevel=2,
                )

        # Build task_id -> example_id mapping (integer indices)
        # Sort for reproducibility
        task_ids = sorted(challenges.keys())
        self.task_id_to_example_id = {tid: idx for idx, tid in enumerate(task_ids)}
        self.num_tasks = len(task_ids)

        # Build examples
        self.examples: list[ARCExample] = []

        for task_id, task_data in challenges.items():
            example_id = self.task_id_to_example_id[task_id]

            for split in splits:
                if split not in task_data:
                    continue

                pairs = task_data[split]
                task_solutions = solutions.get(task_id, [])

                for pair_idx, pair in enumerate(pairs):
                    input_grid = pair["input"]

                    # Get output grid
                    if split == "train":
                        output_grid = pair["output"]
                    elif include_test_outputs and pair_idx < len(task_solutions):
                        output_grid = task_solutions[pair_idx]
                    else:
                        output_grid = None

                    # Apply dihedral augmentations
                    n_transforms = 8 if apply_dihedral else 1

                    for dihedral_idx in range(n_transforms):
                        # Transform grids
                        if dihedral_idx > 0 and output_grid is not None:
                            aug_input, aug_output = apply_dihedral_transform(
                                input_grid, output_grid, dihedral_idx
                            )
                        elif dihedral_idx > 0:
                            aug_input = DIHEDRAL_TRANSFORMS[dihedral_idx](input_grid)
                            aug_output = None
                        else:
                            aug_input, aug_output = input_grid, output_grid

                        # Encode to tokens
                        tokens = encode_example(aug_input, aug_output)

                        # Skip if too long
                        if drop_long and len(tokens) > max_seq_len:
                            continue

                        # Compute 3D positions
                        positions = compute_positions_3d(tokens)

                        self.examples.append(ARCExample(
                            task_id=task_id,
                            example_id=example_id,
                            pair_idx=pair_idx,
                            split=split,
                            tokens=tokens,
                            positions=positions,
                            has_output=output_grid is not None,
                            dihedral_idx=dihedral_idx,
                        ))

        # Sort by length for efficient batching
        self.examples.sort(key=lambda x: len(x.tokens))

        # Precompute lengths for bucketing
        self.lengths = [len(ex.tokens) for ex in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {
            "tokens": ex.tokens,
            "positions": ex.positions,
            "task_id": ex.task_id,
            "example_id": ex.example_id,
            "pair_idx": ex.pair_idx,
            "split": ex.split,
            "has_output": ex.has_output,
            "dihedral_idx": ex.dihedral_idx,
        }


# =============================================================================
# BATCHING
# =============================================================================

class LengthBucketSampler(Sampler):
    """
    Samples batches of similar-length sequences to minimize padding.

    Groups examples into buckets by length, then samples batches
    from within each bucket.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        # Sort indices by length
        indices = np.argsort(self.lengths)

        # Group into batches
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size].tolist()
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        # Shuffle batches (not within batches - keep length grouping)
        if self.shuffle:
            rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self.epoch = epoch


def collate_fn(
    batch: list[dict],
    color_perms: torch.Tensor | None = None,
    color_perm_idx: int = 0,
) -> dict:
    """
    Collate a batch of examples with padding.

    Args:
        batch: List of example dicts
        color_perms: Optional [n_perms, 10] color permutation tensor
        color_perm_idx: Which permutation to apply (0 = identity)

    Returns:
        Collated batch dict with padded tensors
    """
    max_len = max(len(ex["tokens"]) for ex in batch)
    batch_size = len(batch)

    # Pad tokens with PAD_TOKEN_ID (END token), positions with 0
    input_ids = torch.full((batch_size, max_len), PAD_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    positions_3d = torch.zeros(batch_size, max_len, 3, dtype=torch.long)
    example_ids = torch.zeros(batch_size, dtype=torch.long)

    for i, ex in enumerate(batch):
        seq_len = len(ex["tokens"])
        input_ids[i, :seq_len] = torch.tensor(ex["tokens"], dtype=torch.long)
        attention_mask[i, :seq_len] = True
        positions_3d[i, :seq_len] = torch.tensor(ex["positions"], dtype=torch.long)
        example_ids[i] = ex["example_id"]

    # Apply color augmentation if requested
    if color_perms is not None and color_perm_idx > 0:
        input_ids = apply_color_perm_to_tokens(input_ids, color_perms[color_perm_idx])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "positions_3d": positions_3d,
        "example_ids": example_ids,
        "task_ids": [ex["task_id"] for ex in batch],
        "pair_idxs": [ex["pair_idx"] for ex in batch],
        "splits": [ex["split"] for ex in batch],
        "has_outputs": [ex["has_output"] for ex in batch],
        "dihedral_idxs": [ex["dihedral_idx"] for ex in batch],
    }


def create_dataloader(
    dataset: ARCDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    color_perms: torch.Tensor | None = None,
    color_perm_idx: int = 0,
    sampler: torch.utils.data.Sampler | None = None,
) -> DataLoader:
    """Create a DataLoader with length-bucketed batching.

    Args:
        dataset: The ARC dataset
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored if sampler provided)
        num_workers: Number of data loading workers
        color_perms: Color permutation tensor
        color_perm_idx: Which color permutation to use
        sampler: Optional external sampler (e.g., DistributedSampler for DDP)

    Returns:
        DataLoader instance
    """
    def collate(batch):
        return collate_fn(batch, color_perms, color_perm_idx)

    if sampler is not None:
        # Use external sampler (e.g., DistributedSampler for DDP)
        # Note: shuffle is controlled by the sampler in this case
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,  # Important for DDP to keep batch sizes consistent
        )
    else:
        # Use length-bucketed batching
        bucket_sampler = LengthBucketSampler(
            dataset.lengths,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_sampler=bucket_sampler,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=True,
        )


# =============================================================================
# DATASET BUILDING UTILITIES
# =============================================================================

# Dataset source URLs
ARC_DATASET_URLS = {
    "arc1": "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip",
    "arc2": "https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.zip",
    "concept": "https://github.com/victorvikram/ConceptARC/archive/refs/heads/main.zip",
}


def download_dataset(
    dataset: str,
    output_dir: str | Path = "data",
    force: bool = False,
) -> Path:
    """
    Download an ARC dataset variant.

    Args:
        dataset: One of "arc1", "arc2", or "concept"
        output_dir: Directory to save to
        force: If True, re-download even if exists

    Returns:
        Path to the extracted data directory
    """
    import urllib.request
    import zipfile

    if dataset not in ARC_DATASET_URLS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(ARC_DATASET_URLS.keys())}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = ARC_DATASET_URLS[dataset]
    zip_path = output_dir / f"{dataset}.zip"

    # Expected extracted directory names
    extract_names = {
        "arc1": "ARC-AGI-master",
        "arc2": "ARC-AGI-2-main",
        "concept": "ConceptARC-main",
    }
    extract_dir = output_dir / extract_names[dataset]

    if not extract_dir.exists() or force:
        print(f"Downloading {dataset} from {url}...")
        urllib.request.urlretrieve(url, zip_path)

        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)

        zip_path.unlink()
        print(f"Done! Extracted to {extract_dir}")

    # Return path to data subdirectory
    if dataset == "concept":
        return extract_dir / "corpus"
    else:
        return extract_dir / "data"


def load_arc_tasks(
    data_dir: str | Path,
    split: str = "training",
) -> dict[str, dict]:
    """
    Load ARC tasks from a directory.

    Args:
        data_dir: Path to ARC data directory (containing training/evaluation folders)
        split: "training" or "evaluation"

    Returns:
        Dict mapping task_id -> task_data
    """
    data_dir = Path(data_dir)
    split_dir = data_dir / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    tasks = {}
    for json_file in sorted(split_dir.glob("*.json")):
        task_id = json_file.stem
        with open(json_file) as f:
            tasks[task_id] = json.load(f)

    return tasks


def load_concept_arc_tasks(corpus_dir: str | Path) -> dict[str, dict]:
    """
    Load ConceptARC tasks organized by concept.

    ConceptARC structure: corpus/<concept>/<task_id>.json

    Returns:
        Dict mapping "concept_taskid" -> task_data
    """
    corpus_dir = Path(corpus_dir)
    tasks = {}

    for concept_dir in sorted(corpus_dir.iterdir()):
        if not concept_dir.is_dir():
            continue

        concept_name = concept_dir.name
        for json_file in sorted(concept_dir.glob("*.json")):
            task_id = f"{concept_name}_{json_file.stem}"
            with open(json_file) as f:
                tasks[task_id] = json.load(f)

    return tasks


def build_dataset(
    sources: list[tuple[str, str | Path]],
    output_path: str | Path,
    strip_eval_outputs: bool = True,
    prefix_task_ids: bool = True,
) -> dict[str, dict]:
    """
    Build a combined dataset from multiple sources.

    Args:
        sources: List of (source_name, data_path) tuples
                 source_name is used for prefixing task IDs
        output_path: Where to save combined JSON
        strip_eval_outputs: If True, remove outputs from evaluation tasks
        prefix_task_ids: If True, prefix task IDs with source name

    Returns:
        Combined task dictionary

    Example:
        build_dataset([
            ("arc1", "data/ARC-master/data"),
            ("arc2", "data/ARC-AGI-master/data"),
            ("concept", "data/ConceptARC-main/corpus"),
        ], "data/combined.json")
    """
    combined = {}

    for source_name, data_path in sources:
        data_path = Path(data_path)

        # Load based on source type
        if source_name == "concept" or "corpus" in str(data_path):
            tasks = load_concept_arc_tasks(data_path)
        else:
            # Standard ARC format
            for split in ["training", "evaluation"]:
                split_dir = data_path / split
                if not split_dir.exists():
                    continue

                for json_file in sorted(split_dir.glob("*.json")):
                    task_id = json_file.stem
                    with open(json_file) as f:
                        task_data = json.load(f)

                    # Strip eval outputs if requested
                    if strip_eval_outputs and split == "evaluation":
                        for pair in task_data.get("test", []):
                            pair.pop("output", None)

                    # Prefix task ID
                    if prefix_task_ids:
                        full_id = f"{source_name}_{task_id}"
                    else:
                        full_id = task_id

                    combined[full_id] = task_data

            continue  # Skip the common assignment below

        # For ConceptARC (already loaded above)
        for task_id, task_data in tasks.items():
            if prefix_task_ids:
                full_id = f"{source_name}_{task_id}"
            else:
                full_id = task_id
            combined[full_id] = task_data

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Built dataset with {len(combined)} tasks -> {output_path}")
    return combined


def build_training_solutions(
    data_dir: str | Path,
    output_path: str | Path,
) -> dict[str, list]:
    """
    Build solutions file for training set test pairs.

    This extracts test outputs from the training split (where they exist)
    for use during training. Does NOT include evaluation set solutions.

    Args:
        data_dir: Path to ARC data directory
        output_path: Where to save solutions JSON

    Returns:
        Solutions dict mapping task_id -> list of test outputs
    """
    data_dir = Path(data_dir)
    solutions = {}

    training_dir = data_dir / "training"
    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")

    for json_file in sorted(training_dir.glob("*.json")):
        task_id = json_file.stem
        with open(json_file) as f:
            task_data = json.load(f)

        # Extract test outputs
        test_outputs = []
        for pair in task_data.get("test", []):
            if "output" in pair:
                test_outputs.append(pair["output"])

        if test_outputs:
            solutions[task_id] = test_outputs

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(solutions, f, indent=2)

    print(f"Built training solutions for {len(solutions)} tasks -> {output_path}")
    return solutions


# Legacy function for backward compatibility
def download_arc_dataset(output_dir: str | Path = "data") -> Path:
    """Download the official ARC-AGI dataset. Returns path to data directory."""
    return download_dataset("arc2", output_dir)


def build_combined_dataset(
    arc_data_dir: str | Path,
    output_path: str | Path,
    include_eval_outputs: bool = False,
) -> None:
    """
    Build a combined challenges JSON from ARC data directory.

    IMPORTANT: By default, evaluation outputs are NOT included.
    This is critical for preventing data leakage.
    """
    arc_data_dir = Path(arc_data_dir)

    combined = {}

    # Load training challenges and solutions
    train_challenges_dir = arc_data_dir / "training"
    for json_file in train_challenges_dir.glob("*.json"):
        task_id = json_file.stem
        with open(json_file) as f:
            task_data = json.load(f)
        combined[task_id] = task_data

    # Load evaluation challenges (NO outputs by default)
    eval_challenges_dir = arc_data_dir / "evaluation"
    for json_file in eval_challenges_dir.glob("*.json"):
        task_id = json_file.stem
        with open(json_file) as f:
            task_data = json.load(f)

        if not include_eval_outputs:
            # Strip outputs from test pairs
            for pair in task_data.get("test", []):
                pair.pop("output", None)

        combined[task_id] = task_data

    # Save combined dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(combined, f)

    print(f"Built combined dataset with {len(combined)} tasks -> {output_path}")


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    # Test tokenization
    print("Testing tokenization...")
    input_grid = [[1, 2, 3], [4, 5, 6]]
    output_grid = [[7, 8], [9, 0]]

    tokens = encode_example(input_grid, output_grid)
    print(f"Input grid: {input_grid}")
    print(f"Output grid: {output_grid}")
    print(f"Tokens: {tokens}")

    decoded = decode_output_grid(tokens)
    print(f"Decoded output: {decoded}")
    assert decoded == output_grid, "Decode mismatch!"

    # Test positions
    positions = compute_positions_3d(tokens)
    print(f"Positions shape: {positions.shape}")
    print(f"First few positions (x,y,z): {positions[:5]}")

    # Test dihedral transforms
    print("\nTesting dihedral transforms...")
    grid = [[1, 2], [3, 4]]
    for i, transform in enumerate(DIHEDRAL_TRANSFORMS):
        print(f"  Transform {i}: {transform(grid)}")

    # Test color permutations
    print("\nTesting color permutations...")
    perms = generate_color_permutations(5, seed=42)
    print(f"Color perms shape: {perms.shape}")
    print(f"First 3 perms:\n{perms[:3]}")

    print("\nAll tests passed!")
