# src/datasets/manifest_dataset.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Any
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    """
    A thin wrapper that exposes only the subset of (rec_path, grasp_idx) pairs
    listed in a manifest.  All manifest paths should be *relative* to the dataset
    root, but we canonicalise aggressively so that absolute / duplicated-root
    forms still work.
    """

    def __init__(self,
                 entries: List[Tuple[str, int]],
                 base_dataset: Any):
        super().__init__()
        self.base = base_dataset
        root = Path(base_dataset.source_root).resolve()
        root_tail = root.parts[-4:]                          # ('data', 'student_grasping', …)

        # ------------------------------------------------------------------ helpers
        def canonical(p: str) -> str:
            """Return an absolute path under <root> with any duplicated
            '<data/student_grasping/studentGrasping/student_grasps_v1>' tail removed."""
            p = Path(p)
            if not p.is_absolute():
                p = root / p
            p = p.resolve()

            # make p relative to root (even if root appears twice)
            rel_parts = Path(*p.parts[len(root.parts):]).parts

            # strip *all* leading copies of the root-tail (defensive against triple copies)
            while rel_parts[:len(root_tail)] == root_tail:
                rel_parts = rel_parts[len(root_tail):]

            return str(root.joinpath(*rel_parts))

        # ------------------------------------------------------------------ lookup table
        idx_map = { (canonical(path), gi): idx
                    for idx, (path, gi) in enumerate(base_dataset.samples) }

        # ------------------------------------------------------------------ manifest → indices
        try:
            self.idxs = [ idx_map[(canonical(path), gi)] for path, gi in entries ]
        except KeyError as err:
            bad_path, _ = err.args[0]
            raise RuntimeError(
                f"Path in manifest not found after canonicalisation:\n{bad_path}\n"
                f"Dataset root: {root}"
            ) from None

    # ---------------------------------------------------------------- Dataset API
    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, i):
        return self.base[self.idxs[i]]
