# scripts/load_backbone.py
import torch

def load_pretrained_backbone(model, ckpt_path: str):
    """
    Load tokenizer, positional MLP, encoder and pool weights from a JEPA
    checkpoint. Anything named 'head.*' is ignored.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)          # works for Lightning & plain
    keep = {k: v for k, v in state.items()
            if k.startswith(("tokenizer.", "positional_encoding.",
                             "encoder.", "pool."))}
    missing, unexpected = model.load_state_dict(keep, strict=False)
    print(f"[JEPA] loaded {len(keep)} tensors "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
