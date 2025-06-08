import torch
from ext.point2vec.tokenizer import PointCloudTokenizer

# 1) load the Lightning checkpoint
ckpt = torch.load(
    "checkpoints/pre_point2vec-epoch.799-step.64800.ckpt",
    map_location="cpu"
)
sd = ckpt.get("state_dict", ckpt)

# 2) pull out only the tokenizer keys
tok_sd = {
    k.replace("tokenizer.", ""): v
    for k, v in sd.items()
    if k.startswith("tokenizer.")
}

# 3) instantiate a fresh tokenizer with the same hyperparams
tokenizer = PointCloudTokenizer(
    num_groups=512,
    group_size=32,
    group_radius=None,
    token_dim=384,
)

# 4) load the filtered state dict
tokenizer.load_state_dict(tok_sd, strict=True)

# 5) save the standalone tokenizer
torch.save(
    tokenizer.state_dict(),
    "checkpoints/point2vec_tokenizer_only.pt"
)
print("✅ Extracted tokenizer weights!")