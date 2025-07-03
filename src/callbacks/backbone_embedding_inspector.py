# callbacks/backbone_embedding_inspector.py
from __future__ import annotations
import os

import pytorch_lightning as pl
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb
from hydra.utils import get_original_cwd


class BackboneEmbeddingInspector(pl.Callback):
    def __init__(self, num_batches: int = 16):
        super().__init__()
        self.num_batches = num_batches

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.eval()
        feats, labels = [], []

        dl = trainer.datamodule.val_dataloader()
        device = pl_module.device

        with torch.no_grad():
            for bi, batch in enumerate(dl):
                if bi >= self.num_batches:
                    break
                pts  = batch["points"].to(device)
                meta = batch["meta"]  # now exists

                # get the backbone embedding
                tok, cen = pl_module.tokenizer(pts)
                pos      = pl_module.positional_encoding(cen)
                z        = pl_module.encoder(tok, pos).last_hidden_state
                emb      = pl_module.pool(z)

                feats.append(emb.cpu())

                # EXTEND with the *string* synset for each sample
                labels.extend(meta["synset"])

        feats = torch.cat(feats).numpy()
        emb2d = TSNE(
            n_components=2, perplexity=30, metric="cosine",
            init="random", random_state=0
        ).fit_transform(feats)

        # Plot and annotate each cluster at its centroid
        plt.figure(figsize=(8, 8))
        for syn in sorted(set(labels)):
            idx = [i for i, l in enumerate(labels) if l == syn]
            xs, ys = emb2d[idx, 0], emb2d[idx, 1]
            plt.scatter(xs, ys, s=12, alpha=0.6)
            cx, cy = xs.mean(), ys.mean()
            plt.text(
                cx, cy, syn,
                fontsize=10, weight="bold",
                ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, pad=1)
            )

        plt.axis("off")
        plt.title("t-SNE of frozen PointJEPA embeddings")

        # log to W&B
        img = wandb.Image(plt.gcf(), caption="PointJEPA t-SNE by synset")
        trainer.logger.experiment.log({"backbone_tsne_by_synset": img}, step=0)

        # also save locally
        proj = get_original_cwd()
        out = os.path.join(proj, trainer.default_root_dir or "artifacts")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "tsne_by_synset.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"→ saved t-SNE to {path}")

        plt.close()
        pl_module.train()
