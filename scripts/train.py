import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mmcv import Config
from src.models.encoders import build_encoder, ENCODERS
from src.models.heads.grasp_head import GraspHead
from src.models.utils.metrics import binary_accuracy
from src.data.tsdf_dataset import TSDFVoxelDataset


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    for batch in loader:
        x = batch['points'].to(device)
        y = batch['label'].to(device).float().unsqueeze(1)
        preds = model(x)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_acc += binary_accuracy(preds, y) * x.size(0)
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch['points'].to(device)
            y = batch['label'].to(device).float().unsqueeze(1)
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item() * x.size(0)
            total_acc += binary_accuracy(preds, y) * x.size(0)
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--encoder', choices=list(ENCODERS.keys()), required=True)
    args = parser.parse_args()

    # Load config\    cfg = Config.fromfile(args.config)

    # Prepare device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build encoder and head
    enc_cfg = cfg.encoder
    encoder = build_encoder(
        args.encoder,
        pretrained=enc_cfg.pretrained,
        checkpoint_path=enc_cfg.checkpoint,
    )
    head = GraspHead(
        in_dim=cfg.model.head.in_dim,
        hidden_dim=cfg.model.head.hidden_dim,
    )
    model = nn.Sequential(encoder, head).to(device)

    # Data loaders\    train_dataset = TSDFVoxelDataset(root=cfg.data.train.root)
    val_dataset = TSDFVoxelDataset(root=cfg.data.val.root)
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.samples_per_gpu,
                              shuffle=True, num_workers=cfg.data.workers_per_gpu)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.samples_per_gpu,
                            shuffle=False, num_workers=cfg.data.workers_per_gpu)

    # Loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        model.parameters(), lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.get('weight_decay', 0)
    )

    best_val_acc = 0.0
    for epoch in range(1, cfg.runner.max_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{cfg.runner.max_epochs} "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_fp = os.path.join('checkpoints', f"best_{args.encoder}.pth")
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), ckpt_fp)
    print("Training complete. Best Val Acc:", best_val_acc)

if __name__ == '__main__':
    main()