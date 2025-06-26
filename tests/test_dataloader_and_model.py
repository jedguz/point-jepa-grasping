import torch

from scripts.dlrhand2_score_datamodule import DLRHand2DataModule
from scripts.score_regressor import GraspRegressor


def test_forward_backward(tmp_dataset):
    dm = DLRHand2DataModule(
        root_dir=str(tmp_dataset),
        batch_size=2,
        num_workers=0,
        num_points=512,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    pc, grasps, scores = batch

    model = GraspRegressor(
        num_points=512,
        tokenizer_groups=16,
        tokenizer_group_size=32,
        tokenizer_radius=0.05,
        encoder_dim=64,
        encoder_depth=2,
        encoder_heads=4,
        encoder_dropout=0.1,
        encoder_attn_dropout=0.1,
        encoder_drop_path_rate=0.0,
        pooling_type="mean",
        pooling_heads=4,
        head_hidden_dims=[128, 64],
        head_output_dim=1,
        grasp_dim=19,
        lr_backbone=1e-3,
        lr_head=1e-2,
    )

    out = model(pc, grasps)
    assert out.shape == (2, 1)

    loss = torch.nn.functional.mse_loss(out.squeeze(-1), scores)
    loss.backward()
