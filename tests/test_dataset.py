import torch

from misc.dlrhand2_score_datamodule import DLRHand2Dataset


def test_dataset_shapes(tmp_dataset):
    ds = DLRHand2Dataset(root_dir=str(tmp_dataset), num_points=1024)
    pc, g, s = ds[0]

    assert pc.shape == (1024, 3)
    assert g.shape == (19,)
    assert s.ndim == 0
    assert pc.dtype == torch.float32
