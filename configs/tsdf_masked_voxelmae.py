_base_ = '../ext/voxel-mae/configs/sst_masked/tsdf_masked_default.py'

# change data root to your TSDF folder
data_root = '/path/to/your/tsdf_npy_folder'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='TSDFVoxelDataset',
        root=data_root,
        pipeline=[
            dict(type='LoadTSDFFromFile', key='points'),    # your custom loader name
            dict(type='MaskTensors', keys=['points'], mask_ratio=0.75),
            dict(type='ToTensor', keys=['points']),
        ]
    ),
    val=dict(
        type='TSDFVoxelDataset',
        root=data_root,
        pipeline=[
            dict(type='LoadTSDFFromFile', key='points'),
            dict(type='ToTensor', keys=['points']),
        ]
    )
)

# optimizer & schedule (tweak as desired)
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.05)
lr_config = dict(policy='step', step=[100], gamma=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=10)
log_config = dict(interval=20)