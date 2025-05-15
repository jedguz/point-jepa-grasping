_base_ = './base_config.py'

# Dataset settings
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='TSDFVoxelDataset',
        root='/path/to/tsdf_npy',
        pipeline=[
            dict(type='LoadTSDFFromFile', key='points'),
            dict(type='ToTensor', keys=['points']),
        ],
    ),
    val=dict(
        type='TSDFVoxelDataset',
        root='/path/to/tsdf_npy',
        pipeline=[
            dict(type='LoadTSDFFromFile', key='points'),
            dict(type='ToTensor', keys=['points']),
        ],
    ),
)

# Encoder specification
encoder = dict(
    name='jepa3d',
    pretrained=True,
    checkpoint='ext/jepa3d/pretrained/jepa3d.pth',
    out_dim=512,  # adjust to your model's embedding size
)

# Model composition: backbone + grasp head
model = dict(
    type='TwoStageModel',
    encoder=encoder,
    head=dict(
        type='GraspHead',
        in_dim=encoder['out_dim'],
        hidden_dim=128,
        num_classes=1,
    ),
)

# Training hyperparameters
optimizer = dict(type='Adam', lr=1e-4, weight_decay=0.0)

lr_config = dict(policy='step', step=[50], gamma=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=10)
log_config = dict(interval=20)