# configs/jepa_grasp_config.py

_base_ = './base_config.py'

# ------------------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------------------
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='FeaturizedPointCloudDataset',   # you’ll implement this
        root='/path/to/your/feats/',
        pipeline=[
            # must load exactly these four keys:
            dict(type='LoadNpy',    keys=['features_clip','features_dino','rgb','points']),
            dict(type='ToTensor',   keys=['features_clip','features_dino','rgb','points']),
        ],
    ),
    val=dict(...),  # same idea
)

# ------------------------------------------------------------------------------
# ENCODER
# ------------------------------------------------------------------------------
encoder = dict(
    type='JEPA3DEncoderWrapper',
    input_feat_dim = 512,      # match how you compute DINO+CLIP dim
    embed_dim       = 768,      # PTv3 hidden dim in their pretrain
    rgb_proj_dim    = 128,      # number of harmonics→proj channels
    ptv3_args       = dict(
        num_layers=12,
        num_heads=12,
        # …match the paper’s or the pretrained ckpt’s hyperparams
    ),
    voxel_size      = 0.05,     # 5cm voxels
    checkpoint_path = 'ext/jepa3d/pretrained/jepa3d.pth',
)

# ------------------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------------------
model = dict(
    type='TwoStageModel',
    encoder=encoder,            # our wrapper
    head=dict(
        type='GraspHead',
        in_dim    = encoder['embed_dim'],
        hidden_dim= 128,
        num_classes= 1,
    ),
)

# ------------------------------------------------------------------------------
# OPT + SCHED
# ------------------------------------------------------------------------------
optimizer = dict(type='Adam', lr=1e-4, weight_decay=0.0)
lr_config = dict(policy='step', step=[50], gamma=0.1)
runner    = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=10)
log_config        = dict(interval=20)
