point2vec:
  checkpoint: "checkpoints/point2vec_tokenizer_only.pt"
  num_groups: 256
  group_size: 32
  group_radius: None
  token_dim: 128

jepa3d:
  checkpoint: "checkpoints/pre_point2vec-epoch.799-step.64800.ckpt"
  input_feat_dim: 128
  embed_dim: 768
  rgb_proj_dim: 64
  num_rgb_harmonic_functions: 16
  ptv3_args:
    depth: 12
    heads: 12
    mlp_ratio: 4
  voxel_size: 0.05
