import torch
import numpy as np
import torch.nn as nn
from ext.jepa3d.models.encoder_3djepa import Encoder3DJEPA
from ext.point2vec.tokenizer import PointCloudTokenizer

class Point2Vec3DJEPA(nn.Module):
    def __init__(
        self,
        load_src: str,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device

        self.model_3djepa = Encoder3DJEPA.from_pretrained("facebook/3d-jepa").cuda()
        self.p2v_output, self.p2v_positions = self.load_and_tokenize(load_src)


    def create_featurized_scene_dict(self, num_points=2048):
        """
        Create a featurized scene dictionary that matches the expected input format
        for the 3D-JEPA encoder based on the forward method.
        
        Args:
            num_points: Number of points in the scene
            device: Device to create tensors on
            model: The model instance to check expected dimensions
        
        Returns:
            dict: featurized_scene_dict with all required keys
        """
        device = self.device

        floor_mask = torch.rand(num_points, device=device) < 0.3
        ceiling_mask = (~floor_mask) & (torch.rand(num_points, device=device) < 0.2) 
        remaining_mask = ~floor_mask & ~ceiling_mask
        
        # Create RGB colors (0-1 range - model will multiply by 255)
        rgb = torch.rand(num_points, 3, device=device) 
        rgb[floor_mask] = torch.tensor([0.4, 0.3, 0.2], device=device) + torch.rand((floor_mask.sum(), 3), device=device) * 0.3  
        rgb[ceiling_mask] = torch.tensor([0.8, 0.8, 0.8], device=device) + torch.rand((ceiling_mask.sum(), 3), device=device) * 0.2
        rgb = torch.clamp(rgb, 0, 1)
        
        # Determine feature dimensions based on model if available
        if self.model_3djepa is not None and hasattr(self.model_3djepa, 'input_feat_dim'):
            total_feat_dim = self.model_3djepa.input_feat_dim
            print(f"Model expects {total_feat_dim} total feature dimensions")
            
            # Split roughly evenly between CLIP and DINO
            clip_feat_dim = total_feat_dim // 2
            dino_feat_dim = total_feat_dim - clip_feat_dim
            print(f"Using CLIP: {clip_feat_dim}, DINO: {dino_feat_dim}")
        else:
            # The error showed we need 896 total dimensions
            # Current attempt: 512 + 384 = 896
            total_feat_dim = 896
            clip_feat_dim = 512  
            dino_feat_dim = 384  
            print(f"Using default dimensions - CLIP: {clip_feat_dim}, DINO: {dino_feat_dim}, Total: {total_feat_dim}")
        
        # Create CLIP and DINO features with correct dimensions from the point2vec output
        features_clip = self.p2v_output[:, :768].clone().detach().to(device)
        features_dino = self.p2v_output[:, 768:].clone().detach().to(device)
        
        xyz = self.p2v_positions.clone().detach().to(device) * 3

        # Create the featurized scene dictionary
        featurized_scene_dict = {
            "features_clip": features_clip,      # Shape: (num_points, clip_feat_dim)
            "features_dino": features_dino,      # Shape: (num_points, dino_feat_dim)
            "rgb": rgb,                          # Shape: (num_points, 3) in [0,1] range
            "points": xyz,                       # Shape: (num_points, 3)
        }
        
        return featurized_scene_dict


    def load_and_tokenize(self, str: load_src):
        data = np.load(load_src)
        print("Keys in file:", list(data.keys()))

        pointcloud_np = data['points'] 
        pointcloud_tensor = torch.from_numpy(pointcloud_np)

        print(f"Pointcloud shape: {pointcloud_tensor.shape}")

        # TODO: find a way to use tokenizer.embedding pretrained weights from loaded checkpoint
        tokenizer = PointCloudTokenizer(2048, 1, None, 1536)

        tokens, centers = tokenizer(pointcloud_tensor.reshape(1, 2048, 3)) 
        print(f"Tokens shape: {tokens.shape}")
        print(f"Centers shape: {centers.shape}")

        print("Pointcloud statistics:")
        print(f"  Mean: {centers.mean().item():.4f}")
        print(f"  Std: {centers.std().item():.4f}")
        print(f"  Min: {centers.min().item():.4f}")
        print(f"  Max: {centers.max().item():.4f}")

        p2v_output = tokens.squeeze(0)
        p2v_positions = centers.squeeze(0)

        # self.p2v_output = p2v_output
        # self.p2v_positions = p2v_positions

        return p2v_output, p2v_positions


    def jepa_encoder(self):
        model_3djepa = self.model_3djepa

        if hasattr(model_3djepa, 'zero_token'):
            model_3djepa.zero_token = model_3djepa.zero_token.cuda()

        featurized_scene_dict = self.create_featurized_scene_dict(
            num_points=2048, 
            model=model_3djepa
        )

        output = model_3djepa(featurized_scene_dict)

        return output

