#!/usr/bin/env python3
# Not finished: adapted from Dominiks code
"""
Usage: python3 vis_grasp.py \
            --checkpoint configs/checkpoints/jepa_no_FT.ckpt \
            --data_dir student_grasps_v1/02808440/148ec8030e52671d44221bef0fa3c36b/0/ \
            --urdf urdfs/dlr2.urdf
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import pybullet as p
import pybullet_data
from scripts.joint_regressor import JointRegressor
from scripts.dlrhand2_joint_datamodule import _sample_mesh_as_pc

# ---------------------------------------------------------------------------- #
#                                 ARGUMENTS                                     #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True,
                    help="Path to your trained .ckpt file")
parser.add_argument("--data_dir",   type=Path, required=True,
                    help="Folder containing mesh.obj + recording.npz")
parser.add_argument("--urdf",       type=Path, required=True,
                    help="Path to your DLR hand URDF")
parser.add_argument("--num_points", type=int, default=2048,
                    help="Number of mesh points to sample")
args = parser.parse_args()

# ---------------------------------------------------------------------------- #
#                                  LOAD MODEL                                   #
# ---------------------------------------------------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: JointRegressor = JointRegressor.load_from_checkpoint(
    str(args.checkpoint),
    map_location=device
)
model.eval().to(device)
print(f"[INFO] Loaded model → {sum(p.numel() for p in model.parameters())/1e6:.1f} M params")

# ---------------------------------------------------------------------------- #
#                           LOAD MESH & POINT CLOUD                            #
# ---------------------------------------------------------------------------- #
mesh_path = args.data_dir / "mesh.obj"
if not mesh_path.is_file():
    raise FileNotFoundError(f"Mesh not found: {mesh_path}")
print(f"[INFO] Sampling {args.num_points} points from mesh")
pc = _sample_mesh_as_pc(str(mesh_path), n=args.num_points)  # (N,3) np.float32
# convert once to tensor batch
pc_t = torch.from_numpy(pc).unsqueeze(0).to(device)         # (1,N,3)

# ---------------------------------------------------------------------------- #
#                          LOAD RECORDED GRASPS                                 #
# ---------------------------------------------------------------------------- #
rec = np.load(args.data_dir / "recording.npz")
grasps = rec["grasps"].astype(np.float32)   # (G, 19) = pose7 + joints12
scores = rec["scores"].astype(np.float32)   # (G,)
idxs   = np.argsort(scores)[::-1]
print(f"[INFO] Found {len(grasps)} grasps, showing top {len(idxs)}")

# ---------------------------------------------------------------------------- #
#                              START PYBULLET                                   #
# ---------------------------------------------------------------------------- #
print("[INFO] Launching PyBullet…")
p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

# Load object as both visual+collision
col = p.createCollisionShape(p.GEOM_MESH, fileName=str(mesh_path), meshScale=[1]*3)
vis = p.createVisualShape   (p.GEOM_MESH, fileName=str(mesh_path), meshScale=[1]*3)
obj = p.createMultiBody(baseMass=1.0,
                        baseCollisionShapeIndex=col,
                        baseVisualShapeIndex=vis,
                        basePosition=[0,0,0.05])

# Load DLR hand
hand_id = p.loadURDF(str(args.urdf),
                     basePosition=[0,0,0.2],
                     baseOrientation=p.getQuaternionFromEuler([0,0,0]),
                     useFixedBase=True,
                     flags=p.URDF_MAINTAIN_LINK_ORDER)
n_joints = p.getNumJoints(hand_id)
assert n_joints >= 12, f"Expected ≥12 joints in URDF, got {n_joints}"

# ---------------------------------------------------------------------------- #
#                         VISUALIZE EACH PREDICTION                             #
# ---------------------------------------------------------------------------- #
for i in idxs:
    pose7 = grasps[i, :7]        # x,y,z + qw,qx,qy,qz
    # Run the network to predict joints
    pose_t = torch.from_numpy(pose7).unsqueeze(0).to(device)  # (1,7)
    with torch.no_grad(), torch.cuda.amp.autocast(device_type=device.type):
        pred12 = model(pc_t, pose_t)[0].cpu().numpy()        # (12,)
    print(f"> Grasp {i}: score {scores[i]:.3f}, pred_joints (rad) = {pred12.round(3)}")

    # Reset hand base to the recorded pose
    p.resetBasePositionAndOrientation(
        bodyUniqueId=hand_id,
        posObj=pose7[:3].tolist(),
        ornObj=pose7[3:].tolist()
    )

    # Apply predicted joint angles
    # assuming joints [1,2,3, 7,8,9, 13,14,15, 19,20,21] map to fingers
    finger_joints = [1,2,3, 7,8,9, 13,14,15, 19,20,21]
    for k, jidx in enumerate(finger_joints):
        p.resetJointState(hand_id, jidx, targetValue=float(pred12[k]), targetVelocity=0.0)
        # if the URDF uses coupled joints, mirror as needed:
        if jidx in [3, 9, 15, 21]:
            p.resetJointState(hand_id, jidx+1, targetValue=float(pred12[k]), targetVelocity=0.0)

    input("Press [Enter] for next grasp…")

# clean up
p.disconnect()
