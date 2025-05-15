# scripts/download_checkpoint.py
import os, requests
url = 'https://github.com/georghess/voxel-mae/releases/download/v0.1/voxelmae_nuscenes.pth'
out_dir = 'ext/voxel-mae/pretrained'
os.makedirs(out_dir, exist_ok=True)
r = requests.get(url, allow_redirects=True)
with open(os.path.join(out_dir, 'voxelmae_nuscenes.pth'), 'wb') as f:
    f.write(r.content)
print('Downloaded checkpoint to', out_dir)