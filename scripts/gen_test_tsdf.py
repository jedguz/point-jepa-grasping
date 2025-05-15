import os
import numpy as np

def create_sphere_tsdf(shape=(64,64,64), radius=20):
    # Center coordinates
    D, H, W = shape
    cz, cy, cx = D//2, H//2, W//2
    # Grid of coordinates
    z = np.arange(D)[:,None,None]
    y = np.arange(H)[None,:,None]
    x = np.arange(W)[None,None,:]
    # Signed distance: positive outside, negative inside
    dist = np.sqrt((z-cz)**2 + (y-cy)**2 + (x-cx)**2) - radius
    # Truncate distances to [-radius, radius]
    tsdf = np.clip(dist, -radius, radius)
    # Normalize to [-1,1]
    tsdf = tsdf / radius
    return tsdf

if __name__ == '__main__':
    out_dir = 'data/tsdf_npy'
    # Ensure directory exists
    os.makedirs(out_dir, exist_ok=True)
    tsdf = create_sphere_tsdf()
    out_path = os.path.join(out_dir, 'test_sphere.npy')
    np.save(out_path, tsdf)
    print(f'Saved synthetic TSDF: {out_path}')