## Setup

### 1. Create a new VM:

gcloud compute instances create train-regression \
  --zone=europe-west4-b \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --local-ssd=interface=nvme \
  --maintenance-policy=TERMINATE \
  --restart-on-failure \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --tags=allow-ssh

### 2. Mount the SSD disk:

```bash
sudo mkfs.ext4 -F /dev/nvme0n1
sudo mkdir -p /mnt/disks/ssd
sudo mount /dev/nvme0n1 /mnt/disks/ssd
sudo chown -R $USER:$USER /mnt/disks/ssd
```

### 3. Install the requirements:

```bash
pip install -r requirements.txt --user
pip install pytorch-lightning==2.2.5 --no-deps --user
pip install "torchmetrics>=0.11,<0.12" --no-deps --user
pip install "lightning-utilities>=0.8.0" --user
pip install trimesh --user
```

### 4. Add ~/.local/bin to PATH:

```bash
nano ~/.bashrc
export PATH="$HOME/.local/bin:$PATH" # add as a line at the end of the file
source ~/.bashrc
```
### 5. Wandb login and run the training script:

```bash
wandb login
export WANDB_ENTITY=ADLR2025
python src/scripts/trainer_joint.py
```