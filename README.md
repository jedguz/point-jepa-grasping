## Setup

### *Update 6 July: Changed A100 to V100*

### 1. Create a new VM:

gcloud compute instances create train-regression \\  
  --zone=europe-west4-b \\  
  --machine-type=n1-standard-8 \\  
  --accelerator=type=nvidia-tesla-v100,count=1 \\  
  --local-ssd=interface=nvme \\  
  --maintenance-policy=TERMINATE \\  
  --restart-on-failure \\   
  --image-family=pytorch-latest-gpu \\  
  --image-project=deeplearning-platform-release \\  
  --boot-disk-size=200GB \\  
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


## Alternative Using Snapshots

1. Create a Snapshot  
gcloud compute snapshots create SNAPNAME \\  
  --source-disk=adlr-jepa \\  
  --source-disk-zone=europe-west4-b \\  
  --quiet  

2. Create a New Disk from the Snapshot  
gcloud compute disks create DISKNAME \\  
  --zone=europe-west4-b \\  
  --source-snapshot=SNAPNAME \\  
  --quiet  

3. Launch a VM with V100 GPU  
gcloud compute instances create train-regression \\  
  --zone=europe-west4-b \\  
  --machine-type=n1-standard-8 \\  
  --accelerator=type=nvidia-tesla-v100,count=1 \\  
  --disk=name=DISKNAME,boot=yes,auto-delete=yes \\  
  --local-ssd=interface=nvme \\  
  --maintenance-policy=TERMINATE \\  
  --restart-on-failure \\  
  --tags=allow-ssh  


  ### 4. Mount SSD

  ```bash
sudo mkfs.ext4 -F /dev/nvme0n1
sudo mkdir -p /mnt/disks/ssd
sudo mount /dev/nvme0n1 /mnt/disks/ssd
sudo chown -R $USER:$USER /mnt/disks/ssd
```


## 🚀 Running the full low-data sweep on a GCP GPU VM  

This guide shows how to launch **all 42 runs** (`src/scripts/run_joint_sweep.py`) on a
Compute Engine GPU VM in zone **`europe-west4-b`**, keep them alive with `tmux`,
and check progress any time.

> **Prerequisites**  
> * GPU VM named **`my-gpu-vm`** (rename if yours differs)  
> * Zone **`europe-west4-b`**  
> * Repository cloned at `~/jed-repo/ADLR` **on the VM**  
> * Python virtual-env at `~/jed-repo/ADLR/.venv`  
> * Google Cloud SDK installed locally (`gcloud …` works)

---

1  SSH into the VM
gcloud compute ssh my-gpu-vm --zone=europe-west4-b
2 Install tmux (first time only)
bash
Copy
Edit
sudo apt update && sudo apt install -y tmux
3 Create / attach a persistent tmux session
bash
Copy
Edit
# first run
tmux new -s sweep

# later: re-attach
tmux attach -t sweep
A green status bar at the bottom means you’re inside tmux.

4 Activate the venv & launch the sweep
bash
Copy
Edit
source ~/jed-repo/ADLR/.venv/bin/activate
cd ~/jed-repo/ADLR
python src/scripts/run_joint_sweep.py
All stdout / stderr from every run stays in this pane; Weights & Biases logs sync automatically.

5 Detach & keep it running
bash
Copy
Edit
Ctrl-b  d      # press Ctrl+b, release, then d
exit           # leave SSH
The sweep session—and every Python process inside—continues until the VM stops or
the loop finishes.

6 Check progress later
bash
Copy
Edit
gcloud compute ssh my-gpu-vm --zone=europe-west4-b
tmux attach -t sweep
Detach again with Ctrl-b d.