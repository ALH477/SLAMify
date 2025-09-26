```
 ________  ___       ________  _____ ______   ___  ________ ___    ___ 
|\   ____\|\  \     |\   __  \|\   _ \  _   \|\  \|\  _____\\  \  /  /|
\ \  \___|\ \  \    \ \  \|\  \ \  \\\__\ \  \ \  \ \  \__/\ \  \/  / /
 \ \_____  \ \  \    \ \   __  \ \  \\|__| \  \ \  \ \   __\\ \    / / 
  \|____|\  \ \  \____\ \  \ \  \ \  \    \ \  \ \  \ \  \_| \/  /  /  
    ____\_\  \ \_______\ \__\ \__\ \__\    \ \__\ \__\ \__\__/  / /    
   |\_________\|_______|\|__|\|__|\|__|     \|__|\|__|\|__|\___/ /     
   \|_________|                                           \|___|/      
                                                                       
```
![SLAM]([https://x.com/i/status/1971435267396648964](https://tenor.com/bmIHR.gif))
# DeMoD LLC Stereo Annotation Tool

A modular pipeline for annotating high-frame-rate stereo videos with depth, segmentation, and SLAM trajectories, optimized for egocentric datasets (e.g., trail biking for AR navigation). Supports OpenCV rectification, SAM, and depth models (RAFT, DROID-SLAM, etc.) with ROCm compatibility.

**Authors**: Asher LeRoy, with contributions from Grok 4 (Fast Beta) by xAI.

## Features
- **Stereo Rectification**: Corrects lens distortion using OpenCV.
- **Depth Estimation**: Supports StereoBM/SGBM, RAFT-Stereo, PSMNet, HITNet, DispNet, MonoDepth, and DROID-SLAM.
- **Segmentation**: SAM (VIT-B/H) for object masks.
- **SLAM**: DROID-SLAM for camera pose estimation.
- **TUI**: Interactive curses-based interface with progress bar.
- **Output**: HDF5 with disparities, masks, poses, and frame indices.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/DeMoD-LLC/stereo-annotation-tool.git
   cd stereo-annotation-tool
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install models (see [Model Setup](#model-setup)).
4. Download checkpoints to `checkpoints/` (e.g., `sam_vit_b_01ec64.pth`, `raftstereo-middlebury.pth`).

## Requirements
- Python 3.8+
- PyTorch 2.5 (ROCm for AMD GPUs)
- OpenCV, h5py, numpy, torchvision
- Optional: segment-anything, RAFT-Stereo, DROID-SLAM, PSMNet, HITNet, DispNet, MonoDepth-PyTorch

## Usage
Run the TUI:
```bash
python stereo_annotation.py
```
Navigate with UP/DOWN, select with ENTER, quit with Q. Configure depth/SAM models, video path, and output HDF5. Toggle rectification/SLAM as needed.

### Sample Data
A 1-minute trail biking clip is available in `sample_data/trail_stereo.mkv`.

## Model Setup
- **SAM**: Download `sam_vit_b_01ec64.pth` or `sam_vit_h_4b8939.pth` from [segment-anything](https://github.com/facebookresearch/segment-anything).
- **RAFT-Stereo**: Clone [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) and download `raftstereo-middlebury.pth`.
- **DROID-SLAM**: Clone [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM).
- **Others**: Follow respective repos for PSMNet, HITNet, DispNet, MonoDepth.

## Calibration
Provide a `calib.json` with `K1`, `K2` (3x3 intrinsics), `D1`, `D2` (distortion), `R` (rotation), `T` (translation). Example:
```json
{
  "K1": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "K2": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "D1": [k1, k2, p1, p2, k3],
  "D2": [k1, k2, p1, p2, k3],
  "R": [[...], [...], [...]],
  "T": [tx, ty, tz]
}
```

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. We welcome new models, bug fixes, and optimizations.

## License
[MIT License](LICENSE) © 2025 DeMoD LLC

## Acknowledgments
- Asher LeRoy for primary development.
- Grok 4 (Fast Beta) by xAI for code assistance and optimization.
- Open-source communities of PyTorch, OpenCV, and referenced models.
