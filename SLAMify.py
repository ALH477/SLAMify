# MIT License
#
# Copyright (c) 2025 DeMoD LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

print("""
 ________  ___       ________  _____ ______   ___  ________ ___    ___ 
|\   ____\|\  \     |\   __  \|\   _ \  _   \|\  \|\  _____\\  \  /  /|
\ \  \___|\ \  \    \ \  \|\  \ \  \\\__\ \  \ \  \ \  \__/\ \  \/  / /
 \ \_____  \ \  \    \ \   __  \ \  \\|__| \  \ \  \ \   __\\ \    / / 
  \|____|\  \ \  \____\ \  \ \  \ \  \    \ \  \ \  \ \  \_| \/  /  /  
    ____\_\  \ \_______\ \__\ \__\ \__\    \ \__\ \__\ \__\__/  / /    
   |\_________\|_______|\|__|\|__|\|__|     \|__|\|__|\|__|\___/ /     
   \|_________|                                           \|___|/      
                                                                       
                                                                       
DeMoD LLC - Stereo Annotation Tool v1.0
""")

import curses
import os
import sys
import logging
from enum import Enum
import gc
import time
from threading import Thread, Lock
from queue import Queue, Empty
import json
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose, ToTensor
import h5py
from tqdm import tqdm  # Fallback for non-TUI mode

# For DROID-SLAM (user must install: git clone https://github.com/princeton-vl/DROID-SLAM && cd DROID-SLAM && pip install -e .)
try:
    from droid import DROID
except ImportError:
    DROID = None

# For RAFT-Stereo
try:
    from raftstereo import RAFTStereo
except ImportError:
    RAFTStereo = None

# For FoundationStereo
try:
    from foundationstereo import FoundationStereo
except ImportError:
    FoundationStereo = None

# For PSMNet
try:
    from psmnet import PSMNet
except ImportError:
    PSMNet = None

# For DispNet
try:
    from dispnet import DispNet
except ImportError:
    DispNet = None

# For MonoDepth
try:
    from monodepth_pytorch import Model
except ImportError:
    Model = None

# For HITNet
try:
    from model import HITNet
except ImportError:
    HITNet = None

# For SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    sam_model_registry = None
    SamPredictor = None

class DepthModel(Enum):
    STEREO_BM = "stereobm"
    STEREO_SGBM = "stereosgbm"
    RAFT_STEREO = "raftstereo"
    FOUNDATION_STEREO = "foundationstereo"
    HITNET = "hitnet"
    PSMNET = "psmnet"
    DISP_NET = "dispnet"
    MONODEPTH = "monodepth"
    DROID_SLAM = "droidslam"

class SAMModel(Enum):
    VIT_B = "vit_b"
    VIT_H = "vit_h"

class Config:
    def __init__(self):
        self.depth_model = DepthModel.STEREO_BM
        self.sam_model = SAMModel.VIT_B
        self.batch_size = 8
        self.video_path = "trail_stereo.mkv"
        self.output_h5 = "annotations.h5"
        self.calib_path = "calib.json"
        self.log_level = logging.INFO
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_rectify = True
        self.enable_slam = True
        self.rect_maps = None  # Store rectification maps
        self.intrinsics = None  # Store camera intrinsics for SLAM

# Global config and models
config = Config()
models = {}
models_lock = Lock()
h5_lock = Lock()
slam_lock = Lock()

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
log_queue = Queue()

def log_worker():
    while True:
        try:
            msg = log_queue.get(timeout=1)
            print(msg)
        except Empty:
            continue

log_thread = Thread(target=log_worker, daemon=True)
log_thread.start()

# Stereo Rectification Setup
def load_calibration(calib_path):
    if not os.path.exists(calib_path):
        logger.warning(f"Calibration file not found: {calib_path}. Skipping rectification.")
        return None, None
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    K1 = np.array(calib['K1'])
    K2 = np.array(calib['K2'])
    D1 = np.array(calib['D1'])
    D2 = np.array(calib['D2'])
    R = np.array(calib['R'])
    T = np.array(calib['T'])
    size = (800, 1280)  # Per view size (height, width)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, size, R, T, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_16SC2)
    # Extract intrinsics for SLAM
    intrinsics = K1[:3, :3]  # 3x3 matrix
    return (map1x, map1y), (map2x, map2y), intrinsics

def rectify_frame(left, right, maps):
    if maps is None:
        return left, right
    map1x, map1y = maps[0]
    map2x, map2y = maps[1]
    left_rect = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)
    return left_rect, right_rect

# Model loading functions
def load_sam_model():
    with models_lock:
        if config.sam_model.value in models:
            return models[config.sam_model.value]
    
    checkpoint_map = {
        SAMModel.VIT_B.value: "sam_vit_b_01ec64.pth",
        SAMModel.VIT_H.value: "sam_vit_h_4b8939.pth"
    }
    sam_checkpoint = checkpoint_map.get(config.sam_model.value, "sam_vit_b_01ec64.pth")
    model_type = config.sam_model.value
    
    if sam_model_registry is None or SamPredictor is None:
        logger.error("SAM not installed. Run: pip install segment-anything")
        return None
    
    try:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=config.device)
        if config.device.type == 'cuda':
            sam = sam.half()
        predictor = SamPredictor(sam)
        with models_lock:
            models[config.sam_model.value] = predictor
        logger.info(f"SAM {model_type} loaded")
        return predictor
    except Exception as e:
        logger.error(f"Failed to load SAM {model_type}: {e}")
        return None

def load_raft_stereo():
    with models_lock:
        if "raft" in models:
            return models["raft"]
    
    if RAFTStereo is None:
        logger.error("RAFT-Stereo not installed. Run: git clone https://github.com/princeton-vl/RAFT-Stereo && cd RAFT-Stereo && pip install -e .")
        return None
    
    try:
        model = RAFTStereo.from_pretrained('raftstereo-middlebury.pth').to(device=config.device)
        if config.device.type == 'cuda':
            model = model.half()
        model.eval()
        with models_lock:
            models["raft"] = model
        logger.info("RAFT-Stereo loaded")
        return model
    except Exception as e:
        logger.error(f"Failed to load RAFT-Stereo: {e}")
        return None

def load_foundation_stereo():
    with models_lock:
        if "foundation" in models:
            return models["foundation"]
    
    if FoundationStereo is None:
        logger.error("FoundationStereo not installed. Run: git clone https://github.com/NVlabs/FoundationStereo && cd FoundationStereo && pip install -e .")
        return None
    
    try:
        checkpoint_path = "checkpoints/foundationstereo_vit_large.pth"
        model = FoundationStereo.from_pretrained(checkpoint_path).to(device=config.device)
        if config.device.type == 'cuda':
            model = model.half()
        model.eval()
        with models_lock:
            models["foundation"] = model
        logger.info("FoundationStereo loaded")
        return model
    except Exception as e:
        logger.error(f"Failed to load FoundationStereo: {e}. Ensure checkpoints are downloaded.")
        return None

def load_hitnet():
    with models_lock:
        if "hitnet" in models:
            return models["hitnet"]
    
    if HITNet is None:
        logger.error("HITNet not installed. Run: git clone https://github.com/zjjMaiMai/TinyHITNet && cd TinyHITNet && pip install -e .")
        return None
    
    try:
        model = HITNet()
        checkpoint_path = "ckpt/hitnet_sf_finalpass.ckpt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            logger.warning(f"HITNet checkpoint not found at {checkpoint_path}. Using untrained model.")
        model.to(device=config.device)
        if config.device.type == 'cuda':
            model = model.half()
        model.eval()
        with models_lock:
            models["hitnet"] = model
        logger.info("HITNet loaded")
        return model
    except Exception as e:
        logger.error(f"Failed to load HITNet: {e}. Ensure checkpoints are downloaded.")
        return None

def load_psmnet():
    with models_lock:
        if "psmnet" in models:
            return models["psmnet"]
    
    if PSMNet is None:
        logger.error("PSMNet not installed. Run: git clone https://github.com/JiaRenChang/PSMNet && cd PSMNet && pip install -e .")
        return None
    
    try:
        checkpoint_path = "checkpoints/psmnet_kitti.pth"
        model = PSMNet(maxdisp=192).to(device=config.device)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            logger.warning(f"PSMNet checkpoint not found at {checkpoint_path}. Using untrained model.")
        if config.device.type == 'cuda':
            model = model.half()
        model.eval()
        with models_lock:
            models["psmnet"] = model
        logger.info("PSMNet loaded")
        return model
    except Exception as e:
        logger.error(f"Failed to load PSMNet: {e}. Ensure checkpoints are downloaded.")
        return None

def load_dispnet():
    with models_lock:
        if "dispnet" in models:
            return models["dispnet"]
    
    if DispNet is None:
        logger.error("DispNet not installed. Run: git clone https://github.com/fabiotosi92/DispNet && cd DispNet && pip install -e .")
        return None
    
    try:
        checkpoint_path = "checkpoints/dispnet_kitti.pth"
        model = DispNet().to(device=config.device)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            logger.warning(f"DispNet checkpoint not found at {checkpoint_path}. Using untrained model.")
        if config.device.type == 'cuda':
            model = model.half()
        model.eval()
        with models_lock:
            models["dispnet"] = model
        logger.info("DispNet loaded")
        return model
    except Exception as e:
        logger.error(f"Failed to load DispNet: {e}. Ensure checkpoints are downloaded.")
        return None

def load_monodepth():
    with models_lock:
        if "monodepth" in models:
            return models["monodepth"]
    
    if Model is None:
        logger.error("MonoDepth not installed. Run: git clone https://github.com/OniroAI/MonoDepth-PyTorch && cd MonoDepth-PyTorch && pip install -e .")
        return None
    
    try:
        args = {
            'model': 'resnet18',
            'input_height': 800,
            'input_width': 1280,
            'input_channels': 1,
            'pretrained': True,
            'checkpoint_path': 'checkpoints/monodepth_kitti.pth'
        }
        model = Model(args).to(device=config.device)
        if os.path.exists(args['checkpoint_path']):
            checkpoint = torch.load(args['checkpoint_path'], map_location=config.device)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            logger.warning(f"MonoDepth checkpoint not found at {args['checkpoint_path']}. Using untrained model.")
        if config.device.type == 'cuda':
            model = model.half()
        model.eval()
        with models_lock:
            models["monodepth"] = model
        logger.info("MonoDepth loaded")
        return model
    except Exception as e:
        logger.error(f"Failed to load MonoDepth: {e}. Ensure checkpoints are downloaded.")
        return None

def load_droid_slam():
    with models_lock:
        if "droid" in models:
            return models["droid"]
    
    if DROID is None:
        logger.error("DROID-SLAM not available. Install from https://github.com/princeton-vl/DROID-SLAM")
        return None
    
    try:
        model = DROID(
            max_h=480,
            max_w=640,
            stride=8,
            slowmo=1.0,
            disable_vis=True
        ).to(device=config.device)
        if config.device.type == 'cuda':
            model = model.half()
        model.eval()
        with models_lock:
            models["droid"] = model
        logger.info("DROID-SLAM loaded")
        return model
    except Exception as e:
        logger.error(f"Failed to load DROID-SLAM: {e}")
        return None

# VRAM monitoring
vram_lock = Lock()
def print_vram_usage(prefix=""):
    if torch.cuda.is_available():
        with vram_lock:
            try:
                allocated = torch.cuda.memory_allocated(config.device) / 1024**3
                reserved = torch.cuda.memory_reserved(config.device) / 1024**3
                max_allocated = torch.cuda.max_memory_allocated(config.device) / 1024**3
                logger.info(f"{prefix} - VRAM: Allocated {allocated:.2f}GB, Reserved {reserved:.2f}GB, Max Allocated {max_allocated:.2f}GB")
            except Exception as e:
                logger.warning(f"VRAM monitoring failed: {e}")
    else:
        logger.debug(f"{prefix} - No GPU available for VRAM monitoring")

# Load initial calibration
if config.enable_rectify:
    config.rect_maps, _, config.intrinsics = load_calibration(config.calib_path)

# Processing function
def process_video(progress_queue):
    logger.info("Starting video processing...")
    print_vram_usage("Initial")
    
    if not os.path.exists(config.video_path):
        logger.error(f"Video not found: {config.video_path}")
        return False
    
    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        logger.error("Failed to open video")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Total frames: {total_frames}")
    
    expected_width, expected_height = 2560, 800
    
    hf = None
    try:
        hf = h5py.File(config.output_h5, 'w')
        disparity_dset = hf.create_dataset('disparities', (0, expected_height, expected_width // 2), maxshape=(None, expected_height, expected_width // 2), dtype=np.float32)
        mask_dset = hf.create_dataset('masks', (0, 1, expected_height, expected_width // 2), maxshape=(None, 1, expected_height, expected_width // 2), dtype=np.bool_)
        if config.enable_slam:
            poses_dset = hf.create_dataset('poses', (0, 4, 4), maxshape=(None, 4, 4), dtype=np.float32)
            indices_dset = hf.create_dataset('frame_indices', (0,), maxshape=(None,), dtype=np.int32)
    except Exception as e:
        logger.error(f"HDF5 init failed: {e}")
        if hf:
            hf.close()
        cap.release()
        return False
    
    predictor = load_sam_model()
    if predictor is None:
        hf.close()
        cap.release()
        return False
    
    # Load depth model with fallback
    depth_model = None
    if config.depth_model == DepthModel.DROID_SLAM:
        depth_model = load_droid_slam()
    elif config.depth_model == DepthModel.RAFT_STEREO:
        depth_model = load_raft_stereo()
    elif config.depth_model == DepthModel.FOUNDATION_STEREO:
        depth_model = load_foundation_stereo()
    elif config.depth_model == DepthModel.HITNET:
        depth_model = load_hitnet()
    elif config.depth_model == DepthModel.PSMNET:
        depth_model = load_psmnet()
    elif config.depth_model == DepthModel.DISP_NET:
        depth_model = load_dispnet()
    elif config.depth_model == DepthModel.MONODEPTH:
        depth_model = load_monodepth()
    
    if depth_model is None and config.depth_model not in [DepthModel.STEREO_BM, DepthModel.STEREO_SGBM]:
        logger.warning(f"Depth model {config.depth_model.value} failed to load. Falling back to StereoSGBM.")
        config.depth_model = DepthModel.STEREO_SGBM
    
    left_images = []
    right_images = []
    frame_indices = []
    max_slam_frames = 1000  # Cap to prevent OOM
    
    frames = []
    processed = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame.shape[0] != expected_height or frame.shape[1] != expected_width:
            logger.warning(f"Unexpected frame size: {frame.shape}")
            continue
        
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Rectify if enabled
        left, right = frame[:, :expected_width // 2], frame[:, expected_width // 2:]
        if config.enable_rectify and config.rect_maps:
            left, right = rectify_frame(left, right, config.rect_maps)
        
        # Accumulate for SLAM
        if config.enable_slam and processed % 5 == 0:
            with slam_lock:
                left_norm = left / 255.0
                right_norm = right / 255.0
                left_tensor = torch.from_numpy(left_norm).unsqueeze(0).float().to(config.device)
                right_tensor = torch.from_numpy(right_norm).unsqueeze(0).float().to(config.device)
                left_images.append(left_tensor)
                right_images.append(right_tensor)
                frame_indices.append(processed)
                # Cap SLAM frames
                if len(left_images) > max_slam_frames:
                    left_images.pop(0)
                    right_images.pop(0)
                    frame_indices.pop(0)
        
        frames.append(frame)
        
        if len(frames) == config.batch_size:
            print_vram_usage(f"Before batch {processed // config.batch_size + 1}")
            if process_batch(frames, predictor, depth_model, hf, disparity_dset, mask_dset, expected_width, expected_height):
                processed += len(frames)
            else:
                logger.error("Batch processing failed")
                break
            frames = []
            gc.collect()
            if config.device.type == 'cuda':
                torch.cuda.empty_cache()
            print_vram_usage(f"After batch {processed // config.batch_size + 1}")
            progress_queue.put((processed, total_frames))
    
    # Final batch
    if frames:
        print_vram_usage("Before final batch")
        if process_batch(frames, predictor, depth_model, hf, disparity_dset, mask_dset, expected_width, expected_height):
            processed += len(frames)
        progress_queue.put((processed, total_frames))
        print_vram_usage("After final batch")
    
    # Run SLAM if enabled
    if config.enable_slam and left_images:
        logger.info(f"Running SLAM on {len(left_images)} subsampled frames")
        if depth_model and config.depth_model == DepthModel.DROID_SLAM:
            try:
                with torch.no_grad():
                    intrinsics = torch.from_numpy(config.intrinsics).float().to(config.device).unsqueeze(0).repeat(len(left_images), 1, 1) if config.intrinsics is not None else torch.eye(3, device=config.device).unsqueeze(0).repeat(len(left_images), 1, 1)
                    traj = depth_model.track(
                        images=left_images,
                        depths=None,
                        intrinsics=intrinsics,
                        poses=torch.eye(4, device=config.device).unsqueeze(0).repeat(len(left_images), 1, 1)
                    )
                    poses = traj['poses']
                # Save poses and indices
                poses_array = np.array([p.cpu().numpy() for p in poses])
                with h5_lock:
                    current_len = poses_dset.shape[0]
                    poses_dset.resize((current_len + len(poses_array), 4, 4))
                    poses_dset[current_len:] = poses_array
                    indices_dset.resize((current_len + len(frame_indices),))
                    indices_dset[current_len:] = np.array(frame_indices)
                logger.info(f"Saved {len(poses_array)} SLAM poses with indices")
            except Exception as e:
                logger.error(f"SLAM computation failed: {e}")
    
    cap.release()
    hf.close()
    logger.info(f"Processing complete. Processed {processed} frames.")
    print_vram_usage("Final")
    return True

def process_batch(frames, predictor, depth_model, hf, disparity_dset, mask_dset, expected_width, expected_height):
    with h5_lock:
        try:
            batch_left = np.stack([f[:, :expected_width // 2] for f in frames], axis=0).astype(np.float32)
            batch_right = np.stack([f[:, expected_width // 2:] for f in frames], axis=0).astype(np.float32)
        except Exception as e:
            logger.error(f"Batch stacking failed: {e}")
            return False
        
        batch_disparities = []
        for i in range(len(frames)):
            try:
                left = batch_left[i]
                right = batch_right[i]
                
                if config.depth_model in [DepthModel.RAFT_STEREO, DepthModel.FOUNDATION_STEREO, DepthModel.HITNET, DepthModel.PSMNET, DepthModel.DISP_NET]:
                    left_norm = left / 255.0
                    right_norm = right / 255.0
                else:
                    left_norm, right_norm = left, right
                
                left_tensor = torch.from_numpy(left_norm).unsqueeze(0).unsqueeze(0).to(config.device, dtype=torch.float16 if config.device.type == 'cuda' else torch.float32)
                right_tensor = torch.from_numpy(right_norm).unsqueeze(0).unsqueeze(0).to(config.device, dtype=torch.float16 if config.device.type == 'cuda' else torch.float32)
                
                if config.depth_model == DepthModel.DROID_SLAM and depth_model:
                    batch_disparities.append(np.zeros((expected_height, expected_width // 2), dtype=np.float32))
                    logger.debug("SLAM mode: Deferring depth to sequence processing")
                elif config.depth_model == DepthModel.RAFT_STEREO and depth_model:
                    with torch.no_grad():
                        disp = depth_model(left_tensor, right_tensor)
                    batch_disparities.append(disp.squeeze().cpu().numpy())
                elif config.depth_model == DepthModel.FOUNDATION_STEREO and depth_model:
                    with torch.no_grad():
                        disp = depth_model(left_tensor, right_tensor)
                    batch_disparities.append(disp.squeeze().cpu().numpy())
                elif config.depth_model == DepthModel.HITNET and depth_model:
                    with torch.no_grad():
                        disp = depth_model(left_tensor, right_tensor)
                    batch_disparities.append(disp.squeeze().cpu().numpy())
                elif config.depth_model == DepthModel.PSMNET and depth_model:
                    with torch.no_grad():
                        disp = depth_model(left_tensor, right_tensor)
                    batch_disparities.append(disp.squeeze().cpu().numpy())
                elif config.depth_model == DepthModel.DISP_NET and depth_model:
                    with torch.no_grad():
                        disp = depth_model(left_tensor, right_tensor)
                    batch_disparities.append(disp.squeeze().cpu().numpy())
                elif config.depth_model == DepthModel.MONODEPTH and depth_model:
                    with torch.no_grad():
                        depth = depth_model(left_tensor)
                    batch_disparities.append(depth.squeeze().cpu().numpy())
                    logger.debug("Using monocular depth as disparity proxy (scale may vary)")
                elif config.depth_model == DepthModel.STEREO_SGBM:
                    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16, blockSize=15, P1=8*3*15**2, P2=32*3*15**2)
                    batch_disparities.append(stereo.compute(left, right))
                else:  # StereoBM
                    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
                    batch_disparities.append(stereo.compute(left, right))
            except Exception as e:
                logger.warning(f"Depth for frame {i}: {e}")
                batch_disparities.append(np.zeros((expected_height, expected_width // 2), dtype=np.float32))
        
        # Append disparities
        current_len = disparity_dset.shape[0]
        disparity_dset.resize((current_len + len(batch_disparities), expected_height, expected_width // 2))
        disparity_dset[current_len:] = np.array(batch_disparities)
        
        # SAM batch
        batch_masks = []
        transform = Compose([ToTensor()])
        for frame in frames:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                rgb_tensor = transform(rgb_frame).to(config.device, dtype=torch.float16 if config.device.type == 'cuda' else torch.float32)
                
                predictor.set_image(rgb_tensor)
                
                input_points = np.array([[400, 400]])
                if config.device.type == 'cuda':
                    input_points = torch.tensor(input_points, device=config.device)
                input_labels = np.array([1])
                
                masks, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=False)
                batch_masks.append(masks[0].cpu().numpy())
            except Exception as e:
                logger.warning(f"SAM for frame: {e}")
                batch_masks.append(np.zeros((expected_height, expected_width), dtype=np.bool_))
            
            del rgb_tensor
            if config.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Append masks
        current_len = mask_dset.shape[0]
        mask_dset.resize((current_len + len(batch_masks), 1, expected_height, expected_width // 2))
        mask_dset[current_len:] = np.expand_dims(np.array([m[:expected_width // 2] for m in batch_masks]), axis=1)
        
        del batch_left, batch_right, batch_disparities
        gc.collect()
        if config.device.type == 'cuda':
            torch.cuda.empty_cache()
        return True

# Curses TUI with error popup
def curses_tui(stdscr):
    curses.curs_set(1)
    stdscr.nodelay(True)
    stdscr.clear()
    stdscr.refresh()
    
    # Colors
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    
    # Main menu state
    current_menu = "main"
    selected_item = 0
    menus = {
        "main": ["Configure Depth Model", "Configure SAM Model", "Set Batch Size", "Set Video Path", "Set Output Path", "Set Calibration Path", "Toggle Rectification", "Toggle SLAM", "Set Log Level", "Run Processing", "Quit"],
        "depth": [e.value for e in DepthModel],
        "sam": [e.value for e in SAMModel],
        "log": ["DEBUG", "INFO", "WARNING", "ERROR"]
    }
    
    # Input buffers
    batch_input = str(config.batch_size)
    video_input = config.video_path
    output_input = config.output_h5
    calib_input = config.calib_path
    log_input = logging.getLevelName(config.log_level)
    
    # Queues for TUI updates
    status_queue = Queue()
    progress_queue = Queue()
    
    def draw_menu(stdscr, menu_name, selected):
        stdscr.clear()
        stdscr.addstr(0, 0, f"=== {menu_name.upper()} ===", curses.A_BOLD)
        items = menus[menu_name]
        for i, item in enumerate(items):
            attr = curses.color_pair(1) if i == selected else 0
            status = "ON" if (menu_name == "main" and ("Rectification" in item and config.enable_rectify or "SLAM" in item and config.enable_slam)) else ""
            stdscr.addstr(i + 2, 0, f"{item} [{status}]", attr)
        stdscr.addstr(len(items) + 3, 0, "Press UP/DOWN to navigate, ENTER to select, Q to back")
        stdscr.addstr(len(items) + 4, 0, f"Current Config: Depth={config.depth_model.value}, SAM={config.sam_model.value}, Batch={config.batch_size}")
        stdscr.refresh()
    
    def messagebox(stdscr, message):
        stdscr.clear()
        lines = message.split('\n')
        max_len = max(len(line) for line in lines)
        height = len(lines) + 4
        width = max_len + 4
        win = curses.newwin(height, width, (curses.LINES - height) // 2, (curses.COLS - width) // 2)
        win.border()
        for i, line in enumerate(lines):
            win.addstr(i + 2, 2, line, curses.color_pair(2))
        win.addstr(height - 2, 2, "Press any key to continue")
        win.refresh()
        stdscr.nodelay(False)
        win.getch()
        stdscr.nodelay(True)
    
    def input_dialog(stdscr, title, default, is_int=False):
        stdscr.nodelay(False)
        stdscr.clear()
        stdscr.addstr(0, 0, title, curses.A_BOLD)
        stdscr.addstr(2, 0, default)
        curses.echo()
        stdscr.move(2, len(default))
        stdscr.refresh()
        input_str = stdscr.getstr(2, 0, 50).decode('utf-8')
        curses.noecho()
        stdscr.nodelay(True)
        if is_int:
            try:
                return int(input_str)
            except ValueError:
                messagebox(stdscr, "Invalid integer input")
                return default
        return input_str
    
    def draw_progress(stdscr, processed, total):
        if total == 0:
            return
        percent = (processed / total) * 100
        bar_width = 40
        filled = int(bar_width * percent // 100)
        bar = '#' * filled + '-' * (bar_width - filled)
        stdscr.addstr(12, 0, f"Progress: [{bar}] {percent:.1f}% ({processed}/{total})", curses.color_pair(3))
        stdscr.refresh()
    
    def run_processing_thread():
        success = process_video(progress_queue)
        status_queue.put("Processing complete" if success else "Processing failed")
    
    running = False
    
    while True:
        if not running:
            if current_menu == "main":
                draw_menu(stdscr, "main", selected_item)
                key = stdscr.getch()
                if key == curses.KEY_UP:
                    selected_item = (selected_item - 1) % len(menus["main"])
                elif key == curses.KEY_DOWN:
                    selected_item = (selected_item + 1) % len(menus["main"])
                elif key == 10:  # ENTER
                    item = menus["main"][selected_item]
                    if "Depth" in item:
                        current_menu = "depth"
                        selected_item = menus["depth"].index(config.depth_model.value)
                    elif "SAM" in item:
                        current_menu = "sam"
                        selected_item = menus["sam"].index(config.sam_model.value)
                    elif "Batch" in item:
                        new_batch = input_dialog(stdscr, "Batch Size:", batch_input, is_int=True)
                        config.batch_size = new_batch
                        batch_input = str(new_batch)
                    elif "Video" in item:
                        new_video = input_dialog(stdscr, "Video Path:", video_input)
                        config.video_path = new_video
                        video_input = new_video
                    elif "Output" in item:
                        new_output = input_dialog(stdscr, "Output HDF5:", output_input)
                        config.output_h5 = new_output
                        output_input = new_output
                    elif "Calibration" in item:
                        new_calib = input_dialog(stdscr, "Calibration JSON Path:", calib_input)
                        config.calib_path = new_calib
                        calib_input = new_calib
                        if config.enable_rectify:
                            config.rect_maps, _, config.intrinsics = load_calibration(config.calib_path)
                    elif "Rectification" in item:
                        config.enable_rectify = not config.enable_rectify
                        logger.info(f"Rectification {'enabled' if config.enable_rectify else 'disabled'}")
                        if config.enable_rectify:
                            config.rect_maps, _, config.intrinsics = load_calibration(config.calib_path)
                    elif "SLAM" in item:
                        config.enable_slam = not config.enable_slam
                        logger.info(f"SLAM {'enabled' if config.enable_slam else 'disabled'}")
                    elif "Log" in item:
                        current_menu = "log"
                        selected_item = menus["log"].index(log_input)
                    elif "Run" in item:
                        numeric_level = getattr(logging, log_input, logging.INFO)
                        logging.getLogger().setLevel(numeric_level)
                        config.log_level = numeric_level
                        running = True
                        proc_thread = Thread(target=run_processing_thread, daemon=True)
                        proc_thread.start()
                        stdscr.addstr(10, 0, "Processing...")
                        stdscr.refresh()
                        processed = 0
                        total = 0
                    elif "Quit" in item:
                        sys.exit(0)
                elif key == ord('q') or key == 27:
                    sys.exit(0)
            
            elif current_menu == "depth":
                draw_menu(stdscr, "depth", selected_item)
                key = stdscr.getch()
                if key == curses.KEY_UP:
                    selected_item = (selected_item - 1) % len(menus["depth"])
                elif key == curses.KEY_DOWN:
                    selected_item = (selected_item + 1) % len(menus["depth"])
                elif key == 10:
                    config.depth_model = DepthModel(menus["depth"][selected_item])
                    logger.info(f"Depth model set to {config.depth_model.value}")
                    current_menu = "main"
                    selected_item = 0
                elif key == ord('q'):
                    current_menu = "main"
                    selected_item = 0
            
            elif current_menu == "sam":
                draw_menu(stdscr, "sam", selected_item)
                key = stdscr.getch()
                if key == curses.KEY_UP:
                    selected_item = (selected_item - 1) % len(menus["sam"])
                elif key == curses.KEY_DOWN:
                    selected_item = (selected_item + 1) % len(menus["sam"])
                elif key == 10:
                    config.sam_model = SAMModel(menus["sam"][selected_item])
                    logger.info(f"SAM model set to {config.sam_model.value}")
                    current_menu = "main"
                    selected_item = 1
                elif key == ord('q'):
                    current_menu = "main"
                    selected_item = 1
            
            elif current_menu == "log":
                draw_menu(stdscr, "log", selected_item)
                key = stdscr.getch()
                if key == curses.KEY_UP:
                    selected_item = (selected_item - 1) % len(menus["log"])
                elif key == curses.KEY_DOWN:
                    selected_item = (selected_item + 1) % len(menus["log"])
                elif key == 10:
                    log_input = menus["log"][selected_item]
                    config.log_level = getattr(logging, log_input, logging.INFO)
                    logger.info(f"Log level set to {log_input}")
                    current_menu = "main"
                    selected_item = 8
                elif key == ord('q'):
                    current_menu = "main"
                    selected_item = 8
        else:
            key = stdscr.getch()
            if key == ord('q'):
                running = False
                stdscr.addstr(10, 0, "Cancelled", curses.color_pair(2))
                stdscr.refresh()
                continue
            
            try:
                status = status_queue.get_nowait()
                stdscr.addstr(10, 0, status, curses.color_pair(2) if "failed" in status.lower() else 0)
                running = False
            except Empty:
                pass
            
            try:
                processed, total = progress_queue.get_nowait()
                draw_progress(stdscr, processed, total)
            except Empty:
                stdscr.addstr(12, 0, "Initializing...", curses.color_pair(3))
            
            stdscr.refresh()
            time.sleep(0.1)

if __name__ == "__main__":
    curses.wrapper(curses_tui)
