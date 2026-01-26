# ==========================
# SECTION 0: Imports
# ==========================
import os
import glob
from typing import Tuple, List, Dict, Any, Optional, Callable, Union
import time
import traceback # For detailed error printing
import random # For dataset shuffling (optional)
import json

# Import RL and ML Libraries
import gymnasium
from gymnasium.spaces import Discrete, Box, Tuple as TupleSpace, Dict as DictSpace # Renamed Tuple to avoid conflict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch # Alias for torch.nn.functional
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as F_vision # Alias for torchvision.transforms.functional
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw # Added ImageDraw
import cv2
# Try to import Python Optimal Transport library (POT) or GeomLoss
try:
    import ot
    POT_AVAILABLE = True
except ImportError:
    POT_AVAILABLE = False
    print("Warning: Python Optimal Transport (POT) library not found. SWD will be used. For Sinkhorn, install POT: pip install pot")

try:
    import geomloss
    GEOM_LOSS_AVAILABLE = True
    print("GeomLoss library found. Will be used for Sinkhorn distance if enabled.")
except ImportError:
    GEOM_LOSS_AVAILABLE = False
    if not POT_AVAILABLE: # Only print if neither is available and Sinkhorn is an option
        print("Warning: GeomLoss library not found. For Sinkhorn, install geomloss: pip install geomloss")


# --- GradCAM Imports ---
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus # MODIFIED: Added GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("Please install pytorch-grad-cam: pip install pytorch-grad-cam")
    exit()

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from tqdm import tqdm


# ==========================
# SECTION 1: Configuration
# ==========================
# Define base path for data
BASE_DATA_PATH = '/home/safaei_d@WMGDS.WMG.WARWICK.AC.UK/hpc_new_mvp/new_mvp/new_mvp/data' # User specific path
# Define paths to KITTI and V-KITTI datasets
KITTI_DATA_DIR = '/home/safaei_d@WMGDS.WMG.WARWICK.AC.UK/hpc_new_mvp/new_mvp/new_mvp/data/data_for_new_mvp/KITTI' # User specific path
VKITTI_DATA_DIR = '/home/safaei_d@WMGDS.WMG.WARWICK.AC.UK/hpc_new_mvp/new_mvp/new_mvp/data/data_for_new_mvp/V-KITTI' # User specific path
# Define path to segmentation masks
SEGMENTATION_DATA_DIR = '/home/safaei_d@WMGDS.WMG.WARWICK.AC.UK/hpc_new_mvp/new_mvp/new_mvp/data/class_segmentation' # User specific path
# Define V-KITTI sequences to use
VKITTI_SEQUENCES_TO_USE = ['01', '02', '06', '18', '20']
# Define path to the pre-trained steering model
STEERING_MODEL_PATH = '/home/safaei_d@WMGDS.WMG.WARWICK.AC.UK/hpc_new_mvp/new_mvp/new_mvp/data/final_model.pth' # User specific path
# Define path to dummy VLM embeddings (replace with actual paths or generation logic)
VLM_EMBEDDING_DIR = '/home/safaei_d@WMGDS.WMG.WARWICK.AC.UK/hpc_new_mvp/new_mvp/new_mvp/data/vlm_embeddings_optimized_v2.json' # User specific path

# Define input size for the steering model
STEERING_MODEL_INPUT_SIZE = (66, 200) # Height, Width
HIGH_RES_INPUT_SIZE = (256, 512) # Height, Width, for perceptual models

# Define preprocessing transformations for the steering model
preprocess_for_steering_model = transforms.Compose([
    transforms.Resize(STEERING_MODEL_INPUT_SIZE),
    transforms.ToTensor(), # Converts to tensor and scales pixels to [0.0, 1.0]
])
preprocess_for_high_res = transforms.Compose([
    transforms.Resize(HIGH_RES_INPUT_SIZE),
    transforms.ToTensor(),
])

# Define the target layer in the steering model for GradCAM
TARGET_LAYER_IN_STEERING_MODEL = lambda model: model.conv5
# Define CAM targets for GradCAM
CAM_TARGETS = [RawScoresOutputTarget()]

# Define the device to use for computation (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# Define the target size for heatmaps (Width, Height)
HEATMAP_TARGET_SIZE = (64, 64) # W, H
# Define the number of projections for Sliced Wasserstein Distance (SWD)
N_SWD_PROJECTIONS = 128

# Define V-KITTI color map for segmentation masks
VKITTI_COLOR_MAP = [
    (0,0,0),(210,0,200),(90,200,255),(0,199,0),(90,240,0),
    (140,140,140),(100,60,100),(250,100,255),(255,255,0),(200,200,0),
    (255,130,0),(80,80,80),(160,60,60),(255,127,80),(0,139,139)
]
N_CLASSES_IN_MASK = len(VKITTI_COLOR_MAP) # Number of classes in segmentation masks

# --- New constants for enhancements ---
MAX_SWD = 5.0 # Max value for SWD/Sinkhorn for clipping
DELTA_WEIGHT = 1.0
TARGET_SWD = 0.2 # Target SWD/Sinkhorn for success bonus
SUCCESS_BONUS = 5.0
REPEAT_ACTION_PENALTY = -0.1 # Penalty for repeating the exact same discrete action tuple

# --- Curriculum Learning constants ---
CURRICULUM_INIT_SWD_THRESHOLD = 5.0
CURRICULUM_RAMP_STEPS = 10000
CURRICULUM_MAX_SWD_ADDITION = 8.0

# --- New Action Space Definitions ---
NUM_OBJ_CLASSES_TO_ADD = 4 # e.g., Car, Pedestrian, Truck, TrafficSign
MAX_INSTANCES_TO_REMOVE = 10 # Max number of instances trackable by InstanceManager for removal

# --- Reward Weights ---
R_VLM_WEIGHT = 1.0
R_SEMSEG_WEIGHT = 0.1 # Lambda_sem
R_DEPTH_WEIGHT = 0.1  # Lambda_depth
R_ADV_WEIGHT = 0.05   # Lambda_adv
R_TEMP_WEIGHT = 0.1   # Lambda_temp
R_COMPLEX_PENALTY = -0.05

VLM_EMBEDDING_DIM = 512

# Initialize these as None or with a default value. They will be properly initialized in `if __name__ == "__main__":`.
SWD_GRID_COORDS = None
SWD_PROJECTIONS = None
SINKHORN_COST_MATRIX = None

# --- Action Space Configuration ---
# Indices for the flattened action vector
CAT_IDX = 0
GEOM_TYPE_IDX = 1
ADD_OBJ_CLASS_IDX = 2
REMOVE_INST_ID_IDX = 3
WEATHER_TYPE_IDX = 4
COLOR_PARAMS_START_IDX = 5 # dB, dC, dS, dH
COLOR_PARAMS_END_IDX = COLOR_PARAMS_START_IDX + 4
GEOM_STRENGTH_IDX = COLOR_PARAMS_END_IDX
ADD_OBJ_POS_SCALE_START_IDX = GEOM_STRENGTH_IDX + 1 # x_norm, y_norm, scale
ADD_OBJ_POS_SCALE_END_IDX = ADD_OBJ_POS_SCALE_START_IDX + 3
ADD_OBJ_ROT_IDX = ADD_OBJ_POS_SCALE_END_IDX
WEATHER_INTENSITY_IDX = ADD_OBJ_ROT_IDX + 1
# --- EXPANDED ACTION SPACE PLACEHOLDERS (Illustrative) ---
# If adding more params, these indices would shift and FLATTENED_ACTION_SIZE would grow.
# For example, if adding Lighting (Category 6) with 7 params:
# LIGHTING_OP_TYPE_IDX = WEATHER_INTENSITY_IDX + 1
# LIGHTING_PARAMS_START_IDX = LIGHTING_OP_TYPE_IDX + 1
# LIGHTING_PARAMS_END_IDX = LIGHTING_PARAMS_START_IDX + 7 (e.g. for a point light)
# FLATTENED_ACTION_SIZE = LIGHTING_PARAMS_END_IDX
# For now, keeping the original size from "Code 2"
FLATTENED_ACTION_SIZE = WEATHER_INTENSITY_IDX + 1


# ========================================================================
# SECTION 1.5: Steering Prediction Model Definition (Same as original)
# ========================================================================
class AutonomousDrivingModel(nn.Module):
    def __init__(self):
        super(AutonomousDrivingModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3) # Target Layer for GradCAM
        self.fc1 = nn.Linear(1152, 1164) # Calculated for 66x200 input after conv5: 64 * 1 * 18 = 1152
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x)); x = self.dropout(x)
        x = self.relu(self.fc2(x)); x = self.dropout(x)
        x = self.relu(self.fc3(x)); x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def get_rl_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y = self.relu(self.conv1(x))
            y = self.relu(self.conv2(y))
            y = self.relu(self.conv3(y))
            y = self.relu(self.conv4(y))
            y = self.relu(self.conv5(y))
        return y

# ==========================
# SECTION 1.6: New Helper Models and Classes
# ==========================
class InstanceManager:
    def __init__(self, initial_seg_mask_path: str, image_hw: Tuple[int, int]):
        self.image_h, self.image_w = image_hw
        self.current_seg_mask_pil: Optional[Image.Image] = None
        self.instances: Dict[int, Dict[str, Any]] = {} # instance_id -> {class_id, bbox, ...}
        self.next_instance_id = 1000

        if os.path.exists(initial_seg_mask_path):
            self.current_seg_mask_pil = Image.open(initial_seg_mask_path).convert("RGB")
            self.current_seg_mask_pil = self.current_seg_mask_pil.resize((self.image_w, self.image_h), Image.NEAREST)
        else:
            print(f"Warning: Initial segmentation mask not found at {initial_seg_mask_path}. InstanceManager starts with empty mask.")
            self.current_seg_mask_pil = Image.new("RGB", (self.image_w, self.image_h), (0,0,0))

        self.object_assets_rgba: Dict[int, Image.Image] = {}
        for i in range(NUM_OBJ_CLASSES_TO_ADD):
            try:
                asset_img = Image.new("RGBA", (50,50), (random.randint(50,200), random.randint(50,200), random.randint(50,200), 200 + random.randint(0,55)))
                self.object_assets_rgba[i] = asset_img
            except Exception: # Broad exception for placeholder
                print(f"Warning: Could not create placeholder asset for object class {i}.")


    def add_instance(self, base_img_tensor: torch.Tensor, class_id: int,
                     x_norm: float, y_norm: float, scale: float, rotation: float) -> torch.Tensor:
        # print(f"InstanceManager: Adding obj class {class_id} at ({x_norm:.2f},{y_norm:.2f}), scale {scale:.2f}, rot {rotation:.1f}")
        if class_id not in self.object_assets_rgba:
            # print(f"Warning: No asset for class_id {class_id}.")
            return base_img_tensor

        base_img_pil = F_vision.to_pil_image(base_img_tensor.cpu().clamp(0,1)) # Ensure tensor is on CPU and clamped for PIL
        asset_pil = self.object_assets_rgba[class_id]
        asset_w, asset_h = asset_pil.size
        new_w = int(asset_w * scale * (self.image_w / 512))
        new_h = int(asset_h * scale * (self.image_h / 256))
        if new_w <=0 or new_h <=0: return base_img_tensor
        
        scaled_asset = asset_pil.resize((new_w, new_h), Image.LANCZOS)
        rotated_asset = scaled_asset.rotate(rotation, resample=Image.BICUBIC, expand=True)
        pos_x = int(x_norm * self.image_w - rotated_asset.width / 2)
        pos_y = int(y_norm * self.image_h - rotated_asset.height / 2)

        # Create a temporary canvas to paste the base image, then the asset
        # This avoids issues if base_img_pil is not RGBA
        temp_canvas = Image.new("RGBA", base_img_pil.size)
        temp_canvas.paste(base_img_pil.convert("RGBA")) # Paste base image
        temp_canvas.paste(rotated_asset, (pos_x, pos_y), rotated_asset) # Paste asset using its alpha

        # Update segmentation mask (simplified)
        if self.current_seg_mask_pil:
            draw = ImageDraw.Draw(self.current_seg_mask_pil)
            # This is a very rough approximation for the mask update
            # In reality, you'd rasterize the rotated asset's alpha onto the mask with a new instance ID/color
            # For simplicity, we're not perfectly updating the instance segmentation here.
            # new_instance_bbox = (pos_x, pos_y, pos_x + rotated_asset.width, pos_y + rotated_asset.height)
            # draw.rectangle(new_instance_bbox, fill=VKITTI_COLOR_MAP[class_id % len(VKITTI_COLOR_MAP)])

        self.instances[self.next_instance_id] = {
            "class_id": class_id, "bbox_xywh": (pos_x, pos_y, rotated_asset.width, rotated_asset.height),
            "scale": scale, "rotation": rotation
        }
        self.next_instance_id += 1
        return F_vision.to_tensor(temp_canvas.convert("RGB")).to(base_img_tensor.device) # Convert back to RGB tensor

    def remove_instance(self, base_img_tensor: torch.Tensor, instance_id_to_remove: int) -> torch.Tensor:
        # print(f"InstanceManager: Removing instance {instance_id_to_remove}")
        if instance_id_to_remove not in self.instances:
            # print(f"Warning: Instance ID {instance_id_to_remove} not found.")
            return base_img_tensor

        instance_info = self.instances[instance_id_to_remove]
        img_np = base_img_tensor.permute(1, 2, 0).cpu().numpy() * 255
        img_np = img_np.astype(np.uint8)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        if "bbox_xywh" in instance_info:
            x, y, w, h = instance_info["bbox_xywh"]
            mask_to_inpaint = np.zeros((self.image_h, self.image_w), dtype=np.uint8)
            # Ensure bbox is within image bounds for safety
            cv2.rectangle(mask_to_inpaint, 
                          (max(0,x), max(0,y)), 
                          (min(self.image_w-1, x+w), min(self.image_h-1, y+h)), 
                          255, -1)

            inpainted_cv = cv2.inpaint(img_cv, mask_to_inpaint, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            inpainted_rgb_np = cv2.cvtColor(inpainted_cv, cv2.COLOR_BGR2RGB)
            final_tensor = torch.from_numpy(inpainted_rgb_np / 255.0).permute(2,0,1).float().to(base_img_tensor.device)
            
            if self.current_seg_mask_pil:
                draw = ImageDraw.Draw(self.current_seg_mask_pil)
                draw.rectangle((x,y,x+w,y+h), fill=VKITTI_COLOR_MAP[0]) # Fill with background class

            del self.instances[instance_id_to_remove]
            return final_tensor
        else:
            # print(f"InstanceManager: No bbox for instance {instance_id_to_remove}, cannot inpaint precisely.")
            return base_img_tensor


class VLMModel(nn.Module): # Placeholder
    def __init__(self):
        super().__init__()
        self.model = None # In a real case: clip.load("ViT-B/32", device=DEVICE)
        print("VLMModel initialised (Note: Actual model loading and inference are placeholders).")

    def run_vlm(self, image_tensor_batch: torch.Tensor) -> torch.Tensor:
        if self.model is None: # Dummy implementation
            return torch.randn(image_tensor_batch.shape[0], VLM_EMBEDDING_DIM, device=image_tensor_batch.device)
        # Actual VLM inference would go here
        return torch.randn(image_tensor_batch.shape[0], VLM_EMBEDDING_DIM, device=image_tensor_batch.device)

def load_vlm_embedding(json_path: str, device: torch.device = DEVICE) -> Optional[torch.Tensor]:
    if not os.path.exists(json_path): return None
    try:
        with open(json_path, 'r') as f: data = json.load(f)
        embedding = torch.tensor(data["embedding"], dtype=torch.float32).to(device)
        return embedding if embedding.shape[0] == VLM_EMBEDDING_DIM else None
    except Exception: return None

def cosine_similarity_torch(embed1: torch.Tensor, embed2: torch.Tensor, eps=1e-8) -> torch.Tensor:
    if embed1 is None or embed2 is None: return torch.tensor(0.0, device=DEVICE if embed1 is None else embed1.device)
    if embed1.ndim == 1: embed1 = embed1.unsqueeze(0)
    if embed2.ndim == 1: embed2 = embed2.unsqueeze(0)
    num = torch.bmm(embed1.unsqueeze(1), embed2.unsqueeze(2)).squeeze()
    den = embed1.norm(p=2, dim=1) * embed2.norm(p=2, dim=1)
    return num / (den + eps)

# --- SIMULATED VLM GUIDANCE FUNCTION ---
def get_vlm_guidance(real_img_tensor_high_res: Optional[torch.Tensor], 
                     current_synth_img_tensor_high_res: Optional[torch.Tensor],
                     step_number: int) -> List[int]:
    """
    Simulates VLM insight to recommend relevant action categories.
    In a real system, this would involve VLM inference on the images/state
    and sophisticated logic to derive recommendations.

    Args:
        real_img_tensor_high_res: The real target image.
        current_synth_img_tensor_high_res: The current synthetic image.
        step_number: Current step in the episode.

    Returns:
        A list of recommended action category indices.
        Example: [0, 1, 5] means Colour, Geometry, and No-op are recommended.
    """
    # Placeholder logic:
    # - For the first few steps, allow more exploration (e.g., color and geometry).
    # - Later, it might suggest more specific things or become more restrictive.
    # - This is highly simplified and not based on actual image content.
    
    # All categories: 0: colour, 1: geom, 2: add_obj, 3: remove_obj, 4: weather, 5: no_op
    all_categories = list(range(6))

    if step_number < 3: # Early in episode, broader suggestions
        # Could be e.g. focus on color and basic geometry
        recommended = [0, 1, 5] 
    elif step_number < 7:
        # Could be e.g. allow object manipulation and weather
        recommended = [0, 1, 2, 3, 4, 5] 
    else:
        # Later, maybe more focused or even allow everything for RL to fine-tune
        # For simulation, let's say it becomes more focused on color/weather or no-op
        recommended = [0, 4, 5] 
    
    # print(f"Simulated VLM Guidance (step {step_number}): Recommending categories {recommended}")
    return recommended


# ==========================
# SECTION 1.7: Precompute SWD and Sinkhorn Assets
# ==========================
def _precompute_swd_assets(target_size: Tuple[int, int], n_projections: int) -> Tuple[np.ndarray, np.ndarray]:
    print(f"Precomputing SWD assets for size {target_size} and {n_projections} projections...")
    target_w, target_h = target_size # W, H
    y_coords_pix = np.arange(target_h); x_coords_pix = np.arange(target_w)
    yv, xv = np.meshgrid(y_coords_pix, x_coords_pix, indexing='ij')
    grid_coords = np.stack((yv.flatten(), xv.flatten()), axis=1).astype(np.float64) # N_pixels x 2 (y,x)
    angles = np.linspace(0, np.pi, n_projections, endpoint=False)
    projections = np.stack((np.cos(angles), np.sin(angles)), axis=1).astype(np.float64) # N_proj x 2
    print("SWD assets precomputed.")
    return grid_coords, projections

def _precompute_sinkhorn_cost_matrix(heatmap_size_wh: Tuple[int,int], device: torch.device) -> torch.Tensor:
    print(f"Precomputing Sinkhorn cost matrix for heatmap size {heatmap_size_wh}...")
    W, H = heatmap_size_wh
    coords_x = torch.arange(W, device=device).float()
    coords_y = torch.arange(H, device=device).float()
    grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing='ij') # H, W
    coords = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1) # (H*W, 2) i.e. (N_pixels, 2) with (x,y)
    cost_matrix = torch.cdist(coords, coords, p=2.0)**2 # Pairwise squared Euclidean distances
    cost_matrix_norm = cost_matrix / (cost_matrix.max() + 1e-9)
    print(f"Sinkhorn cost matrix of shape {cost_matrix_norm.shape} precomputed and normalised.")
    return cost_matrix_norm


# ==========================
# SECTION 2: Dataset Class
# ==========================
class KittiVKittiPairDataset(Dataset):
    def __init__(self, kitti_dir: str, vkitti_dir: str, seg_dir: str, vlm_embed_dir: str,
                 vkitti_sequences: List[str], preprocess_func_steering: Callable, preprocess_func_high_res: Callable):
        self.kitti_dir = kitti_dir; self.vkitti_dir = vkitti_dir;
        self.seg_dir = seg_dir; self.vlm_embed_dir = vlm_embed_dir;
        self.preprocess_steering = preprocess_func_steering
        self.preprocess_high_res = preprocess_func_high_res
        self.image_pairs = []
        print("Building dataset index (strict path checking)...")
        for seq_id_str in vkitti_sequences:
            vkitti_scene_path = os.path.join(vkitti_dir, f"scene_{seq_id_str}")
            vkitti_rgb_path = os.path.join(vkitti_scene_path, "frames", "rgb", "Camera_0")
            if not os.path.isdir(vkitti_rgb_path): continue
            try: kitti_seq_id_int = int(seq_id_str)
            except ValueError: continue
            kitti_seq_path_base = os.path.join(kitti_dir, f"{kitti_seq_id_int:04d}")
            potential_kitti_paths = [
                os.path.join(kitti_seq_path_base, 'image_02', 'data'),
                os.path.join(kitti_seq_path_base, 'image_2'),
                kitti_seq_path_base ]
            found_kitti_path = next((p for p in potential_kitti_paths if os.path.isdir(p)), None)
            if not found_kitti_path: continue
            seg_seq_base_path = os.path.join(self.seg_dir, f"Scene{seq_id_str}")
            if not os.path.isdir(seg_seq_base_path): continue

            vkitti_frames = sorted(glob.glob(os.path.join(vkitti_rgb_path, "rgb_*.jpg")))
            for vkitti_frame_path in vkitti_frames:
                try:
                    frame_name = os.path.basename(vkitti_frame_path)
                    frame_id_str_vkitti = frame_name.replace('rgb_', '').replace('.jpg', '')
                    frame_id_int = int(frame_id_str_vkitti)
                    kitti_frame_path = os.path.join(found_kitti_path, f"{frame_id_int:06d}.png")
                    vkitti_seg_frame_path = os.path.join(seg_seq_base_path, f"classgt_{frame_id_str_vkitti}.png")
                    vlm_json_filename = f"kitti_{seq_id_str}_{frame_id_int:06d}_vlm.json"
                    real_vlm_embed_path = os.path.join(self.vlm_embed_dir, seq_id_str, vlm_json_filename)

                    if os.path.exists(kitti_frame_path) and os.path.exists(vkitti_seg_frame_path):
                        self.image_pairs.append({
                            "kitti_img": kitti_frame_path, "vkitti_img": vkitti_frame_path,
                            "vkitti_seg": vkitti_seg_frame_path,
                            "real_vlm_embed_path": real_vlm_embed_path if os.path.exists(real_vlm_embed_path) else None,
                            "seq_id": seq_id_str, "frame_id": frame_id_int
                        })
                except ValueError: continue
                except Exception as e: print(f"Error processing {vkitti_frame_path}: {e}")
        print(f"Dataset initialised with {len(self.image_pairs)} image triplets.")
        if not self.image_pairs: raise ValueError("No image triplets found. Check paths and naming conventions.")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair_info = self.image_pairs[idx]
        try:
            real_img_pil = Image.open(pair_info["kitti_img"]).convert("RGB")
            synth_img_pil = Image.open(pair_info["vkitti_img"]).convert("RGB")
            real_img_tensor_steering = self.preprocess_steering(real_img_pil)
            synth_img_tensor_steering = self.preprocess_steering(synth_img_pil)
            real_img_tensor_high_res = self.preprocess_high_res(real_img_pil)
            synth_img_tensor_high_res = self.preprocess_high_res(synth_img_pil)
            info_dict = {k: v for k, v in pair_info.items()}
            info_dict["seg_mask_path"] = pair_info["vkitti_seg"]
            return (real_img_tensor_steering, synth_img_tensor_steering,
                    real_img_tensor_high_res, synth_img_tensor_high_res,
                    pair_info["vkitti_seg"], info_dict)
        except Exception as e:
            print(f"Error loading data index {idx} (Paths: {pair_info}): {e}"); traceback.print_exc()
            raise RuntimeError(f"Error loading data for index {idx}") from e

# ==========================
# SECTION 3: GradCAM Helper
# ==========================
def get_steering_heatmap(
    steering_model: nn.Module, gradcam_instance: Union[GradCAM, GradCAMPlusPlus], image_tensor: torch.Tensor, # MODIFIED type hint
    device: torch.device, cam_targets: List, heatmap_target_size: Tuple[int, int] = HEATMAP_TARGET_SIZE # W,H
    ) -> Optional[np.ndarray]:
    steering_model.eval()
    resized_heatmap = np.zeros((heatmap_target_size[1], heatmap_target_size[0]), dtype=np.float32) # H, W
    try:
        input_tensor_for_cam = image_tensor.clone().unsqueeze(0).to(device).requires_grad_(True)
        grayscale_cam = gradcam_instance(input_tensor=input_tensor_for_cam, targets=cam_targets)
        if grayscale_cam is not None and grayscale_cam.shape[0] > 0:
            heatmap_np = grayscale_cam[0, :]
            if heatmap_np is not None and heatmap_np.size > 0:
                resized_heatmap_cv = cv2.resize(heatmap_np, heatmap_target_size, cv2.INTER_LINEAR)
                min_val, max_val = resized_heatmap_cv.min(), resized_heatmap_cv.max()
                if max_val > min_val: resized_heatmap = (resized_heatmap_cv - min_val) / (max_val - min_val)
                else: resized_heatmap = np.zeros_like(resized_heatmap_cv)
    except Exception: # Removed verbose print
        pass 
    return resized_heatmap.astype(np.float32)

# ==========================
# SECTION 4: SWD / Sinkhorn Reward
# ==========================
def calculate_swd_deterministic(
    heatmap_real: np.ndarray, heatmap_enhanced_synthetic: np.ndarray, # Both H,W
    grid_coords: np.ndarray, projections: np.ndarray, epsilon: float = 1e-9,
    heatmap_target_size_wh: Tuple[int, int] = HEATMAP_TARGET_SIZE # W, H
) -> float:
    try:
        target_w, target_h = heatmap_target_size_wh
        expected_shape_hw = (target_h, target_w)
        if not (isinstance(heatmap_real, np.ndarray) and heatmap_real.shape == expected_shape_hw):
            heatmap_real = np.zeros(expected_shape_hw, dtype=np.float64)
        if not (isinstance(heatmap_enhanced_synthetic, np.ndarray) and heatmap_enhanced_synthetic.shape == expected_shape_hw):
            heatmap_enhanced_synthetic = np.zeros(expected_shape_hw, dtype=np.float64)

        heatmap_real = np.nan_to_num(np.maximum(heatmap_real, 0.0))
        heatmap_enhanced_synthetic = np.nan_to_num(np.maximum(heatmap_enhanced_synthetic, 0.0))
        h_real_flat = heatmap_real.flatten().astype(np.float64);
        h_synth_flat = heatmap_enhanced_synthetic.flatten().astype(np.float64)
        sum_real = h_real_flat.sum(); sum_synth = h_synth_flat.sum()

        if sum_real < epsilon and sum_synth < epsilon: return 0.0
        if sum_real < epsilon or sum_synth < epsilon: return MAX_SWD + 1.0

        p_real = h_real_flat / sum_real; p_synth = h_synth_flat / sum_synth
        total_w1d = 0.0
        for i in range(projections.shape[0]):
            proj_vector = projections[i]
            projected_coords = grid_coords @ proj_vector
            sort_indices = np.argsort(projected_coords)
            sorted_projected_coords = projected_coords[sort_indices]
            sorted_p_real = p_real[sort_indices]; sorted_p_synth = p_synth[sort_indices]
            cdf_diff = np.cumsum(sorted_p_real) - np.cumsum(sorted_p_synth)
            w1d_dist = np.sum(np.abs(cdf_diff[:-1]) * np.diff(sorted_projected_coords))
            total_w1d += w1d_dist
        swd_value = total_w1d / projections.shape[0]
        return float(swd_value) if np.isfinite(swd_value) and swd_value >= 0 else MAX_SWD + 1.0
    except Exception: return MAX_SWD + 1.0

def get_sinkhorn_distance(
    p_np: np.ndarray, q_np: np.ndarray, C: torch.Tensor,
    epsilon_sinkhorn: float = 0.01, device: torch.device = DEVICE, eps_for_stability: float = 1e-10
) -> float:
    if not (GEOM_LOSS_AVAILABLE or POT_AVAILABLE): return MAX_SWD + 1.0
    try:
        p_flat = torch.from_numpy(p_np.astype(np.float32).flatten()).to(device)
        q_flat = torch.from_numpy(q_np.astype(np.float32).flatten()).to(device)
        sum_p = p_flat.sum(); sum_q = q_flat.sum()
        if sum_p < 1e-9 and sum_q < 1e-9: return 0.0
        if sum_p < 1e-9 or sum_q < 1e-9: return MAX_SWD + 1.0
        p_norm = p_flat / (sum_p + 1e-9); q_norm = q_flat / (sum_q + 1e-9)
        num_iterations_sinkhorn = 200

        if POT_AVAILABLE: # Prefer POT
            p_pot = p_norm.cpu().numpy() + eps_for_stability; p_pot /= p_pot.sum()
            q_pot = q_norm.cpu().numpy() + eps_for_stability; q_pot /= q_pot.sum()
            C_cpu = C.cpu().numpy()
            transport_plan = ot.sinkhorn(p_pot, q_pot, C_cpu, reg=epsilon_sinkhorn, numItermax=num_iterations_sinkhorn, stopThr=1e-5)
            sinkhorn_val = np.sum(transport_plan * C_cpu)
            return float(sinkhorn_val) if np.isfinite(sinkhorn_val) else MAX_SWD + 1.0
        elif GEOM_LOSS_AVAILABLE: # Fallback to GeomLoss if POT not available
            # print("Warning: Using GeomLoss for Sinkhorn. Ensure compatibility with fixed support.")
            # This part would need careful implementation with GeomLoss for fixed support.
            # For now, returning high value if only GeomLoss is available for this specific setup.
            return MAX_SWD + 2.0 
    except Exception: return MAX_SWD + 1.0
    return MAX_SWD + 1.0

# =============================================
# SECTION 6: New Enhancement Helper Functions
# =============================================
def tensor_to_pil(tensor_img: torch.Tensor) -> Image.Image:
    return F_vision.to_pil_image(tensor_img.cpu().clamp(0,1))

def pil_to_tensor(pil_img: Image.Image, device: torch.device) -> torch.Tensor:
    return F_vision.to_tensor(pil_img).to(device)

def apply_colour_tweak(img_tensor: torch.Tensor, dB: float, dC: float, dS: float, dH: float) -> torch.Tensor:
    img_tensor = F_vision.adjust_brightness(img_tensor, brightness_factor=max(0, 1.0 + dB))
    img_tensor = F_vision.adjust_contrast(img_tensor, contrast_factor=max(0, 1.0 + dC))
    img_tensor = F_vision.adjust_saturation(img_tensor, saturation_factor=max(0, 1.0 + dS))
    img_tensor = F_vision.adjust_hue(img_tensor, hue_factor=np.clip(dH * 0.5, -0.5, 0.5))
    return img_tensor

def apply_gaussian_blur(img_tensor: torch.Tensor, strength: float) -> torch.Tensor:
    if strength == 0: return img_tensor
    kernel_size = int(strength * 10) * 2 + 1
    return F_vision.gaussian_blur(img_tensor, kernel_size=kernel_size)

def apply_motion_blur(img_tensor: torch.Tensor, strength: float, angle: Optional[float]=None) -> torch.Tensor:
    if strength == 0: return img_tensor
    kernel_size = int(strength * 20) + 1
    if angle is None: angle = random.uniform(0, 360)
    center = kernel_size // 2
    M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[center, 0:kernel_size] = np.ones(kernel_size)
    kernel_motion_blur = cv2.warpAffine(kernel_motion_blur, M, (kernel_size, kernel_size))
    kernel_motion_blur = kernel_motion_blur / (kernel_motion_blur.sum() + 1e-9)
    img_np_uint8 = (img_tensor.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    blurred_np = cv2.filter2D(img_np_uint8, -1, kernel_motion_blur)
    return pil_to_tensor(Image.fromarray(blurred_np), img_tensor.device)

def apply_perspective_warp(img_tensor: torch.Tensor, strength: float) -> torch.Tensor:
    if strength == 0: return img_tensor
    h, w = img_tensor.shape[-2:]
    max_offset = int(strength * min(h,w) * 0.1)
    pts_original = np.float32([[0,0], [w-1,0], [0,h-1], [w-1,h-1]])
    dw = [random.randint(-max_offset, max_offset) for _ in range(4)]
    dh = [random.randint(-max_offset, max_offset) for _ in range(4)]
    pts_warped = np.float32([
        [np.clip(0+dw[0],0,w-1), np.clip(0+dh[0],0,h-1)], 
        [np.clip(w-1+dw[1],0,w-1), np.clip(0+dh[1],0,h-1)],
        [np.clip(0+dw[2],0,w-1), np.clip(h-1+dh[2],0,h-1)], 
        [np.clip(w-1+dw[3],0,w-1), np.clip(h-1+dh[3],0,h-1)]
    ])
    img_np_uint8 = (img_tensor.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    matrix = cv2.getPerspectiveTransform(pts_original, pts_warped)
    warped_np = cv2.warpPerspective(img_np_uint8, matrix, (w,h), borderMode=cv2.BORDER_REPLICATE)
    return pil_to_tensor(Image.fromarray(warped_np), img_tensor.device)

def apply_unsharp_mask(img_tensor: torch.Tensor, strength: float) -> torch.Tensor:
    if strength == 0: return img_tensor
    radius = int(strength * 5)
    percent = int(100 + strength * 150)
    img_pil = tensor_to_pil(img_tensor)
    sharpened_pil = img_pil.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=3))
    return pil_to_tensor(sharpened_pil, img_tensor.device)

def apply_rain(img_tensor: torch.Tensor, intensity: float) -> torch.Tensor:
    if intensity == 0: return img_tensor
    rain_effect = torch.rand_like(img_tensor) * intensity * 0.2
    img_tensor = img_tensor * (1.0 - intensity * 0.3)
    return torch.clamp(img_tensor - rain_effect, 0.0, 1.0)

def apply_fog(img_tensor: torch.Tensor, intensity: float) -> torch.Tensor:
    if intensity == 0: return img_tensor
    fog_color = torch.ones_like(img_tensor) * 0.75
    return torch.clamp(torch.lerp(img_tensor, fog_color, intensity * 0.8), 0.0, 1.0)

def apply_snow(img_tensor: torch.Tensor, intensity: float) -> torch.Tensor:
    if intensity == 0: return img_tensor
    num_flakes = int(intensity * img_tensor.numel() * 0.005)
    snow_mask = torch.zeros_like(img_tensor)
    for _ in range(num_flakes):
        c = random.randint(0,img_tensor.shape[0]-1)
        h_idx = random.randint(0, img_tensor.shape[1]-1)
        w_idx = random.randint(0, img_tensor.shape[2]-1)
        snow_mask[c, h_idx, w_idx] = 1.0
    return torch.clamp(img_tensor * (1.0 - intensity*0.1) + snow_mask * intensity, 0.0, 1.0)

def apply_sunflare(img_tensor: torch.Tensor, intensity: float) -> torch.Tensor:
    if intensity == 0: return img_tensor
    h, w = img_tensor.shape[1], img_tensor.shape[2]
    flare_y, flare_x = random.randint(0,h-1), random.randint(0,w-1)
    radius = int(min(h,w) * 0.2 * intensity) + 1 # ensure radius > 0
    yy, xx = torch.meshgrid(torch.arange(h, device=DEVICE), torch.arange(w, device=DEVICE), indexing='ij')
    dist_sq = (yy - flare_y)**2 + (xx - flare_x)**2
    flare_mask = torch.exp(-dist_sq / (2 * radius**2 + 1e-6)).unsqueeze(0)
    flare_color = torch.tensor([1.0, 0.9, 0.7], device=DEVICE).view(3,1,1) * intensity
    return torch.clamp(img_tensor * (1.0 - flare_mask * 0.8) + flare_color * flare_mask, 0.0, 1.0)

# ==========================
# SECTION 7: RL Environment
# ==========================
class ImageEnhancementEnv(gymnasium.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(
        self, kitti_dir: str, vkitti_dir: str, seg_dir: str, vlm_embed_dir: str,
        vkitti_sequences: List[str], steering_model: nn.Module,
        gradcam_instance_steering: Union[GradCAM, GradCAMPlusPlus], # MODIFIED type hint
        preprocess_func_steering: Callable, preprocess_func_high_res: Callable,
        cam_targets_steering: List, max_steps: int,
        heatmap_target_size_wh: Tuple[int, int], device: torch.device,
        swd_grid_coords: np.ndarray, swd_projections: np.ndarray,
        sinkhorn_cost_matrix: torch.Tensor, use_sinkhorn_instead_of_swd: bool = False
    ):
        super().__init__()
        self.max_steps = max_steps; self.device = device;
        self.steering_model = steering_model.to(device);
        for param in self.steering_model.parameters(): param.requires_grad = False
        self.steering_model.eval()
        self.gradcam_steering = gradcam_instance_steering
        self.preprocess_steering = preprocess_func_steering
        self.preprocess_high_res = preprocess_func_high_res
        self.cam_targets_steering = cam_targets_steering
        self.heatmap_target_size_wh = heatmap_target_size_wh
        self.heatmap_target_size_hw = heatmap_target_size_wh[::-1]

        self.swd_grid_coords = swd_grid_coords; self.swd_projections = swd_projections
        self.sinkhorn_cost_matrix = sinkhorn_cost_matrix.to(device)
        self.use_sinkhorn_instead_of_swd = use_sinkhorn_instead_of_swd
        if self.use_sinkhorn_instead_of_swd and not (GEOM_LOSS_AVAILABLE or POT_AVAILABLE):
            print("Warning: Sinkhorn requested but no library found. Falling back to SWD.")
            self.use_sinkhorn_instead_of_swd = False

        try:
            self.dataset = KittiVKittiPairDataset(
                kitti_dir, vkitti_dir, seg_dir, vlm_embed_dir, vkitti_sequences,
                self.preprocess_steering, self.preprocess_high_res
            )
            assert len(self.dataset) > 0, "Dataset is empty!"
        except Exception as e: print(f"FATAL: Dataset init failed: {e}"); traceback.print_exc(); raise

        self.action_space = Box(low=-1.0, high=1.0, shape=(FLATTENED_ACTION_SIZE,), dtype=np.float32)
        print(f"Env Action Space (Flattened Box): {self.action_space}")

        self._feature_extractor_method = self.steering_model.get_rl_features
        self.feature_vector_length = self._determine_feature_length()
        self.heatmap_diff_size = self.heatmap_target_size_wh[0] * self.heatmap_target_size_wh[1]

        self.observation_space = DictSpace({
            "visual_features": Box(-np.inf, np.inf, shape=(2 * self.feature_vector_length,), dtype=np.float32),
            "heatmap_diff": Box(-1.0, 1.0, shape=(self.heatmap_diff_size,), dtype=np.float32),
            "vlm_similarity": Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        })
        print(f"Env Observation space: {self.observation_space}")

        self.vlm_model = VLMModel().to(device)
        # Env State Variables (initialized in reset)
        self.current_step = 0
        self.real_img_tensor_steering: Optional[torch.Tensor] = None
        self.initial_synth_img_tensor_steering: Optional[torch.Tensor] = None
        self.current_synth_img_tensor_steering: Optional[torch.Tensor] = None
        self.real_img_tensor_high_res: Optional[torch.Tensor] = None
        self.initial_synth_img_tensor_high_res: Optional[torch.Tensor] = None
        self.current_synth_img_tensor_high_res: Optional[torch.Tensor] = None
        self.seg_mask_path: Optional[str] = None
        self.instance_mgr: Optional[InstanceManager] = None
        self.real_heatmap: Optional[np.ndarray] = None
        self.last_heatmap_diff_flat = np.zeros(self.heatmap_diff_size, dtype=np.float32)
        self.info_dict_from_dataset = {}
        self.vlm_real_embed: Optional[torch.Tensor] = None
        self.prev_vlm_sim: float = 0.0
        self.initial_raw_swd_or_sinkhorn: float = MAX_SWD + 1.0
        self.previous_clipped_swd_or_sinkhorn: float = MAX_SWD
        self.last_decoded_action_dict: Optional[Dict] = None
        self.current_max_initial_swd_for_curriculum: float = CURRICULUM_INIT_SWD_THRESHOLD
        self.allowed_action_categories: List[int] = list(range(6)) # Initially all 6 categories allowed
        self.current_vlm_recommended_categories: List[int] = list(range(6)) # Initially allow all

    def set_curriculum_max_initial_swd(self, threshold: float):
        self.current_max_initial_swd_for_curriculum = threshold

    def set_allowed_action_categories(self, allowed_list: List[int]):
        self.allowed_action_categories = allowed_list
        # print(f"Env: Curriculum - Allowed action categories updated to: {self.allowed_action_categories}")

    def _determine_feature_length(self) -> int:
        try:
            # print("Determining steering feature vector length...")
            (sample_real_steering, _, _, _, _, _) = self.dataset[0]
            dummy_input = sample_real_steering.unsqueeze(0).to(self.device)
            self.steering_model.eval()
            with torch.no_grad(): features = self._feature_extractor_method(dummy_input)
            if len(features.shape) > 2:
                pooled_features = torch.mean(features, dim=list(range(2, len(features.shape))))
                feature_length = pooled_features.shape[1]
            elif len(features.shape) == 2: feature_length = features.shape[1]
            else: raise ValueError(f"Unexpected feature map shape: {features.shape}")
            # print(f"Determined steering feature length: {feature_length}")
            return feature_length
        except Exception as e:
            print(f"FATAL: Failed to determine steering feature length: {e}"); traceback.print_exc()
            return 64

    def _extract_steering_features(self, image_tensor_steering: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if image_tensor_steering is None: return None
        try:
            self.steering_model.eval()
            with torch.no_grad():
                input_batch = image_tensor_steering.unsqueeze(0).to(self.device)
                features = self._feature_extractor_method(input_batch)
            if len(features.shape) > 2:
                pooled_features = torch.mean(features, dim=list(range(2, len(features.shape)))).squeeze(0)
            elif len(features.shape) == 2: pooled_features = features.squeeze(0)
            else: return None
            return pooled_features
        except Exception: return None # Removed verbose print

    def _get_obs(self) -> Dict[str, np.ndarray]:
        default_obs_dict = {
            "visual_features": np.zeros(2 * self.feature_vector_length, dtype=np.float32),
            "heatmap_diff": np.zeros(self.heatmap_diff_size, dtype=np.float32),
            "vlm_similarity": np.zeros(1, dtype=np.float32)
        }
        real_steer_feat = self._extract_steering_features(self.real_img_tensor_steering)
        synth_steer_feat = self._extract_steering_features(self.current_synth_img_tensor_steering)
        if real_steer_feat is None or synth_steer_feat is None:
            visual_features_flat = default_obs_dict["visual_features"]
        else:
            visual_features_flat = torch.cat(
                (real_steer_feat.flatten(), synth_steer_feat.flatten()), dim=0
            ).cpu().numpy().astype(np.float32)
            if visual_features_flat.shape[0] != 2 * self.feature_vector_length:
                visual_features_flat = default_obs_dict["visual_features"]
        
        heatmap_diff_obs = self.last_heatmap_diff_flat
        if heatmap_diff_obs.shape[0] != self.heatmap_diff_size:
            heatmap_diff_obs = default_obs_dict["heatmap_diff"]
        
        vlm_similarity_obs = np.array([self.prev_vlm_sim], dtype=np.float32)
        obs_dict = {
            "visual_features": visual_features_flat,
            "heatmap_diff": heatmap_diff_obs,
            "vlm_similarity": vlm_similarity_obs
        }
        for key in obs_dict:
            if obs_dict[key].shape != self.observation_space.spaces[key].shape:
                obs_dict[key] = np.zeros(self.observation_space.spaces[key].shape, dtype=self.observation_space.spaces[key].dtype.as_numpy_dtype)
        return obs_dict

    def _decode_action(self, flat_action_vector: np.ndarray) -> Dict[str, Any]:
        decoded = {}
        def scale_action(val, min_val, max_val):
            return min_val + ((val + 1.0) / 2.0) * (max_val - min_val)
        def discretize_action(val, num_bins):
            scaled_val = ((val + 1.0) / 2.0) * num_bins
            return np.clip(int(scaled_val), 0, num_bins - 1)

        num_categories = 6 # 0: colour, 1: geom, 2: add_obj, 3: remove_obj, 4: weather, 5: no_op
        agent_intended_category = discretize_action(flat_action_vector[CAT_IDX], num_categories)
        
        # --- Apply Curriculum and VLM Guidance ---
        # 1. Curriculum Check: Is the agent's intended category allowed by the general curriculum?
        if agent_intended_category not in self.allowed_action_categories:
            final_category = 5 # Force no-op if curriculum disallows
            # print(f"Action Decode: Category {agent_intended_category} disallowed by curriculum. Forced no-op.")
        else:
            # 2. VLM Guidance Check: If curriculum allows, does VLM recommend it?
            if agent_intended_category != 5 and agent_intended_category not in self.current_vlm_recommended_categories:
                final_category = 5 # Force no-op if VLM does not recommend (and it's not already a no-op)
                # print(f"Action Decode: Category {agent_intended_category} (allowed by curriculum) not in VLM recommendations {self.current_vlm_recommended_categories}. Forced no-op.")
            else:
                final_category = agent_intended_category # Allowed by both curriculum and VLM (or is a no-op)
        
        decoded["category"] = final_category
        
        # Decode all parameters based on agent's *original* intent,
        # as the policy network was trained to output these.
        # The `final_category` will determine if these params are actually used.
        color_params_raw = flat_action_vector[COLOR_PARAMS_START_IDX:COLOR_PARAMS_END_IDX]
        decoded["dB"] = scale_action(color_params_raw[0], -0.2, 0.2)
        decoded["dC"] = scale_action(color_params_raw[1], -0.2, 0.2)
        decoded["dS"] = scale_action(color_params_raw[2], -0.2, 0.2)
        decoded["dH"] = scale_action(color_params_raw[3], -0.2, 0.2)

        num_geom_types = 4
        decoded["geom_type"] = discretize_action(flat_action_vector[GEOM_TYPE_IDX], num_geom_types)
        decoded["geom_strength"] = scale_action(flat_action_vector[GEOM_STRENGTH_IDX], 0.0, 1.0)

        decoded["object_class"] = discretize_action(flat_action_vector[ADD_OBJ_CLASS_IDX], NUM_OBJ_CLASSES_TO_ADD)
        add_obj_pos_scale_raw = flat_action_vector[ADD_OBJ_POS_SCALE_START_IDX:ADD_OBJ_POS_SCALE_END_IDX]
        decoded["x_norm"] = scale_action(add_obj_pos_scale_raw[0], 0.0, 1.0)
        decoded["y_norm"] = scale_action(add_obj_pos_scale_raw[1], 0.0, 1.0)
        decoded["scale"] = scale_action(add_obj_pos_scale_raw[2], 0.1, 1.5)
        decoded["rotation"] = scale_action(flat_action_vector[ADD_OBJ_ROT_IDX], -15.0, 15.0)

        decoded["instance_id"] = discretize_action(flat_action_vector[REMOVE_INST_ID_IDX], MAX_INSTANCES_TO_REMOVE)

        num_weather_types = 4
        decoded["weather_type"] = discretize_action(flat_action_vector[WEATHER_TYPE_IDX], num_weather_types)
        decoded["weather_intensity"] = scale_action(flat_action_vector[WEATHER_INTENSITY_IDX], 0.0, 1.0)
        
        return decoded

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        loaded_successfully = False; max_retries = min(20, len(self.dataset) * 2)
        for attempt in range(max_retries):
            try:
                idx = self.np_random.integers(0, len(self.dataset))
                (real_steer, synth_steer, real_high, synth_high, seg_path, info_dict) = self.dataset[idx]
                self.real_img_tensor_steering = real_steer.to(self.device)
                self.initial_synth_img_tensor_steering = synth_steer.to(self.device)
                self.real_img_tensor_high_res = real_high.to(self.device)
                self.initial_synth_img_tensor_high_res = synth_high.to(self.device)

                self.real_heatmap = get_steering_heatmap(
                    self.steering_model, self.gradcam_steering, self.real_img_tensor_steering, self.device,
                    self.cam_targets_steering, heatmap_target_size=self.heatmap_target_size_wh)
                if self.real_heatmap is None or self.real_heatmap.sum() < 1e-6: continue

                vlm_path = info_dict.get("real_vlm_embed_path")
                self.vlm_real_embed = load_vlm_embedding(vlm_path, self.device) if vlm_path else None
                if self.vlm_real_embed is None:
                    self.vlm_real_embed = self.vlm_model.run_vlm(self.real_img_tensor_high_res.unsqueeze(0)).squeeze(0)

                self.current_synth_img_tensor_steering = self.initial_synth_img_tensor_steering.clone()
                self.current_synth_img_tensor_high_res = self.initial_synth_img_tensor_high_res.clone()
                self.seg_mask_path = seg_path; self.info_dict_from_dataset = info_dict
                self.instance_mgr = InstanceManager(self.seg_mask_path, HIGH_RES_INPUT_SIZE)

                initial_synth_heatmap = get_steering_heatmap(
                    self.steering_model, self.gradcam_steering, self.initial_synth_img_tensor_steering, self.device,
                    self.cam_targets_steering, heatmap_target_size=self.heatmap_target_size_wh)
                if initial_synth_heatmap is None: initial_synth_heatmap = np.zeros(self.heatmap_target_size_hw, dtype=np.float32)

                if self.use_sinkhorn_instead_of_swd:
                    current_initial_raw_dist = get_sinkhorn_distance(
                        self.real_heatmap, initial_synth_heatmap, self.sinkhorn_cost_matrix, device=self.device)
                else:
                    current_initial_raw_dist = calculate_swd_deterministic(
                        self.real_heatmap, initial_synth_heatmap, self.swd_grid_coords, self.swd_projections,
                        heatmap_target_size_wh=self.heatmap_target_size_wh)
                
                if current_initial_raw_dist > self.current_max_initial_swd_for_curriculum:
                    if attempt % 20 == 0 : print(f"Curriculum Note: SWD/Sinkhorn {current_initial_raw_dist:.2f} > {self.current_max_initial_swd_for_curriculum:.2f}. Retrying sample.")
                    continue

                self.initial_raw_swd_or_sinkhorn = current_initial_raw_dist
                self.previous_clipped_swd_or_sinkhorn = min(self.initial_raw_swd_or_sinkhorn, MAX_SWD)
                heatmap_diff = (self.real_heatmap - initial_synth_heatmap).flatten().astype(np.float32)
                if heatmap_diff.shape[0] == self.heatmap_diff_size: self.last_heatmap_diff_flat = heatmap_diff
                else: self.last_heatmap_diff_flat = np.zeros(self.heatmap_diff_size, dtype=np.float32)

                if self.vlm_real_embed is not None:
                    initial_synth_vlm_embed = self.vlm_model.run_vlm(self.initial_synth_img_tensor_high_res.unsqueeze(0)).squeeze(0)
                    self.prev_vlm_sim = cosine_similarity_torch(self.vlm_real_embed, initial_synth_vlm_embed).item()
                else: self.prev_vlm_sim = 0.0
                loaded_successfully = True; break
            except RuntimeError: continue # Removed verbose print
            except Exception: traceback.print_exc(); continue # Removed verbose print

        if not loaded_successfully: # Fallback if curriculum or other errors persist
            print(f"Warning: Failed to load ideal sample after {max_retries} retries. Forcing load of last attempted sample.")
            # Ensure all necessary variables are initialized even in this fallback
            if 'idx' not in locals(): idx = self.np_random.integers(0, len(self.dataset)) # if idx was never set
            (real_steer, synth_steer, real_high, synth_high, seg_path, info_dict) = self.dataset[idx]
            self.real_img_tensor_steering = real_steer.to(self.device)
            self.initial_synth_img_tensor_steering = synth_steer.to(self.device)
            self.real_img_tensor_high_res = real_high.to(self.device)
            self.initial_synth_img_tensor_high_res = synth_high.to(self.device)
            self.current_synth_img_tensor_steering = self.initial_synth_img_tensor_steering.clone()
            self.current_synth_img_tensor_high_res = self.initial_synth_img_tensor_high_res.clone()
            self.seg_mask_path = seg_path; self.info_dict_from_dataset = info_dict
            self.instance_mgr = InstanceManager(self.seg_mask_path, HIGH_RES_INPUT_SIZE)
            self.real_heatmap = get_steering_heatmap(self.steering_model, self.gradcam_steering, self.real_img_tensor_steering, self.device, self.cam_targets_steering, self.heatmap_target_size_wh)
            if self.real_heatmap is None: self.real_heatmap = np.zeros(self.heatmap_target_size_hw, dtype=np.float32)
            vlm_path = info_dict.get("real_vlm_embed_path")
            self.vlm_real_embed = load_vlm_embedding(vlm_path, self.device) if vlm_path else self.vlm_model.run_vlm(self.real_img_tensor_high_res.unsqueeze(0)).squeeze(0)
            initial_synth_heatmap = get_steering_heatmap(self.steering_model, self.gradcam_steering, self.initial_synth_img_tensor_steering, self.device, self.cam_targets_steering, self.heatmap_target_size_wh)
            if initial_synth_heatmap is None: initial_synth_heatmap = np.zeros(self.heatmap_target_size_hw, dtype=np.float32)
            if self.use_sinkhorn_instead_of_swd: self.initial_raw_swd_or_sinkhorn = get_sinkhorn_distance(self.real_heatmap, initial_synth_heatmap, self.sinkhorn_cost_matrix, device=self.device)
            else: self.initial_raw_swd_or_sinkhorn = calculate_swd_deterministic(self.real_heatmap, initial_synth_heatmap, self.swd_grid_coords, self.swd_projections, heatmap_target_size_wh=self.heatmap_target_size_wh)
            self.previous_clipped_swd_or_sinkhorn = min(self.initial_raw_swd_or_sinkhorn, MAX_SWD)
            self.last_heatmap_diff_flat = (self.real_heatmap - initial_synth_heatmap).flatten().astype(np.float32)
            if self.last_heatmap_diff_flat.shape[0] != self.heatmap_diff_size: self.last_heatmap_diff_flat = np.zeros(self.heatmap_diff_size, dtype=np.float32)
            if self.vlm_real_embed is not None:
                initial_synth_vlm_embed = self.vlm_model.run_vlm(self.initial_synth_img_tensor_high_res.unsqueeze(0)).squeeze(0)
                self.prev_vlm_sim = cosine_similarity_torch(self.vlm_real_embed, initial_synth_vlm_embed).item()
            else: self.prev_vlm_sim = 0.0

        self.current_step = 0
        self.last_decoded_action_dict = None
        # Get VLM guidance for the initial state
        self.current_vlm_recommended_categories = get_vlm_guidance(
            self.real_img_tensor_high_res, self.current_synth_img_tensor_high_res, self.current_step
        )
        observation = self._get_obs()
        clean_info = {k: v for k, v in self.info_dict_from_dataset.items() if isinstance(v, (str, int, float, bool))}
        clean_info["seg_mask_path"] = str(self.seg_mask_path) if self.seg_mask_path else "None"
        clean_info["initial_raw_dist_at_reset"] = self.initial_raw_swd_or_sinkhorn
        return observation, clean_info

    def _apply_enhancement_dispatch(self, action_dict: Dict[str, Any]):
        category = action_dict["category"] # This category is already after curriculum and VLM guidance override
        img_tensor = self.current_synth_img_tensor_high_res
        original_img_tensor_for_no_op = img_tensor.clone() # For true no-op

        if category == 0: # colour_tweak
            img_tensor = apply_colour_tweak(img_tensor, action_dict["dB"], action_dict["dC"], action_dict["dS"], action_dict["dH"])
        elif category == 1: # geometry_tweak
            geom_type = action_dict["geom_type"]; strength = action_dict["geom_strength"]
            if geom_type == 0: img_tensor = apply_gaussian_blur(img_tensor, strength)
            elif geom_type == 1: img_tensor = apply_motion_blur(img_tensor, strength)
            elif geom_type == 2: img_tensor = apply_perspective_warp(img_tensor, strength)
            elif geom_type == 3: img_tensor = apply_unsharp_mask(img_tensor, strength)
        elif category == 2: # add_object
            if self.instance_mgr:
                img_tensor = self.instance_mgr.add_instance(
                    img_tensor, action_dict["object_class"], action_dict["x_norm"],
                    action_dict["y_norm"], action_dict["scale"], action_dict["rotation"])
        elif category == 3: # remove_object
            if self.instance_mgr:
                img_tensor = self.instance_mgr.remove_instance(img_tensor, action_dict["instance_id"])
        elif category == 4: # weather_change
            weather_type = action_dict["weather_type"]; intensity = action_dict["weather_intensity"]
            if weather_type == 0: img_tensor = apply_rain(img_tensor, intensity)
            elif weather_type == 1: img_tensor = apply_fog(img_tensor, intensity)
            elif weather_type == 2: img_tensor = apply_snow(img_tensor, intensity)
            elif weather_type == 3: img_tensor = apply_sunflare(img_tensor, intensity)
        elif category == 5: # no_op
            img_tensor = original_img_tensor_for_no_op # Ensure it's a true no-op

        self.current_synth_img_tensor_high_res = torch.clamp(img_tensor, 0.0, 1.0)
        pil_high_res = F_vision.to_pil_image(self.current_synth_img_tensor_high_res.cpu().clamp(0,1))
        self.current_synth_img_tensor_steering = self.preprocess_steering(pil_high_res).to(self.device)

    def step(self, action_vector: np.ndarray):
        # Get VLM guidance for the current step *before* decoding the action that uses it
        self.current_vlm_recommended_categories = get_vlm_guidance(
            self.real_img_tensor_high_res, self.current_synth_img_tensor_high_res, self.current_step
        )
        action_dict = self._decode_action(action_vector) # VLM guidance is used inside _decode_action
        
        try: self._apply_enhancement_dispatch(action_dict)
        except Exception as e: print(f"Error applying enhancement: {e}"); traceback.print_exc()

        total_reward = 0.0; reward_info = {}
        enhanced_steering_heatmap = get_steering_heatmap(
            self.steering_model, self.gradcam_steering, self.current_synth_img_tensor_steering, self.device,
            self.cam_targets_steering, heatmap_target_size=self.heatmap_target_size_wh)
        if enhanced_steering_heatmap is None: enhanced_steering_heatmap = np.zeros(self.heatmap_target_size_hw, dtype=np.float32)

        current_raw_dist = 0.0
        if self.real_heatmap is not None and self.real_heatmap.sum() > 1e-6 :
            if self.use_sinkhorn_instead_of_swd:
                current_raw_dist = get_sinkhorn_distance(self.real_heatmap, enhanced_steering_heatmap, self.sinkhorn_cost_matrix, device=self.device)
            else:
                current_raw_dist = calculate_swd_deterministic(self.real_heatmap, enhanced_steering_heatmap, self.swd_grid_coords, self.swd_projections, heatmap_target_size_wh=self.heatmap_target_size_wh)
        
        current_clipped_dist = min(current_raw_dist, MAX_SWD)
        r_attn = - (current_clipped_dist / MAX_SWD); total_reward += r_attn
        reward_info["r_attn"] = r_attn; reward_info["raw_dist_metric"] = current_raw_dist; reward_info["clipped_dist_metric"] = current_clipped_dist
        
        delta_dist_improvement = self.previous_clipped_swd_or_sinkhorn - current_clipped_dist
        r_delta_dist = DELTA_WEIGHT * delta_dist_improvement; total_reward += r_delta_dist
        reward_info["r_delta_dist"] = r_delta_dist
        self.previous_clipped_swd_or_sinkhorn = current_clipped_dist

        if self.vlm_real_embed is not None:
            current_synth_vlm_embed = self.vlm_model.run_vlm(self.current_synth_img_tensor_high_res.unsqueeze(0)).squeeze(0)
            sim_current = cosine_similarity_torch(self.vlm_real_embed, current_synth_vlm_embed).item()
            delta_vlm_sim = sim_current - self.prev_vlm_sim
            r_vlm = R_VLM_WEIGHT * delta_vlm_sim; total_reward += r_vlm
            self.prev_vlm_sim = sim_current
            reward_info.update({"r_vlm": r_vlm, "vlm_sim_current": sim_current})
        else: reward_info.update({"r_vlm": 0.0, "vlm_sim_current": 0.0})

        r_complex = R_COMPLEX_PENALTY if action_dict["category"] in [2,3] else 0.0
        total_reward += r_complex; reward_info["r_complex"] = r_complex
        
        r_repeat = 0.0
        # Compare relevant parts of action_dict for repeat penalty, excluding no-op from causing penalty if it was intended
        # A more robust check would compare the raw action_vector if the policy is deterministic for same obs
        if self.last_decoded_action_dict and action_dict["category"] != 5: 
            # Simple check: if category and key params are same (this is a heuristic)
            if self.last_decoded_action_dict["category"] == action_dict["category"]:
                 r_repeat = REPEAT_ACTION_PENALTY
        if action_dict["category"] != 5: # Only update last_action if it wasn't a no-op forced by VLM/Curriculum
            self.last_decoded_action_dict = action_dict 
        else: # If it was a no-op, don't consider it for repeat penalty against a future *meaningful* action
            self.last_decoded_action_dict = None 

        total_reward += r_repeat; reward_info["r_repeat"] = r_repeat
        reward_info.update({"r_semseg":0, "r_depth":0, "r_adv":0, "r_temp":0}) # Placeholders

        if self.real_heatmap is not None:
            heatmap_diff_updated = (self.real_heatmap - enhanced_steering_heatmap).flatten().astype(np.float32)
            if heatmap_diff_updated.shape[0] == self.heatmap_diff_size: self.last_heatmap_diff_flat = heatmap_diff_updated
            else: self.last_heatmap_diff_flat = np.zeros(self.heatmap_diff_size, dtype=np.float32)
        else: self.last_heatmap_diff_flat = np.zeros(self.heatmap_diff_size, dtype=np.float32)

        self.current_step += 1
        terminated = False; truncated = self.current_step >= self.max_steps
        info_to_return_to_agent = {}
        if terminated or truncated:
            success = (current_clipped_dist < TARGET_SWD)
            if success: total_reward += SUCCESS_BONUS; reward_info["r_success_bonus"] = SUCCESS_BONUS
            else: reward_info["r_success_bonus"] = 0.0
            info_to_return_to_agent['initial_dist_metric'] = self.initial_raw_swd_or_sinkhorn
            info_to_return_to_agent['final_dist_metric'] = current_raw_dist
            info_to_return_to_agent['final_clipped_dist_metric'] = current_clipped_dist
            info_to_return_to_agent['delta_dist_metric'] = self.initial_raw_swd_or_sinkhorn - current_raw_dist
            info_to_return_to_agent['success'] = float(success)
            for r_key, r_value in reward_info.items(): info_to_return_to_agent[f"final_ep_{r_key}"] = r_value
        
        info_to_return_to_agent.update(reward_info)
        info_to_return_to_agent["action_category"] = action_dict["category"] # Log the final applied category
        info_to_return_to_agent["vlm_recommended_categories_count"] = len(self.current_vlm_recommended_categories)


        observation = self._get_obs()
        return observation, total_reward, terminated, truncated, info_to_return_to_agent

    def close(self):
        # print("Closing ImageEnhancementEnv.")
        del self.steering_model, self.vlm_model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

# ==========================
# SECTION 8: Callbacks
# ==========================
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0): super().__init__(verbose)
    def _on_step(self) -> bool:
        if self.locals.get('infos') and self.locals['infos'][0]:
            info = self.locals['infos'][0]
            for key, value in info.items():
                if key.startswith("r_") or key.startswith("vlm_") or key.startswith("raw_") or \
                   key.startswith("clipped_") or key == "action_category" or key == "vlm_recommended_categories_count":
                    if isinstance(value, (float, int)): self.logger.record(f"rollout_step/{key}", value)
            if "episode" in info:
                ep_info = info["episode"]
                for key, value in ep_info.items():
                    if key not in ['r','l', 't'] and isinstance(value, (float, int)):
                        self.logger.record(f"episode/{key}", value)
        return True

class SWDCurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.curriculum_ramp_steps = CURRICULUM_RAMP_STEPS
        self.init_swd_threshold = CURRICULUM_INIT_SWD_THRESHOLD
        self.max_swd_addition = CURRICULUM_MAX_SWD_ADDITION
        self._last_logged_threshold = -1
    def _on_step(self) -> bool:
        timesteps_done = self.num_timesteps
        curriculum_fraction = min(1.0, timesteps_done / self.curriculum_ramp_steps)
        current_max_swd = self.init_swd_threshold + self.max_swd_addition * curriculum_fraction
        if hasattr(self.training_env, 'env_method'):
            try: self.training_env.env_method("set_curriculum_max_initial_swd", current_max_swd)
            except Exception: # Removed verbose print
                if isinstance(self.training_env.envs, list):
                    for env_i in self.training_env.envs:
                        if hasattr(env_i, "set_curriculum_max_initial_swd"):
                            env_i.set_curriculum_max_initial_swd(current_max_swd)
        if timesteps_done % (self.curriculum_ramp_steps // 20) == 0 or abs(current_max_swd - self._last_logged_threshold) > 0.01 :
            self.logger.record("curriculum/swd_max_initial_dist", current_max_swd)
            self.logger.record("curriculum/swd_fraction", curriculum_fraction)
            self._last_logged_threshold = current_max_swd
        return True

class ActionComplexityCurriculumCallback(BaseCallback):
    def __init__(self, T1_steps=10000, T2_steps=30000, T3_steps=60000, verbose=0):
        super().__init__(verbose)
        self.T1 = T1_steps; self.T2 = T2_steps; self.T3 = T3_steps
        self._current_stage = -1
    def _on_step(self) -> bool:
        timesteps_done = self.num_timesteps; new_stage = -1; allowed_categories = []
        if timesteps_done < self.T1: new_stage = 0; allowed_categories = [0, 1, 5]
        elif timesteps_done < self.T2: new_stage = 1; allowed_categories = [0, 1, 2, 5]
        elif timesteps_done < self.T3: new_stage = 2; allowed_categories = [0, 1, 2, 3, 5]
        else: new_stage = 3; allowed_categories = [0, 1, 2, 3, 4, 5]

        if new_stage != self._current_stage:
            self._current_stage = new_stage
            print(f"Timestep {timesteps_done}: Action Curriculum stage {new_stage}. Allowed by curriculum: {allowed_categories}")
            if hasattr(self.training_env, 'env_method'):
                try: self.training_env.env_method("set_allowed_action_categories", allowed_categories)
                except Exception: # Removed verbose print
                    if isinstance(self.training_env.envs, list):
                        for env_i in self.training_env.envs:
                            if hasattr(env_i, "set_allowed_action_categories"):
                                env_i.set_allowed_action_categories(allowed_categories)
            self.logger.record("curriculum/action_complexity_stage", new_stage)
        return True

# ==========================
# SECTION 9: Main Training (PPO)
# ==========================
if __name__ == "__main__":
    print("Starting RL PPO Image Enhancement Script with GradCAM++ and Simulated VLM Guidance")
    print(f"Using device: {DEVICE}")

    print("Precomputing SWD assets globally...")
    SWD_GRID_COORDS, SWD_PROJECTIONS = _precompute_swd_assets(HEATMAP_TARGET_SIZE, N_SWD_PROJECTIONS)
    print("Precomputing Sinkhorn cost matrix globally...")
    SINKHORN_COST_MATRIX = _precompute_sinkhorn_cost_matrix(HEATMAP_TARGET_SIZE, DEVICE)

    paths_ok = True
    # Simplified path checking for brevity in this example
    for p_val in [KITTI_DATA_DIR, VKITTI_DATA_DIR, SEGMENTATION_DATA_DIR, STEERING_MODEL_PATH, VLM_EMBEDDING_DIR]:
        if not (os.path.isdir(p_val) or os.path.exists(p_val)):
             # Allow VLM_EMBEDDING_DIR to be created if its parent exists
            if p_val == VLM_EMBEDDING_DIR and os.path.exists(os.path.dirname(p_val)):
                if not os.path.exists(p_val): os.makedirs(p_val, exist_ok=True); print(f"Created dir: {p_val}")
            else:
                print(f"FATAL: Path not found or invalid: {p_val}"); paths_ok = False; break
    if not paths_ok: exit(1)
    print("Dataset and model paths verified (or placeholders acknowledged).")

    try:
        print("Loading steering model...")
        steering_model_instance = AutonomousDrivingModel()
        state_dict = torch.load(STEERING_MODEL_PATH, map_location=DEVICE)
        steering_model_instance.load_state_dict(state_dict)
        steering_model_instance.to(DEVICE).eval()
        print(f"Steering model '{os.path.basename(STEERING_MODEL_PATH)}' loaded.")
        target_layers_objects = [TARGET_LAYER_IN_STEERING_MODEL(steering_model_instance)]
        # MODIFIED: Use GradCAMPlusPlus
        gradcam_instance_steering = GradCAMPlusPlus(model=steering_model_instance, target_layers=target_layers_objects)
        print("GradCAM++ setup complete.")
    except Exception as e: print(f"FATAL: Steering Model/GradCAM++ setup failed: {e}"); traceback.print_exc(); exit(1)

    MAX_EPISODE_STEPS = 15
    TOTAL_TIMESTEPS = 200_000 # Adjust as needed
    N_ENVS = 1 # For simplicity with simulated VLM and debugging. Increase for actual training.
    
    PPO_N_STEPS = 256 
    PPO_MINI_BATCH_SIZE = 64 
    PPO_N_EPOCHS = 10 
    PPO_GAMMA = 0.99; PPO_GAE_LAMBDA = 0.95; PPO_CLIP_RANGE = 0.2
    PPO_ENT_COEF = 0.005; PPO_VF_COEF = 0.5; PPO_MAX_GRAD_NORM = 0.5
    LEARNING_RATE = 3e-4

    def make_env_fn(rank: int, seed: int = 0, env_is_eval=False):
        def _init():
            try:
                env_instance = ImageEnhancementEnv(
                    kitti_dir=KITTI_DATA_DIR, vkitti_dir=VKITTI_DATA_DIR,
                    seg_dir=SEGMENTATION_DATA_DIR, vlm_embed_dir=VLM_EMBEDDING_DIR,
                    vkitti_sequences=VKITTI_SEQUENCES_TO_USE,
                    steering_model=steering_model_instance,
                    gradcam_instance_steering=gradcam_instance_steering, # Pass GradCAM++ instance
                    preprocess_func_steering=preprocess_for_steering_model,
                    preprocess_func_high_res=preprocess_for_high_res,
                    cam_targets_steering=CAM_TARGETS, max_steps=MAX_EPISODE_STEPS,
                    heatmap_target_size_wh=HEATMAP_TARGET_SIZE, device=DEVICE,
                    swd_grid_coords=SWD_GRID_COORDS, swd_projections=SWD_PROJECTIONS,
                    sinkhorn_cost_matrix=SINKHORN_COST_MATRIX,
                    use_sinkhorn_instead_of_swd=True
                )
                env_instance.reset(seed=seed + rank)
                log_dir_suffix = "eval" if env_is_eval else f"train_env_{rank}"
                log_dir = f"./tmp/gym_logs_ppo_vlm/{log_dir_suffix}/" # New log dir
                os.makedirs(log_dir, exist_ok=True)
                info_keywords_for_monitor = (
                    'initial_dist_metric', 'final_dist_metric', 'final_clipped_dist_metric',
                    'delta_dist_metric', 'success', 'action_category', 'vlm_recommended_categories_count',
                    'final_ep_r_attn', 'final_ep_r_delta_dist', 'final_ep_raw_dist_metric', 
                    'final_ep_clipped_dist_metric', 'final_ep_r_vlm', 'final_ep_vlm_sim_current',
                    'final_ep_r_complex', 'final_ep_r_repeat', 'final_ep_r_success_bonus'
                )
                env_instance = Monitor(env_instance, log_dir, info_keywords=info_keywords_for_monitor, allow_early_resets=True)
                return env_instance
            except Exception as e_init: print(f"FATAL: Env init rank {rank} failed: {e_init}"); traceback.print_exc(); raise
        set_random_seed(seed)
        return _init

    print(f"\nCreating {N_ENVS} training environment(s)...")
    if N_ENVS > 1 and os.name != 'nt': # SubprocVecEnv preferred if N_ENVS > 1 and not Windows or properly guarded
         train_env = SubprocVecEnv([make_env_fn(i, seed=i*10) for i in range(N_ENVS)])
    else:
        if N_ENVS > 1 and os.name == 'nt': print("Warning: Using DummyVecEnv for N_ENVS > 1 on Windows due to potential SubprocVecEnv issues in __main__.")
        train_env = DummyVecEnv([make_env_fn(0, seed=0)])
    print(f"Training environment created: {type(train_env)}")

    LOG_DIR_BASE = "ppo_enhancement_logs_vlm_gradcampp" # New log dir base
    TENSORBOARD_LOG_DIR = f"./{LOG_DIR_BASE}/tensorboard/"
    EVAL_LOG_DIR = f"./{LOG_DIR_BASE}/eval_logs/"
    CHECKPOINT_LOG_DIR = f"./{LOG_DIR_BASE}/checkpoints/"
    for d in [TENSORBOARD_LOG_DIR, EVAL_LOG_DIR, CHECKPOINT_LOG_DIR]: os.makedirs(d, exist_ok=True)

    print("\nInitialising PPO agent...")
    policy_kwargs = {} # Using default MlpPolicy features extractor for MultiInputPolicy
    model = PPO(
        "MultiInputPolicy", train_env, learning_rate=LEARNING_RATE, n_steps=PPO_N_STEPS,
        batch_size=PPO_MINI_BATCH_SIZE, n_epochs=PPO_N_EPOCHS, gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA, clip_range=PPO_CLIP_RANGE, ent_coef=PPO_ENT_COEF,
        vf_coef=PPO_VF_COEF, max_grad_norm=PPO_MAX_GRAD_NORM,
        tensorboard_log=TENSORBOARD_LOG_DIR, policy_kwargs=policy_kwargs, verbose=1, device=DEVICE
    )
    print(f"PPO Agent initialised. Action space: {model.action_space}, Obs space: {model.observation_space}")

    reward_log_callback = RewardLoggerCallback()
    swd_curriculum_callback = SWDCurriculumCallback()
    action_curriculum_callback = ActionComplexityCurriculumCallback(
        T1_steps=TOTAL_TIMESTEPS // 10, T2_steps=TOTAL_TIMESTEPS // 3, T3_steps=int(TOTAL_TIMESTEPS // 1.5)
    )
    eval_env_instance = DummyVecEnv([make_env_fn(rank=N_ENVS + 1, seed=420, env_is_eval=True)])
    eval_freq_timesteps = max(PPO_N_STEPS * 5 * N_ENVS , 5000) # Eval freq in agent steps
    eval_callback = EvalCallback(
        eval_env_instance, best_model_save_path=os.path.join(EVAL_LOG_DIR, 'best_model_ppo'),
        log_path=EVAL_LOG_DIR, eval_freq=eval_freq_timesteps // N_ENVS, # eval_freq is per env step for SB3
        n_eval_episodes=10, deterministic=True, render=False, warn=True )
    save_freq_timesteps = max(PPO_N_STEPS * 20 * N_ENVS, 20000) # Checkpoint freq in agent steps
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_timesteps // N_ENVS, # save_freq is per env step for SB3
        save_path=CHECKPOINT_LOG_DIR, name_prefix='enhancer_ppo_vlm' )
    callbacks_list = [swd_curriculum_callback, action_curriculum_callback, reward_log_callback, checkpoint_callback, eval_callback]
    print("Callbacks configured.")

    print(f"\n--- Starting PPO Training for {TOTAL_TIMESTEPS} Timesteps ---")
    print(f"Log Interval (PPO n_steps * n_envs): {PPO_N_STEPS * N_ENVS}")
    print(f"TensorBoard: tensorboard --logdir \"{os.path.abspath(TENSORBOARD_LOG_DIR)}\"")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, callback=callbacks_list,
            tb_log_name="PPO_Enhancer_VLM_GradCAMpp", progress_bar=True
        )
    except Exception as e_train: print(f"FATAL ERROR during training: {e_train}"); traceback.print_exc()
    finally:
        print("\n--- Training Finished or Interrupted ---")
        MODEL_SAVE_PATH = f"{LOG_DIR_BASE}/enhancer_ppo_vlm_final"
        try: model.save(MODEL_SAVE_PATH); print(f"Final model saved to {MODEL_SAVE_PATH}.zip")
        except Exception as e_save: print(f"Error saving final model: {e_save}")
        print("Closing environments...");
        if 'train_env' in locals() and train_env is not None: train_env.close()
        if 'eval_env_instance' in locals() and eval_env_instance is not None: eval_env_instance.close()
        del steering_model_instance, gradcam_instance_steering
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("Script finished.")
        for d_name, d_val in [("TensorBoard", TENSORBOARD_LOG_DIR), ("Evaluation", EVAL_LOG_DIR), ("Checkpoints", CHECKPOINT_LOG_DIR)]:
            print(f"{d_name} logs: {os.path.abspath(d_val)}")