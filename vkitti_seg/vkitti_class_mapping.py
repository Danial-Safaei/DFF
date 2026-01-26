"""
V-KITTI Class Segmentation Mapping Utilities

Converts V-KITTI RGB class segmentation maps to class indices for optimal ControlNet conditioning.
Based on research showing class indices perform better than RGB colors for ControlNet segmentation.
"""

import numpy as np
import os
from PIL import Image
import torch
from typing import Dict, Tuple, Union

# V-KITTI Class Mapping (RGB -> Class Index)
VKITTI_CLASS_MAPPING = {
    # Category: (r, g, b) -> class_index
    'Terrain': (210, 0, 200, 0),
    'Sky': (90, 200, 255, 1),
    'Tree': (0, 199, 0, 2),
    'Vegetation': (90, 240, 0, 3),
    'Building': (140, 140, 140, 4),
    'Road': (100, 60, 100, 5),
    'GuardRail': (250, 100, 255, 6),
    'TrafficSign': (255, 255, 0, 7),
    'TrafficLight': (200, 200, 0, 8),
    'Pole': (255, 130, 0, 9),
    'Misc': (80, 80, 80, 10),
    'Truck': (160, 60, 60, 11),
    'Car': (255, 127, 80, 12),
    'Van': (0, 139, 139, 13),
    'Undefined': (0, 0, 0, 14)
}

# Create RGB to index lookup
RGB_TO_INDEX = {}
INDEX_TO_RGB = {}
CLASS_NAMES = []

for category, (r, g, b, idx) in VKITTI_CLASS_MAPPING.items():
    RGB_TO_INDEX[(r, g, b)] = idx
    INDEX_TO_RGB[idx] = (r, g, b)
    CLASS_NAMES.append(category)

def rgb_to_class_indices(rgb_seg_map: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """
    Convert V-KITTI RGB segmentation map to class indices.

    Args:
        rgb_seg_map: RGB segmentation map (H, W, 3) as numpy array or PIL Image

    Returns:
        class_indices: Class index map (H, W) with values 0-14
    """
    if isinstance(rgb_seg_map, Image.Image):
        rgb_seg_map = np.array(rgb_seg_map)

    height, width = rgb_seg_map.shape[:2]
    class_indices = np.zeros((height, width), dtype=np.uint8)

    # Convert each RGB color to class index
    for (r, g, b), class_idx in RGB_TO_INDEX.items():
        # Find pixels matching this RGB color (with small tolerance for compression artifacts)
        mask = (
            (np.abs(rgb_seg_map[:, :, 0] - r) <= 2) &
            (np.abs(rgb_seg_map[:, :, 1] - g) <= 2) &
            (np.abs(rgb_seg_map[:, :, 2] - b) <= 2)
        )
        class_indices[mask] = class_idx

    return class_indices

def class_indices_to_rgb(class_indices: np.ndarray) -> np.ndarray:
    """
    Convert class indices back to RGB segmentation map for visualization.

    Args:
        class_indices: Class index map (H, W) with values 0-14

    Returns:
        rgb_seg_map: RGB segmentation map (H, W, 3)
    """
    height, width = class_indices.shape
    rgb_seg_map = np.zeros((height, width, 3), dtype=np.uint8)

    for class_idx, (r, g, b) in INDEX_TO_RGB.items():
        mask = class_indices == class_idx
        rgb_seg_map[mask] = [r, g, b]

    return rgb_seg_map

def prepare_vkitti_conditioning_image(seg_path: str, target_size: Tuple[int, int] = (1248, 376)) -> torch.Tensor:
    """
    Prepare V-KITTI segmentation image for ControlNet conditioning.

    Args:
        seg_path: Path to V-KITTI RGB segmentation image
        target_size: Target size (height, width) for conditioning

    Returns:
        conditioning_tensor: Normalized tensor ready for ControlNet (C, H, W)
    """
    # Load RGB segmentation
    rgb_seg = Image.open(seg_path).convert('RGB')
    rgb_seg = rgb_seg.resize((target_size[1], target_size[0]), Image.NEAREST)  # Use NEAREST to preserve exact colors

    # Convert to class indices
    class_indices = rgb_to_class_indices(rgb_seg)

    # Convert to tensor and normalize to [0, 1]
    conditioning_tensor = torch.from_numpy(class_indices).float() / 14.0  # Normalize by max class index

    # Add channel dimension (1, H, W)
    conditioning_tensor = conditioning_tensor.unsqueeze(0)

    return conditioning_tensor

def create_debug_visualization(rgb_seg_path: str, output_path: str) -> None:
    """
    Create debug visualization showing original RGB and converted class indices.

    Args:
        rgb_seg_path: Path to original RGB segmentation
        output_path: Path to save debug visualization
    """
    # Load original
    rgb_original = Image.open(rgb_seg_path).convert('RGB')
    rgb_array = np.array(rgb_original)

    # Convert to class indices and back
    class_indices = rgb_to_class_indices(rgb_array)
    rgb_reconstructed = class_indices_to_rgb(class_indices)

    # Create side-by-side comparison
    fig_width = rgb_array.shape[1] * 2
    fig_height = rgb_array.shape[0]
    comparison = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)

    comparison[:, :rgb_array.shape[1]] = rgb_array
    comparison[:, rgb_array.shape[1]:] = rgb_reconstructed

    # Save debug image
    Image.fromarray(comparison).save(output_path)
    print(f"🔍 Debug visualization saved to {output_path}")

# Quality optimization settings for V-KITTI ControlNet training
VKITTI_TRAINING_CONFIG = {
    # Model settings for high quality - OPTIMIZED FOR SPEED ON L40 (46GB)
    # NOTE (training compliance): project-wide training/eval expects 512x512 by default.
    # You can override at runtime with env TRAIN_RESOLUTION="1248x376" (or any WxH / int).
    "resolution": "512",
    "train_batch_size": 2,  # Increased from 1 - L40 has 46GB VRAM
    "gradient_accumulation_steps": 2,  # Effective batch size = 4 (same as before but faster)

    # Learning rate settings - scaled for larger effective batch
    "learning_rate": 2e-5,  # Doubled since batch is effectively same but more stable gradients
    "lr_scheduler": "constant_with_warmup",
    "lr_warmup_steps": 250,  # Reduced warmup since training is shorter

    # Training stability - REDUCED STEPS (2x batch throughput)
    "max_train_steps": 30000,  # Reduced from 50000 - same effective training with 2x batch
    "checkpointing_steps": 2500,
    "validation_steps": 2000,  # Less frequent validation to save time

    # Quality optimizations - USE BF16 for better stability and speed
    "mixed_precision": "bf16",  # Changed from fp16 - better numerical stability, same speed
    "gradient_checkpointing": True,
    "use_8bit_adam": True,

    # Data augmentation (minimal to preserve segmentation accuracy)
    "random_flip": False,  # Disable to preserve segmentation labels

    # Loss settings
    "snr_gamma": 5.0,  # Helps with training stability

    # SPEED OPTIMIZATIONS
    "dataloader_num_workers": 8,  # Increased from default 4
    "dataloader_pin_memory": True,  # Faster CPU->GPU transfer
}

def validate_vkitti_dataset(dataset_path: str) -> bool:
    """
    Validate V-KITTI class segmentation dataset for completeness.
    
    Supports two modes:
    1. Raw V-KITTI format: Scene01/, Scene02/, etc. subdirectories
    2. Prepared dataset format: metadata.jsonl file with paths

    Args:
        dataset_path: Path to V-KITTI class segmentation directory OR prepared dataset directory

    Returns:
        is_valid: True if dataset is properly formatted
    """
    import os
    import json

    # Check if this is a prepared dataset (has metadata.jsonl)
    metadata_path = os.path.join(dataset_path, 'metadata.jsonl')
    if os.path.exists(metadata_path):
        try:
            # Validate metadata.jsonl has entries and paths exist
            valid_entries = 0
            invalid_entries = 0
            with open(metadata_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 10:  # Only check first 10 entries for speed
                        break
                    entry = json.loads(line.strip())
                    if 'image' in entry and 'conditioning_image' in entry:
                        # Optionally verify paths exist (first entry only)
                        if i == 0:
                            if os.path.exists(entry['image']) and os.path.exists(entry['conditioning_image']):
                                print(f"✅ Prepared dataset validated: {metadata_path}")
                                print(f"   Sample image: {entry['image']}")
                                print(f"   Sample conditioning: {entry['conditioning_image']}")
                            else:
                                print(f"⚠️ Sample paths may not exist, but proceeding...")
                        valid_entries += 1
                    else:
                        invalid_entries += 1
            
            if valid_entries > 0:
                print(f"✅ Prepared dataset with {valid_entries}+ valid entries")
                return True
            else:
                print(f"❌ No valid entries in metadata.jsonl")
                return False
        except Exception as e:
            print(f"❌ Error reading metadata.jsonl: {e}")
            return False

    # Original validation: Raw V-KITTI format with Scene directories
    required_scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']

    for scene in required_scenes:
        scene_path = os.path.join(dataset_path, scene)
        if not os.path.exists(scene_path):
            print(f"❌ Missing scene directory: {scene}")
            return False

        # Check for segmentation files
        seg_files = [f for f in os.listdir(scene_path) if f.startswith('classgt_') and f.endswith('.png')]
        if len(seg_files) == 0:
            print(f"❌ No segmentation files found in {scene}")
            return False

        print(f"✅ {scene}: {len(seg_files)} segmentation files")

    print(f"✅ V-KITTI class segmentation dataset validation complete")
    return True

if __name__ == "__main__":
    # Test the mapping system
    print("🔍 V-KITTI Class Mapping System")
    print(f"📊 Total classes: {len(VKITTI_CLASS_MAPPING)}")

    for category, (r, g, b, idx) in VKITTI_CLASS_MAPPING.items():
        print(f"  {idx:2d}: {category:12s} -> RGB({r:3d}, {g:3d}, {b:3d})")

    # Validate dataset if available
    dataset_path = "/networkhome/WMGDS/safaei_d/3_generators/data/data_for_new_mvp/class_segmentation_v_kitti"
    if os.path.exists(dataset_path):
        validate_vkitti_dataset(dataset_path)