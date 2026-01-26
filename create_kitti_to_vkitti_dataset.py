#!/usr/bin/env python3
"""
Create KITTI→V-KITTI Translation Dataset

Creates the proper dataset structure for direct KITTI→V-KITTI image translation:
- Input: KITTI image (what user provides)
- Conditioning: V-KITTI class segmentation (structural guidance)
- Target: V-KITTI image (desired output style)

This enables true KITTI→V-KITTI style transfer with segmentation-guided conditioning.
"""

import os
import json
from pathlib import Path
from PIL import Image
import numpy as np

def verify_image_triplet(kitti_path, seg_path, vkitti_path):
    """Verify that all three images in a triplet exist and are valid."""
    try:
        # Check file existence
        if not all(os.path.exists(p) for p in [kitti_path, seg_path, vkitti_path]):
            return False, "Missing file(s)"

        # Verify images can be loaded
        kitti_img = Image.open(kitti_path)
        seg_img = Image.open(seg_path)
        vkitti_img = Image.open(vkitti_path)

        # Basic size validation
        if any(img.size[0] < 100 or img.size[1] < 100 for img in [kitti_img, seg_img, vkitti_img]):
            return False, "Image too small"

        return True, "Valid triplet"
    except Exception as e:
        return False, f"Error: {str(e)}"

def create_kitti_to_vkitti_dataset():
    """
    Create KITTI→V-KITTI translation dataset with triplet structure.

    Dataset Structure:
    - input_image: KITTI real photo (user input)
    - conditioning_image: V-KITTI segmentation (structural guidance)
    - target_image: V-KITTI synthetic image (desired output)
    """

    # Paths (override with env if needed)
    kitti_path = os.environ.get("KITTI_PATH", "/path/to/KITTI")
    vkitti_images_path = os.environ.get("VKITTI_IMAGES_PATH", "/path/to/V-KITTI")
    vkitti_seg_path = os.environ.get("VKITTI_SEG_PATH", "/path/to/V-KITTI/segmentation")
    output_dir = os.environ.get(
        "OUTPUT_DIR",
        str(Path(__file__).resolve().parent / "controlnet_kitti_to_vkitti_dataset")
    )
    print(f"Using KITTI_PATH={kitti_path}")
    print(f"Using VKITTI_IMAGES_PATH={vkitti_images_path}")
    print(f"Using VKITTI_SEG_PATH={vkitti_seg_path}")
    print(f"Using OUTPUT_DIR={output_dir}")

    # Scene mapping
    scene_mapping = {
        "0001": ("Scene01", "scene_01"),
        "0002": ("Scene02", "scene_02"),
        "0006": ("Scene06", "scene_06"),
        "0018": ("Scene18", "scene_18"),
        "0020": ("Scene20", "scene_20")
    }

    os.makedirs(output_dir, exist_ok=True)

    dataset_entries = []
    total_triplets = 0
    valid_triplets = 0

    print("🎯 Creating KITTI→V-KITTI Translation Dataset")
    print("Input: KITTI → Conditioning: Segmentation → Target: V-KITTI")
    print("=" * 70)

    for kitti_seq, (seg_scene, vkitti_scene) in scene_mapping.items():
        print(f"\n📁 Processing Sequence {kitti_seq} → {seg_scene} → {vkitti_scene}")

        # Paths for this sequence
        kitti_seq_path = os.path.join(kitti_path, kitti_seq)
        seg_scene_path = os.path.join(vkitti_seg_path, seg_scene)
        vkitti_scene_path = os.path.join(vkitti_images_path, vkitti_scene, "frames", "rgb", "Camera_0")

        # Verify all directories exist
        missing_dirs = []
        if not os.path.exists(kitti_seq_path):
            missing_dirs.append(f"KITTI: {kitti_seq_path}")
        if not os.path.exists(seg_scene_path):
            missing_dirs.append(f"Segmentation: {seg_scene_path}")
        if not os.path.exists(vkitti_scene_path):
            missing_dirs.append(f"V-KITTI: {vkitti_scene_path}")

        if missing_dirs:
            print(f"❌ Missing directories:")
            for missing in missing_dirs:
                print(f"   {missing}")
            continue

        # Get all KITTI images for this sequence
        kitti_images = [f for f in os.listdir(kitti_seq_path) if f.endswith('.png')]
        kitti_images.sort()

        sequence_triplets = 0
        sequence_valid = 0

        for kitti_file in kitti_images:
            # Extract frame number (000000.png -> 0)
            frame_num = int(kitti_file.replace('.png', ''))

            # Build paths for triplet
            kitti_image_path = os.path.join(kitti_seq_path, kitti_file)
            seg_file = f"classgt_{frame_num:05d}.png"
            seg_image_path = os.path.join(seg_scene_path, seg_file)
            vkitti_file = f"rgb_{frame_num:05d}.jpg"
            vkitti_image_path = os.path.join(vkitti_scene_path, vkitti_file)

            # Verify triplet validity
            is_valid, reason = verify_image_triplet(kitti_image_path, seg_image_path, vkitti_image_path)

            if is_valid:
                # Create dataset entry for KITTI→V-KITTI translation
                entry = {
                    "input_image": kitti_image_path,        # KITTI real photo (input)
                    "conditioning_image": seg_image_path,   # V-KITTI segmentation (conditioning)
                    "target_image": vkitti_image_path,      # V-KITTI synthetic (target output)
                    "prompt": f"Transform real KITTI driving scene to V-KITTI virtual style, sequence {kitti_seq}, frame {frame_num}",
                    "sequence": kitti_seq,
                    "frame": frame_num,
                    "scene_mapping": f"{kitti_seq}→{seg_scene}→{vkitti_scene}"
                }

                dataset_entries.append(entry)
                sequence_valid += 1
            else:
                print(f"   ⚠️  Frame {frame_num}: {reason}")

            sequence_triplets += 1

        print(f"✅ {sequence_valid}/{sequence_triplets} valid triplets in {kitti_seq}")
        valid_triplets += sequence_valid
        total_triplets += sequence_triplets

    # Write dataset metadata
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_path, 'w') as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry) + '\n')

    # Write dataset info
    info_path = os.path.join(output_dir, "dataset_info.json")
    dataset_info = {
        "dataset_type": "kitti_to_vkitti_translation",
        "description": "KITTI→V-KITTI image translation with segmentation conditioning",
        "total_triplets": valid_triplets,
        "total_attempted": total_triplets,
        "success_rate": f"{valid_triplets/total_triplets*100:.1f}%" if total_triplets > 0 else "0%",
        "structure": {
            "input_image": "KITTI real photo (user input)",
            "conditioning_image": "V-KITTI segmentation (structural guidance)",
            "target_image": "V-KITTI synthetic image (desired output)"
        },
        "scene_mappings": scene_mapping,
        "created": "2025-09-25"
    }

    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    # Create sample visualization script
    sample_script_path = os.path.join(output_dir, "visualize_triplet.py")
    with open(sample_script_path, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""
Visualize KITTI→V-KITTI dataset triplets
"""
import json
import random
import matplotlib.pyplot as plt
from PIL import Image

def visualize_random_triplet():
    with open("metadata.jsonl", 'r') as f:
        entries = [json.loads(line) for line in f]

    entry = random.choice(entries)

    # Load images
    kitti_img = Image.open(entry["input_image"])
    seg_img = Image.open(entry["conditioning_image"])
    vkitti_img = Image.open(entry["target_image"])

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'KITTI→V-KITTI Triplet: {entry["scene_mapping"]}, Frame {entry["frame"]}', fontsize=14)

    axes[0].imshow(kitti_img)
    axes[0].set_title('Input: KITTI Real Photo')
    axes[0].axis('off')

    axes[1].imshow(seg_img)
    axes[1].set_title('Conditioning: V-KITTI Segmentation')
    axes[1].axis('off')

    axes[2].imshow(vkitti_img)
    axes[2].set_title('Target: V-KITTI Synthetic')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_random_triplet()
''')

    print(f"\n🎉 KITTI→V-KITTI Translation Dataset Created!")
    print(f"📊 Statistics:")
    print(f"   Total valid triplets: {valid_triplets}")
    print(f"   Success rate: {valid_triplets/total_triplets*100:.1f}%")
    print(f"📁 Output directory: {output_dir}")
    print(f"📝 Metadata: {metadata_path}")
    print(f"ℹ️  Info: {info_path}")
    print(f"🔍 Sample viz: {sample_script_path}")

    print(f"\n🎯 Dataset Structure:")
    print(f"   Input: KITTI real photos (what user provides)")
    print(f"   Conditioning: V-KITTI segmentation (structural guidance)")
    print(f"   Target: V-KITTI synthetic images (desired style output)")

    return output_dir, valid_triplets

if __name__ == "__main__":
    create_kitti_to_vkitti_dataset()
