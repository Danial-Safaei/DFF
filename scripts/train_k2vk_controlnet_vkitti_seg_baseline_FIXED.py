#!/usr/bin/env python3
"""
V-KITTI SEGMENTATION BASELINE - PRODUCTION READY

Robust V-KITTI class segmentation ControlNet training with comprehensive validation,
proper error handling, and template-based script generation.

FIXES APPLIED:
- Replaced fragile string replacement with template-based architecture
- Added comprehensive validation and error handling
- Fixed all import and context issues
- Complete implementation with proper argument handling

Usage:
    python train_k2vk_controlnet_vkitti_seg_baseline_FIXED.py \
        --vkitti_seg_path /path/to/vkitti/segmentation \
        --kitti_images_path /path/to/kitti/images \
        --output_dir /path/to/output \
        --steps 45000
"""

import os
import sys
import pathlib

# Add current directory to path for local imports
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_VKITTI_SEG_DIR = _THIS_DIR.parent / "vkitti_seg"
if str(_VKITTI_SEG_DIR) not in sys.path:
    sys.path.insert(0, str(_VKITTI_SEG_DIR))

from vkitti_training_base import VKITTITrainingBase, create_robust_argument_parser
from model_loading import initialize_external_models  # New unified loader


class VKITTIBaselineTrainer(VKITTITrainingBase):
    """
    V-KITTI Baseline segmentation trainer with robust architecture.
    """

    def __init__(self):
        super().__init__("baseline")

    def apply_variant_modifications(self, base_content: str, **kwargs) -> str:
        """
        Apply baseline-specific modifications using robust template approach.
        """
        self.logger.info("🔧 Applying baseline V-KITTI modifications...")
        
        # CRITICAL: These wrappers patch an upstream HuggingFace script using string markers.
        # If HF changes, we must FAIL FAST (not silently skip) to avoid "training succeeded" while
        # the intended logic never ran.
        def must_contain(content: str, marker: str, label: str) -> None:
            if marker not in content:
                raise RuntimeError(f"Required patch marker not found ({label}): {marker!r}")

        # Define modification markers and replacements
        modifications = []

        # 1. Add V-KITTI imports at the top
        import_marker = "import diffusers"
        vkitti_imports = '''import diffusers

# V-KITTI Baseline Imports
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# V-KITTI Class Mapping for RGB -> Class Index conversion
VKITTI_CLASS_MAPPING = {
    'Terrain': (210, 0, 200, 0), 'Sky': (90, 200, 255, 1), 'Tree': (0, 199, 0, 2),
    'Vegetation': (90, 240, 0, 3), 'Building': (140, 140, 140, 4), 'Road': (100, 60, 100, 5),
    'GuardRail': (250, 100, 255, 6), 'TrafficSign': (255, 255, 0, 7), 'TrafficLight': (200, 200, 0, 8),
    'Pole': (255, 130, 0, 9), 'Misc': (80, 80, 80, 10), 'Truck': (160, 60, 60, 11),
    'Car': (255, 127, 80, 12), 'Van': (0, 139, 139, 13), 'Undefined': (0, 0, 0, 14)
}

# Create RGB to index lookup
RGB_TO_INDEX = {(r, g, b): idx for _, (r, g, b, idx) in VKITTI_CLASS_MAPPING.items()}

def rgb_to_class_indices(rgb_seg_map):
    """Convert V-KITTI RGB segmentation to class indices."""
    if hasattr(rgb_seg_map, 'convert'):  # PIL Image
        rgb_seg_map = np.array(rgb_seg_map)

    height, width = rgb_seg_map.shape[:2]
    class_indices = np.zeros((height, width), dtype=np.uint8)

    for (r, g, b), class_idx in RGB_TO_INDEX.items():
        mask = (
            (np.abs(rgb_seg_map[:, :, 0] - r) <= 2) &
            (np.abs(rgb_seg_map[:, :, 1] - g) <= 2) &
            (np.abs(rgb_seg_map[:, :, 2] - b) <= 2)
        )
        class_indices[mask] = class_idx

    return class_indices'''

        modifications.append((import_marker, vkitti_imports))

        # 2. Add KITTI→V-KITTI translation preprocessing function
        dataset_prep_marker = "train_dataset = prepare_train_dataset(train_dataset, accelerator)"
        must_contain(base_content, dataset_prep_marker, "dataset prep")
        # Instead of replacing the line (which may disturb indentation or later logic), we inject
        # the preprocessing block immediately BEFORE it while keeping the original line intact.
        preprocessing_block = (
            "# === KITTI→V-KITTI Translation Training ===\n"
            "print('🎯 KITTI→V-KITTI: KITTI input + segmentation conditioning → V-KITTI style output')\n"
            "from datasets import Image as DatasetImage\n"
            "# Map dataset columns for triplet structure\n"
            "train_dataset = train_dataset.cast_column('target_image', DatasetImage())  # V-KITTI targets\n"
            "train_dataset = train_dataset.cast_column('conditioning_image', DatasetImage())  # Segmentation conditioning\n"
            "train_dataset = train_dataset.cast_column('input_image', DatasetImage())  # KITTI inputs\n\n"
            "def preprocess_kitti_vkitti_proper(examples):\n"
            "    '''Process KITTI→V-KITTI with PROPER ControlNet architecture.'''\n"
            "    conditioning_images = []\n"
            "    enhanced_prompts = []\n"
            "\n"
            "    # Process each triplet (KITTI input, segmentation, V-KITTI target)\n"
            "    for i in range(len(examples[args.conditioning_image_column])):\n"
            "        # Load KITTI input for content analysis\n"
            "        kitti_path = examples['input_image'][i]\n"
            "        if isinstance(kitti_path, str):\n"
            "            kitti_img = Image.open(kitti_path).convert('RGB')\n"
            "        else:\n"
            "            kitti_img = kitti_path.convert('RGB')\n"
            "\n"
            "        # Load segmentation conditioning (pure structural guidance)\n"
            "        seg_path = examples[args.conditioning_image_column][i]\n"
            "        if isinstance(seg_path, str):\n"
            "            seg_img = Image.open(seg_path).convert('RGB')\n"
            "        else:\n"
            "            seg_img = seg_path.convert('RGB')\n"
            "\n"
            "        # Resize segmentation conditioning to match training resolution\n"
            "        # IMPORTANT: do NOT hardcode KITTI native size; follow --resolution.\n"
            "        seg_resized = seg_img.resize((args.resolution_width, args.resolution_height), Image.NEAREST)\n"
            "\n"
            "        conditioning_images.append(seg_resized)\n"
            "\n"
            "        # Enhance prompt with lightweight KITTI content analysis\n"
            "        base_prompt = examples['prompt'][i] if i < len(examples.get('prompt', [])) else 'Transform to V-KITTI style'\n"
            "\n"
            "        kitti_resized = kitti_img.resize((256, 256), Image.LANCZOS)\n"
            "        import numpy as np\n"
            "        kitti_array = np.array(kitti_resized)\n"
            "\n"
            "        avg_brightness = np.mean(kitti_array)\n"
            "        green_ratio = np.mean(kitti_array[:, :, 1]) / (np.mean(kitti_array) + 1e-6)\n"
            "        blue_ratio = np.mean(kitti_array[:, :, 2]) / (np.mean(kitti_array) + 1e-6)\n"
            "\n"
            "        content_hints = []\n"
            "        if avg_brightness > 140: content_hints.append('bright daylight scene')\n"
            "        elif avg_brightness < 80: content_hints.append('overcast or shadowy scene')\n"
            "        else: content_hints.append('balanced lighting scene')\n"
            "\n"
            "        if green_ratio > 1.15: content_hints.append('abundant vegetation')\n"
            "        if blue_ratio > 1.1: content_hints.append('clear sky visible')\n"
            "\n"
            "        enhanced_prompt = f\"{base_prompt}, {', '.join(content_hints)}, photorealistic V-KITTI synthetic rendering\"\n"
            "        enhanced_prompts.append(enhanced_prompt)\n"
            "\n"
            "    examples[args.conditioning_image_column] = conditioning_images\n"
            "    examples['prompt'] = enhanced_prompts\n"
            "    return examples\n\n"
            "train_dataset = train_dataset.map(preprocess_kitti_vkitti_proper, batched=True)\n"
            "print(f'✅ Applied PROPER KITTI→V-KITTI architecture preprocessing to {len(train_dataset)} samples')\n"
            "print('   - Pure segmentation as ControlNet conditioning')\n"
            "print('   - Content-aware prompt enhancement from KITTI analysis')\n"
            "# === End V-KITTI Segmentation Preprocessing ===\n"
        )

        # Defer injection: store a tuple indicating we want to insert before marker
        modifications.append((dataset_prep_marker, preprocessing_block, 'inject_before'))

        # (Quality optimization block removed for stability; can be re-added with safe indentation logic later.)

        # Apply all modifications sequentially
        modified_content = base_content
        for mod in modifications:
            if len(mod) == 2:
                marker, replacement = mod
                mode = 'replace'
            else:
                marker, replacement, mode = mod

            if marker not in modified_content:
                # Fail fast: do not allow silent downgrades.
                raise RuntimeError(f"Required patch marker not found: {marker[:120]}")

            if mode == 'replace':
                modified_content = modified_content.replace(marker, replacement)
                self.logger.info(f"✅ Applied modification (replace): {marker[:40]}...")
            elif mode == 'inject_before':
                # Derive indentation from marker occurrence (characters between previous newline and marker start)
                idx = modified_content.find(marker)
                line_start = modified_content.rfind('\n', 0, idx) + 1
                existing_indent = ''
                while line_start + len(existing_indent) < len(modified_content) and modified_content[line_start + len(existing_indent)] in (' ', '\t'):
                    existing_indent += modified_content[line_start + len(existing_indent)]

                def indent_block(block: str, indent: str) -> str:
                    lines = block.split('\n')
                    indented = []
                    for l in lines:
                        if l.strip() == '':
                            indented.append(l)
                        else:
                            indented.append(indent + l)
                    return '\n'.join(indented)

                injected = indent_block(replacement.rstrip('\n'), existing_indent) + '\n'
                modified_content = modified_content.replace(marker, injected + existing_indent + marker)
                indent_repr = existing_indent.replace('\t', '\\t')
                self.logger.info(f"✅ Applied modification (inject_before): {marker[:40]} with indent '{indent_repr}' ...")
            else:
                self.logger.warning(f"⚠️ Unknown modification mode '{mode}' for marker snippet: {marker[:40]}")
                continue

        self.logger.info("✅ Baseline modifications applied successfully")

        # --- Post-processing fix: ensure quality optimization block keeps indentation level ---
        lines = modified_content.split('\n')
        for i, line in enumerate(lines):
            if line.strip() == '# V-KITTI Quality Optimizations':
                # Determine required indent by scanning upwards for the nearest non-empty preceding line
                for j in range(i - 1, -1, -1):
                    prev = lines[j]
                    if prev.strip():
                        prev_indent = prev[:len(prev) - len(prev.lstrip(' \t'))]
                        current_indent = line[:len(line) - len(line.lstrip(' \t'))]
                        # If current indent shorter than previous indent and previous indent length is multiple of 4, adjust block
                        if len(prev_indent) > len(current_indent):
                            k = i
                            # Re-indent contiguous block until blank line or dedent to earlier than current
                            while k < len(lines) and lines[k].strip():
                                content = lines[k].lstrip(' \t')
                                lines[k] = prev_indent + content
                                k += 1
                        break
        modified_content = '\n'.join(lines)
        return modified_content

    def get_training_arguments(self, **kwargs) -> list:
        """Get baseline-specific training arguments."""
        # Baseline uses standard ControlNet arguments only
        return []


def main():
    """Main execution function with comprehensive error handling."""

    # Parse arguments with validation
    baseline_args = {
        # Optional: resume from an existing ControlNet directory (exact path used in your notebook)
        'controlnet_model_name_or_path': {
            'type': str,
            'required': False,
            'help': 'Path to an existing ControlNet directory to initialize weights from (resume)'
        }
    }
    args, parser = create_robust_argument_parser("baseline", baseline_args)

    # Create trainer instance
    trainer = VKITTIBaselineTrainer()

    # Execute training
    success = trainer.run_training(
        vkitti_seg_path=args.vkitti_seg_path,
        kitti_images_path=args.kitti_images_path,
        output_dir=args.output_dir,
        steps=args.steps,
        # Pass through optional ControlNet init path if provided (else env CONTROLNET_PATH or default will be used)
        controlnet_model_name_or_path=args.controlnet_model_name_or_path
    )

    if success:
        trainer.logger.info("🎉 V-KITTI Baseline training completed successfully!")
        sys.exit(0)
    else:
        trainer.logger.error("💥 V-KITTI Baseline training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()