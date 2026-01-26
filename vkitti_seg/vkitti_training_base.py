"""
V-KITTI Training Base Infrastructure

Robust, template-based training infrastructure for V-KITTI segmentation ControlNet training.
Replaces fragile string replacement with solid template-based architecture.
"""

import os
import sys
import json
import shutil
import pathlib
import datetime
import subprocess
import urllib.request
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod

# Add current directory to path for local imports
sys.path.append(str(pathlib.Path(__file__).parent))

from vkitti_class_mapping import VKITTI_TRAINING_CONFIG, validate_vkitti_dataset
from utils_robust import setup_logging, create_output_directory, RobustSubprocessRunner


class VKITTITrainingBase(ABC):
    """
    Base class for V-KITTI ControlNet training with robust architecture.
    Provides template-based script generation instead of fragile string replacement.
    """

    def __init__(self, variant_name: str):
        self.variant_name = variant_name
        self.logger = setup_logging(f"vkitti_seg_{variant_name}")
        self.script_template = None

    def validate_environment(self) -> bool:
        """Validate training environment and dependencies."""
        try:
            # Check Python environment
            import torch
            import diffusers
            import accelerate

            self.logger.info(f"✅ Environment validation passed")
            self.logger.info(f"   - PyTorch: {torch.__version__}")
            self.logger.info(f"   - Diffusers: {diffusers.__version__}")
            self.logger.info(f"   - Accelerate: {accelerate.__version__}")

            return True

        except ImportError as e:
            self.logger.error(f"❌ Missing dependency: {e}")
            return False

    def validate_model_paths(self, **model_paths) -> bool:
        """Validate all required model paths exist."""
        for name, path in model_paths.items():
            if path and not os.path.exists(path):
                self.logger.error(f"❌ {name} not found: {path}")
                return False
            elif path:
                self.logger.info(f"✅ {name} validated: {path}")
        return True

    def create_vkitti_dataset_jsonl(self, vkitti_seg_path: str, kitti_images_path: str, output_jsonl: str) -> int:
        """
        Create JSONL dataset with robust error handling and validation.
        """
        self.logger.info("🔄 Creating V-KITTI dataset JSONL with enhanced validation...")

        pairs = []
        scene_mapping = {
            'Scene01': '0001', 'Scene02': '0002', 'Scene06': '0006',
            'Scene18': '0018', 'Scene20': '0020'
        }

        for vkitti_scene, kitti_scene in scene_mapping.items():
            seg_scene_path = os.path.join(vkitti_seg_path, vkitti_scene)
            kitti_scene_path = os.path.join(kitti_images_path, kitti_scene)

            if not os.path.exists(seg_scene_path):
                self.logger.warning(f"⚠️ V-KITTI scene missing: {seg_scene_path}")
                continue

            if not os.path.exists(kitti_scene_path):
                self.logger.warning(f"⚠️ KITTI scene missing: {kitti_scene_path}")
                continue

            seg_files = sorted([f for f in os.listdir(seg_scene_path)
                              if f.startswith('classgt_') and f.endswith('.png')])

            scene_pairs = 0
            for seg_file in seg_files:
                frame_id = seg_file.replace('classgt_', '').replace('.png', '')
                kitti_frame_id = frame_id.zfill(6)
                kitti_image_path = os.path.join(kitti_scene_path, f"{kitti_frame_id}.png")

                if os.path.exists(kitti_image_path):
                    pairs.append({
                        "image": kitti_image_path,
                        "conditioning_image": os.path.join(seg_scene_path, seg_file),
                        "prompt": self.get_training_prompt(vkitti_scene)
                    })
                    scene_pairs += 1

            self.logger.info(f"✅ {vkitti_scene}: {scene_pairs} pairs found")

        # Write JSONL with validation
        with open(output_jsonl, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair) + '\n')

        self.logger.info(f"✅ Created dataset with {len(pairs)} total image pairs")
        return len(pairs)

    def get_training_prompt(self, scene: str) -> str:
        """Generate training prompt for given scene."""
        # CRITICAL: keep prompt exactly aligned with how you run evaluation.
        # Default is the canonical project prompt; override with TRAIN_PROMPT if you explicitly want per-scene prompts.
        # This prevents "silent drift" where training used one prompt and evaluation used another.
        prompt_override = os.environ.get("TRAIN_PROMPT")
        if prompt_override:
            return prompt_override
        return "virtual kitti style"

    def download_base_script(self) -> pathlib.Path:
        """Download HuggingFace base script with validation."""
        script_url = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/controlnet/train_controlnet_sdxl.py"
        script_path = pathlib.Path("ControlNet/train_controlnet_sdxl_original.py")

        if not script_path.exists():
            self.logger.info("📥 Downloading HuggingFace ControlNet training script...")
            script_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with urllib.request.urlopen(script_url) as response:
                    with open(script_path, 'wb') as f:
                        f.write(response.read())
                self.logger.info("✅ Base script downloaded successfully")
            except Exception as e:
                self.logger.error(f"❌ Download failed: {e}")
                raise

        return script_path

    def create_training_script(self, **kwargs) -> pathlib.Path:
        """
        Create specialized training script using template approach.
        This replaces the fragile string replacement method.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_name = f"train_controlnet_sdxl_vkitti_{self.variant_name}_{timestamp}.py"
        script_path = pathlib.Path("ControlNet") / script_name

        # Get base script
        base_script_path = self.download_base_script()

        # Read base script
        with open(base_script_path, 'r') as f:
            base_content = f.read()

        # Fix version compatibility issue - replace strict version check
        base_content = self.fix_version_compatibility(base_content)

        # Apply variant-specific modifications
        modified_content = self.apply_variant_modifications(base_content, **kwargs)

        # Write specialized script
        with open(script_path, 'w') as f:
            f.write(modified_content)

        self.logger.info(f"✅ Created training script: {script_path}")
        return script_path

    def fix_version_compatibility(self, script_content: str) -> str:
        """Fix Diffusers version compatibility issues and disable checkpoint saving."""
        self.logger.info("🔧 Fixing Diffusers version compatibility and disabling checkpoints...")

        # Replace strict version check with compatible one
        if 'check_min_version("0.36.0.dev0")' in script_content:
            script_content = script_content.replace(
                'check_min_version("0.36.0.dev0")',
                'check_min_version("0.35.0.dev0")'
            )
            self.logger.info("✅ Fixed version check: 0.36.0.dev0 -> 0.35.0.dev0")

        # Alternative: Comment out version check entirely if needed
        elif 'check_min_version(' in script_content:
            import re
            # Find and comment out any version check
            pattern = r'(\s*)check_min_version\([^)]+\)'
            script_content = re.sub(pattern, r'\1# check_min_version() # Disabled for compatibility', script_content)
            self.logger.info("✅ Disabled strict version check for compatibility")

        # Disable checkpoint saving by replacing the checkpoint condition with a never-true condition
        import re

        # Replace the checkpoint saving condition to never trigger
        script_content = script_content.replace(
            'if global_step % args.checkpointing_steps == 0:',
            'if False:  # CHECKPOINT DISABLED - never save intermediate checkpoints'
        )

        # Also set a very high default for checkpointing_steps to prevent issues
        script_content = script_content.replace(
            'default=500,',
            'default=999999999,  # Disabled - no intermediate checkpoints'
        )

        self.logger.info("✅ Disabled all intermediate checkpoint saving - only final model will be saved")

        # Add non-square resolution support
        script_content = self.add_nonsquare_resolution_support(script_content)

        # Add CUDA allocator safeguards early to reduce fragmentation-related OOMs
        try:
            import re
            # Insert after the first os import if present; otherwise after top-level imports
            if 'import os' in script_content:
                script_content = script_content.replace(
                    'import os',
                    'import os\n# Reduce CUDA memory fragmentation during training\n' \
                    'os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")\n' \
                    'os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")'
                )
            else:
                # Fallback: prepend at top
                script_content = (
                    '# Runtime env safeguards for CUDA allocator\n'
                    'import os\n'
                    'os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")\n'
                    'os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")\n'
                ) + script_content
        except Exception:
            # Non-fatal if injection fails
            pass

        # Ensure generated script can import our local helpers (absolute + relative fallback)
        try:
            # Inject sys.path updates near the top (after TF_CPP_MIN_LOG_LEVEL guard if present)
            anchor = 'os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")'
            injection = (
                '\n# Ensure local training helpers are importable (absolute + relative)\n'
                'import sys, pathlib\n'
                'abs_vkitti_path = "/networkhome/WMGDS/safaei_d/3_generators/training/vkitti_seg_final7/vkitti_seg"\n'
                'if abs_vkitti_path not in sys.path:\n'
                '    sys.path.insert(0, abs_vkitti_path)\n'
                'try:\n'
                '    _here = pathlib.Path(__file__).resolve()\n'
                '    _fallback = _here.parent.parent / "vkitti_seg"\n'
                '    if str(_fallback) not in sys.path:\n'
                '        sys.path.append(str(_fallback))\n'
                'except Exception:\n'
                '    pass\n'
            )
            if anchor in script_content:
                script_content = script_content.replace(anchor, anchor + injection)
            else:
                # Prepend if anchor not found
                script_content = injection + script_content
        except Exception:
            pass

        # Align dtype of encoder_hidden_states to UNet input to avoid mixed precision issues
        # CRITICAL: Must inject BEFORE ControlNet call (which happens before UNet call)
        try:
            # First, try to inject before ControlNet conditioning (earliest usage point)
            if '# ControlNet conditioning.' in script_content:
                script_content = script_content.replace(
                    '                # ControlNet conditioning.',
                    (
                        '                # Align dtype for encoder hidden states to match UNet/sample dtype\n'
                        '                try:\n'
                        '                    _ehs = batch["prompt_ids"].to(noisy_latents.dtype) if hasattr(batch["prompt_ids"], "to") else batch["prompt_ids"]\n'
                        '                except Exception:\n'
                        '                    _ehs = batch["prompt_ids"]\n'
                        '\n'
                        '                # ControlNet conditioning.'
                    ),
                    1  # Only replace first occurrence
                )
                script_content = script_content.replace(
                    'encoder_hidden_states=batch["prompt_ids"]',
                    'encoder_hidden_states=_ehs'
                )
            # Fallback: If no ControlNet, inject before UNet call
            elif 'model_pred = unet(' in script_content:
                script_content = script_content.replace(
                    'model_pred = unet(',
                    (
                        '                # Align dtype for encoder hidden states to match UNet/sample dtype\n'
                        '                try:\n'
                        '                    _ehs = batch["prompt_ids"].to(noisy_latents.dtype) if hasattr(batch["prompt_ids"], "to") else batch["prompt_ids"]\n'
                        '                except Exception:\n'
                        '                    _ehs = batch["prompt_ids"]\n'
                        '                model_pred = unet('
                    )
                )
                script_content = script_content.replace(
                    'encoder_hidden_states=batch["prompt_ids"]',
                    'encoder_hidden_states=_ehs'
                )
        except Exception:
            pass

        # Enable VAE memory optimizations (slicing/tiling) right before scheduler setup,
        # so we don't split if/else blocks that build VAE differently.
        try:
            import re
            pattern = r"^([ \t]*)pipeline\.scheduler\s*=\s*UniPCMultistepScheduler\.from_config\(pipeline\.scheduler\.config\)"  # noqa: E501
            def _inject_before_scheduler(match: re.Match) -> str:
                indent = match.group(1)
                extra = (
                    f"{indent}# Memory optimizations for VAE (reduces VRAM)\n"
                    f"{indent}try:\n"
                    f"{indent}    if hasattr(vae, 'enable_slicing'):\n"
                    f"{indent}        vae.enable_slicing()\n"
                    f"{indent}    if hasattr(vae, 'enable_tiling'):\n"
                    f"{indent}        vae.enable_tiling()\n"
                    f"{indent}except Exception:\n"
                    f"{indent}    pass\n\n"
                )
                return extra + match.group(0)
            script_content = re.sub(pattern, _inject_before_scheduler, script_content, count=1, flags=re.MULTILINE)
        except Exception:
            # If pattern not found, skip silently
            pass

        return script_content

    def add_nonsquare_resolution_support(self, script_content: str) -> str:
        """Add support for non-square resolutions in WxH format."""
        self.logger.info("🔧 Adding non-square resolution support (WxH format)...")

        # 1. Modify argument parser to handle WxH format
        # Handle multiple possible argument parser formats
        import re

        # Pattern 1: Multi-line format
        pattern1 = r'parser\.add_argument\(\s*\n\s*"--resolution",\s*type=int,'
        replacement1 = lambda m: m.group(0).replace('type=int,', 'type=str,')

        if re.search(pattern1, script_content):
            script_content = re.sub(pattern1, replacement1, script_content)
            self.logger.info("✅ Modified resolution argument to accept string format (multi-line)")

        # Pattern 2: Single-line format
        pattern2 = r'parser\.add_argument\("--resolution",\s*type=int,'
        replacement2 = r'parser.add_argument("--resolution", type=str,'

        if re.search(pattern2, script_content):
            script_content = re.sub(pattern2, replacement2, script_content)
            self.logger.info("✅ Modified resolution argument to accept string format (single-line)")

        # 2. Add resolution parsing logic after argument parser
        resolution_parser_code = '''

    # Parse resolution (support both int and WxH format)
    if 'x' in str(args.resolution):
        args.resolution_width, args.resolution_height = map(int, str(args.resolution).split('x'))
    else:
        # Square resolution
        res = int(args.resolution)
        args.resolution_width, args.resolution_height = res, res'''

        # Insert after argument parsing
        if 'args = parser.parse_args()' in script_content:
            script_content = script_content.replace(
                'args = parser.parse_args()',
                f'args = parser.parse_args(){resolution_parser_code}'
            )
            self.logger.info("✅ Added resolution parsing logic")

        # 3. Update resolution validation
        resolution_validation_old = 'if args.resolution % 8 != 0:'
        resolution_validation_new = 'if args.resolution_width % 8 != 0 or args.resolution_height % 8 != 0:'
        if resolution_validation_old in script_content:
            script_content = script_content.replace(resolution_validation_old, resolution_validation_new)

            # Update the error message
            script_content = script_content.replace(
                '"`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."',
                '"`--resolution` width and height must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."'
            )
            self.logger.info("✅ Updated resolution validation for WxH format")

        # 4. Update image transforms to use non-square dimensions
        # Replace (args.resolution, args.resolution) with (args.resolution_height, args.resolution_width)
        script_content = script_content.replace(
            'transforms.Resize(args.resolution, interpolation=interpolation)',
            'transforms.Resize((args.resolution_height, args.resolution_width), interpolation=interpolation)'
        )
        script_content = script_content.replace(
            'transforms.CenterCrop(args.resolution)',
            'transforms.CenterCrop((args.resolution_height, args.resolution_width))'
        )
        script_content = script_content.replace(
            'transforms.Resize(args.resolution, interpolation=interpolation_mode)',
            'transforms.Resize((args.resolution_height, args.resolution_width), interpolation=interpolation_mode)'
        )

        # 5. Update original_size and target_size for SDXL
        script_content = script_content.replace(
            'original_size = (args.resolution, args.resolution)',
            'original_size = (args.resolution_height, args.resolution_width)'
        )
        script_content = script_content.replace(
            'target_size = (args.resolution, args.resolution)',
            'target_size = (args.resolution_height, args.resolution_width)'
        )

        self.logger.info("✅ Updated image transforms and SDXL parameters for non-square resolution")
        return script_content

    @abstractmethod
    def apply_variant_modifications(self, base_content: str, **kwargs) -> str:
        """Apply variant-specific modifications to base script."""
        pass

    @abstractmethod
    def get_training_arguments(self, **kwargs) -> List[str]:
        """Get variant-specific training arguments."""
        pass

    def execute_training(self, script_path: pathlib.Path, output_dir: pathlib.Path,
                        dataset_dir: pathlib.Path, steps: int, **kwargs) -> bool:
        """Execute training with comprehensive error handling."""

        # Determine correct column names based on variant and dataset
        if self.variant_name == "baseline":
            # Baseline uses KITTI→V-KITTI dataset with different column structure
            image_column = "target_image"  # V-KITTI synthetic images (targets)
        else:
            # Other variants use standard dataset structure
            image_column = "image"

        # Build command with optional FAST overrides via env vars
        # Env overrides: TRAIN_BS, GRAD_ACCUM, MIXED_PRECISION, DATALOADER_WORKERS, DISABLE_GC
        env_train_bs = os.environ.get("TRAIN_BS")
        env_grad_accum = os.environ.get("GRAD_ACCUM")
        env_mp = os.environ.get("MIXED_PRECISION")
        env_workers = os.environ.get("DATALOADER_WORKERS")
        disable_gc = os.environ.get("DISABLE_GC", "0").lower() in ("1", "true", "yes")

        train_batch_size = int(env_train_bs) if env_train_bs else int(VKITTI_TRAINING_CONFIG["train_batch_size"])
        grad_accum_steps = int(env_grad_accum) if env_grad_accum else int(VKITTI_TRAINING_CONFIG["gradient_accumulation_steps"])
        mixed_precision = env_mp if env_mp else VKITTI_TRAINING_CONFIG["mixed_precision"]
        dataloader_workers = int(env_workers) if env_workers else VKITTI_TRAINING_CONFIG.get("dataloader_num_workers", 8)

        # Allow overriding pretrained model (base SDXL) and controlnet model (for resume from trained ControlNet)
        pretrained_override = kwargs.get("pretrained_model_name_or_path")
        pretrained_arg = pretrained_override if pretrained_override else "stabilityai/stable-diffusion-xl-base-1.0"

        controlnet_override_kw = kwargs.get("controlnet_model_name_or_path")
        controlnet_override_env = os.environ.get("CONTROLNET_PATH")
        controlnet_arg = (
            controlnet_override_kw
            if controlnet_override_kw
            else (controlnet_override_env if controlnet_override_env else "SargeZT/sdxl-controlnet-seg")
        )

        # Allow overriding training resolution without code edits.
        # Keep this in sync with evaluation defaults (project convention is 512x512 unless explicitly overridden).
        train_resolution = os.environ.get("TRAIN_RESOLUTION") or str(VKITTI_TRAINING_CONFIG["resolution"])

        command = [
            "accelerate", "launch", str(script_path),
            "--train_data_dir", str(dataset_dir.resolve()),
            "--output_dir", str(output_dir.resolve()),
            "--pretrained_model_name_or_path", str(pretrained_arg),
            "--controlnet_model_name_or_path", str(controlnet_arg),
            "--image_column", image_column,  # Correct column based on dataset structure
            "--conditioning_image_column", "conditioning_image",  # Segmentation conditioning
            "--caption_column", "prompt",
            "--resolution", str(train_resolution),
            "--train_batch_size", str(train_batch_size),
            "--gradient_accumulation_steps", str(grad_accum_steps),
            "--max_train_steps", str(steps),
            # Note: checkpointing_steps removed - only save final weights
            "--validation_steps", str(VKITTI_TRAINING_CONFIG["validation_steps"]),
            "--learning_rate", str(VKITTI_TRAINING_CONFIG["learning_rate"]),
            "--lr_scheduler", VKITTI_TRAINING_CONFIG["lr_scheduler"],
            "--lr_warmup_steps", str(VKITTI_TRAINING_CONFIG["lr_warmup_steps"]),
            "--mixed_precision", mixed_precision,
            "--report_to", "wandb",
            "--use_8bit_adam",
            "--set_grads_to_none",
            "--dataloader_num_workers", str(dataloader_workers),
            "--allow_tf32",  # Enable TF32 for faster matmul on Ampere+ GPUs (L40)
            # Note: --snr_gamma removed as it's not supported in current diffusers version
        ]

        # Conditionally enable/disable gradient checkpointing
        if not disable_gc and VKITTI_TRAINING_CONFIG.get("gradient_checkpointing", True):
            command.append("--gradient_checkpointing")

        # Add variant-specific arguments
        command.extend(self.get_training_arguments(**kwargs))

        # Create log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_dir / f"vkitti_{self.variant_name}_log_{timestamp}.txt"

        # Log execution details
        self.logger.info(f"📂 Output: {output_dir.resolve()}")
        self.logger.info(f"📝 Log: {log_file}")
        self.logger.info(f"🎯 Steps: {steps}")
        self.logger.info(f"🔧 Variant: {self.variant_name}")
        self.logger.info(f"⚙️  Batch/Accum/MixedPrecision/Workers: {train_batch_size}/{grad_accum_steps}/{mixed_precision}/{dataloader_workers}")
        self.logger.info(f"📦 Pretrained: {pretrained_arg}")
        self.logger.info(f"🧠 Gradient Checkpointing: {'OFF' if disable_gc else 'ON'}")
        self.logger.info(f"🧩 ControlNet init: {controlnet_arg}")
        self.logger.info("="*80)

        result = None
        training_failed = False

        try:
            runner = RobustSubprocessRunner(self.logger)

            # Write initial log
            with open(log_file, "w") as f:
                f.write(f"V-KITTI {self.variant_name} Training Started: {datetime.datetime.now()}\n")
                f.write(f"Command: {' '.join(command)}\n")
                f.write("="*80 + "\n")

            # Execute training
            # Ensure MKL + OpenMP threading layer is compatible in subprocess
            # Add CUDA/PyTorch speed optimizations
            env_overrides = {
                "MKL_THREADING_LAYER": os.environ.get("MKL_THREADING_LAYER", "GNU"),
                # Enable TF32 for faster matmul on Ampere+ GPUs (L40 supports this)
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
                # cuDNN optimizations
                "TORCH_CUDNN_V8_API_ENABLED": "1",
                # Disable debug sync for speed
                "CUDA_LAUNCH_BLOCKING": "0",
                # Pin memory for faster data transfer
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:256",
            }
            result = runner.run_command(command, timeout=86400, env=env_overrides)  # 24 hours (reduced steps but safety margin)

            # Write completion log
            with open(log_file, "a") as f:
                f.write(f"\nTraining completed: {datetime.datetime.now()}\n")
                f.write(f"Return code: {result.returncode}\n")

            self.logger.info("✅ Training completed successfully!")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Training failed with code: {e.returncode}")
            training_failed = True
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error: {e}")
            training_failed = True
            return False
        finally:
            # Cleanup temporary script unless preservation requested or error occurred
            keep_flag = os.environ.get("KEEP_PATCHED", "0") in ("1", "true", "True")
            keep_on_error = training_failed or (result and result.returncode != 0)

            if script_path.exists():
                if keep_flag or keep_on_error:
                    self.logger.info(f"🧪 Retaining generated script for debugging: {script_path}")
                    if keep_on_error:
                        self.logger.info(f"💡 To inspect the generated script: cat {script_path}")
                else:
                    script_path.unlink()
                    self.logger.info(f"✅ Cleaned up: {script_path}")

    def run_training(self, vkitti_seg_path: str, kitti_images_path: str,
                    output_dir: str, steps: int, **kwargs) -> bool:
        """
        Main training pipeline with comprehensive validation.
        """
        self.logger.info(f"🚀 Starting V-KITTI {self.variant_name} ControlNet Training")

        # 1. Environment validation
        if not self.validate_environment():
            return False

        # 2. Dataset validation
        if not validate_vkitti_dataset(vkitti_seg_path):
            self.logger.error("❌ V-KITTI dataset validation failed")
            return False

        # 3. Model path validation
        if not self.validate_model_paths(**kwargs):
            return False

        # 4. Create output directory
        try:
            output_path = create_output_directory(pathlib.Path(output_dir))
        except Exception as e:
            self.logger.error(f"❌ Output directory creation failed: {e}")
            return False

        # 5. Create dataset JSONL
        jsonl_path = f"vkitti_{self.variant_name}_pairs.jsonl"
        if not os.path.exists(jsonl_path):
            num_pairs = self.create_vkitti_dataset_jsonl(vkitti_seg_path, kitti_images_path, jsonl_path)
            if num_pairs == 0:
                self.logger.error("❌ No training pairs found")
                return False

        # 6. Prepare dataset directory
        if self.variant_name == "baseline":
            # Use KITTI→V-KITTI translation dataset for baseline training.
            # This dataset is created by `create_kitti_to_vkitti_dataset.py` and lives alongside
            # the vkitti training utilities (not in the repo root).
            dataset_dir = pathlib.Path(__file__).resolve().parent / "controlnet_kitti_to_vkitti_dataset"
            if not dataset_dir.exists():
                # Attempt automatic creation using project helper
                try:
                    creator = pathlib.Path(__file__).resolve().parents[1] / "create_kitti_to_vkitti_dataset.py"
                    if creator.exists():
                        self.logger.info("❌ KITTI→V-KITTI dataset not found. Attempting to create it now...")
                        subprocess.run([sys.executable, str(creator)], check=False)
                    else:
                        self.logger.warning(f"Create script not found: {creator}")
                except Exception as e:
                    self.logger.warning(f"Dataset auto-create failed: {e}")
                if not dataset_dir.exists():
                    self.logger.error("❌ KITTI→V-KITTI translation dataset not found after auto-create attempt")
                    return False
            self.logger.info("✅ Using KITTI→V-KITTI translation dataset (KITTI input → V-KITTI output)")
        else:
            dataset_dir = pathlib.Path(f"controlnet_vkitti_{self.variant_name}_dataset")
            dataset_dir.mkdir(exist_ok=True)
            shutil.copy(jsonl_path, dataset_dir / "metadata.jsonl")

        # 7. Create and execute training
        try:
            script_path = self.create_training_script(**kwargs)
            return self.execute_training(script_path, output_path, dataset_dir, steps, **kwargs)
        except Exception as e:
            self.logger.error(f"❌ Training setup failed: {e}")
            return False


def create_robust_argument_parser(variant_name: str, additional_args: Optional[Dict] = None) -> tuple:
    """Create robust argument parser with validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description=f"Train V-KITTI {variant_name} ControlNet model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Standard arguments
    parser.add_argument("--vkitti_seg_path", type=str, required=True,
                       help="Path to V-KITTI class segmentation directory")
    parser.add_argument("--kitti_images_path", type=str, required=True,
                       help="Path to KITTI images directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for trained model")
    parser.add_argument("--steps", type=int, default=45000,
                       help="Number of training steps")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                       default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="Path to pretrained model or model identifier")

    # Add variant-specific arguments
    if additional_args:
        for arg_name, arg_config in additional_args.items():
            parser.add_argument(f"--{arg_name}", **arg_config)

    args = parser.parse_args()

    # Validate required paths
    for path_arg in ['vkitti_seg_path', 'kitti_images_path']:
        path = getattr(args, path_arg)
        if not os.path.exists(path):
            parser.error(f"Path does not exist: {path}")

    return args, parser
