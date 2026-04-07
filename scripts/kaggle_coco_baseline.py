import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["image_id", "prompt", "evaluation_seed", "width", "height"]
REQUIRED_MODULES = [
    "torch",
    "accelerate",
    "diffusers",
    "clip",
    "cleanfid",
    "prettytable",
    "pandas",
]


def check_modules():
    missing = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    return missing


def check_gpu():
    try:
        import torch

        return torch.cuda.is_available(), torch.cuda.device_count()
    except Exception:
        return False, 0


def validate_csv(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing CSV columns: {missing_cols}")
    return df


def resolve_reference_folder(coco_image_folder: Path):
    saved_images = coco_image_folder / "saved_images"
    if saved_images.is_dir():
        return saved_images
    if coco_image_folder.is_dir():
        return coco_image_folder
    raise FileNotFoundError(f"COCO image folder not found: {coco_image_folder}")


def validate_reference_images(df: pd.DataFrame, ref_folder: Path):
    sample_id = int(df.iloc[0]["image_id"])
    expected_path = ref_folder / f"{sample_id}.jpg"
    return expected_path.exists(), expected_path


def preflight(args):
    print("=== Preflight checks ===")
    missing_modules = check_modules()
    if missing_modules:
        print("[FAIL] Missing modules:", ", ".join(missing_modules))
        return False
    print("[OK] Python dependencies are available")

    has_gpu, gpu_count = check_gpu()
    if not has_gpu:
        print("[FAIL] CUDA GPU is not available")
        return False
    print(f"[OK] CUDA available with {gpu_count} GPU(s)")

    csv_path = Path(args.data_path)
    df = validate_csv(csv_path)
    print(f"[OK] CSV is valid: {csv_path} ({len(df)} rows)")

    ref_folder = resolve_reference_folder(Path(args.coco_image_folder))
    print(f"[OK] Reference folder found: {ref_folder}")

    found_sample, expected_path = validate_reference_images(df, ref_folder)
    if not found_sample:
        print(f"[FAIL] Sample reference image not found: {expected_path}")
        return False
    print(f"[OK] Sample reference image found: {expected_path}")

    print("=== Preflight passed ===")
    return True


def build_dry_run_csv(df: pd.DataFrame, output_csv: Path, dry_run_count: int):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.head(dry_run_count).to_csv(output_csv, index=False)
    return output_csv


def run_cmd(cmd):
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run COCO SD1.4 baseline on Kaggle")
    parser.add_argument("--coco-image-folder", required=True, help="Folder containing saved_images/{id}.jpg")
    parser.add_argument("--data-path", default="benchmark/coco_30k.csv", help="CSV path for COCO prompts")
    parser.add_argument("--img-save-path", default="benchmark/generated_imgs/sd14_baseline")
    parser.add_argument("--save-path", default="benchmark/results/sd14_baseline")
    parser.add_argument("--base-model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--clip-batch-size", type=int, default=128)
    parser.add_argument("--dry-run-count", type=int, default=0, help="Use first N samples if > 0")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    ok = preflight(args)
    if not ok:
        sys.exit(1)

    if args.preflight_only:
        return

    data_path = Path(args.data_path)
    if args.dry_run_count > 0:
        df = validate_csv(data_path)
        dry_csv = Path("benchmark") / f"coco_{args.dry_run_count}.csv"
        build_dry_run_csv(df, dry_csv, args.dry_run_count)
        data_path = dry_csv
        print(f"Using dry-run CSV: {data_path}")

    task_args = [
        f"coco_image_folder={args.coco_image_folder}",
        f"data_path={data_path}",
        f"clip_model={args.clip_model}",
        f"clip_batch_size={args.clip_batch_size}",
    ]

    gen_cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        str(args.num_processes),
        "evaluate_task.py",
        "--task",
        "coco",
        "--task_args",
        *task_args,
        "--img_save_path",
        args.img_save_path,
        "--save_path",
        args.save_path,
        "--base_model",
        args.base_model,
        "--gen_only",
    ]

    eval_cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        str(args.num_processes),
        "evaluate_task.py",
        "--task",
        "coco",
        "--task_args",
        *task_args,
        "--img_save_path",
        args.img_save_path,
        "--save_path",
        args.save_path,
        "--eval_only",
    ]

    run_cmd(gen_cmd)
    run_cmd(eval_cmd)

    print("\nDone. Check outputs in:")
    print(f"- {args.save_path}/coco-fid.json")
    print(f"- {args.save_path}/coco-metrics.json")


if __name__ == "__main__":
    main()
