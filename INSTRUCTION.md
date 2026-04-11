# COCO Baseline (SD v1.4) Instructions

This guide describes how to run the updated COCO baseline pipeline with both FID and CLIP cosine (CS) metrics.

## 1. Reference COCO Image Folder Format

Set one folder path named `coco_image_folder`.

Expected structure:

```text
{coco_image_folder}/
  saved_images/
    179765.jpg
    391895.jpg
    ...
```

Notes:
- The filename must be `{image_id}.jpg`.
- Example: for `image_id=179765` in `benchmark/coco_30k.csv`, the path is `{coco_image_folder}/saved_images/179765.jpg`.

## 2. What the Updated Pipeline Computes

For `task=coco`, evaluation now computes:
- `FID`: between generated COCO images and reference images in `coco_image_folder/saved_images`.
- `CLIPCosine`: mean raw image-text cosine similarity between generated image and its corresponding COCO caption.

Also exported:
- `num_expected`, `num_found`, `num_missing` for coverage checks.
- seed range info from CSV (`evaluation_seed`) in output metadata.

## 3. Generation (SD v1.4 baseline)

Use `--gen_only` first to generate COCO images.

```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task coco \
  --task_args coco_image_folder=/kaggle/input/your-coco-folder data_path=benchmark/coco_30k.csv \
  --img_save_path benchmark/generated_imgs/sd14_baseline \
  --save_path benchmark/results/sd14_baseline \
  --base_model CompVis/stable-diffusion-v1-4 \
  --gen_only
```

Generated files are saved to:
- `benchmark/generated_imgs/sd14_baseline/coco30k/COCO_val2014_000000XXXXXX.jpg`

## 4. Evaluation (FID + CLIPCosine)

Run evaluation only after generation:

```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task coco \
  --task_args coco_image_folder=/kaggle/input/your-coco-folder data_path=benchmark/coco_30k.csv clip_model=ViT-B/32 clip_batch_size=128 \
  --img_save_path benchmark/generated_imgs/sd14_baseline \
  --save_path benchmark/results/sd14_baseline \
  --eval_only
```

Output files:
- `benchmark/results/sd14_baseline/coco-fid.json`
- `benchmark/results/sd14_baseline/coco-metrics.json`
- `benchmark/results/sd14_baseline/coco-missing-image-ids.json` (only if missing exists)

## 5. Dry-Run (Recommended)

Before full 30k run, test on a subset.

Create a 200-sample CSV:

```python
import pandas as pd

src = "benchmark/coco_30k.csv"
dst = "benchmark/coco_200.csv"

pd.read_csv(src).head(200).to_csv(dst, index=False)
print("saved", dst)
```

Then run generation + evaluation with:
- `data_path=benchmark/coco_200.csv`

## 6. Reproducibility Notes

- Generation already uses `evaluation_seed` from CSV (one seed per row).
- To keep fair comparison across methods, reuse:
  - the same `data_path` CSV,
  - the same generation config,
  - the same `coco_image_folder` reference set.

## 7. Quick Validation Checklist

- `num_missing == 0` in `coco-metrics.json`.
- Number of generated files in `.../coco30k` matches row count in CSV.
- `FID` and `CLIPCosine` both exist in `coco-metrics.json`.

## 8. Preflight Environment Checklist

Run these checks before launching full 30k on Kaggle:

- [ ] Python can import: `torch`, `accelerate`, `diffusers`, `clip`, `cleanfid`, `prettytable`, `pandas`.
- [ ] GPU is enabled and visible from PyTorch (`torch.cuda.is_available() == True`).
- [ ] `benchmark/coco_30k.csv` exists and has columns: `image_id,prompt,evaluation_seed,width,height`.
- [ ] `coco_image_folder/saved_images` exists.
- [ ] At least one sample id from CSV exists in `saved_images/{id}.jpg`.
- [ ] Enough disk space for generated images + outputs.

## 9. One-Command Script (Preflight + Run)

Use `scripts/kaggle_coco_baseline.py` to run preflight, optional dry-run, generation, and evaluation.

Preflight only:

```bash
python scripts/kaggle_coco_baseline.py \
  --coco-image-folder /kaggle/input/your-coco-folder \
  --preflight-only
```

Dry-run 200 samples:

```bash
python scripts/kaggle_coco_baseline.py \
  --coco-image-folder /kaggle/input/your-coco-folder \
  --dry-run-count 200
```

Full 30k run:

```bash
python scripts/kaggle_coco_baseline.py \
  --coco-image-folder /kaggle/input/your-coco-folder
```

## 10. Concept Baseline Tasks

Two concept modes are available in `evaluate_task.py`.

### General Concept

Use this for concepts like Superman, Snoopy, Batman, or retain concepts.

```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task general_concept \
  --task_args concepts=[superman,snoopy] num_samples=20 seed=42 \
  --img_save_path benchmark/generated_imgs/general_baseline \
  --save_path benchmark/results/general_baseline \
  --base_model CompVis/stable-diffusion-v1-4
```

This uses `ClipTemplateDataset` and `ClipEvaluator` with template prompts from `src/misc/clip_templates.py`.

### Artist Concept

Use this for style concepts like Van Gogh, Picasso, and Caravaggio.

```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task artist_concept \
  --task_args datasets=[vangogh] num_samples=20 default_seed=42 \
  --img_save_path benchmark/generated_imgs/artist_baseline \
  --save_path benchmark/results/artist_baseline \
  --base_model CompVis/stable-diffusion-v1-4
```

This uses `ArtworkDataset` and `ArtworkEvaluator` with prompt CSVs such as `benchmark/vangogh_prompts.csv`.

### Sample Count Convention

- COCO: full `coco_30k.csv`
- General concepts: `num_samples=20` maps to 20 template prompts
- Artist concepts: `num_samples=20` maps to 20 prompt rows from the CSV
