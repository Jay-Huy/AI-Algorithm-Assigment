# Run Commands (Minimal)

## 1. COCO

### Full 30k
```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task coco \
  --task_args coco_image_folder=/kaggle/input/your-coco-folder data_path=benchmark/coco_30k.csv clip_model=ViT-B/32 clip_batch_size=16 \
  --img_save_path benchmark/generated_imgs/sd14_baseline \
  --save_path benchmark/results/sd14_baseline \
  --base_model CompVis/stable-diffusion-v1-4
```

### Dry-run (test only)
- Use a smaller CSV first, for example `benchmark/coco_200.csv`.
```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task coco \
  --task_args coco_image_folder=/kaggle/input/your-coco-folder data_path=benchmark/coco_200.csv clip_model=ViT-B/32 clip_batch_size=16 \
  --img_save_path benchmark/generated_imgs/sd14_baseline_dry \
  --save_path benchmark/results/sd14_baseline_dry \
  --base_model CompVis/stable-diffusion-v1-4
```

## 2. One General Concept

Default concept run uses `num_samples=20`.

### Normal run (default 20 samples)
```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task general_concept \
  --task_args concepts=[superman] num_samples=20 num_images_per_template=1 seed=42 \
  --img_save_path benchmark/generated_imgs/general_superman \
  --save_path benchmark/results/general_superman \
  --base_model CompVis/stable-diffusion-v1-4
```

### Dry-run
```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task general_concept \
  --task_args concepts=[superman] num_samples=5 num_images_per_template=1 seed=42 \
  --img_save_path benchmark/generated_imgs/general_superman_dry \
  --save_path benchmark/results/general_superman_dry \
  --base_model CompVis/stable-diffusion-v1-4
```

## 3. One Artist Concept (from CSV)

Default concept run uses `num_samples=20` from the artist CSV.

### Normal run (default 20 samples)
```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task artist_concept \
  --task_args datasets=[vangogh] num_samples=20 num_images_per_prompt=1 default_seed=42 \
  --img_save_path benchmark/generated_imgs/artist_vangogh \
  --save_path benchmark/results/artist_vangogh \
  --base_model CompVis/stable-diffusion-v1-4
```

### Dry-run
```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task artist_concept \
  --task_args datasets=[vangogh] num_samples=5 num_images_per_prompt=1 default_seed=42 \
  --img_save_path benchmark/generated_imgs/artist_vangogh_dry \
  --save_path benchmark/results/artist_vangogh_dry \
  --base_model CompVis/stable-diffusion-v1-4
```
