# UCE Continual Commands

## 1. Train UCE Stages (V1 -> V2 -> V3)

### V1: Erase Superman
```bash
python UCE-file/uce_sd_erase.py \
  --model_id CompVis/stable-diffusion-v1-4 \
  --edit_concepts "superman" \
  --guide_concepts "" \
  --concept_type object \
  --device cuda:0 \
  --erase_scale 1 --preserve_scale 1 --lamb 0.5 \
  --uce_save_path checkpoints/uce/v1_superman.safetensors
```

### V2: Load V1, erase Van Gogh
```bash
python UCE-file/uce_sd_erase.py \
  --model_id CompVis/stable-diffusion-v1-4 \
  --previous_uce_path checkpoints/uce/v1_superman.safetensors \
  --edit_concepts "van gogh" \
  --guide_concepts "art" \
  --concept_type art \
  --device cuda:0 \
  --erase_scale 1 --preserve_scale 1 --lamb 0.5 \
  --uce_save_path checkpoints/uce/v2_superman_vangogh.safetensors
```

### V3: Load V2, erase Snoopy
```bash
python UCE-file/uce_sd_erase.py \
  --model_id CompVis/stable-diffusion-v1-4 \
  --previous_uce_path checkpoints/uce/v2_superman_vangogh.safetensors \
  --edit_concepts "snoopy" \
  --guide_concepts "" \
  --concept_type object \
  --device cuda:0 \
  --erase_scale 1 --preserve_scale 1 --lamb 0.5 \
  --uce_save_path checkpoints/uce/v3_superman_vangogh_snoopy.safetensors
```

## 2. Evaluate Each Stage (Optional for V1/V2, required for final V3)

### V1 eval
```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task general_concept \
  --task_args concepts=[superman] num_samples=20 num_images_per_template=1 seed=42 \
  --img_save_path benchmark/generated_imgs/uce_v1_superman \
  --save_path benchmark/results/uce_v1_superman \
  --base_model CompVis/stable-diffusion-v1-4 \
  --uce_path checkpoints/uce/v1_superman.safetensors
```

### V2 eval
```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task artist_concept \
  --task_args datasets=[vangogh] num_samples=20 num_images_per_prompt=1 default_seed=42 reference_folder=benchmark/generated_imgs/sd14_baseline_artist \
  --img_save_path benchmark/generated_imgs/uce_v2_vangogh \
  --save_path benchmark/results/uce_v2_vangogh \
  --base_model CompVis/stable-diffusion-v1-4 \
  --uce_path checkpoints/uce/v2_superman_vangogh.safetensors
```

### V3 final eval (example: both concept + utility)
```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task general_concept \
  --task_args concepts=[superman,snoopy] num_samples=20 num_images_per_template=1 seed=42 reference_folder=benchmark/generated_imgs/sd14_baseline_general \
  --img_save_path benchmark/generated_imgs/uce_v3_general \
  --save_path benchmark/results/uce_v3_general \
  --base_model CompVis/stable-diffusion-v1-4 \
  --uce_path checkpoints/uce/v3_superman_vangogh_snoopy.safetensors
```

```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task artist_concept \
  --task_args datasets=[vangogh] num_samples=20 num_images_per_prompt=1 default_seed=42 reference_folder=benchmark/generated_imgs/sd14_baseline_artist \
  --img_save_path benchmark/generated_imgs/uce_v3_artist \
  --save_path benchmark/results/uce_v3_artist \
  --base_model CompVis/stable-diffusion-v1-4 \
  --uce_path checkpoints/uce/v3_superman_vangogh_snoopy.safetensors
```

```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task coco \
  --task_args coco_image_folder=/kaggle/input/your-coco-folder data_path=benchmark/coco_30k.csv clip_model=ViT-B/32 clip_batch_size=16 \
  --img_save_path benchmark/generated_imgs/uce_v3_coco \
  --save_path benchmark/results/uce_v3_coco \
  --base_model CompVis/stable-diffusion-v1-4 \
  --uce_path checkpoints/uce/v3_superman_vangogh_snoopy.safetensors
```
