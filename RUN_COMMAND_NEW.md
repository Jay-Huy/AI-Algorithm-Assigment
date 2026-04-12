# RUN COMMANDS FOR NEW METHOD (Localized SPEED)

## 1) Train continual stages

### V1: erase superman
```bash
python UCE-file/handle-problem.py \
  --model_id CompVis/stable-diffusion-v1-4 \
  --edit_concepts "superman" \
  --preserve_concepts "" \
  --k_layers 5 \
  --lamb 1e-6 \
  --save_dir checkpoints/new \
  --exp_name v1_superman
```

### V2: load V1 then erase van gogh
```bash
python UCE-file/handle-problem.py \
  --model_id CompVis/stable-diffusion-v1-4 \
  --previous_ours_path checkpoints/new/v1_superman.safetensors \
  --edit_concepts "van gogh" \
  --preserve_concepts "" \
  --k_layers 5 \
  --lamb 1e-6 \
  --save_dir checkpoints/new \
  --exp_name v2_superman_vangogh
```

### V3: load V2 then erase snoopy
```bash
python UCE-file/handle-problem.py \
  --model_id CompVis/stable-diffusion-v1-4 \
  --previous_ours_path checkpoints/new/v2_superman_vangogh.safetensors \
  --edit_concepts "snoopy" \
  --preserve_concepts "" \
  --k_layers 5 \
  --lamb 1e-6 \
  --save_dir checkpoints/new \
  --exp_name v3_superman_vangogh_snoopy
```

## 2) Evaluate checkpoints with evaluate_task.py

### V1 concept eval
```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task general_concept \
  --task_args concepts=[superman] num_samples=20 num_images_per_template=1 seed=42 reference_folder=benchmark/generated_imgs/sd14_superman \
  --img_save_path benchmark/generated_imgs/new_v1_superman \
  --save_path benchmark/results/new_v1_superman \
  --base_model CompVis/stable-diffusion-v1-4 \
  --ours_path checkpoints/new/v1_superman.safetensors
```

### V2 artist eval
```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task artist_concept \
  --task_args datasets=[vangogh] num_samples=20 num_images_per_prompt=1 default_seed=42 reference_folder=benchmark/generated_imgs/sd14_artists \
  --img_save_path benchmark/generated_imgs/new_v2_vangogh \
  --save_path benchmark/results/new_v2_vangogh \
  --base_model CompVis/stable-diffusion-v1-4 \
  --ours_path checkpoints/new/v2_superman_vangogh.safetensors
```

### V3 final eval: general concepts
```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task general_concept \
  --task_args concepts=[superman,snoopy] num_samples=20 num_images_per_template=1 seed=42 reference_folder=benchmark/generated_imgs/sd14_general \
  --img_save_path benchmark/generated_imgs/new_v3_general \
  --save_path benchmark/results/new_v3_general \
  --base_model CompVis/stable-diffusion-v1-4 \
  --ours_path checkpoints/new/v3_superman_vangogh_snoopy.safetensors
```

### V3 utility eval: coco
```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task coco \
  --task_args coco_image_folder=/kaggle/input/your-coco-folder data_path=benchmark/coco_30k.csv clip_model=ViT-B/32 clip_batch_size=16 \
  --img_save_path benchmark/generated_imgs/new_v3_coco \
  --save_path benchmark/results/new_v3_coco \
  --base_model CompVis/stable-diffusion-v1-4 \
  --ours_path checkpoints/new/v3_superman_vangogh_snoopy.safetensors
```

## Notes
- Do not pass both --uce_path and --ours_path in one run.
- Use + for spaces in task_args lists when needed (example: van+gogh).
- If you already generated images, use --eval_only to skip generation.
