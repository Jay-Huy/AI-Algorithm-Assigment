# Run Commands (One Round)

## 1) COCO: SD v1.4 vs COCO (CLIPCosine + FID)
```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task coco \
  --task_args coco_image_folder=/kaggle/input/your-coco-folder data_path=benchmark/coco_30k.csv clip_model=ViT-B/32 clip_batch_size=16 \
  --img_save_path benchmark/generated_imgs/sd14_coco \
  --save_path benchmark/results/sd14_coco \
  --base_model CompVis/stable-diffusion-v1-4
```

## 2) Superheroes: self-reference FID + cosine
```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task general_concept \
  --task_args concepts=[superman,batman,thor,wonder+woman,shazam] num_samples=20 num_images_per_template=1 seed=42 reference_folder=benchmark/generated_imgs/sd14_superheroes \
  --img_save_path benchmark/generated_imgs/sd14_superheroes \
  --save_path benchmark/results/sd14_superheroes \
  --base_model CompVis/stable-diffusion-v1-4
```

## 3) Artists: self-reference FID + cosine
```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task general_concept \
  --task_args concepts=[van+gogh,picasso,monet,paul+gauguin,caravaggio] num_samples=20 num_images_per_template=1 seed=42 reference_folder=benchmark/generated_imgs/sd14_artists \
  --img_save_path benchmark/generated_imgs/sd14_artists \
  --save_path benchmark/results/sd14_artists \
  --base_model CompVis/stable-diffusion-v1-4
```

## 4) Cartoon characters: self-reference FID + cosine
```bash
accelerate launch --num_processes 1 evaluate_task.py \
  --task general_concept \
  --task_args concepts=[snoopy,mickey,spongebob,pikachu,hello+kitty] num_samples=20 num_images_per_template=1 seed=42 reference_folder=benchmark/generated_imgs/sd14_cartoons \
  --img_save_path benchmark/generated_imgs/sd14_cartoons \
  --save_path benchmark/results/sd14_cartoons \
  --base_model CompVis/stable-diffusion-v1-4
```

Notes:
- No mode flag means default behavior: generate first, then evaluate.
- Use `+` for concept names with spaces.
