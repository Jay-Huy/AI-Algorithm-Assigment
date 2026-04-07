import json
import os
from argparse import ArgumentParser

import pandas as pd
from prettytable import PrettyTable
from cleanfid import fid

from src.configs.generation_config import GenerationConfig
from .eval_util import clip_score

from .evaluator import Evaluator, GenerationDataset


class Coco30kGenerationDataset(GenerationDataset):
    """
    Dataset for COCO-30k Caption dataset.
    """
    def __init__(
        self,
        save_folder: str = "benchmark/generated_imgs/",
        base_cfg: GenerationConfig = GenerationConfig(),
        data_path: str = "benchmark/coco_30k.csv",
        **kwargs
    ) -> None:
        df = pd.read_csv(data_path)
        self.data = []
        for idx, row in df.iterrows():
            cfg = base_cfg.copy()
            cfg.prompts = [row["prompt"]]
            cfg.negative_prompt = ""
            # fix width & height to be divisible by 8
            cfg.width = row["width"] - row["width"] % 8
            cfg.height = row["height"] - row["height"] % 8
            cfg.seed = int(row["evaluation_seed"])
            cfg.generate_num = 1
            cfg.save_path = os.path.join(
                save_folder,
                "coco30k",
                "COCO_val2014_" + "%012d" % row["image_id"] + ".jpg",
            )
            self.data.append(cfg.dict())


class CocoEvaluator(Evaluator):
    """
    Evaluator on COCO-30k Caption dataset.
    """
    def __init__(
        self,
        save_folder: str = "benchmark/generated_imgs/",
        output_path: str = "benchmark/results/",
        data_path: str = "benchmark/coco_30k.csv",
        coco_image_folder: str = "",
        clip_model: str = "ViT-B/32",
        clip_batch_size: int = 128,
    ):
        super().__init__(save_folder=save_folder, output_path=output_path)

        self.data_path = data_path
        self.coco_image_folder = coco_image_folder
        self.clip_model = clip_model
        self.clip_batch_size = int(clip_batch_size)

    def _get_reference_folder(self) -> str:
        if not self.coco_image_folder:
            raise ValueError("coco_image_folder must be provided for COCO FID evaluation.")

        saved_images_path = os.path.join(self.coco_image_folder, "saved_images")
        if os.path.isdir(saved_images_path):
            return saved_images_path
        if os.path.isdir(self.coco_image_folder):
            return self.coco_image_folder
        raise FileNotFoundError(f"COCO image folder not found: {self.coco_image_folder}")

    def _collect_generated_pairs(self, generated_folder: str):
        df = pd.read_csv(self.data_path)
        image_paths = []
        prompts = []
        seeds = []
        missing_ids = []

        for _, row in df.iterrows():
            image_id = int(row["image_id"])
            image_name = f"COCO_val2014_{image_id:012d}.jpg"
            image_path = os.path.join(generated_folder, image_name)
            if os.path.exists(image_path):
                image_paths.append(image_path)
                prompts.append(str(row["prompt"]))
                seeds.append(int(row["evaluation_seed"]))
            else:
                missing_ids.append(image_id)

        return image_paths, prompts, seeds, missing_ids

    def _compute_clip_cosine(self, image_paths, prompts):
        if len(image_paths) == 0:
            return 0.0

        total_score = 0.0
        total_count = 0
        batch_size = max(1, self.clip_batch_size)

        for start in range(0, len(image_paths), batch_size):
            end = min(start + batch_size, len(image_paths))
            batch_scores = clip_score(
                image_paths[start:end],
                prompts[start:end],
                clip_model=self.clip_model,
                use_weight=False,
                clamp_min_zero=False,
            )
            total_score += float(batch_scores.sum())
            total_count += len(batch_scores)

        return total_score / total_count

    def evaluation(self):
        print("Evaluating on COCO-30k Caption dataset...")
        generated_folder = os.path.join(self.save_folder, "coco30k")
        if not os.path.isdir(generated_folder):
            raise FileNotFoundError(f"Generated COCO folder not found: {generated_folder}")

        reference_folder = self._get_reference_folder()
        fid_value = fid.compute_fid(generated_folder, reference_folder)

        image_paths, prompts, seeds, missing_ids = self._collect_generated_pairs(generated_folder)
        clip_cosine = self._compute_clip_cosine(image_paths, prompts)

        # metrics = torch_fidelity.calculate_metrics(
        #     input1=os.path.join(self.save_folder, "coco30k"),
        #     input2=self.data_path,
        #     cuda=True,
        #     fid=True,
        #     samples_find_deep=True)

        pt = PrettyTable()
        pt.field_names = ["Metric", "Value"]
        pt.add_row(["FID", fid_value])
        pt.add_row(["CLIPCosine", clip_cosine])
        pt.add_row(["NumExpected", len(image_paths) + len(missing_ids)])
        pt.add_row(["NumFound", len(image_paths)])
        pt.add_row(["NumMissing", len(missing_ids)])
        print(pt)

        metrics = {
            "FID": float(fid_value),
            "CLIPCosine": float(clip_cosine),
            "num_expected": len(image_paths) + len(missing_ids),
            "num_found": len(image_paths),
            "num_missing": len(missing_ids),
            "clip_model": self.clip_model,
            "clip_batch_size": self.clip_batch_size,
            "seed_column": "evaluation_seed",
            "seed_min": int(min(seeds)) if len(seeds) > 0 else None,
            "seed_max": int(max(seeds)) if len(seeds) > 0 else None,
        }

        with open(os.path.join(self.output_path, "coco-fid.json"), "w") as f:
            json.dump({"FID": float(fid_value)}, f)
        with open(os.path.join(self.output_path, "coco-metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        if len(missing_ids) > 0:
            with open(os.path.join(self.output_path, "coco-missing-image-ids.json"), "w") as f:
                json.dump(missing_ids, f)
