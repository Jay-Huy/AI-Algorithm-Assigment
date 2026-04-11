import os
import argparse
from io import BytesIO

import requests
from PIL import Image
from tqdm.auto import tqdm


def parse_image_id(url: str) -> int:
    # URL mẫu: http://images.cocodataset.org/val2014/COCO_val2014_000000391895.jpg
    file_name = url.strip().split("/")[-1]
    return int(file_name.replace("COCO_val2014_", "").replace(".jpg", ""))


def load_urls(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url_file", type=str, default="missing_urls.txt")
    parser.add_argument("--save_dir", type=str, default="coco/saved_images")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    urls = load_urls(args.url_file)
    failed = []

    for url in tqdm(urls, desc="Downloading"):
        try:
            image_id = parse_image_id(url)
            img_path = os.path.join(args.save_dir, f"{image_id}.jpg")

            if os.path.exists(img_path):
                continue

            response = requests.get(url, timeout=args.timeout)
            img = Image.open(BytesIO(response.content))
            img.save(img_path)

        except Exception as e:
            failed.append((url, str(e)))

    print(f"Done. Total URLs: {len(urls)}")
    print(f"Failed: {len(failed)}")

    if failed:
        with open("failed_urls.txt", "w", encoding="utf-8") as f:
            for url, err in failed:
                f.write(f"{url}\t{err}\n")
        print("Da luu loi vao failed_urls.txt")


if __name__ == "__main__":
    main()
