import csv
import requests
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
import os
import json
import argparse

def build_lookup(images, annotations):
    images_dict = {img['id']: img for img in images}
    annotations_dict = {ann['image_id']: ann for ann in annotations}
    return images_dict, annotations_dict

def find_image_caption(id, images_dict, annotations_dict):
    img_info = images_dict.get(id)
    ann_info = annotations_dict.get(id)
    if img_info and ann_info:
        try:
            image_url = 'http://images.cocodataset.org/val2014/' + img_info['file_name']
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            return img, ann_info['caption'], img_info['file_name']
        except Exception as e:
            print(f'Image ID: {id} | URL: {image_url} | Error: {e}')
            return None, None, None
    else:
        print(f'Image ID: {id} not found')
        return None, None, None

def process_csv(csv_file, images, annotations, output_dir="coco/saved_images", json_file="coco/data_infor.json", flag = None):
    images_dict, annotations_dict = build_lookup(images, annotations)
    results = []
    ids = []

    os.makedirs(output_dir, exist_ok=True)

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row['image_id']))

    mapping = {}

    for i, id in enumerate(tqdm(ids)):
      if flag != None:
          if i == flag:
            break
      img, caption, file_name = find_image_caption(id, images_dict, annotations_dict)
      if img is not None and caption is not None:
          image_path = output_dir + "/" + f"{id}.jpg"
          img.save(image_path)
          results.append((img, caption))
          mapping[id] = {
              "path": image_path,
              "caption": caption
          }

    if len(results) == len(ids):
        print("Tất cả id trong file CSV đều tìm được ảnh và caption.")
    else:
        print(f"Chỉ tìm được {len(results)} trên tổng {len(ids)} id.")

    # lưu mapping ra file JSON
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(mapping, jf, ensure_ascii=False, indent=2)

    print(f"Đã lưu thông tin vào {json_file}, ảnh nằm trong thư mục {output_dir}")
    return results

# Khai báo argument parser
parser = argparse.ArgumentParser(description="Process COCO captions and images")
parser.add_argument("--caption_val_path", 
                    type=str, 
                    default ='captions_val2014.json',
                    help="Path tới file caption JSON (ví dụ: captions_val2014.json)")

parser.add_argument("--csv", 
                    type=str,
                    default="benchmark/coco_30k.csv", 
                    help="Path tới file CSV chứa image_id")

parser.add_argument("--flag", 
                    type=int, 
                    default=None,
                    help="Số lượng ảnh muốn xử lý (tùy chọn)")

args = parser.parse_args()

# Load dữ liệu từ file caption JSON
with open(args.caption_val_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Gọi hàm process_csv
results = process_csv(
    csv_file=args.csv,
    images=data['images'],
    annotations=data['annotations'],
    flag=args.flag
)
