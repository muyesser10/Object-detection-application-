import os
import shutil
import argparse
from collections import defaultdict
from pycocotools.coco import COCO
import random

def select_n_samples_per_class(coco, img_ids, n=1000, labels=None):
    class_counts = defaultdict(int)
    selected_img_ids = []

    # Get category ids for the provided labels
    label_ids = []
    for label in labels:
        category_id = coco.getCatIds(catNms=[label])
        if category_id:
            label_ids.append(category_id[0])  # Assume single category per label

    # Filter selected images for each class
    class_img_map = defaultdict(list)
    used_image_ids = set()

    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        classes_in_img = set(ann['category_id'] for ann in anns)

        # Only consider images that contain the specified labels
        if classes_in_img & set(label_ids):
            for category_id in classes_in_img:
                if category_id in label_ids:
                    class_img_map[category_id].append(img_id)

    # Ensure we have exactly n samples per class with unique images
    final_selected_img_ids = []
    for class_id, img_ids in class_img_map.items():
        # Remove images already selected for other classes
        unique_imgs = [img_id for img_id in img_ids if img_id not in used_image_ids]

        # Select exactly n samples if enough images are available
        if len(unique_imgs) >= n:
            selected_class_imgs = random.sample(unique_imgs, n)
        else:
            selected_class_imgs = unique_imgs  # Use all available images if fewer than n

        # Update used_image_ids to avoid duplicates
        used_image_ids.update(selected_class_imgs)
        final_selected_img_ids.extend(selected_class_imgs)

        # Print the count of selected images for each class
        category_name = coco.loadCats(class_id)[0]['name']
        print(f"Sınıf '{category_name}' ({class_id}) için {len(selected_class_imgs)} benzersiz görsel seçildi.")

    return final_selected_img_ids

def main(coco_path, output_path, labels, n_samples):
    annFile = os.path.join(coco_path, 'annotations', 'instances_train2017.json')
    coco = COCO(annFile)

    # Get all image ids
    img_ids = coco.getImgIds()

    # Select n samples per class
    selected_img_ids = select_n_samples_per_class(coco, img_ids, n=n_samples, labels=labels)

    # Load selected images
    images = coco.loadImgs(selected_img_ids)

    # Create output directories
    os.makedirs(os.path.join(output_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'train'), exist_ok=True)

    # Copy images and create label files
    for img in images:
        # Copy image
        src = os.path.join(coco_path, 'train2017', img['file_name'])
        dst = os.path.join(output_path, 'images', 'train', img['file_name'])
        shutil.copy(src, dst)

        # Create label file
        ann_ids = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(ann_ids)
        label_file = os.path.join(output_path, 'labels', 'train', img['file_name'].replace('.jpg', '.txt'))
        with open(label_file, 'w') as f:
            for ann in anns:
                category_id = ann['category_id']
                bbox = ann['bbox']
                # Convert bbox to YOLO format
                x_center = (bbox[0] + bbox[2] / 2) / img['width']
                y_center = (bbox[1] + bbox[3] / 2) / img['height']
                width = bbox[2] / img['width']
                height = bbox[3] / img['height']
                f.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

    # Print the total number of selected images
    print(f"Toplamda {len(selected_img_ids)} görsel seçildi.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create YOLO dataset from COCO dataset")
    parser.add_argument("--coco_path", type=str, required=True, help="Path to COCO dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output YOLO dataset")
    parser.add_argument("--labels", type=str, required=True, help="Comma separated list of labels")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples per class")
    args = parser.parse_args()

    main(args.coco_path, args.output_path, args.labels.split(','), args.n_samples)
