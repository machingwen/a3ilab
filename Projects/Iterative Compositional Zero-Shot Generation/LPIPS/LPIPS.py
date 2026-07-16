# lpips_pairwise_diversity.py

import os
import itertools
import lpips
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
import shutil  # ← 加在檔案開頭
# -------------------- Configuration --------------------
IMAGE_DIRECTORY = "/workspace/DiT/LPIPS/LPIPS/Itr/Gamma"
LPIPS_NET_TYPE = 'vgg'  # 'vgg' for accuracy, 'alex' for speed
USE_GPU = True
RESIZE_IMAGES = False
TARGET_RESIZE_DIM = (128, 128)

# File output
SAVE_DISTANCES_FILE = "gamma_itr.txt"
SUMMARY_FILE = "results_gamma_itr.txt"


# -------------------- Function Definitions --------------------
def load_and_preprocess_image(image_path, device, resize_images_flag, target_dim):
    img = Image.open(image_path).convert('RGB')
    transform_list = []
    if resize_images_flag:
        transform_list.append(T.Resize(target_dim))
    transform_list.append(T.ToTensor())
    transform_list.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    preprocess = T.Compose(transform_list)
    return preprocess(img).unsqueeze(0).to(device)

def calculate_average_pairwise_lpips(image_dir, net_type, use_gpu_flag, 
                                     resize_flag, target_dimensions):
    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found: {image_dir}")
        return None, None, None

    device = torch.device("cuda:2" if use_gpu_flag and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        loss_fn = lpips.LPIPS(net=net_type, verbose=False).to(device)
        loss_fn.eval()
    except Exception as e:
        print(f"Error initializing LPIPS model: {e}")
        return None, None, None

    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]

    if len(image_files) < 2:
        print("Need at least two images to calculate pairwise LPIPS.")
        return None, None, None

    print(f"Found {len(image_files)} images in {image_dir}.")
    images_tensor_list = []
    print("Preprocessing images...")
    for img_path in tqdm(image_files, desc="Loading & Preprocessing Images"):
        try:
            images_tensor_list.append(
                load_and_preprocess_image(img_path, device, resize_flag, target_dimensions)
            )
        except Exception as e:
            print(f"\nWarning: Could not load or preprocess image {img_path}: {e}")
            print("Skipping this image.")

    if len(images_tensor_list) < 2:
        print("Need at least two valid images to calculate pairwise LPIPS after loading attempts.")
        return None, None, None

    print(f"Successfully preprocessed {len(images_tensor_list)} images.")
    all_lpips_distances = []
    num_images = len(images_tensor_list)
    total_pairs = num_images * (num_images - 1) // 2

    print(f"Calculating LPIPS for {total_pairs} unique pairs...")
    pair_indices_iterator = itertools.combinations(range(num_images), 2)

    with torch.no_grad():
        for i, j in tqdm(pair_indices_iterator, total=total_pairs, desc="Calculating LPIPS Distances"):
            img1_tensor = images_tensor_list[i]
            img2_tensor = images_tensor_list[j]
            try:
                distance = loss_fn(img1_tensor, img2_tensor).item()
                all_lpips_distances.append(distance)
            except Exception as e:
                print(f"\nWarning: Could not calculate LPIPS for pair "
                      f"({os.path.basename(image_files[i])}, {os.path.basename(image_files[j])}): {e}")
                print("Skipping this pair.")

    if not all_lpips_distances:
        print("No LPIPS distances were successfully calculated.")
        return None, None, None

    average_lpips = np.mean(all_lpips_distances)
    std_lpips = np.std(all_lpips_distances)
    min_lpips = np.min(all_lpips_distances)
    max_lpips = np.max(all_lpips_distances)

    # --- 結果輸出區塊 ---
    summary_lines = [
        "\n--- LPIPS Diversity Results ---",
        f"Successfully processed {len(all_lpips_distances)} unique pairs.",
        f"Average Pairwise LPIPS: {average_lpips:.4f}",
        f"Standard Deviation of LPIPS: {std_lpips:.4f}",
        f"Min LPIPS: {min_lpips:.4f}",
        f"Max LPIPS: {max_lpips:.4f}"
    ]
    for line in summary_lines:
        print(line)

    try:
        with open(SUMMARY_FILE, "w") as f:
            for line in summary_lines:
                f.write(line + "\n")
        print(f"LPIPS summary saved to: {SUMMARY_FILE}")
    except Exception as e:
        print(f"Error saving summary results to file: {e}")
    
     # ------------------- 新增區塊：計算每張圖的 LPIPS 多樣性 -------------------
    print("Calculating per-image LPIPS diversity...")
    per_image_lpips = np.zeros(num_images)
    counts = np.zeros(num_images)
    dist_idx = 0
    for i, j in itertools.combinations(range(num_images), 2):
        d = all_lpips_distances[dist_idx]
        per_image_lpips[i] += d
        per_image_lpips[j] += d
        counts[i] += 1
        counts[j] += 1
        dist_idx += 1
    per_image_lpips = per_image_lpips / counts

    # 儲存每張圖的 LPIPS 分數
    per_image_lpips_file = os.path.join(image_dir, "per_image_lpips.txt")
    with open(per_image_lpips_file, "w") as f:
        for path, score in zip(image_files, per_image_lpips):
            f.write(f"{os.path.basename(path)}\t{score:.4f}\n")
    print(f"Per-image LPIPS scores saved to: {per_image_lpips_file}")

    # 篩選高多樣性圖片（LPIPS 高於 25% 百分位）
    threshold = np.percentile(per_image_lpips, 25)
    print(f"Filtering and copying images with LPIPS >= {threshold:.4f}")

    filtered_dir = os.path.join(image_dir, "filtered_high_diversity")
    os.makedirs(filtered_dir, exist_ok=True)

    high_diversity_images = []
    for idx, lpips_val in enumerate(per_image_lpips):
        if lpips_val >= threshold:
            src_path = image_files[idx]
            dst_path = os.path.join(filtered_dir, os.path.basename(src_path))
            try:
                shutil.copy2(src_path, dst_path)
                high_diversity_images.append(dst_path)
            except Exception as e:
                print(f"⚠️ Failed to copy {src_path} → {dst_path}: {e}")

    print(f"✅ Copied {len(high_diversity_images)} high-diversity images to: {filtered_dir}")

    with open(os.path.join(image_dir, "high_diversity_images.txt"), "w") as f:
        for path in high_diversity_images:
            f.write(path + "\n")
    print("High-diversity image list saved.")

    return average_lpips, std_lpips, all_lpips_distances

# -------------------- Execution Block --------------------
if __name__ == "__main__":
    if IMAGE_DIRECTORY == "path/to/your/image_dataset" or not os.path.isdir(IMAGE_DIRECTORY):
        print("❌ ERROR: IMAGE_DIRECTORY is not correctly set or does not exist.")
    else:
        print("Starting LPIPS diversity calculation...")
        start_time = time.time()

        avg_lpips, std_dev_lpips, all_distances = calculate_average_pairwise_lpips(
            image_dir=IMAGE_DIRECTORY,
            net_type=LPIPS_NET_TYPE,
            use_gpu_flag=USE_GPU,
            resize_flag=RESIZE_IMAGES,
            target_dimensions=TARGET_RESIZE_DIM
        )

        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds.")

        if avg_lpips is not None and SAVE_DISTANCES_FILE and all_distances:
            try:
                with open(SAVE_DISTANCES_FILE, "w") as f:
                    for dist in all_distances:
                        f.write(f"{dist}\n")
                print(f"All pairwise distances saved to: {SAVE_DISTANCES_FILE}")
            except Exception as e:
                print(f"Error saving distances to file: {e}")
        
        # ------------------- 再次對 filtered_high_diversity 資料夾做 LPIPS -------------------
        filtered_dir = os.path.join(IMAGE_DIRECTORY, "filtered_high_diversity")
        if os.path.exists(filtered_dir) and os.path.isdir(filtered_dir):
            print("\n🔍 Calculating LPIPS for filtered high-diversity images...")
            filtered_avg, filtered_std, _ = calculate_average_pairwise_lpips(
                image_dir=filtered_dir,
                net_type=LPIPS_NET_TYPE,
                use_gpu_flag=USE_GPU,
                resize_flag=RESIZE_IMAGES,
                target_dimensions=TARGET_RESIZE_DIM
            )
            print(f"📊 Filtered images - Avg LPIPS: {filtered_avg:.4f} | Std: {filtered_std:.4f}")
        else:
            print(f"⚠️ filtered_high_diversity 資料夾不存在：{filtered_dir}")
