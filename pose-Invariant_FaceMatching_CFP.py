"""
CFP → DECA → ArcFace similarity full pipeline (Merged Final)

Steps:
1) Randomly select N persons from CFP dataset (profile + frontal).
2) Copy 1 profile + 1 frontal to input folder.
3) Run DECA inference automatically on copied images (via run_deca_inference).
4) Read DECA output folders and extract UV + frontalized PNGs.
5) Compute ArcFace embeddings and:
   - per-person CSV + visualization
   - pairwise comparisons and CSV + pair visualizations
   - contact sheet generator
"""

import os
import sys
import argparse
import random
import shutil
import csv
import itertools
from glob import glob
import subprocess
import math

import cv2
import numpy as np
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# --------------------------------------------------------
#               CFP RANDOM PAIR EXTRACTOR
# --------------------------------------------------------
def extract_random_cfp_pairs(dataset_path, output_path, num_persons=5):
    """
    Select N random persons from CFP dataset and copy:
       - 1 profile image
       - 1 frontal image
    """
    os.makedirs(output_path, exist_ok=True)

    persons = sorted([
        p for p in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, p))
    ])

    if len(persons) == 0:
        raise RuntimeError(f"No persons found in {dataset_path}")

    num_persons = min(num_persons, len(persons))
    selected = random.sample(persons, num_persons)

    copied = []

    for person in selected:
        profile_dir = os.path.join(dataset_path, person, "profile")
        frontal_dir = os.path.join(dataset_path, person, "frontal")

        if not os.path.exists(profile_dir) or not os.path.exists(frontal_dir):
            print(f"⚠️ Missing CFP folders for {person}, skipping...")
            continue

        profile_imgs = [f for f in os.listdir(profile_dir)
                        if f.lower().endswith(('.jpg', '.png'))]
        frontal_imgs = [f for f in os.listdir(frontal_dir)
                        if f.lower().endswith(('.jpg', '.png'))]

        if len(profile_imgs) == 0 or len(frontal_imgs) == 0:
            print(f"⚠️ Missing images for {person}, skipping...")
            continue

        p_img = random.choice(profile_imgs)
        f_img = random.choice(frontal_imgs)

        shutil.copy(
            os.path.join(profile_dir, p_img),
            os.path.join(output_path, f"{person}_profile_{p_img}")
        )
        shutil.copy(
            os.path.join(frontal_dir, f_img),
            os.path.join(output_path, f"{person}_frontal_{f_img}")
        )

        copied.append(person)
        print(f"✔ Copied: {person} → {p_img}, {f_img}")

    print(f"\n[OK] {len(copied)} persons copied to {output_path}")
    return copied

# --------------------------------------------------------
#                DECA INFERENCE (INTERNAL)
# --------------------------------------------------------
try:
    from deca_infer import run_deca_inference
except ImportError:
    print("❌ ERROR: Could not import run_deca_inference from deca_infer.py")
    run_deca_inference = None
    # don't sys.exit here; allow user to skip DECA if desired


# -------------------- ArcFace ONNX utilities --------------------
ONNX_MODEL_URL = "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-11-int8/resolve/main/arcfaceresnet100-11-int8.onnx"
ONNX_MODEL_LOCAL = "arcfaceresnet100-int8.onnx"
ONNX_MODEL_ARG = ONNX_MODEL_LOCAL

import onnxruntime as ort

def download_arcface(url=ONNX_MODEL_URL, save_path=ONNX_MODEL_LOCAL):
    if os.path.exists(save_path):
        print("[INFO] ArcFace ONNX already exists:", save_path)
        return save_path
    print("[INFO] Downloading ArcFace ONNX ...")
    subprocess.run(["wget", url, "-O", save_path, "--no-check-certificate"], check=True)
    return save_path

def get_onnx_session(model_path):
    available = ort.get_available_providers()
    use_cuda = 'CUDAExecutionProvider' in available
    if use_cuda:
        providers = [
            ('CUDAExecutionProvider', {
                "device_id": 0,
            }),
            'CPUExecutionProvider'
        ]
    else:
        providers = ['CPUExecutionProvider']

    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"[INFO] ONNX session created. Providers: {sess.get_providers()}")
    print(f"[INFO] Input: {input_name}  Output: {output_name}")
    return sess, input_name, output_name

def prewhiten(img: np.ndarray):
    x = img.astype(np.float32)
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = (x - mean) / std_adj
    return y

def arcface_preprocess_from_bgr(img_bgr, target_size=(112, 112)):
    if img_bgr is None:
        raise ValueError("Input image is None")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    img_pw = prewhiten(img_resized)
    img_chw = np.transpose(img_pw, (2, 0, 1)).astype(np.float32)
    img_batch = np.expand_dims(img_chw, axis=0)
    return img_batch

def compute_embedding_onnx(sess, input_name, output_name, img_bgr):
    inp = arcface_preprocess_from_bgr(img_bgr)
    out = sess.run([output_name], {input_name: inp})[0]
    emb = np.asarray(out).reshape(-1)
    norm = np.linalg.norm(emb)
    if norm < 1e-8:
        return emb
    return emb / norm

def cosine_sim(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)

def compute_psnr(img1, img2):
    """Compute PSNR between two BGR images"""
    if img1 is None or img2 is None:
        return float("nan")
    return cv2.PSNR(img1, img2)

def compute_ssim(img1, img2):
    """Compute SSIM between two BGR images"""
    if img1 is None or img2 is None:
        return float("nan")
    # convert to grayscale 
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    s, _ = ssim(g1, g2, full=True)
    return float(s)

# File/folder utilities
def find_input_images(input_dir, person):
    base_id = person.split('_')[0].strip()

    patterns_frontal = [
        f"{base_id}_frontal*.jpg",
        f"{base_id}_frontal*.png",
        f"{base_id}*frontal*.jpg",
        f"{base_id}*frontal*.png",
    ]
    patterns_profile = [
        f"{base_id}_profile*.jpg",
        f"{base_id}_profile*.png",
        f"{base_id}*profile*.jpg",
        f"{base_id}*profile*.png",
    ]

    frontal = None
    profile = None

    for p in patterns_frontal:
        files = glob(os.path.join(input_dir, p))
        if files:
            frontal = files[0]
            break

    for p in patterns_profile:
        files = glob(os.path.join(input_dir, p))
        if files:
            profile = files[0]
            break

    return frontal, profile


def autodetect_persons_from_deca_out(deca_out):
    persons = []
    if not os.path.isdir(deca_out):
        return persons

    for entry in sorted(os.listdir(deca_out)):
        sub = os.path.join(deca_out, entry)
        if not os.path.isdir(sub):
            continue
        pngs = [f for f in os.listdir(sub) if f.lower().endswith('.png')]
        found = False
        for p in pngs:
            if p.endswith('_frontal.png'):
                base = p[:-len('_frontal.png')]
                if base + '.png' in pngs:
                    persons.append(entry)
                    found = True
                    break
        if found:
            continue
        if f"{entry}.png" in pngs and f"{entry}_frontal.png" in pngs:
            persons.append(entry)
            continue
        if pngs:
            persons.append(entry)
    return persons


def group_by_base(persons):
    """
    Return dict base_id -> list of person folder names sharing that base.
    """
    groups = {}
    for p in persons:
        base = p.split('_')[0]
        groups.setdefault(base, []).append(p)
    return groups

# Pairwise processing helpers
def get_uv_and_front_paths(person_folder, person_name):
    """
    Return (uv_path, front_path) for a given person folder/name (best-effort).
    """
    uv_path = os.path.join(person_folder, f"{person_name}.png")
    front_path = os.path.join(person_folder, f"{person_name}_frontal.png")

    if not (os.path.exists(uv_path) and os.path.exists(front_path)):
        pngs = [f for f in os.listdir(person_folder) if f.lower().endswith('.png')]
        base = None
        for p in pngs:
            if p.endswith('_frontal.png'):
                cand_base = p[:-len('_frontal.png')]
                if f"{cand_base}.png" in pngs:
                    base = cand_base
                    break
        if base:
            uv_path = os.path.join(person_folder, f"{base}.png")
            front_path = os.path.join(person_folder, f"{base}_frontal.png")
        else:
            frontal_candidates = [os.path.join(person_folder, f) for f in pngs if f.endswith('_frontal.png')]
            uv_candidates = [os.path.join(person_folder, f) for f in pngs if not f.endswith('_frontal.png')]
            if frontal_candidates and uv_candidates:
                front_path = frontal_candidates[0]
                uv_path = uv_candidates[0]

    if os.path.exists(uv_path) and os.path.exists(front_path):
        return uv_path, front_path
    return None, None

def save_pair_visual(pair_vis_dir, A_name, B_name, img_uv_A, img_uv_B, img_front_A, img_front_B, front_sim, uv_sim, psnr_uv, ssim_uv):
    os.makedirs(pair_vis_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(cv2.cvtColor(img_uv_A, cv2.COLOR_BGR2RGB)); axes[0, 0].set_title(f"UV {A_name}"); axes[0, 0].axis("off")
    axes[0, 1].imshow(cv2.cvtColor(img_uv_B, cv2.COLOR_BGR2RGB)); axes[0, 1].set_title(f"UV {B_name}"); axes[0, 1].axis("off")
    axes[1, 0].imshow(cv2.cvtColor(img_front_A, cv2.COLOR_BGR2RGB)); axes[1, 0].set_title(f"Front {A_name}"); axes[1, 0].axis("off")
    axes[1, 1].imshow(cv2.cvtColor(img_front_B, cv2.COLOR_BGR2RGB)); axes[1, 1].set_title(f"Front {B_name}"); axes[1, 1].axis("off")
    plt.figtext(0.5, 0.01, f"FRONT: {front_sim:.4f} | UV: {uv_sim:.4f} | PSNR(UV): {psnr_uv:.2f} | SSIM(UV): {ssim_uv:.4f}",
        fontsize=12, ha="center")
    out_path = os.path.join(pair_vis_dir, f"{A_name}_vs_{B_name}.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

# Core processing: existing per-person processing (keeps original behavior)
def process_and_visualize(pairs, input_dir, out_dir, output_vis, csv_path, sess, input_name, output_name):
    os.makedirs(output_vis, exist_ok=True)
    csv_rows = []
    print("\n[INFO] Calculating similarities and generating visualizations...")

    for person in pairs:
        print(f"----------------------------------------------")
        print(f"[INFO] Processing person/folder: {person}")

        person_folder = os.path.join(out_dir, person)
        if not os.path.isdir(person_folder):
            print(f"[WARNING] Folder does not exist: {person_folder} -- skipping")
            continue

        uv_path = os.path.join(person_folder, f"{person}.png")
        frontalized_path = os.path.join(person_folder, f"{person}_frontal.png")

        if not (os.path.exists(uv_path) and os.path.exists(frontalized_path)):
            pngs = [f for f in os.listdir(person_folder) if f.lower().endswith('.png')]
            base = None
            for p in pngs:
                if p.endswith('_frontal.png'):
                    cand_base = p[:-len('_frontal.png')]
                    if f"{cand_base}.png" in pngs:
                        base = cand_base
                        break
            if base:
                uv_path = os.path.join(person_folder, f"{base}.png")
                frontalized_path = os.path.join(person_folder, f"{base}_frontal.png")
            else:
                frontal_candidates = [os.path.join(person_folder, f) for f in pngs if f.endswith('_frontal.png')]
                uv_candidates = [os.path.join(person_folder, f) for f in pngs if not f.endswith('_frontal.png')]
                if frontal_candidates and uv_candidates:
                    frontalized_path = frontal_candidates[0]
                    uv_path = uv_candidates[0]

        if not (os.path.exists(uv_path) and os.path.exists(frontalized_path)):
            print(f"[WARNING] Missing DECA outputs in {person_folder}. Need both UV and frontalized PNGs. Skipping.")
            continue

        frontal_input, profile_input = find_input_images(input_dir, person)
        if frontal_input is None or profile_input is None:
            fr_candidates = glob(os.path.join(input_dir, f"{person}*frontal*.jpg")) + \
                            glob(os.path.join(input_dir, f"{person}*frontal*.png"))
            pr_candidates = glob(os.path.join(input_dir, f"{person}*profile*.jpg")) + \
                            glob(os.path.join(input_dir, f"{person}*profile*.png"))
            frontal_input = fr_candidates[0] if fr_candidates else frontal_input
            profile_input = pr_candidates[0] if pr_candidates else profile_input

        if frontal_input is None or profile_input is None:
            print(f"[WARNING] Could not find input frontal/profile images for '{person}' in {input_dir}. Skipping.")
            continue

        img_frontal_in = cv2.imread(frontal_input)
        img_profile_in = cv2.imread(profile_input)
        img_uv = cv2.imread(uv_path)
        img_frontalized = cv2.imread(frontalized_path)

        if img_frontal_in is None or img_profile_in is None or img_uv is None or img_frontalized is None:
            print("[ERROR] One or more images failed to load. Skipping.")
            continue

        try:
            emb_frontal_in = compute_embedding_onnx(sess, input_name, output_name, img_frontal_in)
            emb_profile_in = compute_embedding_onnx(sess, input_name, output_name, img_profile_in)
            emb_uv = compute_embedding_onnx(sess, input_name, output_name, img_uv)
            emb_frontalized = compute_embedding_onnx(sess, input_name, output_name, img_frontalized)
        except Exception as e:
            print(f"[ERROR] Failed computing embeddings for '{person}': {e}")
            continue

        sim_profile = cosine_sim(emb_frontal_in, emb_profile_in)
        sim_uv = cosine_sim(emb_frontal_in, emb_uv)
        sim_front = cosine_sim(emb_frontal_in, emb_frontalized)

        print(f"[SIM] Frontal vs Profile     = {sim_profile:.4f}")
        print(f"[SIM] Frontal vs UV Texture  = {sim_uv:.4f}")
        print(f"[SIM] Frontal vs Frontalized = {sim_front:.4f}")

        csv_rows.append({
            "person": person,
            "sim_profile": f"{sim_profile:.6f}",
            "sim_uv": f"{sim_uv:.6f}",
            "sim_frontalized": f"{sim_front:.6f}"
        })

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(cv2.cvtColor(img_frontal_in, cv2.COLOR_BGR2RGB)); axes[0].set_title("Original Frontal", fontsize=16); axes[0].axis("off")
        axes[1].imshow(cv2.cvtColor(img_profile_in, cv2.COLOR_BGR2RGB)); axes[1].set_title("Original Profile", fontsize=16); axes[1].axis("off")
        axes[2].imshow(cv2.cvtColor(img_uv, cv2.COLOR_BGR2RGB)); axes[2].set_title("UV Texture", fontsize=16); axes[2].axis("off")
        axes[3].imshow(cv2.cvtColor(img_frontalized, cv2.COLOR_BGR2RGB)); axes[3].set_title("Frontalized", fontsize=16); axes[3].axis("off")
        scores = (f"Frontal vs Profile: {sim_profile:.4f} | Frontal vs UV: {sim_uv:.4f} | "
                  f"Frontal vs Front: {sim_front:.4f}")
        plt.figtext(0.5, 0.02, scores, ha="center", fontsize=16)
        out_path = os.path.join(output_vis, f"{person}_comparison.png")
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(out_path, dpi=150)
        plt.close()

    # write per-person CSV
    print(f"----------------------------------------------")
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["person", "sim_profile", "sim_uv", "sim_frontalized"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)
    print(f"[INFO] Comparisons CSV saved: {csv_path}\n")
    return csv_rows

# Pairwise combinations & CSV
def process_pairwise(groups, deca_out, pair_vis_root, sess, input_name, output_name):
    """
    groups: dict base_id -> list of person folder names
    deca_out: root folder containing person subfolders
    pair_vis_root: root dir where vis_pairs/<baseID>/ will be created
    returns list of pair rows
    """
    pair_rows = []
    for base_id, persons in groups.items():
        if len(persons) < 2:
            continue
        persons_sorted = sorted(persons)
        base_dir = os.path.join(pair_vis_root, base_id)
        os.makedirs(base_dir, exist_ok=True)

        # all combinations
        for A, B in itertools.combinations(persons_sorted, 2):
            folder_A = os.path.join(deca_out, A)
            folder_B = os.path.join(deca_out, B)

            uv_A, front_A = get_uv_and_front_paths(folder_A, A)
            uv_B, front_B = get_uv_and_front_paths(folder_B, B)

            if not (uv_A and front_A and uv_B and front_B):
                print(f"[WARN] Missing UV/Front files for pair {A} vs {B}. Skipping pair.")
                continue

            img_uv_A = cv2.imread(uv_A)
            img_uv_B = cv2.imread(uv_B)
            img_front_A = cv2.imread(front_A)
            img_front_B = cv2.imread(front_B)

            if img_uv_A is None or img_uv_B is None or img_front_A is None or img_front_B is None:
                print(f"[WARN] Failed to read images for pair {A} vs {B}. Skipping pair.")
                continue

            try:
                emb_uv_A = compute_embedding_onnx(sess, input_name, output_name, img_uv_A)
                emb_uv_B = compute_embedding_onnx(sess, input_name, output_name, img_uv_B)
                emb_front_A = compute_embedding_onnx(sess, input_name, output_name, img_front_A)
                emb_front_B = compute_embedding_onnx(sess, input_name, output_name, img_front_B)
            except Exception as e:
                print(f"[ERROR] ONNX inference failed for pair {A} vs {B}: {e}")
                continue

            uv_sim = cosine_sim(emb_uv_A, emb_uv_B)
            front_sim = cosine_sim(emb_front_A, emb_front_B)

            # ----- PSNR & SSIM between UV(angle) and UV(REF) -----
            psnr_uv = compute_psnr(img_uv_A, img_uv_B)
            ssim_uv = compute_ssim(img_uv_A, img_uv_B)  

            vis_out = save_pair_visual(base_dir, A, B, img_uv_A, img_uv_B, img_front_A, img_front_B, front_sim, uv_sim, psnr_uv, ssim_uv)

            pair_rows.append({
                "base_id": base_id,
                "entry_A": A,
                "entry_B": B,
                "front_similarity": f"{front_sim:.6f}",
                "uv_similarity": f"{uv_sim:.6f}",
                "psnr_uv": psnr_uv,
                "ssim_uv": ssim_uv
            })

            print("[INFO] Running pairwise UV/Front comparison...")
            print(f"[PAIR] {A} vs {B} : Front sim: {front_sim:.4f} | UV sim: {uv_sim:.4f} | "
                  f"PSNR UV: {psnr_uv:.2f} | SSIM UV: {ssim_uv:.4f}")

    return pair_rows

def write_and_sort_pair_csv(pair_rows, out_csv_path, sort_by="front_similarity", descending=True):
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    fieldnames = ["base_id", "entry_A", "entry_B", "front_similarity", "uv_similarity", "psnr_uv", "ssim_uv"]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in pair_rows:
            writer.writerow(r)

    # Now sort by frontal_similarity (numeric) if requested
    try:
        with open(out_csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        rows_sorted = sorted(rows, key=lambda r: float(r.get(sort_by, 0.0)), reverse=descending)
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_sorted)
        print(f"[INFO] Pairwise CSV written and sorted by '{sort_by}' ({'DESC' if descending else 'ASC'}) -> {out_csv_path}")
    except Exception as e:
        print(f"[ERROR] Failed to sort/write pairwise CSV: {e}")

# Contact sheet generator (merges per-person visuals)
def create_contact_sheet_for_pair(deca_out_root, out_path, thumb_w=360, thumb_h=270):
    """
    Build a contact sheet with 2 rows and 3 columns:
        Row 1: frontal -> UV -> frontalized
        Row 2: profile -> UV -> frontalized
    deca_out_root: path to folder containing DECA outputs (e.g., 'inference_test/out_data')
    out_path: where to save final contact sheet
    """
    # detect persons (frontal + profile)
    person_folders = sorted([f for f in os.listdir(deca_out_root)
                             if os.path.isdir(os.path.join(deca_out_root, f))])
    
    if not person_folders:
        print(f"[INFO] No person folders found in {deca_out_root}")
        return
    
    # Expecting at least 2 folders to form 2 rows
    if len(person_folders) < 2:
        print(f"[WARN] Less than 2 person folders found in {deca_out_root}, cannot create 2-row sheet.")
        return

    imgs_rows = []

    for pf in person_folders[:2]:  # only first two for a single sheet
        folder_path = os.path.join(deca_out_root, pf)
        base_name = pf

        # Original / UV image
        uv_path = None
        frontalized_path = None
        input_path = None

        # try to find uv and frontalized robustly
        for f in os.listdir(folder_path):
            if f.lower().endswith('.png') and not f.endswith('_frontal.png'):
                uv_path = os.path.join(folder_path, f)
            if f.lower().endswith('_frontal.png'):
                frontalized_path = os.path.join(folder_path, f)
            if f.lower().endswith(('.jpg', '.png')) and f not in [os.path.basename(uv_path) if uv_path else '', os.path.basename(frontalized_path) if frontalized_path else '']:
                input_path = os.path.join(folder_path, f)

        imgs = []
        for p in [input_path, uv_path, frontalized_path]:
            if p and os.path.exists(p):
                im = cv2.imread(p)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                imgs.append(im)
            else:
                imgs.append(np.ones((thumb_h, thumb_w, 3), dtype=np.uint8)*255)
        imgs_rows.append(imgs)

    # Combine into a 2x3 contact sheet
    sheet_h = thumb_h * 2
    sheet_w = thumb_w * 3
    sheet = np.ones((sheet_h, sheet_w, 3), dtype=np.uint8) * 255

    for r, row_imgs in enumerate(imgs_rows):
        for c, im in enumerate(row_imgs):
            y = r * thumb_h
            x = c * thumb_w
            sheet[y:y+thumb_h, x:x+thumb_w] = im

    # Save
    sheet_bgr = cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, sheet_bgr)
    print(f"[INFO] Contact sheet saved -> {out_path}")

# main (CLI + orchestration)
def main():
    parser = argparse.ArgumentParser(description="Run DECA then compute similarity, visualizations, CSV using ArcFace ONNX (with pairwise comparisons).")
    parser.add_argument("--cfp_root", type=str, default="/content/Towards-Realistic-Generative-3D-Face-Models/cfp-dataset/Data/Images",
                        help="Path to CFP dataset (if you want to auto-copy inputs)")
    parser.add_argument("--input_dir", type=str, default="inference_test/in_data")
    parser.add_argument("--deca_out", type=str, default="inference_test/out_data")
    parser.add_argument("--output_vis", type=str, default="inference_test/out_data/vis_cfp")
    parser.add_argument("--csv_path", type=str, default="inference_test/out_data/csv/cfp_comparisons.csv")
    parser.add_argument("--pairs", nargs="+", default=[], help="List of person (folder) names to compare (optional)")
    parser.add_argument("--auto", action="store_true", help="Auto-detect persons from deca_out (if no --pairs provided this is used).")
    parser.add_argument("--sort_key", type=str, default=None, choices=[None, "sim_profile", "sim_uv", "sim_frontalized"], help="Column name for sorting CSV")
    parser.add_argument("--sort_desc", action="store_true", help="Sort descending")
    parser.add_argument("--model_path", default=ONNX_MODEL_LOCAL, help="Path to ArcFace ONNX model")
    parser.add_argument("--num_persons", type=int, default=2, help="When --cfp_root provided, how many persons to copy")

    args = parser.parse_args()

    # optionally copy from CFP
    if args.cfp_root:
        print("\n[STEP] Copying random CFP pairs...")
        extract_random_cfp_pairs(args.cfp_root, args.input_dir, args.num_persons)

    # Run DECA
    print("\n[STEP] Running DECA inference (if run_deca_inference available)...")
    if run_deca_inference is not None:
        try:
            run_deca_inference(args.input_dir, args.deca_out)
            print("[INFO] DECA Done.")
        except Exception as e:
            print(f"[WARNING] run_deca_inference() raised an exception: {e}. Continuing (assuming DECA outputs already exist).")
    else:
        print("[WARNING] run_deca_inference() not available in environment. Skipping DECA run.")

    # Prepare persons list
    if args.pairs:
        persons = args.pairs
        print(f"[INFO] Using provided --pairs: {persons}")
    elif args.auto:
        persons = autodetect_persons_from_deca_out(args.deca_out)
        print(f"[INFO] Auto-detected persons/folders: {persons}")
        if not persons:
            print("[WARNING] No persons detected in deca_out; nothing to process.")
            return
    else:
        # default: autodetect
        persons = autodetect_persons_from_deca_out(args.deca_out)
        print(f"[INFO] Auto-detected persons/folders: {persons}")
        if not persons:
            print("[WARNING] No persons detected in deca_out; nothing to process.")
            return

    # Check model
    if not os.path.exists(args.model_path):
        print(f"[ERROR] ONNX model not found at: {args.model_path}")
        return

    # ONNX session
    sess, input_name, output_name = get_onnx_session(args.model_path)

    # ensure output dirs
    os.makedirs(args.output_vis, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)

    # per-person processing and visualization
    csv_rows = process_and_visualize(persons, args.input_dir, args.deca_out, args.output_vis, args.csv_path,
                                     sess, input_name, output_name)

    # Pairwise
    groups = group_by_base(persons)
    pair_vis_root = os.path.join(args.output_vis, "pairs")
    os.makedirs(pair_vis_root, exist_ok=True)
    pair_rows = process_pairwise(groups, args.deca_out, pair_vis_root, sess, input_name, output_name)
    pair_csv_path = os.path.join(args.deca_out, "csv/cfp_pairwise.csv")
    write_and_sort_pair_csv(pair_rows, pair_csv_path, sort_by="front_similarity", descending=True)

    # Contact sheet for the whole deca_out (one sheet combining first two persons)
    contact_path = os.path.join(args.output_vis, "contact_sheet.png")
    create_contact_sheet_for_pair(args.deca_out, contact_path, thumb_w=360, thumb_h=270)

    # Optionally sort the original CSV
    if args.sort_key:
        try:
            with open(args.csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            rows_sorted = sorted(rows, key=lambda r: float(r.get(args.sort_key, 0.0)), reverse=args.sort_desc)
            with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(rows_sorted)
            print(f"[INFO] CSV sorted by '{args.sort_key}' ({'DESC' if args.sort_desc else 'ASC'})")
        except Exception as e:
            print(f"[ERROR] Failed to sort original CSV: {e}")

    print("\n[✅] Pipeline finished")

if __name__ == "__main__":
    main()