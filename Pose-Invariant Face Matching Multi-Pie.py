"""
Multi-PIE -> DECA -> ArcFace(ONNX) similarity pipeline.

Steps:
1. Randomly select a subject from Multi-PIE (HR_128) and copy one image per camera into `inference_test/in_data/<person>`, marking camera 051 as the frontal reference (`_frontal`).
2. Optionally run DECA to generate UV and frontalized outputs for the subject.
3. Compute ArcFace embeddings:
   – Compare each input image to the frontal reference.
   – If DECA outputs exist, compare UV and frontalized images to their DECA reference outputs.
4. Save similarity results to CSV and generate per-person visualization.
5. Run Multi-PIE pairwise comparisons: compare each angle folder against the 051 reference folder.

"""
import os
import random
import shutil
import csv
from glob import glob
import subprocess
import argparse
import math
import itertools

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# try to import run_deca_inference if you have it in a module deca_infer.py
try:
    from deca_infer import run_deca_inference
except Exception:
    run_deca_inference = None

# -------------------- Config paths --------------------
MULTIPIE_ROOT = "Multi_Pie/HR_128"              # Multi-PIE dataset folder
BASE_INPUT_OUT = "inference_test/in_data"      # copied inputs go here (per-person subfolder)
DECA_OUT_ROOT = "inference_test/out_data"      # expected deca outputs here (per-person subfolder)
VIS_ROOT = os.path.join(DECA_OUT_ROOT, "vis_multi_pie")
CSV_OUT = os.path.join(DECA_OUT_ROOT, "csv", "multipie_results.csv")
ONNX_MODEL_URL = "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-11-int8/resolve/main/arcfaceresnet100-11-int8.onnx"
ONNX_MODEL_LOCAL = "arcfaceresnet100-int8.onnx"
ONNX_MODEL_ARG = ONNX_MODEL_LOCAL

# -------------------- ArcFace ONNX utilities --------------------
import onnxruntime as ort

def download_arcface(url=ONNX_MODEL_URL, save_path=ONNX_MODEL_LOCAL):
    if os.path.exists(save_path):
        print("[INFO] ArcFace ONNX already exists:", save_path)
        return save_path
    print("[INFO] Downloading ArcFace ONNX ...")
    subprocess.run(["wget", url, "-O", save_path, "--no-check-certificate"], check=True)
    return save_path

def get_onnx_session(model_path):
    # Create ONNX Runtime session (CUDA if available)
    available = ort.get_available_providers()
    use_cuda = 'CUDAExecutionProvider' in available
    if use_cuda:
        providers = [
            ('CUDAExecutionProvider', {"device_id": 0}),
            'CPUExecutionProvider'
        ]
    else:
        providers = ['CPUExecutionProvider']
    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"[INFO] ONNX session created. Providers: {sess.get_providers()}")
    return sess, input_name, output_name

def prewhiten(img: np.ndarray):
    # Standard FaceNet normalization
    x = img.astype(np.float32)
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / math.sqrt(x.size))
    y = (x - mean) / std_adj
    return y

def arcface_preprocess_from_bgr(img_bgr, target_size=(112, 112)):
    # Resize → RGB → prewhiten → CHW → batch
    if img_bgr is None:
        raise ValueError("Input image is None")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    img_pw = prewhiten(img_resized)
    img_chw = np.transpose(img_pw, (2, 0, 1)).astype(np.float32)
    img_batch = np.expand_dims(img_chw, axis=0)
    return img_batch

def compute_embedding_onnx(sess, input_name, output_name, img_bgr):
    # Forward pass and L2-normalize embedding
    inp = arcface_preprocess_from_bgr(img_bgr)
    out = sess.run([output_name], {input_name: inp})[0]
    emb = np.asarray(out).reshape(-1)
    norm = np.linalg.norm(emb)
    if norm < 1e-8:
        return emb
    return emb / norm

def cosine_sim(a, b):
    # Standard cosine similarity
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)

# -------------------- Random Multi-PIE subject selection --------------------
def select_random_subject_and_copy(dataset_path=MULTIPIE_ROOT, base_output_path=BASE_INPUT_OUT):
    # Scan dataset
    os.makedirs(base_output_path, exist_ok=True)
    all_images = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg'))]
    if not all_images:
        raise RuntimeError(f"No images found in {dataset_path}")
    persons = set(f.split('_')[0] for f in all_images)
    person_id = random.choice(list(persons))
    print(f"[INFO] Selected person: {person_id}")

    output_path = os.path.join(base_output_path, person_id)
    os.makedirs(output_path, exist_ok=True)

    # Select one image per camera
    person_images = [f for f in all_images if f.startswith(person_id)]
    info_dict = {}  # camera_id -> image name
    for fname in person_images:
        parts = fname.split('_')
        # ensure format has enough parts; user code expects >=6
        if len(parts) < 6:
            continue
        # parts mapping based on your filename structure
        # e.g., 206_01_02_051_05_crop_128.png  -> parts[0]=206, parts[1]=01, parts[2]=02, parts[3]=051, parts[4]=05, ...
        s_id = parts[1]
        i_id = parts[2]
        c_id = parts[3]
        pose_id = parts[4]
        if c_id not in info_dict:
            info_dict[c_id] = fname

    copied_count = 0
    print(f"\n[INFO] Copying one image per camera for person {person_id}:")
    # Copy to in_data/<person>
    for c_id, fname in info_dict.items():
        src_path = os.path.join(dataset_path, fname)
        if not os.path.exists(src_path):
            print(f"[WARN] Source not found: {src_path}")
            continue
        # If camera is 051, add _frontal before extension
        name_parts = fname.rsplit('.', 1)
        if c_id == '051':
            dst_fname = f"{name_parts[0]}_frontal.{name_parts[1]}"
        else:
            dst_fname = fname
        dst_path = os.path.join(output_path, dst_fname)
        shutil.copy(src_path, dst_path)
        copied_count += 1
        print(f"{person_id}: camera {c_id} → {dst_fname} copied successfully.")

    print(f"\n✅ Done! {copied_count} images copied for person {person_id}.")
    return person_id, output_path

# -------------------- DECA runner wrapper --------------------
def run_deca_if_available(input_dir, deca_out_root=DECA_OUT_ROOT):
    # Expect per-person folder named same as input subfolder (we'll use that)
    if run_deca_inference is None:
        print("[WARNING] run_deca_inference() not available. Skipping DECA run. Make sure DECA outputs exist.")
        return False
    try:
        # run_deca_inference(input_dir, deca_out_root) expects input dir and output dir,
        # adjust if your deca_infer signature is different.
        run_deca_inference(input_dir, deca_out_root)
        print("[INFO] DECA inference completed.")
        return True
    except Exception as e:
        print(f"[ERROR] run_deca_inference raised: {e}")
        return False

# -------------------- find input frontal/profile in input folder --------------------
def find_reference_input(input_person_folder):
    # find file that endswith '_frontal.png' first
    files = sorted([f for f in os.listdir(input_person_folder) if f.lower().endswith(('.png', '.jpg'))])
    ref = None
    for f in files:
        if f.endswith('_frontal.png'):
            ref = os.path.join(input_person_folder, f)
            break
    # fallback: try to find camera '051' file if naming preserved
    if ref is None:
        for f in files:
            if '_051_' in f or f.startswith('051') or f.endswith('_051.png'):
                ref = os.path.join(input_person_folder, f)
                break
    return ref

# -------------------- Detect uv / frontalized paths from DECA out per-person --------------------
def get_deca_paths_for_person(deca_out_person_folder):
    # assume deca outputs are like: <base>.png (uv), <base>_frontal.png (frontalized)
    pngs = [f for f in os.listdir(deca_out_person_folder) if f.lower().endswith('.png')]
    uv = None
    frontal = None
    for p in pngs:
        if p.endswith('_frontal.png'):
            base = p[:-len('_frontal.png')]
            if f"{base}.png" in pngs:
                uv = os.path.join(deca_out_person_folder, f"{base}.png")
                frontal = os.path.join(deca_out_person_folder, p)
                return uv, frontal
    # fallback heuristics
    if pngs:
        # take first non-frontal as uv and any frontal as frontal
        for p in pngs:
            if p.endswith('_frontal.png'):
                frontal = os.path.join(deca_out_person_folder, p)
            else:
                uv = os.path.join(deca_out_person_folder, p)
    return uv, frontal

# -------------------- Multi-PIE pairwise comparison --------------------
def get_uv_and_frontal_paths_multipie(folder):
    """Detect UV.png and *_frontal.png inside a DECA folder (Multi-PIE version)."""
    pngs = [p for p in os.listdir(folder) if p.endswith(".png")]
    uv = None
    front = None

    # Prefer exact deca naming: base.png + base_frontal.png
    for p in pngs:
        if p.endswith("_frontal.png"):
            base = p[:-len("_frontal.png")]
            uv_candidate = f"{base}.png"
            if uv_candidate in pngs:
                uv = os.path.join(folder, uv_candidate)
                front = os.path.join(folder, p)
                return uv, front

    # fallback: first non-frontal as UV, first frontal as front
    for p in pngs:
        if "_frontal" in p:
            front = os.path.join(folder, p)
        else:
            uv = os.path.join(folder, p)

    return uv, front

def process_multipie_pairwise(person_id, deca_out_root, sess, input_name, output_name, vis_root):
    """Perform pairwise UV/Frontal comparisons for all DECA angle folders of a person."""

    # Collect all DECA output folders matching this person
    person_folders = [
        f for f in os.listdir(deca_out_root)
        if f.startswith(person_id) and os.path.isdir(os.path.join(deca_out_root, f))
    ]

    # Need at least reference folder + one comparison folder
    if len(person_folders) < 2:
        print("[WARN] Not enough DECA folders for person:", person_id)
        return []

    # Detect the frontal reference folder (Multi-PIE camera 051)
    ref_folder = None
    for f in person_folders:
        if "_051_" in f or f.endswith("_051") or "051" in f:
            ref_folder = f
            break

    # No reference → cannot run pairwise comparison
    if ref_folder is None:
        print("[ERROR] No 051 (frontal reference) found for", person_id)
        return []

    # Load UV and frontalized reference outputs
    ref_path = os.path.join(deca_out_root, ref_folder)
    uv_ref, front_ref = get_uv_and_frontal_paths_multipie(ref_path)

    img_uv_ref = cv2.imread(uv_ref) if uv_ref else None
    img_front_ref = cv2.imread(front_ref) if front_ref else None

    # Compute embeddings for the reference UV / frontal images
    emb_uv_ref = compute_embedding_onnx(sess, input_name, output_name, img_uv_ref) if img_uv_ref is not None else None
    emb_front_ref = compute_embedding_onnx(sess, input_name, output_name, img_front_ref) if img_front_ref is not None else None

    print(f"[INFO] Reference folder: {ref_folder}")
    print(f"       UV_REF: {uv_ref}")
    print(f"       FRONT_REF: {front_ref}\n")

    pair_rows = []
    vis_dir = os.path.join(vis_root, person_id)
    os.makedirs(vis_dir, exist_ok=True)

    # Compare each angle folder with the reference 051 folder
    for f in person_folders:
        if f == ref_folder:
            continue  # skip reference itself

        folder_path = os.path.join(deca_out_root, f)
        uv, fr = get_uv_and_frontal_paths_multipie(folder_path)

        # Read UV / frontalized images for this angle
        img_uv = cv2.imread(uv) if uv else None
        img_fr = cv2.imread(fr) if fr else None

        if img_uv is None or img_fr is None:
            print("[WARN] Missing UV or Frontalized in", folder_path)
            continue

        # Compute embeddings
        emb_uv = compute_embedding_onnx(sess, input_name, output_name, img_uv)
        emb_fr = compute_embedding_onnx(sess, input_name, output_name, img_fr)

        # Compute similarity vs the reference folder
        uv_sim = cosine_sim(emb_uv_ref, emb_uv) if emb_uv_ref is not None else float('nan')
        fr_sim = cosine_sim(emb_front_ref, emb_fr) if emb_front_ref is not None else float('nan')

        print(f"[PAIR] {f} vs REF : UV sim: {uv_sim:.4f} | Front sim: {fr_sim:.4f}")

        # -------- Visualization for this angle vs reference --------
        out_path = os.path.join(vis_dir, "pairwise comparisons", f"{f}_vs_REF.png")
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        axes[0][0].imshow(cv2.cvtColor(img_uv, cv2.COLOR_BGR2RGB)); axes[0][0].set_title(f"UV ({f})"); axes[0][0].axis('off')
        axes[0][1].imshow(cv2.cvtColor(img_uv_ref, cv2.COLOR_BGR2RGB)); axes[0][1].set_title(f"UV REF ({ref_folder})"); axes[0][1].axis('off')
        axes[1][0].imshow(cv2.cvtColor(img_fr, cv2.COLOR_BGR2RGB)); axes[1][0].set_title(f"FRONT ({f})"); axes[1][0].axis('off')
        axes[1][1].imshow(cv2.cvtColor(img_front_ref, cv2.COLOR_BGR2RGB)); axes[1][1].set_title(f"FRONT REF ({ref_folder})"); axes[1][1].axis('off')
      
        # Add similarity text
        plt.figtext(0.5, 0.01, f"UV: {uv_sim:.4f} | FRONT: {fr_sim:.4f}", ha="center")
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        out_dir = os.path.join(vis_dir, "pairwise comparisons")
        os.makedirs(out_dir, exist_ok=True)  
        out_path = os.path.join(out_dir, f"{f}_vs_REF.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

        # Append row for CSV aggregation
        pair_rows.append({
            "person": person_id,
            "entry": f,
            "reference": ref_folder,
            "uv_similarity": uv_sim,
            "front_similarity": fr_sim
        })

    return pair_rows

# -------------------- Visualization helper --------------------
def save_comparison_vis(out_vis_dir, person_id, ref_img_path, img_paths, sims_dict):
    """Generate comparison figure: reference on left, grid of other input images on the right."""
    os.makedirs(out_vis_dir, exist_ok=True)

    # Build a wide figure: reference on left, grid of others on right
    ref_img = cv2.cvtColor(cv2.imread(ref_img_path), cv2.COLOR_BGR2RGB)
    n = len(img_paths)
    cols = min(6, n)
    rows = math.ceil(n / cols)

    # Create figure: first column → reference, rest → other inputs
    fig = plt.figure(figsize=(4 + 3*cols, 3*(rows+0.5)))
    ax = fig.add_subplot(rows+1, cols, 1)
    ax.imshow(ref_img); ax.set_title("Reference (051_frontal)"); ax.axis("off")
    # Fill the grid with input images and similarity text
    for i, p in enumerate(img_paths):
        r = (i // cols) + 1
        c = (i % cols) + 1
        axidx = (r+1 - 1) * cols + c
        ax = fig.add_subplot(rows+1, cols, axidx)
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        ax.imshow(im)
        sim = sims_dict.get(os.path.basename(p), None)
        title = os.path.basename(p)
        if sim is not None:
            title += f"\nsim={sim:.4f}"
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    out_path = os.path.join(out_vis_dir,"profile comparisons", f"{person_id}_comparison_vis.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("[INFO] Saved comparison visual ->", out_path)
    return out_path

def process_profile_comparisons(person_id, input_person_folder, deca_out_root,
                               sess, input_name, output_name,
                               csv_out_path, vis_out_dir):

    """Generate Multi-PIE profile comparison figures: original frontal on the left, 
    corresponding profile, UV texture, and frontalized outputs on the right, with similarity scores."""
    os.makedirs(vis_out_dir, exist_ok=True)

    # 1) Locate the reference frontal input image
    frontal_input_path = find_reference_input(input_person_folder)
    if frontal_input_path is None:
        print("[ERROR] No frontal input found (_frontal or 051).")
        return []

    img_frontal = cv2.imread(frontal_input_path)
    emb_frontal = compute_embedding_onnx(sess, input_name, output_name, img_frontal)

    # 2) Remaining input images are considered profile images
    all_inputs = sorted([
        os.path.join(input_person_folder, f)
        for f in os.listdir(input_person_folder)
        if f.lower().endswith(('.png', '.jpg'))
    ])

    profile_inputs = [p for p in all_inputs if p != frontal_input_path]

    results = []

    # 3) Iterate over all DECA-generated folders for this person
    person_folders = [
        f for f in os.listdir(deca_out_root)
        if f.startswith(person_id) and os.path.isdir(os.path.join(deca_out_root, f))
    ]

    for folder in person_folders:
        print(f"----------------------------------------------")
        print(f"[INFO] Processing folder: {folder}")
        angle_folder = os.path.join(deca_out_root, folder)
        uv_path, front_path = get_uv_and_frontal_paths_multipie(angle_folder)

        # Skip folders without both UV and frontalized outputs
        if uv_path is None or front_path is None:
            print("[WARN] Missing UV or front in", angle_folder)
            continue

        img_uv = cv2.imread(uv_path)
        img_front = cv2.imread(front_path)

        # Compute ArcFace embeddings
        emb_uv = compute_embedding_onnx(sess, input_name, output_name, img_uv)
        emb_front = compute_embedding_onnx(sess, input_name, output_name, img_front)

        # 4) Compute similarity scores (equivalent to CFP)
        sim_uv = cosine_sim(emb_frontal, emb_uv)           # frontal_input vs UV texture
        sim_frontalized = cosine_sim(emb_frontal, emb_front)  # frontal_input vs frontalized output

        # Attempt to find the original profile input that matches this angle (same camera id)
        matching_profile = None
        for p in profile_inputs:
            if folder in p:
                matching_profile = p
                break

        # If a matching profile exists, compute similarity
        if matching_profile:
            img_prof = cv2.imread(matching_profile)
            emb_prof = compute_embedding_onnx(sess, input_name, output_name, img_prof)
            sim_profile = cosine_sim(emb_frontal, emb_prof)
        else:
            sim_profile = float('nan')

        print(f"[SIM] Frontal vs Profile     : {sim_profile:.4f}")
        print(f"[SIM] Frontal vs UV Texture  : {sim_uv:.4f}")
        print(f"[SIM] Frontal vs Frontalized : {sim_frontalized:.4f}")

        # Store result row
        results.append({
            "person": person_id,
            "angle": folder,
            "sim_profile": sim_profile,
            "sim_uv": sim_uv,
            "sim_frontalized": sim_frontalized
        })

        # 5) Visualization similar to CFP benchmark
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        fig.subplots_adjust(top=2.0, hspace=0.3, wspace=0.15)

        # Display frontal input
        axes[0].imshow(cv2.cvtColor(img_frontal, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Frontal", fontsize=16, fontweight="bold")
        axes[0].axis("off")

        # Display matching profile or a placeholder
        if matching_profile:
            axes[1].imshow(cv2.cvtColor(img_prof, cv2.COLOR_BGR2RGB))
            axes[1].set_title("Original Profile", fontsize=16, fontweight="bold")
            axes[1].axis("off")
        else:
            axes[1].text(0.5, 0.5, "NO PROFILE", fontsize=16, fontweight="bold", ha="center")
            axes[1].axis("off")

        # UV texture image
        axes[2].imshow(cv2.cvtColor(img_uv, cv2.COLOR_BGR2RGB))
        axes[2].set_title("UV Texture", fontsize=16, fontweight="bold")
        axes[2].axis("off")

        # Frontalized output
        axes[3].imshow(cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB))
        axes[3].set_title("Frontalized", fontsize=16, fontweight="bold")
        axes[3].axis("off")

        # Display similarity scores under the figure
        score_text = (
            f"Frontal vs Profile: {sim_profile:.4f} | "
            f"Frontal vs UV: {sim_uv:.4f} | "
            f"Frontal vs Frontlized: {sim_frontalized:.4f}"
        )
        plt.figtext(0.5, 0.02, score_text, ha="center", fontsize=16)

        out_png = os.path.join(vis_out_dir, f"{folder}.png")
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(out_png, dpi=150)
        plt.close()

    # 6) Save results to CSV
    os.makedirs(os.path.dirname(csv_out_path), exist_ok=True)
    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["person","angle","sim_profile","sim_uv","sim_frontalized"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"----------------------------------------------")
    print("[INFO] Multi-PIE Profile Comparisons CSV saved:", csv_out_path)
    return results

# -------------------- Main pipeline --------------------
def pipeline(multipie_root=MULTIPIE_ROOT, base_input_out=BASE_INPUT_OUT,
             deca_out_root=DECA_OUT_ROOT, vis_root=VIS_ROOT, csv_out=CSV_OUT,
             onnx_model_path=ONNX_MODEL_LOCAL):

    # 1) Select a random Multi-PIE subject and copy one image per camera
    person_id, input_person_folder = select_random_subject_and_copy(multipie_root, base_input_out)

    # 2) Run DECA inference if the environment supports it
    run_deca_if_available(input_person_folder, deca_out_root)

    # 3) Load or download ArcFace ONNX model
    if not os.path.exists(onnx_model_path):
        try:
            download_arcface()  # try downloading if not present
        except Exception as e:
            print(f"[ERROR] Could not download ONNX model: {e}")
            return
    # Initialize ONNX runtime session
    sess, input_name, output_name = get_onnx_session(onnx_model_path)

    # 4) Read the 051_frontal reference input for similarity comparisons
    ref_input_path = find_reference_input(input_person_folder)
    if ref_input_path is None:
        print("[ERROR] Reference frontal (051) not found in input folder:", input_person_folder)
        return
    ref_img = cv2.imread(ref_input_path)
    if ref_img is None:
        print("[ERROR] Failed to read reference image:", ref_input_path)
        return
    # Compute ArcFace embedding of the reference image
    emb_ref_input = compute_embedding_onnx(sess, input_name, output_name, ref_img)

    # 5) Compare all other input images to the reference image
    input_images = sorted([
        os.path.join(input_person_folder, f) 
        for f in os.listdir(input_person_folder)
        if f.lower().endswith(('.png', '.jpg')) and os.path.join(input_person_folder, f) != ref_input_path
    ])

    sim_rows = []         # rows for the main CSV
    sims_for_vis = {}     # to generate per-person visualization

    for in_img_path in input_images:
        img = cv2.imread(in_img_path)
        if img is None:
            print("[WARN] failed read", in_img_path)
            continue
        try:
            emb = compute_embedding_onnx(sess, input_name, output_name, img)
            sim = cosine_sim(emb_ref_input, emb)  # similarity against reference
        except Exception as e:
            print(f"[ERROR] Onnx embedding failed for {in_img_path}: {e}")
            sim = None

        sims_for_vis[os.path.basename(in_img_path)] = sim if sim is not None else float('nan')

        # Record input-image similarity to reference
        sim_rows.append({
            "person": person_id,
            "type": "input",
            "file": os.path.basename(in_img_path),
            "reference": os.path.basename(ref_input_path),
            "similarity": f"{sim:.6f}" if sim is not None else ""
        })

    # 6) Compare DECA outputs (UV/frontalized) to the input reference
    person_folders = [
        f for f in os.listdir(deca_out_root)
        if f.startswith(person_id) and os.path.isdir(os.path.join(deca_out_root, f))
    ]

    if person_folders:
        for folder_name in person_folders:
            deca_person_folder = os.path.join(deca_out_root, folder_name)

            # Locate the UV and frontalized results DECA produced
            uv_ref, frontal_ref = get_deca_paths_for_person(deca_person_folder)

            if uv_ref and frontal_ref:
                img_uv_ref = cv2.imread(uv_ref)
                img_front_ref = cv2.imread(frontal_ref)

                # Compute embeddings for DECA reference outputs
                emb_uv_ref, emb_front_ref = None, None
                if img_uv_ref is not None:
                    try:
                        emb_uv_ref = compute_embedding_onnx(sess, input_name, output_name, img_uv_ref)
                    except Exception as e:
                        print("[WARN] emb_uv_ref failed:", e)

                if img_front_ref is not None:
                    try:
                        emb_front_ref = compute_embedding_onnx(sess, input_name, output_name, img_front_ref)
                    except Exception as e:
                        print("[WARN] emb_front_ref failed:", e)

                # Iterate through all DECA PNG outputs for that person
                deca_pngs = [p for p in os.listdir(deca_person_folder) if p.lower().endswith('.png')]
                for p in deca_pngs:
                    p_path = os.path.join(deca_person_folder, p)

                    # Skip the reference DECA files already used
                    if p_path == uv_ref or p_path == frontal_ref:
                        continue

                    img = cv2.imread(p_path)
                    if img is None:
                        continue

                    # Compute embedding and compare with UV and frontal references
                    try:
                        emb = compute_embedding_onnx(sess, input_name, output_name, img)

                        # UV similarity
                        if emb_uv_ref is not None:
                            sim_uv = cosine_sim(emb_uv_ref, emb)
                            sim_rows.append({
                                "person": person_id,
                                "type": "deca_uv",
                                "file": p,
                                "reference": os.path.basename(uv_ref),
                                "similarity": f"{sim_uv:.6f}"
                            })
                            print(f"[SIM][uv] {p} vs UV_REF -> {sim_uv:.4f}")

                        # Frontal similarity
                        if emb_front_ref is not None:
                            sim_front = cosine_sim(emb_front_ref, emb)
                            sim_rows.append({
                                "person": person_id,
                                "type": "deca_front",
                                "file": p,
                                "reference": os.path.basename(frontal_ref),
                                "similarity": f"{sim_front:.6f}"
                            })
                            print(f"[SIM][front] {p} vs FRONT_REF -> {sim_front:.4f}")

                    except Exception as e:
                        print(f"[WARN] ONNX embedding failed for deca file {p}: {e}")

    else:
        # If no DECA folder exists (DECA disabled or failed)
        print("[INFO] DECA output folder not found for person -> skipping deca output comparisons:", deca_person_folder)

    # 7) Multi-PIE: run CFP-style profile comparisons for the selected person
    print("\n[INFO] Running Multi-PIE Profile Comparisons evaluation...")
    cfp_csv = os.path.join(DECA_OUT_ROOT, "csv", "multipie_comparisons.csv")
    process_profile_comparisons(
        person_id,
        input_person_folder,
        deca_out_root,
        sess, input_name, output_name,
        cfp_csv,
        os.path.join(vis_root, person_id, "profile comparisons")
    )

    # 8) Multi-PIE: pairwise evaluation (every angle folder vs 051_frontal)
    print("\n[INFO] Running Multi-PIE pairwise UV/Front comparison...")
    pair_rows = process_multipie_pairwise(
        person_id, deca_out_root, sess, input_name, output_name, vis_root
    )

    # Save pairwise CSV
    pair_csv = os.path.join(deca_out_root, "csv", "multipie_pairwise.csv")
    os.makedirs(os.path.dirname(pair_csv), exist_ok=True)
    with open(pair_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["person", "entry", "reference", "uv_similarity", "front_similarity"])
        writer.writeheader()
        for r in pair_rows:
            writer.writerow(r)

    print("[INFO] Pairwise CSV saved:", pair_csv)

    # 9) Final combined CSV of (inputs + DECA)
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    with open(csv_out, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["person", "type", "file", "reference", "similarity"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only for new file
        if os.stat(csv_out).st_size == 0:
            writer.writeheader()

        for row in sim_rows:
            writer.writerow(row)

    print("[INFO] CSV updated ->", csv_out)

    # 10) Create a visual comparison of all input images vs reference
    vis_person_dir = os.path.join(vis_root, person_id)
    os.makedirs(vis_person_dir, exist_ok=True)

    save_comparison_vis(
        vis_person_dir, person_id, ref_input_path, input_images, sims_for_vis
    )

    # Per-person extra CSV (not required but helpful)
    per_person_csv = os.path.join(deca_out_root, "csv", f"{person_id}_multipie_profile_comparison.csv")

    print("[✅] Pipeline finished for person:", person_id)
    return True

# -------------------- Entrypoint --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-PIE -> DECA -> ArcFace(ONNX) similarity pipeline")
    parser.add_argument("--multipie_root", default=MULTIPIE_ROOT)
    parser.add_argument("--in_data", default=BASE_INPUT_OUT)
    parser.add_argument("--deca_out", default=DECA_OUT_ROOT)
    parser.add_argument("--onnx", default=ONNX_MODEL_LOCAL, help="Path to ArcFace ONNX model")
    args = parser.parse_args()
    # override config from args if needed
    MULTIPIE_ROOT = args.multipie_root
    BASE_INPUT_OUT = args.in_data
    DECA_OUT_ROOT = args.deca_out
    ONNX_MODEL_LOCAL = args.onnx

    pipeline(MULTIPIE_ROOT, BASE_INPUT_OUT, DECA_OUT_ROOT, VIS_ROOT, CSV_OUT, ONNX_MODEL_LOCAL)