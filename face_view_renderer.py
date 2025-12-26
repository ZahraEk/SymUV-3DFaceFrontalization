import os
import cv2
import torch
import numpy as np
import imageio
from pathlib import Path
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, HardPhongShader, BlendParams, look_at_view_transform
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------- Defaults ----------
FOV_DEG = 20.0
PAD = 0.85
PAD_TOP = 0.95
IMG_SIZE_DEFAULT = 1024

# ---------- Renderer ----------
def make_renderer(center, radius, azim_deg, *, elev_deg=0.0, roll_deg=0.0, image_size=IMG_SIZE_DEFAULT):
    dist = (radius / np.tan(np.deg2rad(FOV_DEG * 0.5))) * PAD

    if center.ndim == 1:
        center_adjusted = center.clone()[None, :]
    elif center.ndim == 2 and center.shape[0] == 1 and center.shape[1] == 3:
        center_adjusted = center.clone()
    else:
        raise ValueError(f"Unsupported center shape: {center.shape}")

    center_adjusted[:, 1] += radius * (PAD_TOP - 1.0)

    R, T = look_at_view_transform(dist=dist, elev=elev_deg, azim=azim_deg, device=device, at=center_adjusted)

    if roll_deg != 0.0:
        roll_rad = np.deg2rad(roll_deg)
        R_roll = torch.tensor([
            [np.cos(roll_rad), -np.sin(roll_rad), 0.0],
            [np.sin(roll_rad),  np.cos(roll_rad), 0.0],
            [0.0,               0.0,              1.0]
        ], dtype=torch.float32, device=device)[None]
        R = torch.bmm(R_roll, R)

    cams = FoVPerspectiveCameras(device=device, R=R, T=T, fov=FOV_DEG)
    raster_settings = RasterizationSettings(
        image_size=int(image_size),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        max_faces_per_bin=80000
    )
    blend_params = BlendParams(gamma=1.0, background_color=(1.0, 1.0, 1.0))
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cams, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cams, blend_params=blend_params)
    )

# ---------- Render helpers ----------
def render_rgb(mesh, center, radius, azim_deg, *, roll_deg=0.0, out_size=IMG_SIZE_DEFAULT, elev_deg=0.0):
    renderer = make_renderer(center, radius, azim_deg, elev_deg=elev_deg, roll_deg=roll_deg, image_size=out_size)
    img = renderer(mesh)[0, ..., :3].detach().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def symmetry_score(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    c = w // 2
    w_crop = int(w * 0.6)
    x0, x1 = max(0, c - w_crop // 2), min(w, c + w_crop // 2)
    roi = gray[:, x0:x1]
    left = roi[:, :roi.shape[1] // 2]
    right_flipped = cv2.flip(roi[:, roi.shape[1] // 2:], 1)
    m = min(left.shape[1], right_flipped.shape[1])
    return float(np.mean(np.abs(left[:, :m].astype(np.float32) - right_flipped[:, :m].astype(np.float32))))

# ---------- Mesh loader ----------
def load_mesh_center_radius(mesh_path):
    mesh = load_objs_as_meshes([mesh_path], device=device)
    V = mesh.verts_packed()
    center = V.mean(0, keepdim=True)  # shape [1,3]
    radius = torch.linalg.norm(V - center, dim=1).max().item()
    return mesh, center, radius

# ---------- Frontal angles ----------
def find_frontal_angles(mesh, center, radius,
                        max_yaw=90, max_roll=20,
                        coarse_step_yaw=3.0, coarse_step_roll=5.0,
                        fine_window_yaw=8.0, fine_step_yaw=0.5,
                        fine_window_roll=5.0, fine_step_roll=1.0):

    yaw_candidates = np.arange(-max_yaw, max_yaw + 1e-6, coarse_step_yaw)
    scores = []
    for y in yaw_candidates:
        img = render_rgb(mesh, center, radius, y, out_size=256)
        score = symmetry_score(img)
        scores.append(score)
    best_yaw = float(yaw_candidates[int(np.argmin(scores))])

    # Fine yaw
    yaw_candidates = np.arange(best_yaw - fine_window_yaw, best_yaw + fine_window_yaw + 1e-6, fine_step_yaw)
    scores = []
    for y in yaw_candidates:
        img = render_rgb(mesh, center, radius, y, out_size=320)
        score = symmetry_score(img)
        scores.append(score)
    best_yaw = float(yaw_candidates[int(np.argmin(scores))])

    # Roll search
    roll_candidates = np.arange(-max_roll, max_roll + 1e-6, coarse_step_roll)
    scores = []
    for r in roll_candidates:
        img = render_rgb(mesh, center, radius, best_yaw, roll_deg=r, out_size=320)
        score = symmetry_score(img)
        scores.append(score)
    best_roll = float(roll_candidates[int(np.argmin(scores))])

    roll_candidates = np.arange(best_roll - fine_window_roll, best_roll + fine_window_roll + 1e-6, fine_step_roll)
    scores = []
    for r in roll_candidates:
        img = render_rgb(mesh, center, radius, best_yaw, roll_deg=r, out_size=400)
        score = symmetry_score(img)
        scores.append(score)
    best_roll = float(roll_candidates[int(np.argmin(scores))])

    return best_yaw, best_roll

# ---------- Image/GIF helpers ----------
def save_frontal_image(mesh_path, frontal_path):
    mesh, center, radius = load_mesh_center_radius(mesh_path)
    yaw, roll = find_frontal_angles(mesh, center, radius)
    img = render_rgb(mesh, center, radius, yaw, roll_deg=roll, out_size=IMG_SIZE_DEFAULT)
    os.makedirs(os.path.dirname(frontal_path), exist_ok=True)
    cv2.imwrite(frontal_path, img)
    print(f"Best pose -> yaw: {yaw:.2f}, roll: {roll:.2f}")
    return frontal_path, yaw, roll

def save_rotation_gif(mesh_path, out_gif, n_frames=30, fps=15, delta_yaw=45.0):
    mesh, center, radius = load_mesh_center_radius(mesh_path)
    yaw, roll = find_frontal_angles(mesh, center, radius)
    os.makedirs(os.path.dirname(out_gif), exist_ok=True)

    forward = np.linspace(yaw - delta_yaw, yaw + delta_yaw, n_frames)
    backward = np.linspace(yaw + delta_yaw, yaw - delta_yaw, n_frames)[1:-1]
    angles = np.concatenate([forward, backward])

    frames = []
    for a in angles:
        img = render_rgb(mesh, center, radius, a, roll_deg=roll, out_size=IMG_SIZE_DEFAULT)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    imageio.mimsave(out_gif, frames, fps=fps)
    print(f"✅Rotation GIF saved at: {out_gif}")
    return out_gif

def save_frontal_and_side_images(mesh_path, out_dir, side_yaw=30.0):
    mesh, center, radius = load_mesh_center_radius(mesh_path)
    yaw, roll = find_frontal_angles(mesh, center, radius)
    os.makedirs(out_dir, exist_ok=True)
    base_name = Path(mesh_path).stem

    frontal_path = os.path.join(out_dir, f"{base_name}_frontal.png")
    cv2.imwrite(frontal_path, render_rgb(mesh, center, radius, yaw, roll_deg=roll, out_size=IMG_SIZE_DEFAULT))

    left_path = os.path.join(out_dir, f"{base_name}_left.png")
    cv2.imwrite(left_path, render_rgb(mesh, center, radius, yaw - side_yaw, roll_deg=roll, out_size=IMG_SIZE_DEFAULT))

    right_path = os.path.join(out_dir, f"{base_name}_right.png")
    cv2.imwrite(right_path, render_rgb(mesh, center, radius, yaw + side_yaw, roll_deg=roll, out_size=IMG_SIZE_DEFAULT))
  
    print(f"Best pose -> yaw: {yaw:.2f}, roll: {roll:.2f}")
    return frontal_path, left_path, right_path, yaw, roll

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Render frontal, side views and GIF from 3D face mesh"
    )
    parser.add_argument("--mesh", type=str, required=True, help="Path to OBJ mesh file")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--n_frames", type=int, default=30)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--side_yaw", type=float, default=30.0, help="Yaw offset for side views")
    parser.add_argument("--delta_yaw", type=float, default=45.0, help="Yaw range for rotation GIF (±degrees)")

    args = parser.parse_args()

    print("=== Render frontal + side views + GIF from 3D face mesh ===")
    os.makedirs(args.out_dir, exist_ok=True)
    base_name = Path(args.mesh).stem

    # ---------- Frontal + Side images ----------
    frontal_path, left_path, right_path, yaw, roll = save_frontal_and_side_images(
        args.mesh, args.out_dir, side_yaw=args.side_yaw
    )
    print("Rendering frontal and side views...")
    print(f"✅Frontal view: {frontal_path}")
    print(f"✅Left view: {left_path}")
    print(f"✅Right view: {right_path}")

    # ---------- Rotation GIF ----------
    print("Rendering rotation GIF...")
    gif_path = os.path.join(args.out_dir, f"{base_name}_rotation.gif")
    save_rotation_gif(
        args.mesh, gif_path, n_frames=args.n_frames, fps=args.fps, delta_yaw=args.delta_yaw
    )
    print("=== Rendering finished ===")


