import cv2
import torch
import numpy as np

# ---------------- Load UV face mask ----------------
uv_mask_path = "/content/Towards-Realistic-Generative-3D-Face-Models/data/uv_mask.png"
uv_face_mask_np = cv2.imread(uv_mask_path, cv2.IMREAD_GRAYSCALE)

if uv_face_mask_np is None:
    raise FileNotFoundError(f"UV face mask not found: {uv_mask_path}")

uv_face_mask_np = (uv_face_mask_np > 128).astype(np.float32)

# ---------------- Mask utilities ----------------
def get_uv_face_mask_tensor(h, w, device):
    """Return resized UV face mask tensor [1,1,H,W]."""
    mask = cv2.resize(uv_face_mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    return mask.to(device)

def compute_mask_z(uv_img_tensor, healthy_side="left", mask_threshold=30):
    """Compute binary mask for damaged half of UV texture."""
    uv_np = (uv_img_tensor[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    h, w, _ = uv_np.shape
    mid_x = w // 2

    left_half = uv_np[:, :mid_x, :]
    right_half = uv_np[:, mid_x:, :]

    if healthy_side == "left":
        left_flipped = cv2.flip(left_half, 1)
        diff = cv2.absdiff(left_flipped, right_half)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, mask_threshold, 255, cv2.THRESH_BINARY)
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[:, mid_x:] = mask
    else:
        right_flipped = cv2.flip(right_half, 1)
        diff = cv2.absdiff(right_flipped, left_half)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, mask_threshold, 255, cv2.THRESH_BINARY)
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[:, :mid_x] = mask

    # Convert to tensor and interpolate to input size
    mask_tensor = torch.tensor(full_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)/255.0
    mask_tensor = torch.nn.functional.interpolate(mask_tensor, size=(uv_img_tensor.shape[2], uv_img_tensor.shape[3]), mode='nearest')
    return mask_tensor

# ---------------- Inpainting ----------------
def inpaint_nasal_region_uv_tensor(uv_tensor, face_mask_tensor):
    """
    Inpaint nasal region of UV texture using OpenCV.
    Returns tensor with prefilled pixels.
    """
    uv = (uv_tensor[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    face_mask = (face_mask_tensor[0,0].cpu().numpy() * 255).astype(np.uint8)

    # HSV-based rough defect detection
    hsv = cv2.cvtColor(uv, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    mask_hsv = ((V < 80) | (V > 250) | (S < 80)).astype(np.uint8)

    # Local high-variance regions
    blur = cv2.GaussianBlur(uv, (5,5), 0)
    diff = np.abs(uv.astype(np.float32) - blur.astype(np.float32))
    diff_gray = np.mean(diff, axis=2)
    thr = np.percentile(diff_gray, 99)
    mask_local = (diff_gray > thr).astype(np.uint8) & (S < 120)
    defect_mask = cv2.bitwise_or(mask_hsv, mask_local)
    defect_mask = cv2.bitwise_and(defect_mask, face_mask)

    # Narrow band along nose
    h, w = defect_mask.shape
    cx = w // 2
    bw = int(w * 0.1)
    band = np.zeros_like(defect_mask)
    band[:, cx-bw:cx+bw] = 1
    defect_mask = cv2.bitwise_and(defect_mask, band)

    # Morphological closing to clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel)

    if defect_mask.sum() < 50:
        return uv_tensor  # nothing to inpaint

    uv_bgr = cv2.cvtColor(uv, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(uv_bgr, (defect_mask*255).astype(np.uint8), 3, cv2.INPAINT_NS)
    uv_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

    uv_out = torch.from_numpy(uv_rgb).float()/255.0
    uv_out = uv_out.permute(2,0,1).unsqueeze(0).to(uv_tensor.device)
    return uv_out

# ---------------- Geometry / Split ----------------
def get_splits_simple(I_out, angle):
    """Return left/right splits of UV tensor for batch size 1."""
    B,C,H,W = I_out.shape
    assert B == 1, "Batch size must be 1"
    mid = W // 2
    left = I_out[:, :, :, :mid]
    right = I_out[:, :, :, mid:]
    return (left, right) if angle > 0 else (right, left)

def seam_feathering(out, left_right_boundary=300):
    """Return blending weights for seam region."""
    B,C,H,W = out.shape
    mid = W//2
    feather = torch.linspace(0,1,steps=left_right_boundary, device=out.device).unsqueeze(0).unsqueeze(0).unsqueeze(2)
    weights = torch.ones((B,1,H,W), device=out.device)
    for i in range(left_right_boundary):
        w = 1.0 - feather[:,:,:,i]
        idx = mid-left_right_boundary+i
        if 0 <= idx < W:
            weights[:,:,:,idx] = w
    return weights
