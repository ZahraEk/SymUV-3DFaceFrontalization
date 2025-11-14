## Copyright © 2023 Human Sensing Lab @ Carnegie Mellon University ##

import os
import torchvision
import torch
from tqdm import tqdm
import math
import numpy as np
import cv2
from skimage import exposure
from decalib.utils.config import cfg as deca_cfg

run_this_file = 0

if run_this_file == 1:
    # from decalib.utils.renderer import SRenderY, set_rasterizer
    from decalib.datasets import datasets 
    # from decalib.utils import util
    from decalib.utils.config import cfg as deca_cfg
    from decalib.deca import DECA

    image_size = 1024
    topology_path = '/content/Towards-Realistic-Generative-3D-Face-Models/data/head_template.obj'
    uv_size = 1024
    rasterizer_type = 'pytorch3d'
    device = 'cuda'
    savefolder = '/content/Towards-Realistic-Generative-3D-Face-Models/inference_test/out_data/uv_tex/'
    inputpath = '/content/Towards-Realistic-Generative-3D-Face-Models/inference_test/in_data/'
    iscrop = True
    detector = 'fan'
    sample_step = 1
    useTex = True
    extractTex = True

    os.makedirs(savefolder, exist_ok=True)

    # Load test images
    testdata = datasets.TestData(inputpath, iscrop=iscrop, face_detector=detector, sample_step=sample_step, crop_size=1024)

    # Initialize DECA
    deca_cfg.model.use_tex = useTex
    deca_cfg.rasterizer_type = rasterizer_type
    deca_cfg.model.extract_tex = extractTex
    deca = DECA(config=deca_cfg, device=device)

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def get_normal(p1, p2, p3):
    return np.cross(p2-p1, p3-p1)

def mesh_angle(vertices, vertex_ids):
    normal = get_normal(np.array(vertices[vertex_ids[0]]), 
                        np.array(vertices[vertex_ids[1]]), 
                        np.array(vertices[vertex_ids[2]]))
    ang = int(angle(normal, [1,0,1])*360/math.pi)
    return ang

def remove_specular_highlights(uv_np):
    gray = cv2.cvtColor(uv_np, cv2.COLOR_BGR2GRAY)
    # Detect very bright pixels (specular highlights)
    _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
    # Smooth the bright regions
    uv_np_clean = cv2.inpaint(uv_np, mask, 5, cv2.INPAINT_TELEA)
    return uv_np_clean
def apply_face_neck_correction(uv_texture_np, mask_path, blend_ratio=0.5):
    """
    Applies color correction (matching) from the face area (white in mask) 
    to the neck/surrounding areas (black in mask) using a modified UV mask.

    Parameters:
    uv_texture_np (numpy.ndarray): The UV texture (H, W, 3) as uint8 (0-255).
    mask_path (str): Path to the modified mask file (e.g., "modified_uv_face_neck_mask.png").
    blend_ratio (float): Blending ratio (0 to 1) for the correction in the surrounding areas.

    Returns:
    numpy.ndarray: The color-corrected UV texture.
    """
    # --- (1) Load and prepare the mask ---
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask file not found at {mask_path}")

    if uv_texture_np.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (uv_texture_np.shape[1], uv_texture_np.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Ensure that the gray area (≈120) is excluded from the mean computation.
    # We assume: white = face (>128), black = 0, gray = 120.
    face_mask_for_mean = (mask > 128).astype(np.uint8)

    # --- (2) Compute the mean color of the face (white mask area) ---
    # Extract face pixels using the new mask
    face_pixels = uv_texture_np[face_mask_for_mean == 1]

    if face_pixels.size == 0:
        print("Warning: Face area in mask is empty. Returning original texture.")
        return uv_texture_np

    # Compute mean color for each channel (R, G, B)
    mean_face_color = np.mean(face_pixels, axis=0).astype(np.float32)

    # --- (3) Create mask for neck/border areas (to be corrected) ---
    protected_mask = (mask > 1).astype(np.float32)  # everything except pure black (0)

    # This soft blur ensures smooth blending at the face/neck border.
    protected_blurred = cv2.GaussianBlur(protected_mask, (35, 35), 0)

    # Correction mask: the region where the correction should be applied.
    correction_mask = 1.0 - protected_blurred

    # Apply blending ratio to the mask
    final_blend_mask = correction_mask[..., None] * blend_ratio

    # --- (4) Apply color correction using NumPy array operations ---
    uv_float = uv_texture_np.astype(np.float32)
    mean_color_image = np.full_like(uv_float, mean_face_color)

    uv_corrected = (mean_color_image * final_blend_mask) + \
                   (uv_float * (1.0 - final_blend_mask))

    # Return the corrected output
    uv_corrected = np.clip(uv_corrected, 0, 255).astype(np.uint8)

    return uv_corrected

def central_merge_nose(img, blend_width=256, center_half=40, bias_left=0.65, nose_y_range=(0.35, 0.7)):
    """
    Symmetric central merge restricted only to the nose area.
    The blending (and central blur) will be applied vertically between
    y = h * nose_y_range[0] and y = h * nose_y_range[1].
    """
    h, w, c = img.shape
    mid_x = w // 2
    blended = img.copy().astype(np.float32)

    # --- Create mask for the nose region ---
    y1 = int(h * nose_y_range[0])
    y2 = int(h * nose_y_range[1])
    nose_mask = np.zeros((h, w), np.uint8)
    nose_mask[y1:y2, :] = 1  # Active only inside the defined nose band

    # --- Part 1: Fill the central gap (helps reconstruct the nose center) ---
    left_strip = blended[y1:y2, mid_x - center_half * 2:mid_x, :]
    right_strip = blended[y1:y2, mid_x:mid_x + center_half * 2, :]

    center_fill = bias_left * np.flip(left_strip, axis=1) + (1 - bias_left) * np.flip(right_strip, axis=1)
    center_fill_uint8 = np.clip(center_fill, 0, 255).astype(np.uint8)

    # Apply Gaussian blur to soften the hard boundary
    blurred_fill = cv2.GaussianBlur(center_fill_uint8, (7, 7), 0).astype(np.float32)

    # Insert blurred central patch
    blended[y1:y2, mid_x - center_half:mid_x + center_half, :] = blurred_fill[:, :center_half * 2, :]

    # --- Part 2: Symmetric blending, restricted to the nose area only ---
    blend_width = int(blend_width)
    arr = np.linspace(0, 1, blend_width).astype(np.float32)
    arr_flip = 1 - arr
    arr = arr[None, :, None]       # For broadcasting
    arr_flip = arr_flip[None, :, None]

    # Left side blend (use mirrored right)
    left_area_to_blend = blended[y1:y2, mid_x - blend_width:mid_x, :]
    mirror_source_right = blended[y1:y2, mid_x:mid_x + blend_width, :]
    mirror_right_flipped = np.flip(mirror_source_right, axis=1)

    blended[y1:y2, mid_x - blend_width:mid_x, :] = (
        left_area_to_blend * arr_flip + mirror_right_flipped * arr
    )

    # Right side blend (use mirrored left)
    right_area_to_blend = blended[y1:y2, mid_x:mid_x + blend_width, :]
    mirror_source_left = blended[y1:y2, mid_x - blend_width:mid_x, :]
    mirror_left_flipped = np.flip(mirror_source_left, axis=1)

    blended[y1:y2, mid_x:mid_x + blend_width, :] = (
        right_area_to_blend * arr_flip + mirror_left_flipped * arr
    )

    # --- Final combination: apply the effect only in the nose region ---
    mask_3c = nose_mask[..., None].astype(np.float32)
    out = blended * mask_3c + img.astype(np.float32) * (1 - mask_3c)
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out


def central_merge_fullheight(img, blend_width=256, center_half=40, bias_left=0.65):
    """
    Full-height symmetric merge with a central blurred fill region.
    """
    h, w, c = img.shape
    mid_x = w // 2
    blended = img.copy().astype(np.float32)

    # 1. Central Fill Calculation (using blur to avoid hard seams)
    # This part must remain to ensure the central vertical gap is filled smoothly.
    left_strip  = blended[:, mid_x - center_half*2:mid_x, :]
    right_strip = blended[:, mid_x:mid_x + center_half*2, :]

    center_fill = bias_left * np.flip(left_strip, axis=1) + (1 - bias_left) * np.flip(right_strip, axis=1)

    # Apply Gaussian blur to soften the boundaries of the central strip
    center_fill_uint8 = np.clip(center_fill, 0, 255).astype(np.uint8)
    blurred_fill = cv2.GaussianBlur(center_fill_uint8, (7, 7), 0).astype(np.float32)

    # Replace central band with blurred content
    blended[:, mid_x - center_half:mid_x + center_half, :] = blurred_fill[:, :center_half*2, :]

    # -----------------------------------------------------------
    # 2. Symmetric blending using weighted gradients
    # This mimics the logic used in tex_merge: original texture × (1 - w) + mirrored texture × w
    blend_width = int(blend_width)

    # Create gradient weight arrays
    arr = np.linspace(0, 1, blend_width).astype(np.float32)
    arr_flip = 1 - arr
    arr = arr[None, :, None]
    arr_flip = arr_flip[None, :, None]

    # --- Left region blending using mirrored right region ---
    left_area_to_blend  = blended[:, mid_x - blend_width:mid_x, :]
    mirror_source_right = blended[:, mid_x:mid_x + blend_width, :]
    mirror_right_flipped = np.flip(mirror_source_right, axis=1)

    blended[:, mid_x - blend_width:mid_x, :] = (
        left_area_to_blend * arr_flip +
        mirror_right_flipped * arr
    )

    # --- Right region blending using mirrored left region ---
    right_area_to_blend = blended[:, mid_x:mid_x + blend_width, :]
    mirror_source_left  = blended[:, mid_x - blend_width:mid_x, :]
    mirror_left_flipped = np.flip(mirror_source_left, axis=1)

    blended[:, mid_x:mid_x + blend_width, :] = (
        right_area_to_blend * arr_flip +
        mirror_left_flipped * arr
    )

    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended

def tex_correction(uv_texture, angle,
                   mask_color_correction="/content/Towards-Realistic-Generative-3D-Face-Models/data/modified_uv_face_eye_mask.png",
                   mask_inpaint_nasal="/content/Towards-Realistic-Generative-3D-Face-Models/data/uv_face_neck_mask.png",
                   gradient_ratio=0.2, center_band=150,
                   auto_gamma=True):
    """
    UV texture correction with histogram alignment, symmetry-based repair,
    soft blending, neck–face color adjustment, and adaptive gamma correction.

    Returns:
        uv_orig : original texture in numpy format
        uv_out  : corrected texture in torch tensor format
    """

    # Convert input texture to uint8 and remove specular artifacts.
    device = uv_texture.device
    uv_orig = (uv_texture.detach().cpu().numpy() * 255).astype(np.uint8)
    uv_orig = remove_specular_highlights(uv_orig)
    uv_np = uv_orig
    h, w, _ = uv_np.shape
    mid_x = w // 2

    # Select the healthy half using head rotation direction.
    if angle < 0:
        healthy_side, mask_side = "right", "left"
    else:
        healthy_side, mask_side = "left", "right"

    # Apply histogram matching to standardize photometric properties.
    left_half, right_half = uv_np[:, :mid_x], uv_np[:, mid_x:]
    if healthy_side == "left":
        right_half = exposure.match_histograms(right_half, left_half, channel_axis=-1)
    else:
        left_half = exposure.match_histograms(left_half, right_half, channel_axis=-1)
    uv_np = np.concatenate([left_half, right_half], axis=1).astype(np.uint8)

    # Create a reliability mask from mirrored differences and add
    # a gradient transition near the midline.
    if mask_side == "right":
        diff = cv2.absdiff(cv2.flip(uv_np[:, :mid_x], 1), uv_np[:, mid_x:])
    else:
        diff = cv2.absdiff(cv2.flip(uv_np[:, mid_x:], 1), uv_np[:, :mid_x])
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask_half = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    mask_half = mask_half.astype(np.float32) / 255.0

    full_mask = np.zeros((h, w), dtype=np.float32)
    gradient_width = int(w * gradient_ratio)
    if mask_side == "right":
        full_mask[:, mid_x:] = mask_half
        grad = np.linspace(0, 1, gradient_width)[None, :]
        full_mask[:, mid_x - gradient_width:mid_x] = mask_half[:, :gradient_width] * grad
    else:
        full_mask[:, :mid_x] = mask_half
        grad = np.linspace(1, 0, gradient_width)[None, :]
        full_mask[:, mid_x:mid_x + gradient_width] = mask_half[:, -gradient_width:] * grad

    # Smooth the mask to achieve seamless blending.
    full_mask[:, mid_x - center_band:mid_x + center_band] = 1.0
    blurred_mask = cv2.GaussianBlur(full_mask, (35, 35), 0)

    # Build a mirrored reference from the healthy side.
    reference = uv_np.copy()
    if healthy_side == "left":
        reference[:, mid_x:] = cv2.flip(uv_np[:, :mid_x], 1)
    else:
        reference[:, :mid_x] = cv2.flip(uv_np[:, mid_x:], 1)

    # Blend original and reference textures according to the smoothed mask.
    blended = reference.astype(np.float32) * blurred_mask[..., None] + \
              uv_np.astype(np.float32) * (1 - blurred_mask[..., None])
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Adjust neck coloration to match facial chromaticity.
    blended = apply_face_neck_correction(blended, mask_color_correction, blend_ratio=0.5)

    # Apply log-based gamma correction to improve tonal balance.
    if auto_gamma:
        hsv = cv2.cvtColor(blended, cv2.COLOR_RGB2HSV)
        V = hsv[:, :, 2].astype(np.float32) / 255.0
        V_safe = np.clip(V, 1e-4, 1.0)
        mean_lin = np.mean(V_safe)
        mean_log = np.mean(np.log(V_safe))
        gamma = np.clip(np.log(mean_lin) / mean_log, 0.6, 2.4)
        blended = np.power(blended / 255.0, 1.0 / gamma)
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    #blended = central_merge_nose(blended, blend_width=120, center_half=40, bias_left=0.50, nose_y_range=(0.20, 0.80))
    #blended = central_merge_fullheight(blended, blend_width=512, center_half=24, bias_left=0.99)

    # Return the corrected texture as a torch tensor.
    uv_out = torch.from_numpy(blended.astype(np.float32) / 255.0).to(device)

    return uv_out, uv_orig

def tex_correction_eye(uv_texture, angle):
    if angle < 0:
        max_pixel = 512
        eye = 1
        arr = np.array(range(max_pixel))/max_pixel
        arr_flip = np.flip(arr, 0)
        uv_texture[:,:max_pixel,:] = torch.flip(uv_texture, (1,))[:,:max_pixel,:]
        uv_texture[:200,:200,:] = eye
    else:
        max_pixel = -512
        eye = uv_texture[:200,-200:,:].clone()
        arr = np.array(range(abs(max_pixel)))/abs(max_pixel)
        arr_flip = np.flip(arr, 0)
        uv_texture[:,max_pixel:,:] = torch.flip(uv_texture, (1,))[:,max_pixel:,:]
        uv_texture[:200,-200:,:] = eye
    return uv_texture

def tex_merge(uv_texture_r, uv_texture_c, uv_texture_l):
    max_pixel = 512
    arr = np.linspace(0, 1, max_pixel)          
    arr_flip = 1 - arr                         
    uv_texture_c[200:,:max_pixel,:] = (
        uv_texture_l[200:,:max_pixel,:] * arr_flip[None,...,None] +
        uv_texture_c[200:,:max_pixel,:] * arr[None,...,None]
    )

    max_pixel = -512
    arr = np.linspace(0, 1, abs(max_pixel))     
    arr_flip = 1 - arr                         
    uv_texture_c[200:,max_pixel:,:] = (
        uv_texture_r[200:,max_pixel:,:] * arr[None,...,None] +
        uv_texture_c[200:,max_pixel:,:] * arr_flip[None,...,None]
    )

    return uv_texture_c

def get_tex_from_img(images, get_cropped_img, deca):
    textures = torch.zeros_like(images).to('cuda')
    count=0

    for img in images:
        data_list = get_cropped_img.__getitem__(img*255)
        img_cropped = data_list['image'].to('cuda')[None,...]

        with torch.no_grad():
            codedict = deca.encode(torchvision.transforms.Resize(224)(img_cropped))
            codedict['images'] = img_cropped
            uv_tex, vertices, uv_face_eye_mask, uv_texture = deca.decode_tex(codedict)

            angle1 = mesh_angle(vertices[0].detach().cpu().numpy(), [3572,3555,2205])
            angle2 = mesh_angle(vertices[0].detach().cpu().numpy(), [3572,723,3555])
            avg_ang = int((angle1+angle2)/2)
            avg_ang = 90-(360-avg_ang)

            corrected_tex, orig_tex = tex_correction(uv_tex[0].permute(1,2,0).detach().cpu(), avg_ang)
 
            correct_tex = corrected_tex.permute(2,0,1)[None,...].to('cuda')
            correct_tex = correct_tex[:,:3,:,:]*uv_face_eye_mask + (uv_texture[:,:3,:,:]*(1-uv_face_eye_mask))
            textures[count] = correct_tex
            count+=1

    return textures

def main():
    # ========== PRECHECK ==========
    try:
        files = os.listdir(inputpath)
        print(f"[INFO] files in inputpath: {files[:50]}")
    except Exception as e:
        print(f"[WARN] cannot list inputpath: {e}")
        files = []

    print(f"[INFO] testdata length: {len(testdata)}")
    os.makedirs(savefolder, exist_ok=True)

    if len(testdata) == 0:
        print("[ERROR] testdata is empty. Make sure inputpath contains images and that datasets.TestData can read them.")
        return

    for i in tqdm(range(len(testdata))):
        try:
            item = testdata[i]
            name = item.get('imagename', f"sample_{i}")
            images = item.get('image', None)
            if images is None:
                print(f"[WARN] item {i} has no 'image' key, skipping.")
                continue

            images = images.to(device)[None, ...]
            with torch.no_grad():
                # Encode and decode
                codedict = deca.encode(torchvision.transforms.Resize(224)(images))
                codedict['images'] = images

                # Get all outputs, ignore extra values
                uv_tex, vertices, *_ = deca.decode_tex(codedict)

                # Compute angle
                angle1 = mesh_angle(vertices[0].detach().cpu().numpy(), [3572,3555,2205])
                angle2 = mesh_angle(vertices[0].detach().cpu().numpy(), [3572,723,3555])
                avg_ang = int((angle1 + angle2) / 2)
                avg_ang = 90 - (360 - avg_ang)
                print(f"\n{'-'*40}\n{name}: 📐avg_angle = {avg_ang}°")

                # Correct UV texture
                corrected_tex, orig_tex = tex_correction(uv_tex[0].permute(1,2,0).detach().cpu(), avg_ang)

                # Create folder for the image
                base_folder = os.path.join(savefolder, name)
                os.makedirs(base_folder, exist_ok=True)

                # Save images in the folder
                cv2.imwrite(os.path.join(base_folder, name + "_orig.png"), cv2.cvtColor(orig_tex, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(base_folder, name + "_corrected.png"), cv2.cvtColor((corrected_tex.cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                print(f"✅SAVED {name} → original and corrected inside {base_folder}")

        except Exception as e:
            print(f"[ERROR] processing item {i} ({name}): {e}")
            continue

if run_this_file == 1:
    main()
