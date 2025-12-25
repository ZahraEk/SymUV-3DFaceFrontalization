import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn as nn
from PIL import Image
from skimage import exposure
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur 

import sys
sys.path.append("/content/Towards-Realistic-Generative-3D-Face-Models/uv_symmetry_gan")

from networks import Generator, DualDiscriminator, Discriminator  
from loss_functions import (
    IdentityLoss,
    VGGPerceptualLoss,
    hinge_d_loss,
    hinge_g_loss,
    total_variation_loss,
    gradient_symmetry_loss
)
from uv_utils import *
from deca_utils import *
from img_2_tex import mesh_angle, tex_correction, apply_face_neck_correction

# ArcFace ONNX model for identity preservation
ONNX_MODEL_LOCAL = "arcfaceresnet100-int8.onnx"

def train_single_uv(img_name, input_dir, out_dir="results", iters=500, uv_size=512, auto_gamma=True):
    """
    Train a UV completion GAN using:
      - Explicit UV target (pose-corrected)
      - Symmetry constraints
      - Dual UV discriminator
      - Image-space adversarial supervision via DECA rendering
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(img_name))[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =============== DECA: 3D Face Reconstruction & UV Unwrapping ===============
    deca = setup_deca(device)
    img_cropped, arcface_inp, _ = load_deca_cropped(img_name, input_dir, device=device, use_mica=True)

    # Extract UV texture and mesh vertices
    uv_tex, vertices, _, _ = run_deca_on_image(deca, img_cropped, arcface_inp=arcface_inp, device=device)

    # ================= Pose =================
    # Head Pose Estimation → Healthy Side Detection
    verts_np = vertices[0].detach().cpu().numpy()
    a1 = mesh_angle(verts_np, [3572,3555,2205])
    a2 = mesh_angle(verts_np, [3572,723,3555])
    avg_ang = int(90 - (360 - (a1 + a2) / 2))

    # Define which facial half is considered healthy
    healthy_side = "left" if avg_ang > 0 else "right"
    print(f"\n[INFO] {base}: avg_angle={avg_ang}°, Healthy side: {healthy_side}\n")

    # ================= UV Preparation (raw + target) =================
    uv_raw_tensor = uv_tex[0].permute(1,2,0).cpu()

    # Pose-aware UV target correction (pseudo-GT)
    uv_target_tensor, _ = tex_correction(uv_raw_tensor, avg_ang)

    resize = transforms.Resize((uv_size, uv_size))
    uv_raw = resize(uv_raw_tensor.permute(2,0,1)).unsqueeze(0).to(device)
    uv_target = resize(uv_target_tensor.permute(2,0,1)).unsqueeze(0).to(device)

    # Normalize UVs to [-1, 1]
    uv_raw_norm = uv_raw * 2 - 1
    uv_target_norm = uv_target * 2 - 1

    # Save UVs for debugging / qualitative inspection
    Image.fromarray((uv_raw[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)) \
        .save(os.path.join(out_dir, f"uv_unwrapped_{base}.png"))
    Image.fromarray((uv_target[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)) \
        .save(os.path.join(out_dir, f"uv_target_{base}.png"))

    # ================= UV Masks (Face / Damaged Region) =================
    _, _, H, W = uv_raw.shape
    face_mask = get_uv_face_mask_tensor(H, W, device)
    mid = W // 2

    # Damage mask inferred via symmetry difference
    mask_z = compute_mask_z(uv_raw, healthy_side, mask_threshold=15)
    mask_z = mask_z.to(face_mask.device) * face_mask
    #mask_z = add_seam_band(mask_z, band_width=80, falloff=20)
    mask_z = mask_z * face_mask

    # Softened version for smoother blending
    mask_z_soft = gaussian_blur(mask_z, kernel_size=31, sigma=8)

    # Mask of preserved (healthy) UV region
    mask_uv = torch.clamp(face_mask - mask_z, 0, 1)

    # Save masks 
    Image.fromarray((mask_z[0,0].cpu().numpy()*255).astype(np.uint8)) \
        .save(os.path.join(out_dir, f"mask_z_{base}.png"))

    # ================= Generator Input Construction =================
    # Input channels: [UV_raw, FaceMask, UV_flipped]
    uv_flip = torch.flip(uv_raw, dims=[3])
    inp = torch.cat([uv_raw, face_mask, uv_flip], dim=1)

    # Split UV into healthy (x_uv) and damaged (x_z)
    x_uv_raw, x_z_raw = get_splits_simple(uv_raw_norm, avg_ang)

    # Save split halves for sanity check
    x_uv_img = (x_uv_raw[0].permute(1,2,0).cpu().numpy() * 0.5 + 0.5)  # [-1,1] -> [0,1]
    x_z_img = (x_z_raw[0].permute(1,2,0).cpu().numpy() * 0.5 + 0.5)
    uv_flip_img = uv_flip[0].permute(1,2,0).cpu().numpy()

    Image.fromarray((x_uv_img*255).astype(np.uint8)).save(os.path.join(out_dir, f"x_uv_{base}.png"))
    Image.fromarray((x_z_img*255).astype(np.uint8)).save(os.path.join(out_dir, f"x_z_{base}.png"))
    Image.fromarray((uv_flip_img*255).astype(np.uint8)).save(os.path.join(out_dir, f"uv_flip_{base}.png"))
    
    # ================= Models =================
    G = Generator(in_ch=7, base=64).to(device)

    # UV-space dual discriminator (global + seam-local)
    D_uv = DualDiscriminator(in_ch=3, base=64).to(device)

    # Image-space discriminator (rendered RGB)
    D_img = Discriminator(in_ch=3, base=64).to(device) 

    opt_g = torch.optim.Adam(G.parameters(), 2e-4, (0.5,0.999))
    opt_d_uv = torch.optim.Adam(D_uv.parameters(), 2e-4, (0.5,0.999))
    opt_d_img = torch.optim.Adam(D_img.parameters(), 2e-4, (0.5,0.999)) 

    # ================= Loss Functions =================
    L1 = nn.L1Loss()
    L_VGG = VGGPerceptualLoss(device=str(device)).to(device)
    L_ID = IdentityLoss(device=str(device), onnx_path=ONNX_MODEL_LOCAL).to(device)

    # ================= Encode Once (DECA Latents) =================
    enc_input = transforms.Resize(224)(img_cropped)
    codedict = deca.encode(enc_input,arcface_inp=arcface_inp)
    codedict['images'] = img_cropped

    # ================= Loss Weights =================
    WEIGHT_ADV_UV = 0.008
    WEIGHT_RENDER_ADV = 0.01       
    WEIGHT_RENDER_REC = 1.0
    WEIGHT_RENDER_VGG = 0.5
    WEIGHT_RENDER_ID = 0.08
    WEIGHT_SYM = 2.0
    WEIGHT_SEAM = 1.0
    WARMUP = 50

    # ================= Training Loop =================
    for i in range(iters):

        # ---------- Generator ----------
        opt_g.zero_grad()

        # Predict full UV
        out = G(inp)

        # Blend output only into damaged region
        final_uv = uv_raw_norm * (1 - mask_z_soft) + out * mask_z_soft 

        # ---------- Render via DECA ----------
        codedict['uv_texture_gt'] = final_uv
        opdict, _ = deca.decode(codedict, name=base)

        render_img = opdict['rendered_images']
        face_mask_img = opdict['alpha_images']

        img_gt = torch.nn.functional.interpolate(img_cropped, render_img.shape[-2:], mode="bilinear", align_corners=False)

        # ---------- Render-space losses ----------
        L_render_L1 = (torch.abs(render_img - img_gt) * face_mask_img).sum() / (face_mask_img.sum()+1e-8)
        L_render_VGG = L_VGG(render_img*face_mask_img, img_gt*face_mask_img)
        L_render_rec = L_render_L1 + WEIGHT_RENDER_VGG * L_render_VGG
        L_render_ID = L_ID(render_img*face_mask_img, img_gt*face_mask_img)

        # ---------- Image GAN (Generator) ----------
        if i > WARMUP:  
            pred_fake_img = D_img(render_img * face_mask_img)
            L_adv_img = hinge_g_loss(pred_fake_img)
        else:
            L_adv_img = torch.tensor(0.0, device=device)

        # ---------- UV-space losses ----------
        L_rec_uv = (torch.abs(final_uv - uv_raw_norm) * mask_uv).sum() / (mask_uv.sum()+1e-8)
        L_rec_z = (torch.abs(out - uv_target_norm) * mask_z).sum() / (mask_z.sum()+1e-8)

        lambda_z = max(0.1 * (1 - i / iters), 0.03) 
        L_rec = L_rec_uv + lambda_z * L_rec_z 

        # Extract damaged half output
        x_z_out = out[:, :, :, mid:] if healthy_side == "left" else out[:, :, :, :mid]

        # Symmetry loss
        L_sym = L1(x_z_out, torch.flip(x_uv_raw, dims=[3]))

        # UV adversarial loss
        d_fake_g, _ = D_uv(x_z_out)
        L_adv_uv = hinge_g_loss(d_fake_g)

        # Seam smoothing loss
        seam_w = seam_feathering(final_uv, left_right_boundary=min(120,mid))  
        #L_seam = (torch.abs(final_uv - uv_target_norm) * seam_w).sum() / (seam_w.sum() + 1e-8)
        L_seam = (torch.abs(final_uv - uv_raw_norm) * seam_w * mask_z).sum() / (seam_w.sum() + 1e-8)

        # ---------- Total Generator Loss ----------
        L_g = (
            #L_rec
            L_rec_uv
            + WEIGHT_ADV_UV * L_adv_uv
            + WEIGHT_RENDER_ADV * L_adv_img  
            + WEIGHT_RENDER_REC * L_render_rec
            + WEIGHT_RENDER_ID * L_render_ID
            + WEIGHT_SYM * L_sym
            + WEIGHT_SEAM * L_seam
        )

        L_g.backward()
        opt_g.step()

        # ---------- UV Discriminator ----------
        opt_d_uv.zero_grad()
        d_real, _ = D_uv(x_uv_raw)
        d_fake, _ = D_uv(x_z_out.detach())
        if i > WARMUP:
            L_d_uv = hinge_d_loss(d_real, d_fake)
            L_d_uv.backward()
            opt_d_uv.step()

        # ---------- Image Discriminator ----------
        if i > WARMUP:  
            opt_d_img.zero_grad()
            pred_real = D_img(img_gt * face_mask_img)
            pred_fake = D_img(render_img.detach() * face_mask_img)
            L_d_img = hinge_d_loss(pred_real, pred_fake)
            L_d_img.backward()
            opt_d_img.step()

        # ---------- Logging  ----------
        final_scaled = (final_uv + 1) / 2
        if i % 50 == 0 or i == iters - 1:
            print(f"\n[{i}/{iters}]: L_g={L_g:.4f}, L_rec_uv={L_rec_uv:.4f}, L_adv_uv={L_adv_uv:.4f}, L_adv_img ={L_adv_img :.4f}\n"
                  f"L_render_rec={L_render_rec:.4f}, L_render_ID={L_render_ID:.4f}, L_sym={L_sym:.4f}, L_seam={L_seam:.4f}\n")

            save = final_scaled[0].cpu().clamp(0,1)
            pil = transforms.ToPILImage()(save)
            pil.save(os.path.join(out_dir,f"iter_{i:4d}_{base}.png"))

    # ---------- Save completed UV ----------
    final_img = final_scaled[0].permute(1,2,0).detach().cpu().clamp(0,1).numpy()
    final_img = (final_img*255).astype(np.uint8)
    raw_uv_path = os.path.join(out_dir, f"uv_complete_{base}.png")
    Image.fromarray(final_img).save(raw_uv_path)

    # ---------- Post-Processing: Face-Neck Correction ----------
    FACE_NECK_MASK_PATH = "/content/Towards-Realistic-Generative-3D-Face-Models/data/uv_face_neck_mask.png"  
    corrected_uv = apply_face_neck_correction(uv_texture_np=final_img, mask_path=FACE_NECK_MASK_PATH, blend_ratio=0.4)

    corrected_path = os.path.join(out_dir, f"uv_complete_neck_correction_{base}.png")
    Image.fromarray(corrected_uv).save(corrected_path)

    # ---------- Post-Processing: Auto-Gamma ----------
    uv_post = corrected_uv.copy()
    if auto_gamma:
       hsv = cv2.cvtColor(uv_post, cv2.COLOR_RGB2HSV)
       V = hsv[:,:,2].astype(np.float32)/255.0
       V_safe = np.clip(V, 1e-4, 1.0)
       mean_lin = np.mean(V_safe)
       mean_log = np.mean(np.log(V_safe))
       gamma = np.clip(np.log(mean_lin)/mean_log, 0.6, 2.4)
       uv_post = np.power(uv_post/255.0, 1.0/gamma)
       uv_post = np.clip(uv_post*255,0,255).astype(np.uint8)

    post_path = os.path.join(out_dir, f"uv_complete_gamma_correction_{base}.png")
    Image.fromarray(uv_post).save(post_path)

    # ================= Save Final OBJ + Frontal Render =================
    with torch.no_grad():

      # decode 
      opdict, visdict = deca.decode(codedict , name=base)

      # UV (only upscale if needed)
      uv_corr_tensor = torch.from_numpy(uv_post).float() / 255.0
      uv_corr_tensor = uv_corr_tensor.permute(2,0,1).unsqueeze(0).to(device)
      if uv_corr_tensor.shape[-1] != 1024:
          uv_corr_tensor = torch.nn.functional.interpolate(uv_corr_tensor, (1024,1024), mode="bicubic", align_corners=False)
         
      opdict['uv_texture_gt'] = uv_corr_tensor

      obj_path = os.path.join(out_dir, f"{base}.obj")
      deca.save_obj(obj_path, opdict, codedict)

    # ================= Final Print =================
    print(f"\n✅[TRAINING COMPLETE] UV completion for '{base}' finished.")
    print(f"✅All outputs (UVs, masks, final images, OBJ) are saved in: '{out_dir}'\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--img", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--uv_size", type=int, default=512)
    args = parser.parse_args()

    base = os.path.splitext(os.path.basename(args.img))[0]
    out_dir = args.out_dir or f"{base}_train_uv_results"

    train_single_uv(args.img, args.input_dir, out_dir, args.iters, args.uv_size, auto_gamma=True)
