import os
import torch
import argparse
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from networks import Generator, DualDiscriminator
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
from img_2_tex import mesh_angle, tex_correction

# optional ArcFace ONNX
ONNX_MODEL_URL = "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-11-int8/resolve/main/arcfaceresnet100-11-int8.onnx"
ONNX_MODEL_LOCAL = "arcfaceresnet100-int8.onnx"
try:
    import onnxruntime as ort
except Exception:
    ort = None
    print("[WARNING] onnxruntime not available; IdentityLoss disabled.")


def train_single_uv(img_name, input_dir, out_dir="results", iters=500, uv_size=512):
    """Train symmetry GAN on a single image UV."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(img_name))[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- DECA init ---
    deca = setup_deca(device)
    if deca is None:
        raise RuntimeError("DECA not available.")

    # --- Load cropped face ---
    img_cropped, item = load_deca_cropped(img_name, input_dir, device=device)

    # --- Run DECA ---
    uv_tex, vertices, uv_texture, uv_face_eye_mask = run_deca_on_image(
        deca, img_cropped, device=device
    )

    # --- Pose / healthy side estimation ---
    verts_np = vertices[0].detach().cpu().numpy()
    angle1 = mesh_angle(verts_np, [3572,3555,2205])
    angle2 = mesh_angle(verts_np, [3572,723,3555])
    avg_ang = int(90 - (360 - (angle1 + angle2) / 2))
    print(f"\n{base}: 📐avg_angle = {avg_ang}°")
    healthy_side = "left" if avg_ang > 0 else "right"
    print(f"[INFO] Healthy side: {healthy_side.capitalize()}\n")

    # --- Prepare UV tensors ---
    uv_tex_np = uv_tex[0].permute(1,2,0).cpu()
    uv_raw_tensor = uv_tex_np.clone()
    uv_corrected_tensor, _ = tex_correction(uv_raw_tensor, avg_ang)

    resize = transforms.Resize((uv_size, uv_size))
    uv_raw = resize(uv_raw_tensor.permute(2,0,1)).unsqueeze(0).to(device)
    uv_target = resize(uv_corrected_tensor.permute(2,0,1)).unsqueeze(0).to(device)

    # --- Face mask ---
    _, _, H, W = uv_target.shape
    face_mask = get_uv_face_mask_tensor(H, W, device)

    # --- Save debug UVs ---
    Image.fromarray((uv_raw[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)) \
        .save(os.path.join(out_dir, f"uv_unwrapped_{base}.png"))
    Image.fromarray((uv_target[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)) \
        .save(os.path.join(out_dir, f"uv_target_{base}.png"))

    # --- Normalize to [-1,1] ---
    uv_raw_norm = uv_raw * 2 - 1
    uv_target_norm = uv_target * 2 - 1

    # --- Generator input ---
    uv_flip = torch.flip(uv_raw, dims=[3])
    inp = torch.cat([uv_raw, face_mask, uv_flip], dim=1)

    # --- Split healthy / damaged halves ---
    x_uv_target, x_z_target = get_splits_simple(uv_target_norm, avg_ang)
    x_uv_img = (x_uv_target[0].permute(1,2,0).cpu().numpy() * 0.5 + 0.5)  # [-1,1] -> [0,1]
    x_z_img = (x_z_target[0].permute(1,2,0).cpu().numpy() * 0.5 + 0.5)
    Image.fromarray((x_uv_img*255).astype(np.uint8)).save(os.path.join(out_dir, f"x_uv_target_{base}.png"))
    Image.fromarray((x_z_img*255).astype(np.uint8)).save(os.path.join(out_dir, f"x_z_target_{base}.png"))

    # --- Models ---
    G = Generator(in_ch=7, base=64).to(device)
    D = DualDiscriminator(in_ch=3, base=64).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))

    L1 = nn.L1Loss()
    L_VGG = VGGPerceptualLoss(device=str(device)).to(device)
    L_ID = IdentityLoss(device=str(device), onnx_path=ONNX_MODEL_LOCAL).to(device)

    # --- Masks ---
    mid = uv_raw.shape[-1] // 2
    mask_z = compute_mask_z(uv_raw, healthy_side=healthy_side, mask_threshold=30)
    mask_z = mask_z.to(face_mask.device) * face_mask
    mask_uv = torch.clamp(face_mask - mask_z, 0, 1)

    # --- Save mask ---
    Image.fromarray((mask_z[0,0].cpu().numpy()*255).astype(np.uint8)) \
        .save(os.path.join(out_dir, f"mask_z_{base}.png"))

    # --- Loss weights ---
    WEIGHT_ADV  = 0.007
    WEIGHT_SYM  = 2.0
    WEIGHT_ID   = 0.01
    WEIGHT_SEAM = 1.0

    # --- Training loop ---
    for i in range(iters):

        # ===== Generator =====
        opt_g.zero_grad()

        out = G(inp)
        final_uv = uv_target_norm * mask_uv + out * mask_z

        # Reconstruction
        L_rec_z  = (torch.abs(out - uv_target_norm) * mask_z).sum() / (mask_z.sum()+1e-8)
        L_rec_uv = (torch.abs(final_uv - uv_target_norm) * mask_uv).sum() / (mask_uv.sum()+1e-8)
        L_rec = L_rec_z + L_rec_uv

        # Symmetry (pixel)
        if healthy_side == "left":
            x_z_out = out[:,:,:,mid:]
        else:
            x_z_out = out[:,:,:,:mid]
        L_sym = L1(x_z_out, torch.flip(x_uv_target, dims=[3]))

        # Symmetry (gradient)
        L_grad_sym = gradient_symmetry_loss(
            out, uv_target_norm, mask_z, mid, healthy_side=healthy_side
        )

        # Adversarial
        d_fake_g, d_fake_l = D(x_z_out)
        L_adv = hinge_g_loss(d_fake_g)
        if d_fake_l is not None:
            L_adv += hinge_g_loss(d_fake_l)

        # Perceptual + ID
        final_scaled = (final_uv + 1) / 2
        target_scaled = (uv_target_norm + 1) / 2
        L_vgg = L_VGG(final_scaled, target_scaled)
        L_id  = L_ID(final_scaled, target_scaled)

        # Seam loss
        seam_w = seam_feathering(out, left_right_boundary=min(120, mid))
        L_seam = (torch.abs(out - uv_target_norm) * seam_w * mask_z).sum() / (seam_w.sum()+1e-8)

        # Total G loss
        L_g = (
            L_rec
            + WEIGHT_ADV * L_adv
            + 0.03 * L_vgg
            + WEIGHT_SYM * L_sym
            + 1.5 * L_grad_sym
            + WEIGHT_ID * L_id
            + WEIGHT_SEAM * L_seam
        )
        L_g.backward()
        opt_g.step()

        # ===== Discriminator =====
        opt_d.zero_grad()
        d_real_g, d_real_l = D(x_uv_target)
        d_fake_g, d_fake_l = D(x_z_out.detach())

        L_d = hinge_d_loss(d_real_g, d_fake_g)
        if d_real_l is not None and d_fake_l is not None:
            L_d += hinge_d_loss(d_real_l, d_fake_l)

        L_d.backward()
        opt_d.step()

        # --- Logging ---
        if i % 50 == 0 or i == iters - 1:
            print(f"[{i}/{iters}] L_rec={L_rec:.4f} L_adv={L_adv:.4f} "
                  f"L_sym={L_sym:.4f} L_seam={L_seam:.4f} L_d={L_d:.4f}")
            save = final_scaled[0].cpu().clamp(0,1)
            pil = transforms.ToPILImage()(save)
            pil.save(os.path.join(out_dir,f"iter_{i:4d}_{base}.png"))

        # --- Final save ---
        final_uv_clamped = final_uv.detach().clamp(0,1)
        final_img = (final_uv_clamped + 1) / 2
        final_np = final_img[0].permute(1,2,0).cpu().numpy()
        Image.fromarray((final_np*255).astype(np.uint8)).save(os.path.join(out_dir, f"uv_complete_{base}.png"))
        print(f"✅ Saved UV Complete: uv_complete_{base}.png")

        with torch.no_grad():
           # encode 
           enc_input = transforms.Resize(224)(img_cropped)
           codedict = deca.encode(enc_input)
           codedict['images'] = img_cropped

           # decode  
           opdict, visdict = deca.decode(codedict, name=base)

           # UV replacement
           opdict['uv_texture_gt'] = final_uv_clamped

           # save OBJ + frontal render
           obj_path = os.path.join(out_dir, f"{base}.obj")
           deca.save_obj(obj_path, opdict, codedict)
           print(f"✅ Saved OBJ + frontal render: {obj_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--uv_size", type=int, default=512)

    args = parser.parse_args()
    base = os.path.splitext(os.path.basename(args.img))[0]
    if args.out_dir is None:
        args.out_dir = f"{base}_train_uv_results"

    train_single_uv(
        img_name=args.img,
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        iters=args.iters,
        uv_size=args.uv_size
    )
