#!/usr/bin/env python3
"""
uv_symmetry_gan.py (DECA-compatible updated)

Usage:
    python uv_symmetry_gan.py --input_dir /path/to/images --img face1.jpg --iters 500 --uv_size 512 --out_dir results

Notes:
- Uses DECA's TestData cropping (face_detector='fan', crop_size=1024) the same way as img_2_tex.py does.
- Expects `img_2_tex.mesh_angle` and `img_2_tex.tex_correction` to be available (we import from img_2_tex.py).
- Optional: decalib (DECA), torchvision models, onnxruntime for IdentityLoss; script falls back when missing.
- Designed for option B: input is a folder; `--img` is the file name within that folder to process.
"""
import os
import subprocess
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.nn.utils import spectral_norm

# local helper file (should be uploaded by user)
try:
    from img_2_tex import mesh_angle, tex_correction
except Exception as e:
    raise RuntimeError("Required module 'img_2_tex' not found or failed to import: {}".format(e))

# decalib / DECA optional
try:
    from decalib.utils.config import cfg as deca_cfg
    from decalib.deca import DECA
    from decalib.datasets import datasets
    import torchvision
    from torchvision import models
except Exception:
    DECA = None
    deca_cfg = None
    datasets = None
    torchvision = None
    models = None
    print("[WARNING] DECA / decalib / torchvision not available. DECA-related functionality will be skipped.")

# onnx runtime optional
ONNX_MODEL_URL = "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-11-int8/resolve/main/arcfaceresnet100-11-int8.onnx"
ONNX_MODEL_LOCAL = "arcfaceresnet100-int8.onnx"
try:
    import onnxruntime as ort
except Exception:
    ort = None
    print("[WARNING] onnxruntime not available; IdentityLoss disabled.")


# ----------------------- Helpers ---------------------------------
def pil_to_tensor(img: Image.Image, size=None):
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    t = transforms.ToTensor()(img)  # C,H,W in [0,1]
    return t

def setup_deca(device):
    if DECA is None:
        return None
    try:
        # ensure texture extraction consistent with img_2_tex.py
        deca_cfg.model.use_tex = True
        deca_cfg.rasterizer_type = 'pytorch3d'
        deca_cfg.model.extract_tex = True
        deca_cfg.model.iscrop = True
        deca = DECA(config=deca_cfg, device=device)
        return deca
    except Exception as e:
        print(f"[FATAL] Could not initialize DECA. Error: {e}")
        return None

def find_item_in_testdata(testdata, target_name):
    target_base = os.path.splitext(os.path.basename(target_name))[0].lower()

    for i in range(len(testdata)):
        try:
            it = testdata[i]
            name = it.get('imagename', "")
            name_base = os.path.splitext(os.path.basename(name))[0].lower()

            # Exact match
            if name == target_name:
                return it

            # Base-name match
            if name_base == target_base:
                return it

            # Loose match (contains)
            if target_base in name_base:
                return it
        except:
            continue

    return None

def load_deca_cropped(img_name, input_dir, device='cuda'):
    """
    Create TestData over input_dir and return the cropped image tensor for img_name.
    """
    if datasets is None:
        raise RuntimeError("decalib.datasets not available; cannot use DECA cropping.")

    # Construct TestData
    testdata = datasets.TestData(input_dir, iscrop=True, face_detector='fan', sample_step=1, crop_size=1024)
    item = find_item_in_testdata(testdata, img_name)
    if item is None:
        # try match by naive filename search
        # build a mapping of imagename -> index quickly
        for i in range(len(testdata)):
            try:
                cand = testdata[i]
                cand_name = cand.get('imagename', None)
                if cand_name is not None and os.path.basename(cand_name) == os.path.basename(img_name):
                    item = cand
                    break
            except Exception:
                continue
    if item is None:
        raise RuntimeError(f"Could not find '{img_name}' inside TestData built from '{input_dir}'. Ensure the file exists and TestData can read it.")
    # item['image'] is typically a HWC floating torch tensor scaled [0,1] but dataset may return CHW
    img_cropped = item['image'].to(device)[None, ...]  # add batch
    return img_cropped, item

def run_deca_on_image(deca, img_cropped, device='cuda'):
    """
    Encode (resize to 224) and decode using the full cropped image (1024).
    Returns uv_tex (B,C,H,W), vertices, uv_face_eye_mask, uv_texture (per-deca outputs).
    """
    with torch.no_grad():
        enc_input = torchvision.transforms.Resize(224)(img_cropped)
        codedict = deca.encode(enc_input)
        codedict['images'] = img_cropped
        # decode_tex in decalib often returns uv_tex, vertices, uv_face_eye_mask, uv_texture (depending on version)
        outputs = deca.decode_tex(codedict)
    # normalize expected outputs robustly
    if len(outputs) >= 4:
        uv_tex, vertices, uv_face_eye_mask, uv_texture = outputs[:4]
    elif len(outputs) == 2:
        uv_tex, vertices = outputs
        uv_face_eye_mask = None
        uv_texture = uv_tex
    else:
        # try to unpack first two and set fallbacks
        uv_tex = outputs[0]
        vertices = outputs[1] if len(outputs) > 1 else None
        uv_face_eye_mask = None
        uv_texture = uv_tex
    return uv_tex, vertices, uv_texture, uv_face_eye_mask

# ----------------------- ArcFace ONNX utils ----------------------
def download_arcface(url=ONNX_MODEL_URL, save_path=ONNX_MODEL_LOCAL):
    if os.path.exists(save_path):
        return save_path
    print("[INFO] Downloading ArcFace ONNX ...")
    subprocess.run(["wget", url, "-O", save_path, "--no-check-certificate"], check=True)
    return save_path

def get_onnx_session(model_path):
    if ort is None:
        raise RuntimeError("onnxruntime not available.")
    providers = ort.get_available_providers()
    use_cuda = 'CUDAExecutionProvider' in providers
    provider_list = [('CUDAExecutionProvider', {"device_id": 0}), 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
    sess = ort.InferenceSession(model_path, providers=provider_list)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"[INFO] ONNX session created. Providers: {sess.get_providers()}")
    return sess, input_name, output_name

# ----------------------- Loss modules ---------------------------
class IdentityLoss(nn.Module):
    def __init__(self, device='cpu', onnx_path=ONNX_MODEL_LOCAL):
        super().__init__()
        self.device = device
        self.sess = None
        self.input_name = None
        self.output_name = None
        self.onnx_path = onnx_path

        if ort is None:
            print("[WARNING] onnxruntime not installed; IdentityLoss will return 0.")
            return
        try:
            if not os.path.exists(self.onnx_path):
                download_arcface(save_path=self.onnx_path)
            self.sess, self.input_name, self.output_name = get_onnx_session(self.onnx_path)
        except Exception as e:
            print(f"[WARNING] Failed to prepare ArcFace ONNX session: {e}")
            self.sess = None

    def preprocess_for_arcface(self, tensor_batch):
        # tensor_batch: N x C x H x W with values in [0,1]
        bs = tensor_batch.shape[0]
        out_list = []
        to_pil = transforms.ToPILImage()
        for i in range(bs):
            t = tensor_batch[i].cpu().clamp(0,1)
            pil = to_pil(t)
            pil = pil.resize((112, 112), Image.BILINEAR)
            arr = np.asarray(pil).astype(np.float32)
            chw = np.transpose(arr, (2,0,1))
            chw = (chw - 127.5) / 128.0
            out_list.append(chw)
        return np.stack(out_list, axis=0).astype(np.float32)

    def forward(self, generated_uv, target_uv):
        if self.sess is None:
            return torch.tensor(0.0).to(generated_uv.device)
        def to_01(x):
            if x.min() < -0.5:
                return (x + 1.0) / 2.0
            return x
        gen = to_01(generated_uv)
        tgt = to_01(target_uv)
        gen_np = self.preprocess_for_arcface(gen)
        tgt_np = self.preprocess_for_arcface(tgt)
        try:
            emb_gen = self.sess.run([self.output_name], {self.input_name: gen_np})[0]
            emb_tgt = self.sess.run([self.output_name], {self.input_name: tgt_np})[0]
        except Exception as e:
            print(f"[WARNING] ONNX run failed: {e}")
            return torch.tensor(0.0).to(generated_uv.device)
        emb_gen_t = F.normalize(torch.from_numpy(emb_gen).float().to(generated_uv.device), dim=1)
        emb_tgt_t = F.normalize(torch.from_numpy(emb_tgt).float().to(generated_uv.device), dim=1)
        cos = F.cosine_similarity(emb_gen_t, emb_tgt_t, dim=1)
        return 1.0 - cos.mean()

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.model = None
        if models is not None:
            try:
                try:
                    vgg = models.vgg19(pretrained=True).features.to(device).eval()
                except Exception:
                    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
                self.model = nn.Sequential(*list(vgg.children())[:30])
                for p in self.model.parameters(): p.requires_grad=False
            except Exception as e:
                print(f"[WARNING] VGG model loading failed: {e}")
        if self.model is None:
            print("[WARNING] VGG Loss will return zero.")
        self.layer_indices = [2,7,12,21,30]

    def forward(self, x, y):
        if self.model is None:
            return torch.tensor(0.0).to(x.device)
        loss = 0.0
        current_x, current_y = x, y
        x_features, y_features = [], []
        for i, layer in enumerate(self.model):
            current_x = layer(current_x)
            current_y = layer(current_y)
            if i in self.layer_indices:
                x_features.append(current_x)
                y_features.append(current_y)
        for xf, yf in zip(x_features, y_features):
            loss += torch.mean(torch.abs(xf - yf))
        return loss

# ----------------------- Networks --------------------------------
class GatedConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.feat = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.gate = nn.Conv2d(in_ch, out_ch, k, s, p)
    def forward(self, x):
        return self.feat(x) * torch.sigmoid(self.gate(x))

class Generator(nn.Module):
    def __init__(self, in_ch=7, base=64):
        super().__init__()
        self.e1 = GatedConv(in_ch, base)
        self.e2 = GatedConv(base, base*2, s=2)
        self.e3 = GatedConv(base*2, base*4, s=2)
        self.e4 = GatedConv(base*4, base*8, s=2)
        self.e5 = GatedConv(base*8, base*8, s=2)
        self.bottleneck = GatedConv(base*8, base*8)
        self.d5 = GatedConv(base*8 + base*8, base*8)
        self.d4 = GatedConv(base*8 + base*4, base*4)
        self.d3 = GatedConv(base*4 + base*2, base*2)
        self.d2 = GatedConv(base*2 + base, base)
        self.d1 = GatedConv(base + base, base)
        self.out = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        b = self.bottleneck(e5)
        d5_up = F.interpolate(b, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d5 = self.d5(torch.cat([d5_up, e4], dim=1))
        d4_up = F.interpolate(d5, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.d4(torch.cat([d4_up, e3], dim=1))
        d3_up = F.interpolate(d4, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.d3(torch.cat([d3_up, e2], dim=1))
        d2_up = F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.d2(torch.cat([d2_up, e1], dim=1))
        d1 = self.d1(torch.cat([d2, e1], dim=1))
        return torch.tanh(self.out(d1))

class Discriminator(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        def block(ic, oc):
            return nn.Sequential(spectral_norm(nn.Conv2d(ic, oc, 4,2,1)), nn.LeakyReLU(0.2, True))
        self.model = nn.Sequential(
            block(in_ch, base),
            block(base, base*2),
            block(base*2, base*4),
            block(base*4, base*8),
            spectral_norm(nn.Conv2d(base*8,1,3,1,1))
        )
    def forward(self,x):
        return self.model(x)

# ----------------------- Masks / utils ---------------------------
def make_uv_mask(uv_np, th=0.07):
    gray = uv_np.mean(axis=2)
    return (gray > th).astype(np.float32)

def get_splits_simple(I_out, angle):
    """Return (x_uv, x_z) for batch size 1 only. Simpler, deterministic."""
    B,C,H,W = I_out.shape
    assert B == 1, "get_splits_simple expects batch size 1"
    mid = W // 2
    left = I_out[:, :, :, :mid]
    right = I_out[:, :, :, mid:]
    if angle > 0:
        return left, right
    else:
        return right, left

def seam_feathering(out, left_right_boundary=16):
    B,C,H,W = out.shape
    mid = W//2
    feather = torch.linspace(0,1,steps=left_right_boundary, device=out.device).unsqueeze(0).unsqueeze(0).unsqueeze(2)
    weights = torch.ones((B,1,H,W), device=out.device)
    for i in range(left_right_boundary):
        w = 1.0 - feather[:,:,:,i]
        idx = mid-left_right_boundary+i
        if idx >= 0 and idx < W:
            weights[:,:,:,idx] = w
    return weights

def total_variation_loss(x):
    dh = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:-1,:]))
    dw = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:-1]))
    return dh + dw

# ----------------------- Training --------------------------------
def train_single_uv(img_name, input_dir, out_dir="results", iters=500, uv_size=512):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(img_name))[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DECA init
    deca = setup_deca(device)
    if deca is None:
        raise RuntimeError("DECA is required for this pipeline but not available. Install decalib and its dependencies.")

    # Load cropped image via DECA TestData (ensures consistent cropping)
    print(f"[INFO] Loading '{img_name}' from '{input_dir}' using TestData cropper...")
    img_cropped, item = load_deca_cropped(img_name, input_dir, device=device)
    print(f"[INFO] img_cropped.shape = {tuple(img_cropped.shape)}")

    # Run DECA: encode(224) + decode_tex(full-res cropped)
    uv_tex, vertices, uv_texture, uv_face_eye_mask = run_deca_on_image(deca, img_cropped, device=device)

    # vertices -> numpy for mesh_angle
    verts_np = vertices[0].detach().cpu().numpy()
    angle1 = mesh_angle(verts_np, [3572, 3555, 2205])
    angle2 = mesh_angle(verts_np, [3572, 723, 3555])
    avg_ang = int(90 - (360 - (angle1 + angle2)/2))
    print(f"\n{base}: 📐avg_angle = {avg_ang}°")
    if avg_ang > 0:
        print("[INFO] The Left side is Healthy, The Right side is Unhealthy.\n")
    else:
        print("[INFO] The Right side is Healthy, The Left side is Unhealthy.\n")

    # uv_tex returned by deca.decode_tex is typically BxCxHxW in [0..1]
    # Convert to HxWxC torch/numpy as in img_2_tex.tex_correction expectation
    uv_tex_permuted = uv_tex[0].permute(1,2,0).detach().cpu()  # H,W,C (torch)
    uv_raw_tensor = uv_tex_permuted.clone()
    uv_corrected_tensor, _aux = tex_correction(uv_raw_tensor, avg_ang)

    # Resize to requested uv_size and move to device (back to BxCxHxW)
    to_resize = transforms.Resize((uv_size, uv_size))
    uv_raw = to_resize(uv_raw_tensor.permute(2,0,1)).unsqueeze(0).to(device)      # 1x3xHxW
    uv_target = to_resize(uv_corrected_tensor.permute(2,0,1)).unsqueeze(0).to(device)# 1x3xHxW

    # Save raw UV/target/mask (for debugging)
    uv_raw_np = uv_raw[0].permute(1,2,0).detach().cpu().numpy()
    Image.fromarray((uv_raw_np * 255).astype(np.uint8)).save(os.path.join(out_dir, f"uv_unwrapped_{base}.png"))
    uv_tar_np = uv_target[0].permute(1,2,0).detach().cpu().numpy()
    Image.fromarray((uv_tar_np * 255).astype(np.uint8)).save(os.path.join(out_dir, f"uv_target_{base}.png"))

    # Normalize to [-1,1] for network
    uv_raw_norm = uv_raw * 2 - 1
    uv_target_norm = uv_target * 2 - 1

    # Build mask from raw UV valid pixels
    uv_np = uv_raw[0].permute(1,2,0).cpu().numpy()
    mask_np = make_uv_mask(uv_np)
    mask = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0).to(device)  # 1x1xHxW
    Image.fromarray((mask_np * 255).astype(np.uint8)).save(os.path.join(out_dir, f"uv_mask_{base}.png"))

    # Construct input: [uv_raw, mask, uv_flip]
    uv_flip = torch.flip(uv_raw, dims=[3])
    inp = torch.cat([uv_raw, mask, uv_flip], dim=1).to(device)  # 1x7xHxW

    # Use deterministic simple splitter for batch=1
    x_uv_target, x_z_target = get_splits_simple(uv_target_norm, avg_ang)

    # Models
    G = Generator(in_ch=7, base=64).to(device)
    D = Discriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))

    L1_loss = nn.L1Loss()
    L_VGG = VGGPerceptualLoss(device=str(device)).to(device)
    L_ID = IdentityLoss(device=str(device), onnx_path=ONNX_MODEL_LOCAL).to(device)

    # Build mask_z based on angle but restrict to valid uv area
    B, C, H, W = uv_raw.shape
    mid = W // 2
    mask_z_np = np.zeros((H, W), dtype=np.float32)
    if avg_ang > 0:
        mask_z_np[:, mid:] = 1.0
    else:
        mask_z_np[:, :mid] = 1.0
    mask_z = torch.tensor(mask_z_np).unsqueeze(0).unsqueeze(0).to(device)
    mask_z = mask_z * mask
    mask_uv = mask * (1.0 - mask_z)

    # Loss weights (same as original)
    WEIGHT_ADV = 0.005
    WEIGHT_SYM = 2.0
    WEIGHT_ID = 0.001
    WEIGHT_SEAM = 1.0

    # Training loop
    for i in range(iters):
        # -------------- Generator update -----------------
        opt_g.zero_grad()
        out = G(inp)

        # Compose final UV (normalized in [-1,1])
        final_uv_norm = uv_target_norm * mask_uv + out * mask_z

        # L_rec_z: only on masked (unhealthy) region
        L_rec_z = (torch.abs(out - uv_target_norm) * mask_z).sum() / (mask_z.sum() + 1e-8)
        L_rec_uv = (torch.abs(final_uv_norm - uv_target_norm) * mask_uv).sum() / (mask_uv.sum() + 1e-8)
        L_rec = L_rec_z + L_rec_uv

        # Symmetry loss: compare generated unhealthy half with flipped healthy half
        if avg_ang > 0:
            x_uv_flip = torch.flip(x_uv_target, dims=[3])
            x_z_out_only = out[:, :, :, mid:]
        else:
            x_uv_flip = torch.flip(x_uv_target, dims=[3])
            x_z_out_only = out[:, :, :, :mid]
        try:
            L_sym = L1_loss(x_z_out_only, x_uv_flip)
        except Exception:
            L_sym = L1_loss(out * mask_z, torch.flip(uv_target_norm * mask_uv, dims=[3]))

        # Adversarial loss on generated patch
        d_fake = D(x_z_out_only)
        L_g_adv = -torch.log(torch.sigmoid(d_fake) + 1e-8).mean()

        # Perceptual + ID
        final_scaled = (final_uv_norm + 1) / 2
        uv_target_scaled = (uv_target_norm + 1) / 2
        L_vgg = L_VGG(final_scaled, uv_target_scaled)
        L_id = L_ID(final_scaled, uv_target_scaled)

        # TV / seam
        L_tv = total_variation_loss(out * mask_z)
        seam_w = seam_feathering(out, left_right_boundary=min(120, mid))
        L_seam = (torch.abs(out - uv_target_norm) * seam_w * mask_z).sum() / (seam_w.sum() + 1e-8)

        # Generator loss aggregation
        L_g = L_rec + WEIGHT_ADV * L_g_adv + 0.03 * L_vgg + WEIGHT_SYM * L_sym + WEIGHT_ID * L_id + 0.003 * L_tv + WEIGHT_SEAM * L_seam

        L_g.backward()
        opt_g.step()

        # -------------- Discriminator update -------------
        opt_d.zero_grad()
        d_real = D(x_uv_target.detach())
        d_fake = D(x_z_out_only.detach())
        L_d = -(torch.log(torch.sigmoid(d_real) + 1e-8).mean() + torch.log(1 - torch.sigmoid(d_fake) + 1e-8).mean())
        L_d.backward()
        opt_d.step()

        # Logging + checkpoints
        if i % 50 == 0 or i == iters-1:
            print(f"[{i}/{iters}] L_rec={float(L_rec):.4f} L_rec_z={float(L_rec_z):.4f} L_adv={float(L_g_adv):.4f} L_sym={float(L_sym):.4f} L_seam={float(L_seam):.4f} L_d={float(L_d):.4f}")
            save = final_scaled[0].cpu().clamp(0, 1)
            pil = transforms.ToPILImage()(save)
            pil.save(os.path.join(out_dir, f"iter_{i:04d}_{base}.png"))

    # Final save
    final_scaled = (final_uv_norm + 1) / 2
    final_img = final_scaled[0].permute(1, 2, 0).detach().cpu().clamp(0,1).numpy()
    final_img = (final_img * 255).astype(np.uint8)
    save_path = os.path.join(out_dir, f"uv_complete_{base}.png")
    Image.fromarray(final_img).save(save_path)
    print(f"[INFO] Saved Completed UV: {save_path}")

# ----------------------- CLI ------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UV Symmetry GAN - Trainer (DECA TestData cropping, Option B)")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing images (TestData will be built from this dir)")
    parser.add_argument("--img", type=str, required=True, help="Image filename inside input_dir to process (e.g. face1.jpg)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: <img>_train_uv_results)")
    parser.add_argument("--iters", type=int, default=500, help="Number of training iterations")
    parser.add_argument("--uv_size", type=int, default=512, help="Resolution for UV output")

    args = parser.parse_args()
    base = os.path.splitext(os.path.basename(args.img))[0]
    if args.out_dir is None:
        args.out_dir = f"{base}_train_uv_results"
    print("\n==========================================")
    print("RUN UV Symmetry GAN ...")
    print("Input dir  :", args.input_dir)
    print("Image      :", args.img)
    print("Iterations :", args.iters)
    print("UV Size    :", args.uv_size)
    print("Output dir :", args.out_dir)
    print("==========================================\n")

    train_single_uv(img_name=args.img, input_dir=args.input_dir, out_dir=args.out_dir, iters=args.iters, uv_size=args.uv_size)
