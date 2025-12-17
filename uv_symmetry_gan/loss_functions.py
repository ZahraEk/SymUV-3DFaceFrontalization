import os
import torch
import subprocess
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# ---------------- GAN losses ----------------
def hinge_d_loss(real, fake):
    """Hinge loss for discriminator."""
    return 0.5 * (F.relu(1 - real).mean() + F.relu(1 + fake).mean())

def hinge_g_loss(fake):
    """Hinge loss for generator."""
    return -fake.mean()

def total_variation_loss(x):
    """Total variation loss for smoothing artifacts."""
    dh = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    dw = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return dh + dw

# ---------------- Symmetry losses ----------------
def gradient_x(img):
    """Horizontal image gradient."""
    return img[:, :, :, 1:] - img[:, :, :, :-1]

def gradient_y(img):
    """Vertical image gradient."""
    return img[:, :, 1:, :] - img[:, :, :-1, :]

def gradient_symmetry_loss(
    out,
    target,
    mask_z,
    mid,
    healthy_side="left",
    weight_x=1.0,
    weight_y=0.5
):
    """
    Gradient-based symmetry loss across UV seam using the healthy side as reference.

    out, target : [B,3,H,W] in [-1,1]
    mask_z      : [B,1,H,W] (1 on damaged side)
    healthy_side: 'left' or 'right'
    """

    # Determine damaged side based on healthy_side
    if healthy_side == "left":
        out_z  = out[:, :, :, mid:]     # damaged: right half
        tgt_uv = target[:, :, :, :mid]  # healthy: left half
        mask   = mask_z[:, :, :, mid:]
    else:  # healthy_side == "right"
        out_z  = out[:, :, :, :mid]     # damaged: left half
        tgt_uv = target[:, :, :, mid:]  # healthy: right half
        mask   = mask_z[:, :, :, :mid]

    # Flip healthy side for symmetry
    tgt_uv_flip = torch.flip(tgt_uv, dims=[3])

    # Apply damaged-region mask
    out_z = out_z * mask

    # Gradients
    gx_out = gradient_x(out_z)
    gx_tgt = gradient_x(tgt_uv_flip)

    gy_out = gradient_y(out_z)
    gy_tgt = gradient_y(tgt_uv_flip)

    Lgx = F.l1_loss(gx_out, gx_tgt)
    Lgy = F.l1_loss(gy_out, gy_tgt)

    return weight_x * Lgx + weight_y * Lgy

# ---------------- Identity (ArcFace ONNX) ----------------
ONNX_MODEL_URL = (
    "https://huggingface.co/onnxmodelzoo/"
    "arcfaceresnet100-11-int8/resolve/main/"
    "arcfaceresnet100-11-int8.onnx"
)
ONNX_MODEL_LOCAL = "arcfaceresnet100-int8.onnx"

try:
    import onnxruntime as ort
except Exception:
    ort = None
    print("[WARNING] onnxruntime not available; IdentityLoss disabled.")

def download_arcface(url=ONNX_MODEL_URL, save_path=ONNX_MODEL_LOCAL):
    """Download ArcFace ONNX model if missing."""
    if os.path.exists(save_path):
        return save_path
    subprocess.run(
        ["wget", url, "-O", save_path, "--no-check-certificate"],
        check=True
    )
    return save_path

def get_onnx_session(model_path):
    """Create ONNXRuntime inference session."""
    if ort is None:
        raise RuntimeError("onnxruntime not available.")

    providers = ort.get_available_providers()
    use_cuda = "CUDAExecutionProvider" in providers
    provider_list = (
        [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        if use_cuda else ["CPUExecutionProvider"]
    )

    sess = ort.InferenceSession(model_path, providers=provider_list)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess, input_name, output_name

class IdentityLoss(nn.Module):
    """ArcFace-based identity loss using ONNXRuntime."""
    def __init__(self, device="cpu", onnx_path=ONNX_MODEL_LOCAL):
        super().__init__()
        self.device = device
        self.sess = None
        self.input_name = None
        self.output_name = None
        self.onnx_path = onnx_path

        if ort is None:
            return

        try:
            if not os.path.exists(self.onnx_path):
                download_arcface(save_path=self.onnx_path)
            self.sess, self.input_name, self.output_name = get_onnx_session(
                self.onnx_path
            )
        except Exception as e:
            print(f"[WARNING] ArcFace init failed: {e}")
            self.sess = None

    def preprocess_for_arcface(self, tensor_batch):
        """Prepare tensor batch for ArcFace input."""
        bs = tensor_batch.shape[0]
        out_list = []
        to_pil = transforms.ToPILImage()

        for i in range(bs):
            t = tensor_batch[i].cpu().clamp(0, 1)
            pil = to_pil(t).resize((112, 112), Image.BILINEAR)
            arr = np.asarray(pil).astype(np.float32)
            chw = np.transpose(arr, (2, 0, 1))
            chw = (chw - 127.5) / 128.0
            out_list.append(chw)

        return np.stack(out_list, axis=0).astype(np.float32)

    def forward(self, generated_uv, target_uv):
        """Compute cosine distance between ArcFace embeddings."""
        if self.sess is None:
            return torch.tensor(0.0, device=generated_uv.device)

        def to_01(x):
            return (x + 1.0) / 2.0 if x.min() < -0.5 else x

        gen = to_01(generated_uv)
        tgt = to_01(target_uv)

        gen_np = self.preprocess_for_arcface(gen)
        tgt_np = self.preprocess_for_arcface(tgt)

        try:
            emb_gen = self.sess.run(
                [self.output_name], {self.input_name: gen_np}
            )[0]
            emb_tgt = self.sess.run(
                [self.output_name], {self.input_name: tgt_np}
            )[0]
        except Exception:
            return torch.tensor(0.0, device=generated_uv.device)

        emb_gen_t = F.normalize(
            torch.from_numpy(emb_gen).to(generated_uv.device), dim=1
        )
        emb_tgt_t = F.normalize(
            torch.from_numpy(emb_tgt).to(generated_uv.device), dim=1
        )

        cos = F.cosine_similarity(emb_gen_t, emb_tgt_t, dim=1)
        return 1.0 - cos.mean()

# ---------------- Perceptual (VGG) ----------------
class VGGPerceptualLoss(nn.Module):
    """VGG19-based perceptual loss."""
    def __init__(self, device="cuda"):
        super().__init__()
        self.model = None

        try:
            try:
                vgg = models.vgg19(pretrained=True).features
            except Exception:
                vgg = models.vgg19(
                    weights=models.VGG19_Weights.IMAGENET1K_V1
                ).features

            self.model = nn.Sequential(*list(vgg.children())[:30]).to(device).eval()
            for p in self.model.parameters():
                p.requires_grad = False
        except Exception as e:
            print(f"[WARNING] VGG loading failed: {e}")
            self.model = None

        self.layer_indices = [2, 7, 12, 21, 30]

    def forward(self, x, y):
        """Compute multi-layer L1 perceptual distance."""
        if self.model is None:
            return torch.tensor(0.0, device=x.device)

        loss = 0.0
        fx, fy = x, y
        feats_x, feats_y = [], []

        for i, layer in enumerate(self.model):
            fx = layer(fx)
            fy = layer(fy)
            if i in self.layer_indices:
                feats_x.append(fx)
                feats_y.append(fy)

        for ax, ay in zip(feats_x, feats_y):
            loss += torch.mean(torch.abs(ax - ay))

        return loss
