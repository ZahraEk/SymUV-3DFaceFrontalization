import os
import torch
import torchvision

import sys
sys.path.append("/content/Towards-Realistic-Generative-3D-Face-Models")

from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg

# ---------- Setup ----------
def setup_deca(device, use_mica=True):
    """
    Proper DECA setup identical to official inference script (MICA-aware).
    """
    deca_cfg.model.use_tex = True
    deca_cfg.model.extract_tex = True
    deca_cfg.model.iscrop = True
    deca_cfg.rasterizer_type = 'pytorch3d'

    deca = DECA(
        config=deca_cfg,
        device=device,
        use_mica=use_mica
    )
    return deca

# ---------- TestData ----------
def load_deca_cropped(img_name, input_dir, device='cuda', use_mica=True):
    """
    Load cropped image AND arcface input via TestData (official way).
    """
    testdata = datasets.TestData(
        input_dir,
        iscrop=True,
        face_detector='fan',
        sample_step=1,
        crop_size=1024,
        use_mica=use_mica
    )

    target_base = os.path.splitext(os.path.basename(img_name))[0].lower()

    item = None
    for i in range(len(testdata)):
        name = os.path.splitext(os.path.basename(testdata[i]['imagename']))[0].lower()
        if name == target_base:
            item = testdata[i]
            break

    if item is None:
        raise RuntimeError(f"[DECA] Image '{img_name}' not found in TestData")

    img_cropped = item['image'].to(device)[None, ...]
    arcface_inp = item.get('arcface_inp', None)
    if arcface_inp is not None:
        arcface_inp = arcface_inp.to(device)[None, ...]

    return img_cropped, arcface_inp, item

# ---------- Run ----------
def run_deca_on_image(deca, img_cropped, arcface_inp=None, device='cuda'):
    """
    MICA-aware DECA encode + decode_tex
    """
    with torch.no_grad():
        enc_input = torchvision.transforms.Resize(224)(img_cropped)

        # ✅ critical: pass arcface_inp
        codedict = deca.encode(enc_input, arcface_inp=arcface_inp)
        codedict['images'] = img_cropped

        outputs = deca.decode_tex(codedict)

    # Normalize outputs
    if len(outputs) >= 4:
        uv_tex, vertices, uv_face_eye_mask, uv_texture = outputs[:4]
    else:
        uv_tex, vertices = outputs[:2]
        uv_face_eye_mask = None
        uv_texture = uv_tex

    return uv_tex, vertices, uv_texture, uv_face_eye_mask
