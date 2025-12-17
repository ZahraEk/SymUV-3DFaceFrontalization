import os
import torch
import torchvision

try:
    import sys
    sys.path.append("/content/Towards-Realistic-Generative-3D-Face-Models")
    from decalib.utils.config import cfg as deca_cfg
    from decalib.deca import DECA
    from decalib.datasets import datasets
except:
    DECA = None
    datasets = None

# ---------- Setup ----------
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

# ---------- TestData ----------
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

# ---------- Run ----------
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
