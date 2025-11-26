import torch
import os
import cv2
import numpy as np
from scipy.io import savemat
import torchvision
from tqdm import tqdm
from PIL import Image
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

def run_deca_inference(inputpath, savefolder, device='cuda', iscrop=True, 
                       detector='fan', sample_step=10, useTex=True, extractTex=True, 
                       rasterizer_type='pytorch3d', render_orig=False, use_mica=True):
    """
    Run DECA inference on a folder of images and save the outputs.

    Args:
        inputpath (str): Path to input image directory.
        savefolder (str): Output directory for saving DECA results.
        device (str): 'cuda' or 'cpu'.
        iscrop (bool): Whether to crop faces before processing.
        detector (str): Face detector type used by DECA. Usually 'fan'.
        sample_step (int): Sample step for TestData (load every N images).
        useTex (bool): Whether to use DECA's texture decoder.
        extractTex (bool): Whether to extract texture maps.
        rasterizer_type (str): Rasterizer backend ('pytorch3d' recommended).
        render_orig (bool): Whether to render on top of original input (unused here).
        use_mica (bool): Whether to use MICA-based identity features.
    """

    # Ensure output directory exists
    os.makedirs(savefolder, exist_ok=True)

    # Select device automatically if CUDA is unavailable
    device = device if torch.cuda.is_available() else 'cpu'

    # Load dataset wrapper that handles cropping, detection, preprocessing
    testdata = datasets.TestData(
        inputpath,
        iscrop=iscrop,
        face_detector=detector,
        sample_step=sample_step,
        crop_size=1024,
        use_mica=use_mica
    )

    # Configure DECA model parameters
    deca_cfg.model.use_tex = useTex
    deca_cfg.model.extract_tex = extractTex
    deca_cfg.rasterizer_type = rasterizer_type

    # Initialize DECA model
    deca = DECA(config=deca_cfg, device=device, use_mica=use_mica)

    # Select which outputs to save
    # (You can enable others as needed)
    saveDepth = False
    saveKpt = False
    saveObj = True       # Save 3D mesh (OBJ)
    saveMat = False
    saveVis = True       # Save visualization images
    saveImages = False

    # Loop through dataset
    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']

        # Load image tensor and add batch dimension
        images = testdata[i]['image'].to(device)[None, ...]

        # Optional: ArcFace input for identity-preserving DECA
        arcface_inp = testdata[i].get('arcface_inp', None)
        if arcface_inp is not None:
            arcface_inp = arcface_inp.to(device)[None, ...]

        # Encode → Decode with DECA (no gradients needed)
        with torch.no_grad():
            # Resize to 224x224 for DECA encoder
            codedict = deca.encode(torchvision.transforms.Resize(224)(images), arcface_inp)

            # Keep original high-resolution image for render/vis
            codedict['images'] = images

            # Decode geometry, texture, rendering, etc.
            opdict, visdict = deca.decode(codedict, name)

        # Create per-image output folder
        img_out_dir = os.path.join(savefolder, name)
        os.makedirs(img_out_dir, exist_ok=True)

        # Save mesh as .obj
        if saveObj:
            deca.save_obj(os.path.join(img_out_dir, f"{name}.obj"), opdict, codedict)

        # Save DECA visualization
        if saveVis:
            vis_img = deca.visualize(visdict)
            cv2.imwrite(os.path.join(img_out_dir, f"{name}_vis.jpg"), vis_img[:, :, ::-1])  # convert RGB→BGR

    print(f"[✅] DECA results saved in: {savefolder}")
    return savefolder