# Towards Realistic Generative 3D Face Models — Extended Fork
This repository is a research-oriented fork of the official implementation of:

Towards Realistic Generative 3D Face Models (AlbedoGAN)
Aashish Rai, Hiresh Gupta*, Ayush Pandey*, Francisco Vicente Carrasco, Shingo Jason Takagi, Amaury Aubel, Daeil Kim, Aayush Prakash, Fernando de la Torre
### Carnegie Mellon University & Meta Reality Labs — WACV 2024

The original project proposes a generative 3D face model that jointly synthesizes high-quality albedo textures and accurate 3D geometry using a StyleGAN2-based framework, enabling photo-realistic rendering.

This fork explores research extensions and experimental modifications for improving texture completion, symmetry enforcement, training stability, and identity preservation in partially observed facial textures.

###🔧 About This Fork

This repository contains independent research modifications by Zahra Ek, with a focus on:

- UV texture completion and symmetry-based inpainting

- Gradient-guided symmetry constraints

- Loss re-weighting schedules and training stabilization

- UV-space and image-space discriminator refinements

- Identity-preserving integration experiments (e.g., DECA, MICA, ArcFace)

- Gamma / illumination correction strategies

⚠️ These changes are experimental and not part of the official WACV 2024 release.


![](figure_1.png)

![](supp_image.png)


## Inference

Conda environment: Refer environment.yml

Download pre-trained models and put in the respective folders. 

Follow [[MICA](https://github.com/Zielon/MICA)] to download insightface and MICA pre-trained models. Put the weights in 'insightface' and 'data/mica_pretrained' folders, respectively.
Follow [[DECA](https://github.com/yfeng95/DECA)] to download DECA pre-trained weights. Put them in the 'data' folder.

Download AlbedoGAN modified weights from the following [[LINK](https://drive.google.com/drive/folders/1nJw8rUBTLcyhvCMTDohE_KcKKtFI6Orm?usp=sharing)]. Put these modified ArcFace backbone and DECA weights to generate better reconstruction results.

- Generate Random 3D Faces (mesh and texture)
    ```
    python demos/demo_generate.py
    ```
    
- Reconstruct 3D Faces from 2D Images
    ```
    python demos/demo_reconstruct.py
    ```

- Generate multi-pose videos
    ```
    python video.py
    ```

## Training code will be released at the earliest convenience. 

## Acknowledgements

This repository is based on the official implementation released by Carnegie Mellon University and Meta Reality Labs for the WACV 2024 paper "Towards Realistic Generative 3D Face Models".

We retain acknowledgements to third-party projects used in the original codebase. Parts of this repository rely on or are inspired by the following works:

1. [[DECA](https://github.com/yfeng95/DECA)]
2. [[MICA](https://github.com/Zielon/MICA)]
3. [[FLAME](https://github.com/soubhiksanyal/FLAME_PyTorch)]

Please refer to the respective license terms of these projects, as well as the X11 license of this repository, before using the code or any pre-trained models.

Additional research extensions and modifications were implemented by Zahra Ek.

## License Terms

The original project is released under the X11 License.

This fork preserves the same license. Please read the license terms available at [[Link](https://github.com/ZahraEk/Towards-Realistic-Generative-3D-Face-Models/blob/main/LICENSE)].

All original code and credit belong to the authors of the WACV 2024 paper.
Modifications and additions are provided by Zahra Ek under the same terms.

## Citation

If you find this code useful, please cite the original paper:

```bibtex
@article{rai2023towards,
  		title={Towards Realistic Generative 3D Face Models},
  		author={Rai, Aashish and Gupta, Hiresh and Pandey, Ayush and Carrasco, Francisco Vicente and Takagi, Shingo Jason and Aubel, Amaury and Kim, Daeil and Prakash, Aayush and De la Torre, Fernando},
  		journal={arXiv preprint arXiv:2304.12483},
  		year={2023}
 		}
```
