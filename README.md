# Multi-temporal Tyrion implementation
## Overview
This is the official repository for the paper Multi-temporal Calving Front Segmentation https://www.sciencedirect.com/science/article/pii/S092427162600300X
The model is designed for multi-temporal calving front segmentation from SAR Imagery. We provide three distinct ways to realize the muli-temporal information exchange
- Convolutional
- LTAE
- GRU

Our implementation is based on the Tyrion implementation from https://github.com/Nora-Go/TYRION and evaluated on the CaFFe Benchmark dataset https://essd.copernicus.org/articles/14/4287/2022/
It is also used in https://github.com/ki7077/Real-World-Tyrion and https://essd.copernicus.org/preprints/essd-2026-273/

## Getting started

### Requirements
We trained our model on 4x A100 Nvidia GPUs with 80GB VRAM each.
Inference can be done on a NVIDIA RTX 3070 with 8GB VRAM. 


### Data
Download the CaFFe Benchmark dataset from https://doi.pangaea.de/10.1594/PANGAEA.940950
Add the splits file into the dataset directory.
Point the employed Config files to the downloaded data directory

### Model
For training the model from scratch download the pretrained Swin V2 checkpoints https://github.com/microsoft/Swin-Transformer (these are image-net pretrained checkpoints not pre-trained on CaFFe)
Then use the trainer file to start the training 
Evaluation is done via the Evaluation file



## Citation
```bibtex
@article{dreier2026multi,
  title={Multi-temporal calving front segmentation},
  author={Dreier, Marcel and Gourmelon, Nora and Pyles, Dakota and Wu, Fei and Braun, Matthias and Seehaus, Thorsten and Maier, Andreas and Christlein, Vincent},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={239},
  pages={276--290},
  year={2026},
  publisher={Elsevier}
}
```
## Acknowledgments
- This research was supported by the Bayerisches Staatsministerium für Wissenschaft und Kunst within the Elite Network Bavaria through the International Doctorate Program “Measuring and Modelling Mountain Glaciers and Ice Caps in a Changing Climate” (IDP M3OCCA); in part by German Research Foundation (DFG) through the Project “Large-Scale Automatic Calving Front Segmentation and Frontal Ablation Analysis of Arctic Glaciers Using Synthetic-Aperture Radar Image Sequences (LASSI)” and the Project “PAGE” within the DFG Emmy Noether Programme
- We thank the providers of the satellite data under various AOs from respective space agencies (DLR, ESA, JAXA, and CSA)
- We acknowledge the computational resources provided by Erlangen National High Performance Computing Center (NHR@FAU), Friedrich–Alexander-Universität Erlangen–Nürnberg (FAU)