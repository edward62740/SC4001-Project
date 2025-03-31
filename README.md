# SC4001 Project

Implementation of DINOv2-based FGVC on Oxford Flowers dataset.

Currently, it is based on the self-supervised DINOv2-B with the IELT attention head ensemble learning mechanism. Additionally, we make the following improvements:

1. Batch-hard triplet loss (see loss.py and train/) [TASK 4]
2. Multi-scale gaussian filter in MHV within IELT, in all blocks {1 .. L-1}, using dilated conv [TASK 1]
3. Mixup (made it worse, maybe it is not implemented right) [TASK 3]

Disambiguation: The existing works directly evaluate on test set during training, and do not use the validation set. For consistency, our results are also evaluated on the test set.

![Test Image](figures/test_images.png)
Figure 1: Example classification results of our models

# Usage

Run `bootstrap.py` to create the dataset and label files.<br>
Model can be trained with `train.py` in the `train/` folder. It is important to define MAX_GRAD_ACCUM_SUB_BATCH such that it is divisible by 3 (for triplet loss), and is used to accumulate gradients (e.g. 32 batch size takes >32 GB VRAM on V100) <br>
The config location is set in `setup.py`. The config file should contain all non-default settings.<br>

In `test/`,<br>
Evaluate the model performance with `test.py`. Provide args to weight directory. <br>
Draw the diagram of example classifications with `test_img.py`.<br>

# Proposed Changes

## Architectural
We propose the use of DINOv2 as a backbone network to replace the original ViT used in the the [IELT](https://github.com/mobulan/IELT) paper.

Additionally, the Multi-head Voting Module (MHV) in IELT currently uses a gaussian or learnable 3x3 kernel to enhance the voting on each attention head map. We propose one that is done at multiple scales with a dialated convolution, such that the final enhancement is defined as the regions maximally enhanced at all scales. It achieves marginally higher performance than the baseline.

## Training
We implement a joint loss function with the standard CE loss, and a triplet loss component. The triplet loss component provides the loss for the triplet set with the hardest negative in a given batch.

# Experiment Results

The following backbones are used:<br>
[ViT-B-16](https://console.cloud.google.com/storage/browser/_details/vit_models/imagenet21k/ViT-B_16.npz) (ImageNet-21k)<br>
[DINOv2-B_14](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) (ImageNet-21k, distilled)
<br>

The results with the specified configurations are shown in Table 1.

Table 1: Results of model training on Oxford Flowers test set
| Backbone    | Optimizer   | Loss         | Augmentation         | Misc.                | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Weights |
|-------------|-------------|--------------|----------------------|----------------------|--------------------|--------------------|---------|
| ViT-B_16    | SGD, 2e-2   | CE           | RandomHorizontalFlip |                      | 99.351 (16)        |                    |         |
| ViT-B_16    | SGD, 2e-2   | CE + Triplet | RandomHorizontalFlip | Multi-Scale Gaussian | 99.401 (12)        |                    |         |
| DINOv2-B_14 | AdamW, 1e-5 | CE + Triplet | RandomHorizontalFlip | Multi-Scale Gaussian | 99.528 (11)        |  99.77            | [Link](https://entuedu-my.sharepoint.com/:f:/g/personal/etan102_e_ntu_edu_sg/Eq49NiW8T3tErMMcyL585mwB7E5GtLjnDfh2WHYd680qwQ?e=Id4xyR)    |



# Citations
```latex
@ARTICLE{10042971,
  author={Xu, Qin and Wang, Jiahui and Jiang, Bo and Luo, Bin},
  journal={IEEE Transactions on Multimedia}, 
  title={Fine-Grained Visual Classification Via Internal Ensemble Learning Transformer}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TMM.2023.3244340}}
  
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023},
  archivePrefix={arXiv},
  eprint={2304.07193}
}
```