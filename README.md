# SC4001 Project

Implementation of **DINOv2-based Multi-scale Ensemble Voting for FGVC**.

We utilize the self-supervised DINOv2-B with the IELT attention head ensemble learning mechanism. Additionally, we make the following improvements:

1. Batch-hard triplet loss (see loss.py and train/)
2. Multi-scale gaussian filter in MHV within IELT, in all blocks {1 .. L-1}, using dilated conv
3. Inattentive token merger

Additional information and ablation can be found in the [paper](). Some example classifications are below.

<img src="figures/test_images.png" alt="Test Image" width="50%" />
Figure 1: Example classification results of our models

# Proposed Methods

## Architectural
DINOv2 is used as the backbone network, and fine-tuned to the specific FGVC task. Tokens are selectively forwarded via the [IELT](https://github.com/mobulan/IELT) mechanism.<br>
<img src="https://github.com/user-attachments/assets/727dca34-bb56-4bb9-affa-0686080227cb" alt="sc4001_arch" width="50%" />

Figure 1: Proposed Model Architecture

Multi-head Voting Module (MHV) in IELT currently uses a gaussian or learnable 3x3 kernel to enhance the voting on each attention head map. We propose one that is done at multiple scales with a dialated convolution, such that the final enhancement is defined as the regions maximally enhanced at all scales. It achieves marginally higher performance than the baseline.

Non-forwarded or "inattentive" tokens are merged by linear combination and forwarded to CLR.

## Training
We implement a joint loss function with the standard CE loss, and a triplet loss component. The triplet loss component provides the loss for the triplet set with the hardest negative in a given batch.

# Experiment Results

The following backbones are used:<br>
[ViT-B-16](https://console.cloud.google.com/storage/browser/_details/vit_models/imagenet21k/ViT-B_16.npz) (ImageNet-21k)<br>
[DINOv2-B_14](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) (ImageNet-21k, distilled)
<br>

The results with the specified configurations are shown in Table 1.

Table 1: Results of model training on Oxford Flowers test set
| **Backbone**      | **Optimizer**   | **Loss**            | **Misc.**              | **Top-1 Acc. (%)** | **Top-5 Acc. (%)** |
|-------------------|------------------|----------------------|-------------------------|--------------------|---------------------|
| ViT-B\_16          | SGD, 2e-2        | CE                   | --                      | 99.32              | 99.84               |
| ViT-B\_16          | SGD, 2e-2        | CE + Triplet         | Multi-Scale Gaussian    | 99.46              | **99.84**           |
| DINOv2-B\_14       | AdamW, 1e-5      | CE + Triplet         | Multi-Scale Gaussian    | **[99.528](https://entuedu-my.sharepoint.com/:u:/r/personal/etan102_e_ntu_edu_sg/Documents/SC4001%20Project%20(Model%20Weights)/dinov2_99.528.bin?csf=1&web=1&e=KpKCy8)**         | 99.77               |
| DINOv2-S\_14       | AdamW, 1e-5      | CE + Triplet         | Multi-Scale Gaussian    | 97.707             | 99.69               |

Table 2: Preliminary results on other FGVC datasets
| **Dataset**       | **Top-1 Acc. (%)** | **Top-5 Acc. (%)** |
|-------------------|--------------------|---------------------|
| CUB_200_2011      | 88.107             | 97.43               |
| Stanford Dogs     | 88.01              | 98.74               |

# Usage

Run `bootstrap.py` to create the dataset and label files.<br>
Model can be trained with `train.py` in the `train/` folder. It is important to define MAX_GRAD_ACCUM_SUB_BATCH such that it is divisible by 3 (for triplet loss), and is used to accumulate gradients (e.g. 32 batch size takes >32 GB VRAM on V100) <br>
The config location is set in `setup.py`. The config file should contain all non-default settings.<br>

In `test/`,<br>
Evaluate the model performance with `test.py`. Provide args to weight directory. <br>
Draw the diagram of example classifications with `test_img.py`.<br>


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
