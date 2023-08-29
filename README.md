
# FFDS-Loss

[Paper]() | [Bibtex](#citation) | [Poster]()

**Introduction**: This repository is the official implementation of ICIP 2023 Paper, [Rethinking Long-Tailed Visual Recognition with Dynamic Probability Smoothing and Frequency Weighted Focusing](). This paper highlights the limitations of existing solutions that combine class- and instance-level re-weighting loss in a naive manner. Specifically, we demonstrate that such solutions result in overfitting the training set, significantly impacting the rare classes. To address this issue, we propose a novel loss function that dynamically reduces the influence of outliers and assigns class-dependent focusing parameters. We also introduce a new long-tailed dataset, ICText-LT, featuring various image qualities and greater realism than artificially sampled datasets. Our method has proven effective, outperforming existing methods through superior quantitative results on CIFAR-LT, Tiny ImageNet-LT, and our new ICText-LT
datasets.

*Codebase Reference: [MiSLAS](https://github.com/dvlab-research/MiSLAS) and [IB-Loss](https://github.com/pseulki/IB-Loss).*

## Installation

### Requirements

- PyTorch 1.10.0
- torchvision 0.11.0
- torchvision 0.15.0
- yacs 0.1.8
- wandb 0.13.11
  
### Create Virtual Environment

```bash
conda create -n FFDS python==3.9
source activate FFDS
```

### Install FFDS

```bash
git clone https://github.com/nwjun/FFDS-Loss.git
cd FFDS-Loss
pip install -r requirements.txt
````

### Dataset

- Imbalanced [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)
- Imbalanced [Tiny-ImageNet](https://drive.google.com/file/d/16fPIr1qhzlw7zWaFGDSgm4VdlR35_Myu/view?usp=sharing)
- Imbalanced [ICText](https://github.com/chunchet-ng/ictext)
  
Download all the datasets (except CIFAR) and place them in the [./data](/data) folder.  If you prefer a different location for the datasets, you can modify the `data_path` parameter in the `config/*/*.yaml` accordingly.

The default configuration expects the following folder structure:

```bash
./data
    ├── cifar10   # This dataset will be downloaded automatically
    ├── cifar100  # This dataset will be downloaded automatically
    ├── ictext2021
    │   ├── train/
    │   ├── val/
    │   └── char_2_idx.json
    └── tiny-imagenet-200
        ├── train/
        ├── val/
        ├── val_annotations.txt
        ├── wnids.txt
        └── words.txt

```

## Training

To train a model, run the following command:
`python train.py --cfg ./config/DFFDS/<DATASETNAME>/<DATASETNAME>_imb<IF>.yaml`

`DATASETNAME`: Can be selected from {`cifar10`, `cifar100`, `tiny_imagenet`, `ictext`} \\
`IF`: Can be selected from {`001`, `002`, `01`} for CIFAR and {`001`, `01`} for Tiny-ImageNet and {`001`,`0`} for ICText
> ICText `IF=0` is the natural distribution with imbalance ratio of 18 after removing lower-case letters. Read the [paper]() or visit [ICText Dataset Repository](https://github.com/chunchet-ng/ictext) for more details.

## Evaluation

To evaluate a trained model from checkpoint, run the following command:
`python eval.py --cfg ./config/DFFDS/<DATASETNAME>/<DATASETNAME>_imb<IF>.yaml  resume /path/to/checkpoint/`

## Results and Models

### CIFAR-10

| IF   | Loss   | Top-1 Acc. | Model        |
|------|--------|:----------:|:------------:|
| 0.01 | FFDS   | 75.60      |              |
|      | D-FFDS | 79.93      | [Link](https://drive.google.com/file/d/1XcEEI1DxEgVbabPM7Y7qkaq2T3NttxiL/view?usp=share_link)     |
| 0.02 | FFDS   | 79.82      |              |
|      | D-FFDS | 82.94      | [Link](https://drive.google.com/file/d/1qfx-7KhJix71XGr5ueVa5S2eoA5UMAZ-/view?usp=share_link)     |
| 0.1  | FFDS   | 87.46      |              |
|      | D-FFDS | 88.48      | [Link](https://drive.google.com/file/d/1tJy_ed-SgyJOhxSzjZP4GGNPoh-bcbXz/view?usp=share_link)     |

### CIFAR-100

| IF   | Loss   | Top-1 Acc. | Model        |
|------|--------|:----------:|:------------:|
| 0.01 | FFDS   | 40.74      |              |
|      | D-FFDS | 43.46      | [Link](https://drive.google.com/file/d/1-Kvgl_AhdOka0ogXSG4XBpOS_OTNLbIB/view?usp=share_link)     |
| 0.02 | FFDS   | 45.67      |              |
|      | D-FFDS | 48.48      |              |
| 0.1  | FFDS   | 58.66      |              |
|      | D-FFDS | 58.82      | [Link](https://drive.google.com/file/d/1ynrNiaBtDn9r0FabS6Cirvp4h6wkwkni/view?usp=share_link)     |

### Tiny-ImageNet

| IF   | Loss   | Top-1 Acc. | Model        |
|------|--------|:----------:|:------------:|
| 0.01 | FFDS   | 42.34      |              |
|      | D-FFDS | 43.86      | [Link](https://drive.google.com/file/d/1FffRx7YFGF9yHMEDuLaBEohCD17UjN1e/view?usp=share_link)     |
| 0.1  | FFDS   | 56.11      |              |
|      | D-FFDS | 58.31      |              |

### ICText

| IF   | Loss   | Top-1 Acc. |
|------|--------|:----------:|
| 0.01 | FFDS   | 76.89      |
|      | D-FFDS | 79.56      |
| 0.0  | FFDS   | 85.40      |
|      | D-FFDS | 85.98      |

## Citation

If you find our paper and repository useful, please cite
```bibtex
@inproceedings{icip2023_ffds,
  author={Nah, Wan Jun and Ng, Chun Chet and Lin, Che-Tsung and Lee, Yeong Khang and Kew, Jie Long and Tan, Zhi Qin and Chan, Chee Seng and Zach, Christopher and Lai, Shang-Hong},
  booktitle={2023 30th IEEE International Conference on Image Processing (ICIP)}, 
  title={Rethinking Long-Tailed Visual Recognition with Dynamic Probability Smoothing and Frequency Weighted Focusing}, 
  year={2023}
```

## Feedback
Suggestions and opinions on this work (both positive and negative) are greatly welcomed. Please contact the authors by sending an email to
`nicolenahwj at gmail.com` or `cs.chan at um.edu.my`.

## License and Copyright
The project is open source under BSD-3 license (see the ``` LICENSE ``` file).

&#169;2023 Universiti Malaya.
