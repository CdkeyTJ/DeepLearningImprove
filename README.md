# A Report on Improvement of Logit Standardization in Knowledge Distillation

## Abstract
This paper explores the application and effectiveness of combining Logit Standardization and Wasserstein Distance in Knowledge Distillation (KD). Traditional KD methods, such as those based on Kullback-Leibler (KL) divergence, face limitations in capturing rich inter-category relationships and adapting to different model capacities. To address these issues, we propose an enhanced KD framework that incorporates Logit Standardization to focus on essential logit relationships and uses Wasserstein Distance for more accurate cross-category comparisons. Our method dynamically adjusts the temperature based on the maximum logit, allowing for flexible adaptation to various model capacities. Through extensive experiments on the CIFAR-100 dataset with different teacher-student model combinations, we demonstrate that our proposed method consistently outperforms state-of-the-art KD techniques in terms of accuracy and generalization. The results highlight the significant contributions of Logit Standardization and Wasserstein Distance in improving KD performance. Future work will focus on optimizing computational efficiency and exploring other potential improvements in KD methodologies. Our code is at [here](https://github.com/CdkeyTJ/DeepLearningImprove/tree/main)

## Usage

The code is built on [mdistiller](<https://github.com/megvii-research/mdistiller>)

### Installation

Environments:

- Python 3.8
- PyTorch 1.7.0

Install the package:

```
sudo pip3 install -r requirements.txt
sudo python setup.py develop
```

## Distilling CNNs

### CIFAR-100

Download the [`cifar_teachers.tar`](https://github.com/megvii-research/mdistiller/releases/tag/checkpoints) and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

`./download_ckpts/cifar_teachers` 
contains CIFAR-100 dataset

when base-temp equals 0 uses dynamic temperature to adjust distillation efficiency

1. For KD

  ```bash
  # KD
  python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml
  
  # KD+Logit_SD
  python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
  
  # KD+Temp
  python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml --base-temp 2 --kd-weight 9 
  
  # KD+Logit_SD+Temp
  python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 0 --kd-weight 9 
  ```

2. For DKD

  ```bash
  # DKD
  python tools/train.py --cfg configs/cifar100/dkd/resnet32x4_resnet8x4.yaml 
  
  # DKD+Logit_SD
  python tools/train.py --cfg configs/cifar100/dkd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
  
  # DKD+Temp
  python tools/train.py --cfg configs/cifar100/dkd/resnet32x4_resnet8x4.yaml --base-temp 0 --kd-weight 9 
  
  # DKD+Logit_SD+Temp
  python tools/train.py --cfg configs/cifar100/dkd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 0 --kd-weight 9 
  ```

3. For KD+WKD

  ```bash
  # WKD
  python tools/train.py --cfg configs/cifar100/wkd/resnet32x4_resnet8x4.yaml
  
  # WKD+Logit_SD
  python tools/train.py --cfg configs/cifar100/wkd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
  
  # WKD+Temp
  python tools/train.py --cfg configs/cifar100/wkd/resnet32x4_resnet8x4.yaml --base-temp 0 --kd-weight 9 
  
  # WKD+Logit_SD+Temp
  python tools/train.py --cfg configs/cifar100/wkd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 0 --kd-weight 9 
  ```

#### Results


| Teacher & Student | ResNet32x4  <br>  <br>ResNet8x4 | Wrn_40_2  <br>  <br>Wrn_16_2 | ResNet32x4  <br>  <br>Wrn_16_2 | ResNet32x4  <br>  <br>Wrn_40_2 | Wrn_40_2  <br>  <br>ResNet8x4 |
| ----------------- | ------------------------------- | ---------------------------- | ------------------------------ | ------------------------------ | ----------------------------- |
| KD                | 73.31                           | 74.92                        | 74.9                           | 77.7                           | 73.97                         |
| KD+Logit_SD       | 76.62                           | 76.11                        | 75.26                          | 77.92                          | 77.11                         |
| KD+Temp           | 76.45                           | 75.4                         | 75.91                          | 78.43                          | 75.86                         |
| KD+WD             | 76.23                           | 77.26                        | 77.51                          | 78.23                          | 76.12                         |
| KD+WD+Temp        | 77.11                           | 78.25                        | 76.84                          | 78.23                          | 76.95                         |



