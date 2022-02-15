## Joint Acne Image Grading and Counting via Label Distribution Learning

### Update
This repository contains updated and refactored version of original Pytorch implementation of "Joint Acne Image Grading and Counting via Label Distribution Learning". Presented version is compatible with Pytorch 1.10.2.  

I refactored the code by implementing training loop and dataset wrapper using [Pytorch Lightning](https://www.pytorchlightning.ai/) and by adding improved logging using [Weights&Biases](https://wandb.ai/).

Following files are added the repository as well:
- [One page summary](https://github.com/ecatherina/LDL/blob/master/review.pdf) of the original paper with key ideas and results
- Detailed [exploratory data analysis](https://github.com/ecatherina/LDL/blob/master/eda.ipynb) of presented in paper dataset ACNE04

--------------------------------------------------------------------------
Pytorch implementation of "Joint Acne Image Grading and Counting via Label Distribution Learning"

This work was accepted by ICCV 2019 [[paper](http://xiaopingwu.cn/assets/paper/iccv2019_ldl.pdf)].

### ACNE04 Dataset

The ACNE04 dataset can be downloaded from [Baidu](https://pan.baidu.com/s/15JQlymnhnEmEt8Q5zpJQDw) (pw: fbrm) or [Google](https://drive.google.com/drive/folders/18yJcHXhzOv7H89t-Lda6phheAicLqMuZ?usp=sharing).

### Additional Information
If you find this work helpful, please cite it as
```
@InProceedings{Wu_2019_ICCV,
  author = {Wu, Xiaoping and Ni, Wen and Jie, Liang and Lai, Yu-Kun and Cheng, Dongyu, She and Ming-Ming and Yang, Jufeng},
  title = {Joint Acne Image Grading and Counting via Label Distribution Learning},
  booktitle = {IEEE International Conference on Computer Vision},
  year = {2019}
}
```

ATTN: This work is free for academic usage. For other purposes, please contact Xiaoping Wu (xpwu95@163.com).
