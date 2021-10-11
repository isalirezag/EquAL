
<div align="center">This repository contains the implementation for the paper:<br />

[Importance of Self-Consistency in Active Learning for Semantic Segmentation (BMVC 2020)](https://arxiv.org/pdf/2008.01860.pdf) 
</div>

#
The code is test on Ubuntu `16.04`, [Pytorch](https://pytorch.org/) `1.5`, and python `3.6.6`.


![GitHub Logo](https://user-images.githubusercontent.com/12434910/88857501-61a1a480-d1c4-11ea-9275-aebb808b9565.png)


![GitHub Logo](https://user-images.githubusercontent.com/12434910/88857535-7847fb80-d1c4-11ea-952e-1bba22396e65.gif)
![GitHub Logo](https://user-images.githubusercontent.com/12434910/88857558-839b2700-d1c4-11ea-9dac-11f383d840d0.gif)

#
The Active learing code in this repository consist of two main steps (folders):<br/>
1) ActiveLearning: Include active learning code. <br/>
2) FinalTraining: Include the code for final training stage after AL process is finished or to compute the upperbond performance.


#
In addition to the resuts in the paper here is also the results via **DeepLabV3** as an backbone (instead of FCN) for our paper.

<details><summary>Results for DeeplabV3 as backbone</summary>

|             Dataset/Model             | Input Resolution | Classes | Batch Size | Epochs | Mean IoU (%) | Budget | Self-Consistency |
|:-------------------------------------:|:----------------:|:-------:|:----------:|:------:|:------------:|--------|:----------------:|
|    Camvid (Fully Trained)-DeepLabV3   |      360x480     |    11   |      5     |   60   |     0.667    | 100%   |         0        |
|  Cityscapes (Fully Trained)-DeepLabV3 |     512x1024     |    19   |      4     |   60   |     0.649    | 100%   |         0        |
|   Camvid (Fully Trained)-DeepLabV3+   |      360x480     |    11   |      5     |   60   |     0.672    | 100%   |         1        |
| Cityscapes (Fully Trained)-DeepLabV3+ |     512x1024     |    19   |      4     |   60   |     0.697    | 100%   |         1        |
|    Camvid (Active Learning)-DeepLabV3   |      360x480     |    11   |      5     |   60   |     0.622    | 12%    |         0        |
|  Cityscapes (Active Learning)-DeepLabV3 |     512x1024     |    19   |      4     |   60   |     0.633    | 12%    |         0        |
|   Camvid (Active Learning)-DeepLabV3+   |      360x480     |    11   |      5     |   60   |     0.634    | 12%    |         1        |
| Cityscapes (Active Learning)-DeepLabV3+ |     512x1024     |    19   |      4     |   60   |     0.674    | 12%    |         1        |
</details>


#
Citation:


```
@article{equal2020,
  title={Importance of Self-Consistency in Active Learning for Semantic Segmentation},
  author={Golestaneh, S. Alireza, Kitani, Kris},
  journal={BMVC},
  year={2020}
}
```

# 
If you have any questions about our work, please do not hesitate to contact us by emails at isalirezag@gmail.com



#
Acknowledgment:
part of the implementation is borrowed from [SegNet](https://github.com/alexgkendall/SegNet-Tutorial) and [Pytorch](https://github.com/pytorch).
