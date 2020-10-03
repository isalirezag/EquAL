
#


**<div align="center">**Please stay tuned, we are working on the licencing and the paper works to be able to release our code for public**.<br />**
  
Importance of Self-Consistency in Active Learning for Semantic Segmentation (BMVC 2020) </div>

#


![GitHub Logo](https://user-images.githubusercontent.com/12434910/88857501-61a1a480-d1c4-11ea-9275-aebb808b9565.png)


![GitHub Logo](https://user-images.githubusercontent.com/12434910/88857535-7847fb80-d1c4-11ea-952e-1bba22396e65.gif)
![GitHub Logo](https://user-images.githubusercontent.com/12434910/88857558-839b2700-d1c4-11ea-9dac-11f383d840d0.gif)

#
In addition to the resuts in the paper here is also the results via DeepLabV3 as an backbone (instead of FCN) for our paper.

<details><summary>Results</summary>

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
