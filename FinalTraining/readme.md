
The main tasks that we do here are:
1) Training the segmentation model via all the labeled data (this will give us the upper bound performance) 
2) Computing the performance of segmentation network when we train it via the data that are labeled from the active learning process.

We use Camvid and Cityscapes dataset. Our model support FCN and DeeplabV3 backbones.

#
## Datasets:
#### CamVid
Download the [CamVid Dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) and put it inside the data folder as follow:
<details>
<summary>Dataset Format</summary>

```bash

|-- /Data/camvid/
           |-- train
                |--0001TP_006690.png
                ...
           |-- trainannot
                |--0001TP_006690.png
                ...
           |-- test
                |--0001TP_008550.png
                ...
           |-- testannot
                |--0001TP_008550.png
                ...
           |-- val
                |--0016E5_07959.png
                ...
           |-- valannot
                |--0016E5_07959.png
                ...
```
</details>

If you have the train and trainannot folder from the active learning process for camvid, copy (or move) those folders from `Data/camvid/`, so it looks as follows:

```
|-- Data/camvid/
           |-- train_al
                ...
           |-- trainannot_al
                ...
```

#### Cityscapes
Download the [Cityscapes Dataset](https://www.cityscapes-dataset.com//) and put it inside the data folder as follow:

<details>
<summary>Dataset Format</summary>
```bash
|-- /Data/cityscapes/
           |-- leftImg8bit
                |--train
                    |--achen
                        |--aachen_000000_000019_leftImg8bit.png
                        ...
                |--test
                    ...
                |--val
                    ...
            |--gtFine
                |--train
                    |--achen
                        |--aachen_000000_000019_gtFine_labelIds.png
                        ...
                |--test
                    ...
                |--val
                    ...
```
</details>

## Training

The scrips that we use for training our model is provided in `FinalTraining/scrips/` depending on the experiment you can copy and move the sctip to `FinalTraining/` and then run it in the terminal  by `sh nameofscript.sh`.
`1.4` to `1.5` we observe that the results  slightly changed.
The results that are provided here are slightly better than the numbers that provided on the paper. 

#### Understandning the script
here is an example to understand the elements of script, you can also read the args for more detail.
<details><summary>Example of running command</summary>

```
DIRECTORY=`dirname $0`
OMP_NUM_THREADS=1 python $DIRECTORY/run.py -m full \
--save_dir $DIRECTORY/SAVE_city_FCN+/  \ 
--dataset_dir  $DIRECTORY/../../cityscapes \ # provide the directory to the datase, e.g. /Data/citiscapes/
--backbone fcn \ # choose the backbone, fcn or deeplab
--dataset cityscapes  \ #select the dataset, cityscapes or camvid
--name model_city \ # choose a name for your model
--optimizer adam   \
--learning_rate 5e-4  --lr_decay 0.1    \
--weight_decay 2e-4  \
--height 512  --width 1024 \ # resize the image to the desire size
--epochs 60  \
--lr_decay_epochs 20 \
--save_val_every_epoch 1 \
--batch_size 4 \
--consistency 1 \ #enable the self-consistency
--gpunum  1 \ # gpu id that you use
--print_step
```
</details>

If you are using the self-consistency for cityscapes with resolusion of 512x1024 you need to have a GPU with memory of `24G`. Please note that using a smaller GPU will lead to lower performance due to the limitation for batch size.

If you want to train the model for the data that you got via active learning, you need to go to `EquAL/FinalTraining/dataloaders/camvid/` or `EquAL/FinalTraining/dataloaders/cityscapes/`  and change the `train_folder` and `train_lbl_folder` to the name that you chose for the name of your model.

Also please note that in `EquAL/FinalTraining/run.py` and `EquAL/ActiveLearning/EquAL.py` we fixed the seed for reproducibility. You can change the seed or comment it, but it is suggested to have the same seed for comparison.



<details><summary>Results</summary>


|             Dataset/Model             | Input Resolution | Classes | Batch Size | Epochs | Mean IoU (%) | Budget | Self-Consistency |
|:-------------------------------------:|:----------------:|:-------:|:----------:|:------:|:------------:|--------|:----------------:|
|       Camvid (Fully Trained)-FCN      |      360x480     |    11   |      5     |   60   |     0.664    | 100%   |         0        |
|     Cityscapes (Fully Trained)-FCN    |     512x1024     |    19   |      4     |   60   |     0.643    | 100%   |         0        |
|      Camvid (Fully Trained)-FCN+      |      360x480     |    11   |      5     |   60   |     0.679    | 100%   |         1        |
|    Cityscapes (Fully Trained)-FCN+    |     512x1024     |    19   |      4     |   60   |     0.677    | 100%   |         1        |
|      Camvid (Active Learning)-FCN     |      360x480     |    11   |      5     |   60   |     0.634    | 12%    |         0        |
|    Cityscapes (Active Learning)-FCN   |     512x1024     |    19   |      4     |   60   |     0.622    | 12%    |         0        |
|     Camvid (Active Learning)-FCN+     |      360x480     |    11   |      5     |   60   |     0.644    | 12%    |         1        |
|   Cityscapes (Active Learning)-FCN+   |     512x1024     |    19   |      4     |   60   |     0.649    | 12%    |         1        |
|             Dataset/Model             | Input Resolution | Classes | Batch Size | Epochs | Mean IoU (%) | Budget | Self-Consistency |
|    Camvid (Fully Trained)-DeepLabV3   |      360x480     |    11   |      5     |   60   |     0.667    | 100%   |         0        |
|  Cityscapes (Fully Trained)-DeepLabV3 |     512x1024     |    19   |      4     |   60   |     0.649    | 100%   |         0        |
|   Camvid (Fully Trained)-DeepLabV3+   |      360x480     |    11   |      5     |   60   |     0.672    | 12%    |         1        |
| Cityscapes (Fully Trained)-DeepLabV3+ |     512x1024     |    19   |      4     |   60   |     0.697    | 12%    |         1        |
|    Camvid (Fully Trained)-DeepLabV3   |      360x480     |    11   |      5     |   60   |     0.622    | 12%    |         0        |
|  Cityscapes (Fully Trained)-DeepLabV3 |     512x1024     |    19   |      4     |   60   |              | 12%    |         0        |
|   Camvid (Fully Trained)-DeepLabV3+   |      360x480     |    11   |      5     |   60   |     0.634    | 12%    |         1        |
| Cityscapes (Fully Trained)-DeepLabV3+ |     512x1024     |    19   |      4     |   60   |              | 12%    |         1        |
</details>





