DIRECTORY=`dirname $0`

OMP_NUM_THREADS=1 python $DIRECTORY/run.py -m full \
--save_dir $DIRECTORY/SAVE_city_FCN_equal+/  \
--dataset_dir  $DIRECTORY/../../cityscapes \
--backbone fcn \
--dataset cityscapes  \
--name model_city \
--optimizer adam   \
--learning_rate 5e-4  --lr_decay 0.1    \
--weight_decay 2e-4  \
--height 512  --width 1024 \
--epochs 60  \
--lr_decay_epochs 20 \
--save_val_every_epoch 1 \
--batch_size 4 \
--consistency 1 \
--gpunum  0 \
--print_step

