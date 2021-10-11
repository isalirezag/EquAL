DIRECTORY=`dirname $0`

OMP_NUM_THREADS=1 python $DIRECTORY/run.py -m full \
--save_dir $DIRECTORY/SAVE_camvid_DL+/  \
--dataset_dir  $DIRECTORY/../../camvid \
--backbone deeplab \
--dataset camvid  \
--name model_camvid_plus \
--optimizer adam   \
--learning_rate 5e-4  --lr_decay 0.1    \
--weight_decay 2e-4  \
--height 360  --width 480 \
--epochs 60  \
--lr_decay_epochs 20 \
--save_val_every_epoch 1 \
--batch_size 5 \
--consistency 1 \
--gpunum  0 \
--print_step

