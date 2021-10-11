
DIRECTORY=`dirname $0`

OMP_NUM_THREADS=1 python $DIRECTORY/EquAL.py  \
--save_dir $DIRECTORY/SAVE_camvid_deeplab_equal_1_12/  \
--dataset_dir $DIRECTORY/../../camvid/  \
--backbone deeplab \
--dataset camvid  \
--name model_camvid \
--equal 1 \
--BUDGET 0.12 \
--optimizer adam \
--learning_rate 5e-4  \
--lr_decay 0.1    \
--weight_decay 2e-4  \
--height 360  --width 480 \
--height_train 360  --width_train 480 \
--Divide_x 5 --Divide_y 5 \
--PatchesToLabelEachRun 1  \
--AL_batch 5 \
--MoreEpoch 5 \
--FinalDesired_Size_y 360 --FinalDesired_Size_x 480 \
--gpunum 1 --workers 4



