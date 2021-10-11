
DIRECTORY=`dirname $0`


# for people with gpu with memory less than 24G

OMP_NUM_THREADS=1 python $DIRECTORY/EquAL.py  \
--save_dir $DIRECTORY/SAVE_cityscapes_deeplab_equal_1_12/  \
--dataset_dir $DIRECTORY/../../cityscapes/  \
--backbone deeplab \
--dataset cityscapes  \
--name model_deeplab \
--equal 1 \
--BUDGET 0.12 \
--optimizer adam \
--learning_rate 5e-4  \
--lr_decay 0.1    \
--weight_decay 2e-4  \
--height 256  --width 512 \
--height_train 256  --width_train 512 \
--Divide_x 8 --Divide_y 4 \
--PatchesToLabelEachRun 5  \
--AL_batch 5 \
--MoreEpoch 5 \
--FinalDesired_Size_y 512 --FinalDesired_Size_x 1024 \
--gpunum 1 --workers 4


# for people with gpu large enough

# OMP_NUM_THREADS=1 python $DIRECTORY/EquAL.py  \
# --save_dir $DIRECTORY/SAVE_cityscapes_deeplab_equal_1_12/  \
# --dataset_dir $DIRECTORY/../../cityscapes/  \
# --backbone deeplab \
# --dataset cityscapes  \
# --name model_deeplab \
# --equal 1 \
# --BUDGET 0.12 \
# --optimizer adam \
# --learning_rate 5e-4  \
# --lr_decay 0.1    \
# --weight_decay 2e-4  \
# --height 512  --width 1024 \
# --height_train 512  --width_train 1024 \
# --Divide_x 8 --Divide_y 4 \
# --PatchesToLabelEachRun 5  \
# --AL_batch 5 \
# --MoreEpoch 5 \
# --FinalDesired_Size_y 512 --FinalDesired_Size_x 1024 \
# --gpunum 1 --workers 4

