#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:23:23 2020

@author: alireza
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import sys
import os
import time
import shutil
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
from args import get_arguments



# Set random seem for reproducibility
manualSeed = 8
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
torch.cuda.manual_seed(manualSeed)

torch_seed = torch.initial_seed()
# print("torch_seed: {}" .format(torch_seed))
# print(torch.rand(5))
# print(torch.rand(5))
# print(torch.rand(5))
print('---------------------------------------------')


from utils import load_dataset_activelearning,Ignoring_voidslabel_camvid,Ignoring_voidslabel_cityscapes
from metric.iou import IoU
from al_val import AL_Val
from utils import EntropyComputing_Prediction,EntropyDivieBlockSampling,SelecingSamples_AL_Sort,load_dataset_al_train


# Get the arguments
args = get_arguments()

# gpu devices
device = torch.device(args.device)

# specific the gpu number to be used
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpunum
print('Using GPU {}'.format(args.gpunum))

# Import the requested dataset
if args.dataset.lower()=='camvid':
    from dataloaders import CamVid as dataset
elif args.dataset.lower()=='cityscapes':
    from dataloaders import Cityscapes as dataset
else:
    print('Your dataset is not in the predefined datasets')
    sys.exit()    


# Loading the data for active learning search and validation 
loaders,  class_encoding = load_dataset_activelearning(dataset,args)
search_train_loader, val_loader = loaders


num_classes = len(class_encoding)

# keep track of labeled samples for annotation
Annotated_Coords = {}

# batch size that is used for training during each iteration
AL_batch = args.AL_batch


TotalBudget = int(len(search_train_loader)*args.Divide_x*args.Divide_y*args.BUDGET)

# number of patches to be label in each iteration
PatchesToLabelEachRun = args.PatchesToLabelEachRun


ChangeTheEpochSize = int((int(0.3*TotalBudget)//args.PatchesToLabelEachRun)*args.PatchesToLabelEachRun)

print('Percent------------------------------------>',args.BUDGET*100)
print('TotalBudget In terms of patches--------------------------->',TotalBudget)
print('TotalBudget/PatchesToLabelEachRun------------------->',TotalBudget//PatchesToLabelEachRun)
print('PartialBudget---------------------------------->',ChangeTheEpochSize)


# Number of ecpoch for 0.3 of iterations
MoreEpoch = args.MoreEpoch

# iteration is when we labeled one samples, so the total number of iterations will be equal to TotalBudget
Iteration  = 0

if args.dataset.lower() == 'camvid':
    Ignoring_voidslabel_camvid(args.dataset_dir + 'trainannot/',args.Divide_y,args.Divide_x)
    Label_Majority_Dict = np.load('Camvid_MetaInfo.npy',allow_pickle='TRUE').item()
    # path for rgb training data
    Dir_RGB = args.dataset_dir + 'train/'   
    # path to anootation  data
    Dir_Ann = args.dataset_dir + 'trainannot/'
    
    Dir_AL_RGB =  args.dataset_dir + 'AL_Iter_RGB' +  '/'
    Dir_AL_Ann =  args.dataset_dir + 'AL_Iter_Ann' + '/'
    
    

if args.dataset.lower() == 'cityscapes':
    Ignoring_voidslabel_cityscapes(args.dataset_dir + 'leftImg8bit/train/',args.Divide_y,args.Divide_x,(args.width_train,args.height_train))
    Label_Majority_Dict = np.load('Cityscapes_MetaInfo.npy',allow_pickle='TRUE').item()
    # path for rgb training data
    Dir_RGB = args.dataset_dir + 'leftImg8bit/train/'  
    # path to anootation  data
    Dir_Ann = args.dataset_dir + 'gtFine/train/'
    
    Dir_AL_RGB =  args.dataset_dir + 'AL_Iter_RGB' +  '/'
    Dir_AL_Ann =  args.dataset_dir + 'AL_Iter_Ann' + '/'
    

class_weights = np.ones(num_classes)
class_weights = torch.from_numpy(class_weights).float().to(device)
ignore_index = list(class_encoding).index('unlabeled')
metric = IoU(num_classes, ignore_index=ignore_index)
class_weights[ignore_index] = 0
print('class_weights',class_weights)


# loss for each iteration
criterion_CE = nn.CrossEntropyLoss(weight=class_weights)
criterion_MSE = nn.MSELoss()

if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)


if args.dataset.lower() == 'cityscapes':
        void_label = 0
elif args.dataset.lower() == 'camvid':
        void_label = 11

#%%
while Iteration<TotalBudget:
    time1 = time.time()
    print('-------------------------------------------')
    print('Iteration', Iteration, 'out of',TotalBudget)
    
    if Iteration ==0:
        if args.backbone.lower() == 'fcn':
            model = torchvision.models.segmentation.fcn_resnet50(num_classes=num_classes).to(device)
        elif args.backbone.lower() == 'deeplab':
            model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=num_classes).to(device)
        
        print("-----Using", torch.cuda.device_count(), "GPUs!")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        
        optimizer = optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
        
        
    print('doing Active learning ...')
    Annotated_Coords_iteration = {}
    model.eval()

#   # In each Iteration we re-evaluate the whole samples
    Scor_Dict_All_Images = {}


    
    # go through the whole dataset
    for step, batch_data in enumerate(search_train_loader):
        startIt = time.time()
        
        inputs = batch_data[0].to(device)
        inputs_name = batch_data[2][0].split('/')[-1][:-4]
        with torch.no_grad():
            
            # compute output
            if args.equal:
                    
                outputs_1_1 = model(inputs)['out']
                outputs_2_1 = torch.flip(model(torch.flip(inputs,[3]))['out'],[3])
            else:
                outputs_1_1 = model(inputs)['out']
                outputs_2_1 = outputs_1_1
                
            numpy_ent1,numpy_ent2, \
                    numpy_ent_total = EntropyComputing_Prediction(outputs_1_1, outputs_2_1,args.dataset.lower())
            size_patch_y = numpy_ent_total.shape[0]//args.Divide_y 
            size_patch_x=  numpy_ent_total.shape[1]//args.Divide_x
            
            patches = numpy_ent_total.unfold(0, size_patch_y, size_patch_y).unfold(1, size_patch_x, size_patch_x)


                
            # height and weight, dividing to 5 parts each h and w
            He,Wi = inputs.shape[2],inputs.shape[3] 
            Divide_x = args.Divide_x
            Divide_y = args.Divide_y
                
            # partitionining and saving the entropy for each partision
            Score_Dict_Each_Img = {}
            # entropy for each image
            Score_Dict_Each_Img = EntropyDivieBlockSampling(patches,numpy_ent_total,inputs_name,Wi,He,
                                                            Divide_x,Divide_y)

            

            Scor_Dict_All_Images.update(Score_Dict_Each_Img)
            
            if step==0:
                
                # plt.axis('off');
                # plt.imshow(inputs[0,:,:,:].cpu().permute(1, 2, 0)); 
                # plt.show();
                # plt.axis('off')
                    

                            
                if not os.path.isdir(args.save_dir+'chose/'):
                    os.makedirs(args.save_dir+'chose/')
                plt.axis('off');
                plt.imshow(numpy_ent_total.cpu().numpy() )
                plt.savefig(args.save_dir+'chose/'+str(Iteration)+'.png')
                # plt.show();
                plt.axis('off')
                
                
            print('step',step,time.time()-startIt,end='\r')
            
            


    print('\n Selecting samples with the highest entropy..')
    
    Annotated_Coords,Annotated_Coords_iteration,Iteration =  SelecingSamples_AL_Sort(Iteration,PatchesToLabelEachRun,\
                                                                             Scor_Dict_All_Images,Label_Majority_Dict,\
                                                                                 Annotated_Coords,\
                                                                                     Annotated_Coords_iteration,\
                                                                                         TotalBudget,args)
    
    Batch_Size = 1
    if len(Annotated_Coords)>2:
        Batch_Size = 2
        
    Annotated_Coords_iteration_Copy = Annotated_Coords.copy()
    if Iteration>AL_batch:
        Annotated_Coords_iteration_Copy = Annotated_Coords.copy()
        Batch_Size = AL_batch
        Epochs_to_Run = 1
    
    if Iteration%ChangeTheEpochSize==0 and Iteration!=0:
        Epochs_to_Run = MoreEpoch
        val = AL_Val(model, val_loader, metric, device)
        
    else:
        Epochs_to_Run = 1
        
    
    print('Acquirering Labels..')
    
    
    if os.path.isdir(Dir_AL_Ann):
        shutil.rmtree(Dir_AL_Ann)
    if os.path.isdir(Dir_AL_RGB):
        shutil.rmtree(Dir_AL_RGB)
        
    if not os.path.isdir(Dir_AL_Ann):
        os.makedirs(Dir_AL_Ann)
    if not os.path.isdir(Dir_AL_RGB):
        os.makedirs(Dir_AL_RGB)
        
    print('Save Training Data')
    if args.dataset.lower() == 'cityscapes':
        for name_id in Annotated_Coords.keys():
            
            Ann_Dir_Save = Dir_Ann + name_id.split('_')[0] + '/' + name_id + '.png'
            Ann_Img = Image.open(Ann_Dir_Save.replace('leftImg8bit','gtFine_labelIds'))
            Ann_Img = Ann_Img.resize((args.width_train,args.height_train), Image.NEAREST)
    
            Ann_Img_Data = np.asarray(Ann_Img)
            
    
            Mask_Data = np.ones(Ann_Img_Data.shape) #* void_label # void class
            Mask_Data[:,:] = void_label
            
            Ann_Data_Copy = Ann_Img_Data.copy()
            
            for coords in Annotated_Coords[name_id]:
                
                Mask_Data[coords[0]:coords[0]+(He//Divide_y),coords[1]:coords[1]+(Wi//Divide_x)] = \
                    Ann_Data_Copy[coords[0]:coords[0]+(He//Divide_y),coords[1]:coords[1]+(Wi//Divide_x)]
                    
        
            # new_mask_img.mode should be either 'RGB' or 'L', we convert it to uint8
            Labeled_Data_ToSave_Ann = Image.fromarray(Mask_Data.astype(np.uint8))#.convert("L")
            NAME = Dir_AL_Ann + name_id + '.png'
            Labeled_Data_ToSave_Ann.save(NAME.replace('leftImg8bit','gtFine_labelIds'), "PNG")
            
            shutil.copy(Dir_RGB + name_id.split('_')[0] + '/' + name_id + '.png',Dir_AL_RGB +  name_id + '.png')
            
    if args.dataset.lower() == 'camvid':
        for name_id in Annotated_Coords.keys():
            
            Ann_Dir_Save = Dir_Ann + name_id + '.png'
            Ann_Img = Image.open(Ann_Dir_Save)
            Ann_Img = Ann_Img.resize((args.width_train,args.height_train), Image.NEAREST)
    
            Ann_Img_Data = np.asarray(Ann_Img)
            
    
            Mask_Data = np.ones(Ann_Img_Data.shape) #* void_label # void class
            Mask_Data[:,:] = void_label
            
            Ann_Data_Copy = Ann_Img_Data.copy()
            
            for coords in Annotated_Coords[name_id]:
                
                Mask_Data[coords[0]:coords[0]+(He//Divide_y),coords[1]:coords[1]+(Wi//Divide_x)] = \
                    Ann_Data_Copy[coords[0]:coords[0]+(He//Divide_y),coords[1]:coords[1]+(Wi//Divide_x)]
                    
        
            # new_mask_img.mode should be either 'RGB' or 'L', we convert it to uint8
            Labeled_Data_ToSave_Ann = Image.fromarray(Mask_Data.astype(np.uint8))#.convert("L")
            NAME = Dir_AL_Ann + name_id + '.png'
            Labeled_Data_ToSave_Ann.save(NAME, "PNG")
            
            shutil.copy(Dir_RGB + name_id + '.png',Dir_AL_RGB +  name_id + '.png')
            
        
        
        
    print('model switch to training mode...')
    
    
    print('Start Training AL')
    
    
    
    model.train()
    print('DataLoading Training Data')
    # dataloader for data in the Dir_AL_RGB folder
    iter_loaders,  class_encoding = load_dataset_al_train(dataset,args,Batch_Size)
    iter_train_loader = iter_loaders   

    print('Number of steps requires (images/batch):',len(iter_train_loader))
    for _ in range(Epochs_to_Run):
            
        for step_al, batch_data_al in enumerate(iter_train_loader):
            
            # Get the inputs and labels
            inputs_al = batch_data_al[0].to(device)
            labels_al = batch_data_al[1].to(device)
            
            if args.backbone.lower() == 'deeplab' and inputs_al.shape[0]<2:
                inputs_al = torch.cat((inputs_al,inputs_al),0)
                labels_al = torch.cat((labels_al,labels_al),0)
            
            
            
            # augmentation
            if  random.randint(0, 1):
                inputs_al = torch.flip(inputs_al, [3])
                labels_al = torch.flip(labels_al, [2])
                
            # model output
            if args.equal:
                    
                outputs_al_1_1 = model(inputs_al)['out']
                outputs_al_1_2 = torch.flip(model(torch.flip(inputs_al,[3]))['out'],[3])
    
    
    
                outputs_al_1_1_dt = outputs_al_1_1.detach().clone()
                outputs_al_1_2_dt = outputs_al_1_2.detach().clone()
                
                loss = criterion_CE(outputs_al_1_1, labels_al)  + \
                        criterion_CE(outputs_al_1_2, labels_al) + \
                            criterion_MSE(outputs_al_1_1, outputs_al_1_2_dt) + \
                                criterion_MSE(outputs_al_1_2, outputs_al_1_1_dt) 
            if not args.equal:
                outputs_al_1_1 = model(inputs_al)['out']      
                loss = criterion_CE(outputs_al_1_1, labels_al)  
                
            
            print('loss {:4.4f}'.format(loss.item()), end='\r')

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print('\ntime...{:4.4f}'.format(time.time()-time1))
         
    if Iteration%ChangeTheEpochSize==0 and Iteration!=0:
        print('do evaluations........')
        (iou, miou) = val.run_epoch()
        
        print('\n')
        # Save arguments
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        summary_filename_performance = os.path.join(args.save_dir, args.name + '_summary_epoch_' + str(str(Iteration) + '.txt'))
        with open(summary_filename_performance, 'w') as summary_file_2:
            summary_file_2.write("\nVALIDATION\n")
            summary_file_2.write("Iteration: {0}\n". format(Iteration))
            summary_file_2.write("Mean IoU: {0}\n". format(miou))
            for key, class_iou in zip(class_encoding.keys(), iou):
               summary_file_2.write("{0}: {1:.4f}\n".format(key, class_iou))

        summary_file_2.close()
        
        # Print per class IoU on last epoch or if best iou
        for key, class_iou in zip(class_encoding.keys(), iou):
            print("{0}: {1:.4f}".format(key, class_iou))
        print('miou',miou)
        

#%%
# torch.save(model.state_dict(), args.save_dir+'AL_MODEL_SAVED')
np.save(args.save_dir+'AL_coorddata.npy', Annotated_Coords)           

print('-----> Starting to save the AL annotations')

Dir_AL_RGB =  args.dataset_dir + 'AL_Iter_RGB' +  '/'
Dir_AL_Ann =  args.dataset_dir + 'AL_Iter_Ann' + '/'


if  os.path.isdir(Dir_AL_Ann):
    shutil.rmtree(Dir_AL_Ann)
# recreate the folder
if not os.path.isdir(Dir_AL_Ann):
    os.makedirs(Dir_AL_Ann)



AllRGBImages = []

if args.dataset.lower() == 'cityscapes':
    for root, dirs, allfiles in os.walk(Dir_RGB):
        for subdir in dirs:
            for root1, dirs1, allfiles1 in os.walk(root + subdir):
                AllRGBImages.extend(allfiles1)
elif args.dataset.lower() == 'camvid':
    for root, dirs, allfiles in os.walk(Dir_RGB):
        AllRGBImages = allfiles
        break
    
    

print('-----> Starting to  creat empty masks')
FinalDesired_Size = (args.FinalDesired_Size_y,args.FinalDesired_Size_x)
SizeThatItIsDone = (args.height_train, args.width_train)


        
for empty_name in AllRGBImages:
    
    mask_data_empty = np.ones((FinalDesired_Size[0], FinalDesired_Size[1])) * void_label
    _empty = Image.fromarray(mask_data_empty.astype(np.uint8))#.convert("L")
    if args.dataset.lower() == 'cityscapes':
        Name = Dir_AL_Ann + empty_name.split('_')[0] + '/'
        if not os.path.exists(Name):
            os.makedirs(Name)
        Name = Name + empty_name
    elif args.dataset.lower() == 'camvid':
        Name = Dir_AL_Ann + empty_name 
        
        
    
    if args.dataset.lower() == 'cityscapes':
        _empty.save(Name.replace('leftImg8bit','gtFine_labelIds'), "PNG")
    elif args.dataset.lower() == 'camvid':
        _empty.save(Name, "PNG")

#%%
print('-----> Starting to  save the AL to the empty mask')

Annotated_Coords = np.load(args.save_dir+'AL_coorddata.npy',allow_pickle='TRUE').item()

He,Wi,Divide_y,Divide_x = FinalDesired_Size[0], FinalDesired_Size[1],args.Divide_y,args.Divide_x

Count_bdg = 0

for rgb_al in Annotated_Coords:
    
    if args.dataset.lower() == 'camvid':
        ann_dir = Dir_Ann  + rgb_al + '.png'
    elif args.dataset.lower() == 'cityscapes':
        ann_dir = Dir_Ann  + rgb_al.split('_')[0] + '/' +  rgb_al + '.png'
        ann_dir = ann_dir.replace('leftImg8bit','gtFine_labelIds')
    
    
    
    
    
    ann = Image.open(ann_dir)
    ann = ann.resize((FinalDesired_Size[1], FinalDesired_Size[0]), Image.NEAREST)
    ann_data = np.asarray(ann)
    
    mask_data_final = np.ones(ann_data.shape) * void_label
    mask_data_final = mask_data_final.astype(np.uint8)
    
    for coord_al in Annotated_Coords[rgb_al]:
        Count_bdg+=1
        
        
        # replace the mask with annotations, i.e. oracle
        coord_al_y = coord_al[0]*int(FinalDesired_Size[0]/SizeThatItIsDone[0])
        coord_al_x = coord_al[1]*int(FinalDesired_Size[1]/SizeThatItIsDone[1])
        NewHe = FinalDesired_Size[0]
        NewWi = FinalDesired_Size[1]
        mask_data_final[coord_al_y:coord_al_y+(NewHe//Divide_y),coord_al_x:coord_al_x+(NewWi//Divide_x)] = ann_data[coord_al_y:coord_al_y+(NewHe//Divide_y),coord_al_x:coord_al_x+(NewWi//Divide_x)]
    
    
        
        new_mask_img = Image.fromarray(mask_data_final.astype(np.uint8))#.convert("L")
        if args.dataset.lower() == 'cityscapes':
            NAME = Dir_AL_Ann + rgb_al.split('_')[0] + '/' + rgb_al + '.png'
            NAME = NAME.replace('leftImg8bit','gtFine_labelIds')
        elif args.dataset.lower() == 'camvid':
            NAME = Dir_AL_Ann +   rgb_al + '.png'

        new_mask_img.save(NAME, "PNG")

print('count budget',Count_bdg)       
shutil.rmtree(Dir_AL_RGB)
shutil.copytree(Dir_RGB, Dir_AL_RGB)




AllRGBImages = []

if args.dataset.lower() == 'cityscapes':
    for root, dirs, allfiles in os.walk(Dir_AL_Ann):
        for subdir in dirs:
            for root1, dirs1, allfiles1 in os.walk(root + subdir):
                AllRGBImages.extend(allfiles1)
elif args.dataset.lower() == 'camvid':
    for root, dirs, allfiles in os.walk(Dir_AL_Ann):
        AllRGBImages = allfiles
        break

    
for img in AllRGBImages:
    if args.dataset.lower() == 'camvid':
        ann_dir = Dir_AL_Ann + img
        rgb_dir = Dir_AL_RGB + img
    if args.dataset.lower() == 'cityscapes':
        
        ann_dir = Dir_AL_Ann +   img.split('_')[0] + '/' + img
        rgb_dir = Dir_AL_RGB +   img.split('_')[0] + '/' + img.replace('gtFine_labelIds','leftImg8bit')
    ann = Image.open(ann_dir)
    ann_data = np.asarray(ann)
    if np.mean(ann_data)==void_label:
        # shutil.rmtree
        os.remove(ann_dir)
        os.remove(rgb_dir)


     
