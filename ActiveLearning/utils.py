import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.stats import entropy


import transforms as ext_transforms
import torchvision.transforms as transforms
from PIL import Image
import torch.utils.data as data

from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters.rank import gradient


import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image
import glob
import os


def Ignoring_voidslabel_camvid(directory_annot,Divide_y,Divide_x):
    
    # Total number of images in the folder
    for root, dirs, AllAnnot in os.walk(directory_annot):
        break
    
    
    Label_Majority_Dict = {}
    for img_ann in AllAnnot:
    
        ann_dir = directory_annot + img_ann
        ann = Image.open(ann_dir)
        
        img_data = np.asarray(ann)
        
        He,Wi = img_data.shape
        
        Wi_split = np.linspace(0, Wi,num=Divide_x, endpoint=False)
        He_split = np.linspace(0, He,num=Divide_y, endpoint=False)
        
        for x_cord in Wi_split:
            for y_cord in He_split:
                Block = img_data[int(y_cord):int(y_cord)+int(He//Divide_y),
                                 int(x_cord):int(x_cord)+(Wi//Divide_x)]
                Freq = np.bincount(Block.flatten()).argmax()
                
                Key_Name = img_ann[:-4]  + '_' +  str(int(y_cord)) + '_' + str(int(x_cord))
                Label_Majority_Dict[Key_Name] = Freq
                
                
    np.save("Camvid_MetaInfo.npy", Label_Majority_Dict)      
    

def Ignoring_voidslabel_cityscapes(directory_rgb,Divide_y,Divide_x,SIZE):
    from dataloaders import utils

    
    # The values associated with the 35 classes
    full_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                    32, 33, -1)
    # The values above are remapped to the following
    new_classes = (0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 5, 0, 0, 0, 6, 0, 7,
                   8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 17, 18, 19, 0)
    
    # Total number of images in the folder
    AllAnnot = []
    for root, dirs, allfiles in os.walk(directory_rgb):
        for subdir in dirs:
            for root1, dirs1, allfiles1 in os.walk(root + subdir):
                AllAnnot.extend(allfiles1)
    
    Label_Majority_Dict = {}
    for step, img_ann in enumerate(AllAnnot[:]):
        
        ann_dir = directory_rgb.replace('leftImg8bit','gtFine') + img_ann.split('_')[0]+'/'+img_ann.replace('leftImg8bit','gtFine_labelIds')
        ann = Image.open(ann_dir)
        ann = utils.remap(ann, full_classes, new_classes)
        ann = ann.resize(SIZE, Image.NEAREST)
        
        img_data = np.asarray(ann)
        
        He,Wi = img_data.shape
        
        Wi_split = np.linspace(0, Wi,num=Divide_x, endpoint=False)
        He_split = np.linspace(0, He,num=Divide_y, endpoint=False)
        
        for x_cord in Wi_split:
            for y_cord in He_split:
                Block = img_data[int(y_cord):int(y_cord)+int(He//Divide_y),
                                 int(x_cord):int(x_cord)+(Wi//Divide_x)]
                Freq = np.bincount(Block.flatten()).argmax()
                
                Key_Name = img_ann[:-4]  + '_' +  str(int(y_cord)) + '_' + str(int(x_cord))
                Label_Majority_Dict[Key_Name] = Freq
                
                
    np.save("Cityscapes_MetaInfo.npy", Label_Majority_Dict)      
    


def EntropyComputing_Prediction(outputs1,outputs2,datasetname):

    
    
    if datasetname == 'cityscapes':
        void_label = 0
    elif datasetname == 'camvid':
        void_label = 11
        
    
    # probability cross the categories
    prob_out1 = torch.nn.functional.softmax(outputs1,dim=1)
    prob_out2 = torch.nn.functional.softmax(outputs2,dim=1)
    
    log_out1 =-1* torch.log(prob_out1)
    log_out1[log_out1 != log_out1] = 0
    log_out1[log_out1 == float("Inf")] = 0
    log_out1[log_out1 == -float("Inf")] = 0
    log_out1[log_out1 == float("-Inf")] = 0
    entropy_computed1 = log_out1*prob_out1
    
    
    entropy_map1 = torch.sum(entropy_computed1,dim=1)
    
    
    
    
    log_out2 =-1* torch.log(prob_out2)
    log_out2[log_out2 != log_out2] = 0
    log_out2[log_out2 == float("Inf")] = 0
    log_out2[log_out2 == -float("Inf")] = 0
    log_out2[log_out2 == float("-Inf")] = 0
    entropy_computed2 = log_out2*prob_out2
    # final entropy map
    entropy_map2 = torch.sum(entropy_computed2,dim=1)
    
    numpy_ent_total = entropy_map1[0,:,:] + entropy_map2[0,:,:]
    
   
    return entropy_map1[0,:,:],entropy_map2[0,:,:], numpy_ent_total 

from numpy.lib.stride_tricks import as_strided

def EntropyDivieBlockSampling(input_data,aaaa,data_name,Wi,He,Divide_x,Divide_y):
        Score_dict = {}
        Wi_split = np.linspace(0, Wi,num=Divide_x, endpoint=False)
        He_split = np.linspace(0, He,num=Divide_y, endpoint=False)


        for x_cord in range(len(Wi_split)):
            for y_cord in range(len(He_split)):
                # print(input_data.shape)
                Block = input_data[int(y_cord),
                                      int(x_cord),:,:]
                # print(Block)
                Ent_Score = torch.mean(Block)#.item()
                # print(Ent_Score)

                # key name for score dict 
                Key_Name = data_name  + '_' +  str(int(He_split[y_cord])) + '_' + str(int(Wi_split[x_cord]))

                
                Score_dict[Key_Name] = Ent_Score


        return Score_dict




def SelecingSamples_AL_Sort(Iteration,BudgetForEachRun,Score_dict_total,Label_Majority_Dict,
                            Annotated_Coords,Annotated_Coords_iteration,TotalBudget,args):
    if args.dataset.lower() == 'cityscapes':
        void_label = 0
    elif args.dataset.lower() == 'camvid':
            void_label = 11
    
    Temp_Budget = 0
    
    Counting = 0
    sorted_x = sorted(Score_dict_total.items(), key=lambda x:x[1])[::-1]

    while Temp_Budget < BudgetForEachRun and  Iteration<TotalBudget:
        MaxValue_Tuple = sorted_x[Counting]
        Counting+=1
        
        # Check to make sure it is not void
        if Label_Majority_Dict[MaxValue_Tuple[0]] != void_label: #not equal void class
        
            #  get the name and coords
            Name_y_x_list = MaxValue_Tuple[0].split('_')
            if args.dataset.lower() == 'camvid':
                FileName = '_'.join(Name_y_x_list[:2])
            elif args.dataset.lower() == 'cityscapes':
                FileName = '_'.join(Name_y_x_list[:4])
            #  get the values
            Ent_Score = MaxValue_Tuple[1]
            
            if args.dataset.lower() == 'camvid':
                
                if FileName in Annotated_Coords.keys():
                    if (int(Name_y_x_list[2]),int(Name_y_x_list[3])) not in Annotated_Coords[FileName]:
                                                
                        Annotated_Coords[FileName].append((int(Name_y_x_list[2]),
                                                           int(Name_y_x_list[3])))
                        
                        if FileName in Annotated_Coords_iteration.keys():
                            Annotated_Coords_iteration[FileName].append((int(Name_y_x_list[2]),
                                                           int(Name_y_x_list[3]),Ent_Score))
                        else:
                            Annotated_Coords_iteration[FileName] = []
                            Annotated_Coords_iteration[FileName].append((int(Name_y_x_list[2]),
                                                                         int(Name_y_x_list[3]),Ent_Score))
                            
                        Temp_Budget += 1
                        Iteration += 1
                else:
                    Annotated_Coords[FileName] = []
                    Annotated_Coords[FileName].append((int(Name_y_x_list[2]),int(Name_y_x_list[3])))
                    
                    
                    Annotated_Coords_iteration[FileName] = []
                    Annotated_Coords_iteration[FileName].append((int(Name_y_x_list[2]),
                                                                          int(Name_y_x_list[3]),Ent_Score))
                        
                    Temp_Budget += 1
                    Iteration += 1
                
            
            
            if args.dataset.lower() == 'cityscapes':            
                # add the new data to oracle
                if FileName in Annotated_Coords.keys():
                    if (int(Name_y_x_list[4]),int(Name_y_x_list[5])) not in Annotated_Coords[FileName]:
                        Annotated_Coords[FileName].append((int(Name_y_x_list[4]),
                                                            int(Name_y_x_list[5])))
                        
                        if FileName in Annotated_Coords_iteration.keys():
                            Annotated_Coords_iteration[FileName].append((int(Name_y_x_list[4]),
                                                            int(Name_y_x_list[5]),Ent_Score))
                        else:
                            Annotated_Coords_iteration[FileName] = []
                            Annotated_Coords_iteration[FileName].append((int(Name_y_x_list[4]),
                                                                          int(Name_y_x_list[5]),Ent_Score))
                            
                        Temp_Budget += 1
                        Iteration += 1
                else:
                    Annotated_Coords[FileName] = []
                    Annotated_Coords[FileName].append((int(Name_y_x_list[4]),int(Name_y_x_list[5])))    
                    
                    Annotated_Coords_iteration[FileName] = []
                    Annotated_Coords_iteration[FileName].append((int(Name_y_x_list[4]),
                                                                          int(Name_y_x_list[5]),Ent_Score))
                    
                    Temp_Budget += 1
                    Iteration += 1
                
                
                
                
                
        print('Temp_Budget',Temp_Budget,'Counting',Counting,end='\r')
        
    #  Annotated_Coords_iteration
    print('\n Ent_Score:',Ent_Score)
    print('Annotated_Coords_iteration',len(Annotated_Coords_iteration.values()))
    
    return Annotated_Coords,Annotated_Coords_iteration,Iteration



def load_dataset_activelearning(dataset,args):
    print("\nLoading dataset...")

    print("Selected Backbone:", args.backbone)
    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

    label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), Image.NEAREST),
        ext_transforms.PILToLongTensor()
    ])


    train_set = dataset(
        args.dataset_dir,
        transform=image_transform,
        label_transform=label_transform)
    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size_AL,
        shuffle=False,
        pin_memory=True,
        # drop_last=True,
        num_workers=args.workers)
 


    val_set = dataset(
        args.dataset_dir,
        mode='val',
        transform=image_transform,
        label_transform=label_transform)
    val_loader = data.DataLoader(
        val_set,
        batch_size=args.batch_size_AL,
        shuffle=False,
        pin_memory=True,
        # drop_last=True,
        num_workers=args.workers)


    # Get encoding between pixel valus in label images and RGB colors
    class_encoding = train_set.color_encoding.copy()

    # Remove the road_marking class from the CamVid dataset as it's merged
    # with the road class
    if args.dataset.lower() == 'camvid':
        del class_encoding['road_marking']

    # Get number of classes to predict
    num_classes = len(class_encoding)

    # Print information for debuggingFs
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Required steps for each epoch: {}".format(len(train_set)//1))
    print("Validation dataset size:", len(val_set))


    # Get a batch of samples to display
    images, labels, _, _ = iter(train_loader).next()
    print("Image size:", images.size())
    print("Label size:", labels.size())
    print("\n")
    print("Class-color encoding:", class_encoding)
    
    
    return (train_loader, val_loader), class_encoding






def load_dataset_al_train(dataset,args,batchsize):
    
    image_transform = transforms.Compose(
        [transforms.Resize((args.height_train, args.width_train)),
         transforms.ToTensor()])

    label_transform = transforms.Compose([
        transforms.Resize((args.height_train, args.width_train), Image.NEAREST),
        ext_transforms.PILToLongTensor()
    ])


    train_al_set = dataset(
        args.dataset_dir,
        mode='train_al',
        transform=image_transform,
        label_transform=label_transform)
    train_al_loader = data.DataLoader(
        train_al_set,
        batch_size=batchsize,
        shuffle=True,
        # drop_last=True,
        num_workers=args.workers)#args.workers)
 
    
     
    
    class_encoding = train_al_set.color_encoding.copy()
    if args.dataset.lower() == 'camvid':
        # if 'road_marking' in class_encoding:
            del class_encoding['road_marking']
    # num_classes = len(class_encoding)


    print("Size of data in the dataloader", len(train_al_set))
    
    
    
    return (train_al_loader), class_encoding

