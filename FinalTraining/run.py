#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:39:53 2020

@author: alireza
"""


import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms


import os
from PIL import Image
import numpy as np
from args import get_arguments
from collections import OrderedDict
import collections

import utils
import transforms as ext_transforms
from metric.iou import IoU
from train import Train
from test import Test



# Set random seem for reproducibility
manualSeed = 8
import random
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
torch.cuda.manual_seed(manualSeed)

# torch_seed = torch.initial_seed()
# print("torch_seed: {}" .format(torch_seed))
# print(torch.rand(5))
# print(torch.rand(5))
# print(torch.rand(5))
print('---------------------------------------------')

# Get the arguments
args = get_arguments()

device = torch.device(args.device)


def load_dataset(dataset):
	print("\nLoading dataset...\n")

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

	# Get selected dataset
	# Load the training set as tensors
	train_set = dataset(
		args.dataset_dir,
		transform=image_transform,
		label_transform=label_transform)
	train_loader = data.DataLoader(
		train_set,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.workers)

	# Load the validation set as tensors
	val_set = dataset(
		args.dataset_dir,
		mode='val',
		transform=image_transform,
		label_transform=label_transform)
	val_loader = data.DataLoader(
		val_set,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.workers)

	# Load the test set as tensors
	test_set = dataset(
		args.dataset_dir,
		mode='test',
		transform=image_transform,
		label_transform=label_transform)
	test_loader = data.DataLoader(
		test_set,
		batch_size=1, #args.batch_size,
		shuffle=False,
		num_workers=args.workers)

	# Get encoding between pixel valus in label images and RGB colors
	class_encoding = train_set.color_encoding

	# Remove the road_marking class from the CamVid dataset as it's merged
	# with the road class
	if args.dataset.lower() == 'camvid':
		del class_encoding['road_marking']

	# Get number of classes to predict
	num_classes = len(class_encoding)

	# Print information for debuggingFs
	print("Number of classes to predict:", num_classes)
	print("Train dataset size:", len(train_set))
	print("Required steps for each epoch: {}".format(len(train_set)//args.batch_size))
	print("Validation dataset size:", len(val_set))
	print("Test dataset size:", len(test_set))

	# Get a batch of samples to display
	if args.mode.lower() == 'test':
		images, labels, _, _ = iter(test_loader).next()
	else:
		images, labels, _, _ = iter(train_loader).next()
	print("Image size:", images.size())
	print("Label size:", labels.size())
	print("\n")
	print("Class-color encoding:", class_encoding)

	# Show a batch of samples and labels
	if args.imshow_batch:
		print("Close the figure window to continue...")
		label_to_rgb = transforms.Compose([
			ext_transforms.LongTensorToRGBPIL(class_encoding),
			transforms.ToTensor()
		])
		color_labels = utils.batch_transform(labels, label_to_rgb)
		utils.imshow_batch(images, color_labels)

	# Get class weights from the selected weighing technique
	print("\nComputing class weights...")
	print("this can take a while depending on the dataset size")

	
	class_weights = np.ones(num_classes)
	class_weights = torch.from_numpy(class_weights).float().to(device)
	ignore_index = list(class_encoding).index('unlabeled')
	class_weights[ignore_index] = 0
	
	
	print("Class weights:", class_weights)

	return (train_loader, val_loader,
			test_loader), class_weights, class_encoding


def train(train_loader, val_loader, class_weights, class_encoding):
	print("\nTraining...\n")

	num_classes = len(class_encoding)
	
	if args.backbone.lower() == 'fcn':
		 model = torchvision.models.segmentation.fcn_resnet50(num_classes=num_classes).to(device)
		# FCN plus resnet weights
		 model_2 =  models.resnet50(pretrained=True).to(device)

		 model_dict = model.state_dict()
		 pretrained_dict = model_2.state_dict()
		 newpretrained_dict = collections.OrderedDict()

		 for key,val in pretrained_dict.items():
			 newpretrained_dict['backbone.'+key] = val

		 # 1. filter out unnecessary keys
		 pretrained_dict = {k: v for k, v in newpretrained_dict.items() if k in model_dict}
		 # 2. overwrite entries in the existing state dict
		 model_dict.update(pretrained_dict) 
		 # 3. load the new state dict
		 model.load_state_dict(model_dict)
		 
	elif args.backbone.lower() == 'deeplab':
		 model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=num_classes).to(device)
		


	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)
		
		
	criterion = nn.CrossEntropyLoss(weight=class_weights)


	if args.optimizer.lower()=='adam':
		optimizer = optim.Adam(
			model.parameters(),
			lr=args.learning_rate,
			weight_decay=args.weight_decay) 


	# Learning rate decay scheduler
	lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs,
									 args.lr_decay)

	# Evaluation metric
	if args.ignore_unlabeled:
		ignore_index = list(class_encoding).index('unlabeled')
	else:
		ignore_index = None
	metric = IoU(num_classes, ignore_index=ignore_index)

	
	start_epoch = 0
	best_miou = 0

	# Start Training
	train = Train(model, train_loader, optimizer, criterion, metric, device,args.backbone.lower(),args.consistency)
	val = Test(model, val_loader, criterion, metric, device,args.backbone.lower())

	for epoch in range(start_epoch, args.epochs):
		
		print(">>>> [Epoch: {0:d}] Training".format(epoch))
		epoch_loss, (iou, miou) = train.run_epoch(iteration_loss=args.print_step, epochnum=epoch)
		lr_updater.step()
		print('\n')
		print(">>>> [Epoch: {0:d}] Training Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, epoch_loss, miou))

		#  show the val results every VAl_num epochs 
		VAl_num = args.save_val_every_epoch
		if (epoch + 1) % VAl_num == 0 or epoch + 1 == args.epochs:

			print(">>>> [Epoch: {0:d}] Validation".format(epoch))

			loss, (iou, miou) = val.run_epoch(iteration_loss=args.print_step, epochnum=epoch)

			print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
				  format(epoch, loss, miou))
						
			

# 			# Save arguments
# 			summary_filename_performance = os.path.join(args.save_dir, args.name + 'summary_epoch_' + str(epoch) + '.txt')
# 			with open(summary_filename_performance, 'w') as summary_file_2:
# 			   
# 				summary_file_2.write("\nVALIDATION\n")
# 				summary_file_2.write("Epoch: {0}\n". format(epoch))
# 				summary_file_2.write("Mean IoU: {0}\n". format(miou))
# 				for key, class_iou in zip(class_encoding.keys(), iou):
# 				   summary_file_2.write("{0}: {1:.4f}\n".format(key, class_iou))

# 			summary_file_2.close()
			utils.save_checkpoint_epoch(model, optimizer, epoch, best_miou,
										args)

			# Print per class IoU on last epoch or if best iou
			if epoch + 1 == args.epochs or miou > best_miou:
				for key, class_iou in zip(class_encoding.keys(), iou):
					print("{0}: {1:.4f}".format(key, class_iou))

			# Save the model if it's the best thus far
			if miou > best_miou:
				print("\nBest model thus far. Saving...\n")
				best_miou = miou
				utils.save_checkpoint(model, optimizer, epoch + 1, best_miou,
									  args)

	return model

torch.cuda.empty_cache()

def test(model, test_loader, class_weights, class_encoding):
	print("\nTesting...\n")

	num_classes = len(class_encoding)


	criterion = nn.CrossEntropyLoss(weight=class_weights)

	# Evaluation metric
	if args.ignore_unlabeled:
		ignore_index = list(class_encoding).index('unlabeled')
	else:
		ignore_index = None
	metric = IoU(num_classes, ignore_index=ignore_index)

	# Test the trained model on the test set
	test = Test(model, test_loader, criterion, metric, device,args.backbone.lower())

	print(">>>> Running test dataset")

	loss, (iou, miou) = test.run_epoch(args.print_step)
	class_iou = dict(zip(class_encoding.keys(), iou))

	print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

	# Print per class IoU
	for key, class_iou in zip(class_encoding.keys(), iou):
		print("{0}: {1:.4f}".format(key, class_iou))


	# Save arguments
	summary_filename_performance = os.path.join(args.save_dir, args.name + '_TEST_' + '.txt')
	with open(summary_filename_performance, 'w') as summary_file_2:
		summary_file_2.write("\nTEST\n")
		summary_file_2.write("Mean IoU: {0}\n". format(miou))
		for key, class_iou in zip(class_encoding.keys(), iou):
			summary_file_2.write("{0}: {1:.4f}\n".format(key, class_iou))
		summary_file_2.close()

	# Show a batch of samples and labels
	if args.imshow_batch_test:
		print("A batch of predictions from the test set...")
		images, gt_labels,_,_ = iter(test_loader).next()
		predict(model, images,gt_labels, class_encoding)
		


def predict(model, images,gt_labels, class_encoding):
	images = images.to(device)

	# Make predictions!
	model.eval()
	with torch.no_grad():
		predictions = model(images)['out']

	# Predictions is one-hot encoded with "num_classes" channels.
	# Convert it to a single int using the indices where the maximum (1) occurs
	_, predictions = torch.max(predictions.data, 1)

	label_to_rgb = transforms.Compose([
		ext_transforms.LongTensorToRGBPIL(class_encoding),
		transforms.ToTensor()
	])
	color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
	color_gt_labels = utils.batch_transform(gt_labels.cpu(), label_to_rgb)
	# utils.imshow_batch(images.data.cpu(), color_predictions)
	utils.imshow_batch_2(images.data.cpu(), color_predictions,color_gt_labels)



if __name__ == '__main__':
	
	import os
	
	os.environ['CUDA_VISIBLE_DEVICES']=args.gpunum
	print('Using GPU {}'.format(args.gpunum))

	
	# Fail fast if the dataset directory doesn't exist
	assert os.path.isdir(
		args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(
		args.dataset_dir)
			
	
			
	# Make the save directory
	if not os.path.isdir(args.save_dir):
		os.mkdir(args.save_dir)
	# Fail fast if the saving directory doesn't exist
	assert os.path.isdir(
		args.save_dir), "The directory \"{0}\" doesn't exist.".format(
			args.save_dir)
			
	# Import the requested dataset
	if args.dataset.lower() == 'camvid':
		from dataloaders import CamVid as dataset
	elif args.dataset.lower() == 'cityscapes':
		from dataloaders import Cityscapes as dataset
		
	else:
		# Should never happen...but just in case it does
		raise RuntimeError("\"{0}\" is not a supported dataset.".format(
			args.dataset))
	
	# Loader cotains the train, val, test, files, their size changes based on batch_size
	loaders, w_class, class_encoding = load_dataset(dataset)
	
	train_loader, val_loader, test_loader = loaders
			
	
	

	print('args.mode.lower() is --> {0}' .format(args.mode.lower()))
	
	if args.mode.lower() in {'train', 'full'}:
		model = train(train_loader, val_loader, w_class, class_encoding)

	if args.mode.lower() in {'test', 'full'}:
		if args.mode.lower() == 'test':

			num_classes = len(class_encoding)
			
			if args.backbone.lower() == 'fcn':
				print('fcn model loaded......')
				model = torchvision.models.segmentation.fcn_resnet50(num_classes=num_classes).to(device)

			if args.backbone.lower() == 'deeplab':
				model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=num_classes).to(device)

		# Initialize a optimizer just so we can retrieve the model from the
		# checkpoint
		optimizer = optim.Adam(model.parameters())

		# Load the previoulsy saved model state 
		model = utils.load_checkpoint(model, optimizer, args.save_dir,
									  args.name,args)[0]

		if args.mode.lower() == 'test':
			print('args.mode.lower() is --> test')
			# print(model)

		test(model, test_loader, w_class, class_encoding)

	print(1)
