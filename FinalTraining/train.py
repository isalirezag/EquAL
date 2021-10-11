import time
import random
import torch

class Train:
	"""Performs the training of ``model`` given a training dataset data
	loader, the optimizer, and the loss criterion.

	Keyword arguments:
	- model (``nn.Module``): the model instance to train.
	- data_loader (``Dataloader``): Provides single or multi-process
	iterators over the dataset.
	- optim (``Optimizer``): The optimization algorithm.
	- criterion (``Optimizer``): The loss criterion.
	- metric (```Metric``): An instance specifying the metric to return.
	- device (``torch.device``): An object representing the device on which
	tensors are allocated.
    modified from     https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.

	"""

	def __init__(self, model, data_loader, optim, criterion, metric, device,modeltype,consistency):
		self.model = model
		self.data_loader = data_loader
		self.optim = optim
		self.criterion = criterion
		self.criterion_MSE = torch.nn.MSELoss()
		self.metric = metric
		self.device = device
		self.modeltype = modeltype
		self.consistency = consistency

	def run_epoch(self, iteration_loss=False, epochnum=0):
		"""Runs an epoch of training.

		Keyword arguments:
		- iteration_loss (``bool``, optional): Prints loss at every step.

		Returns:
		- The epoch loss (float).

		"""
		self.model.train()
		
		time_sum = 0
		epoch_loss = 0.0
		self.metric.reset()
		for step, batch_data in enumerate(self.data_loader):
			
			
			
			
			start_time = time.time()

			# Get the inputs and labels
			inputs = batch_data[0].to(self.device)
			labels = batch_data[1].to(self.device)
			
			# augmentation
			if  random.randint(0, 1):
				inputs = torch.flip(inputs, [3])
				labels = torch.flip(labels, [2])

			# Forward propagation
			if not self.consistency:
				outputs = self.model(inputs)['out']
				loss = self.criterion(outputs, labels) 
			if self.consistency:
				outputs = self.model(inputs)['out']
				outputs_al_1_2 = torch.flip(self.model(torch.flip(inputs,[3]))['out'],[3])
				
				outputs_al_1_1_dt = outputs.detach().clone()
				outputs_al_1_2_dt = outputs_al_1_2.detach().clone()
				
				loss = self.criterion(outputs, labels) + self.criterion(outputs_al_1_2, labels) + \
					self.criterion_MSE(outputs, outputs_al_1_2_dt) + \
						self.criterion_MSE(outputs_al_1_2, outputs_al_1_1_dt) 

			# Backpropagation
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()
			
			end_time = time.time() 
			
			time_sum += end_time-start_time

			# Keep track of loss for current epoch
			epoch_loss += loss.item()

			# Keep track of the evaluation metric
			self.metric.add(outputs.detach(), labels.detach())


			if iteration_loss and step%10==0:
				
				print("[Epoch {} Step: {} out of {}] Iteration loss: {:4.4f} | time {:4.4f}" .format(epochnum,step,len(self.data_loader), loss.item(),time_sum ),end='\r')
				time_sum = 0

		return epoch_loss / len(self.data_loader), self.metric.value()
