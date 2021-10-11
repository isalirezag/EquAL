import torch


class AL_Val:
    """Tests the ``model`` on the specified test dataset using the
    data loader, and loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to test.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    def __init__(self, model, data_loader, metric, device):
        self.model = model
        self.data_loader = data_loader

        self.metric = metric
        self.device = device


    def run_epoch(self):
        """
        Runs an epoch of validation.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float), and the values of the specified metrics

        """
        self.model.eval()
        
        self.metric.reset()
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)


            with torch.no_grad():
                # Forward propagation
                # outputs = self.model(inputs)
                
                outputs = self.model(inputs)['out']
 

            # print('fffff',labels.shape,outputs.shape,inputs.shape,labels.max())
            # Keep track of evaluation the metric
            self.metric.add(outputs.detach(), labels.detach())

            # if iteration_loss:
            #     # print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))
            #     print("[Epoch {} Step: {} out of {}] Iteration loss: {:4.4f} " .format(epochnum,step,len(self.data_loader), loss.item()))


        return   self.metric.value()
