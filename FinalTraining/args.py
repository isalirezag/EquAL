from argparse import ArgumentParser


def get_arguments():
    """
    Defines command-line arguments, and parses them.

    """
    parser = ArgumentParser()

    # Execution mode
    parser.add_argument(
        "--mode",
        "-m",
        choices=['train', 'test', 'full'],
        default='train',
        help=("train: performs training and validation; \
              test: tests the model "
              "found in \"--save_dir\" with name \"--name\" on \"--dataset\"; "
              "full: combines train and test modes. Default: train"))
        
    parser.add_argument(
        "--consistency",
        type=int,
        default=0,
        help="Use the self consistency or not")
        

    # Hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="The batch size. 5 for camvid and 4 for cityscapes")
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs. Default: 300")

    parser.add_argument(
        "--save_val_every_epoch",
        type=int,
        default=10,
        help="Save the results every {} epoch. Default: 10")


    parser.add_argument(
        "--optimizer",
        type=str,
        default='adam',
        help="optmizer type. [adam]")

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="The learning rate.")
    
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.1,
        help="The learning rate decay factor. Default: 0.1")
    
    parser.add_argument(
        "--lr_decay_epochs",
        type=int,
        default=20,
        help="The number of epochs before adjusting the learning rate. "
        "Default: 100")

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=2e-4,
        help="weight decay. Default: 2e-4")

    # Dataset
    parser.add_argument(
        "--dataset",
        choices=['camvid', 'cityscapes'],
        default='camvid',
        help="Dataset to use. Default: camvid")
    

    
    parser.add_argument(
        "--height",
        type=int,
        default=360,
        help="The image height. Default: 360")
    
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="The image width. Default: 480")
    
    
    
    parser.add_argument(
        "--backbone",
        choices=['fcn', 'deeplab'],
        default='fcn',
        help="The semantic segmentation model that is used. "
        "Default: fcn")
    
    parser.add_argument(
        "--with_unlabeled",
        dest='ignore_unlabeled',
        action='store_false',
        help="if use this then the unlabeled class is not ignored.")

    # Settings
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. Default: 4")
    
    parser.add_argument(
        "--print_step",
        action='store_true',
        help="Print loss every step")
    
    parser.add_argument(
        "--imshow_batch",
        action='store_true',
        help=("Displays batch images when loading the dataset and making "
              "predictions."))
    
    parser.add_argument(
        "--imshow_batch_test",
        action='store_true',
        help=("Displays batch images when loading the dataset and making "
              "predictions."))
    
    parser.add_argument(
        "--device",
        default='cuda',
        help="Device on which the network will be trained. Default: cuda")
    
    parser.add_argument(
        "--gpunum",
        default='0',
        help="GPU number to use. Default: 1")

    # Storage settings
    parser.add_argument(
        '--name',
        type=str,
        default='model_camvid',
        help='Name given to the model when saving.')
    
    
    
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/media/alireza/SmithHall210/camvid/",
        help="Path to the root directory of the selected dataset. "
        "Default: data/CamVid")
    
    parser.add_argument(
        "--save_dir",
        type=str,
        default='/media/Code/Save/',
        help="The directory where models are saved. Default: save")
    
    parser.add_argument(
        "--checkpoint_test",
        type=str,
        default='none',
        help="checkpoint that you like to test")

    return parser.parse_args()

if __name__ == '__main__':
    # Get the arguments
    args = get_arguments()

    print(args)

