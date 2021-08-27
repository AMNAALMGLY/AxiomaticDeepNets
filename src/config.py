from argparse import Namespace
import multiprocessing
import torch

args = Namespace(
    
    model = 'googlenet',
    
    # path to input and output image files 
   input_image = "/content/drive/MyDrive/AxiomiticDeepNets/Images/dog.jpg",
   output_result = "/content/drive/MyDrive/AxiomiticDeepNets/results/",

    
  
    #parameters for the LSTM model 
    num_output = 1, # {1 for binary, > 1 for multiclass}
    num_layers = 6,
    num_heads = 8,
    dropout = 0.1,
    dropout_ff = 0.1, # dropout for feedforward layers, 0.1 used for saint_s and 0.8 for saint and saint_i variants
    embed_dim = 32,
    d_ff = 32,
    

    # parameter for the Integrated Gradient
    n_steps = 300,

    # parameters for the dataset
    classesFile='/content/drive/MyDrive/AxiomiticDeepNets/imagenet_classes.txt',
#batch_size = 32, # [32, 256]
    num_workers = 8,
    
    # parameters for preprocessing dataset TEXT
    data_folder = 'data',
    train_split = 0.65,
    validation_split = 0.15,
    test_split = 0.20,
    

    # parameters for training TEXT

    num_epochs = 1, # default is 100
    no_of_gpus = 4,
    seed = 1234, # 4 different seeds used
    resume_checkpoint = None,
   
)

args.num_workers = multiprocessing.cpu_count()
args.no_of_gpus = torch.cuda.device_count()