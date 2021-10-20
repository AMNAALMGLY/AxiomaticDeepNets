#config.py
from argparse import Namespace
import multiprocessing
import torch
import torch.nn as nn
args = Namespace(
    #experiment
    experiment='image',
    
    #model
    model = 'googlenet',
    
    # path to input and output image files 
   input_image = "/content/drive/MyDrive/AxiomiticDeepNets/Images/dog.jpg",
   output_result = "/content/drive/MyDrive/AxiomiticDeepNets/results/",

    #input sentence and embedding vector
    input_sent='I love this movie',
    embedding="glove.6B.100d",
  

    #parameters for the LSTM model 
    size_of_vocab = 2500,
    embedding_dim = 100,
    num_hidden_nodes = 256,
    num_output_nodes = 1,
    num_layers = 2,
    bidirectional = True,
    dropout = 0.2,
   

    

    # parameter for the Integrated Gradient
    n_steps = 300,

    # parameters for the dataset
    classesFile='/content/drive/MyDrive/AxiomiticDeepNets/imagenet_classes.txt',
    batch_size = 64, 
    num_workers = 8,
    
    # parameters for preprocessing dataset TEXT
    data_folder = 'data',
   
    validation_split = 0.9,
   
    

    # parameters for training TEXT
    num_epochs = 5, 
    no_of_gpus = 4,
    seed = 1234, # 4 different seeds used
    model_path = '/content/drive/MyDrive/AxiomiticDeepNets/models',
   
)

args.num_workers = multiprocessing.cpu_count()
args.no_of_gpus = torch.cuda.device_count()
    
