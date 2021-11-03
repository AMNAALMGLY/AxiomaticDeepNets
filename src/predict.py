from __future__ import print_function, division
import os
import urllib
import torch
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(input, model,classesFile='imagenet_classes.txt',transform=True):
    '''
    input is an image
    '''
    if transform:
       preprocess = transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])
       input_tensor2 = preprocess(input)
    else:
      input_tensor2=input

   
    input_batch2 = input_tensor2.unsqueeze(0)
    if torch.cuda.is_available():
      input_batch2 = input_batch2.to('cuda')
      model.to('cuda')
    with torch.no_grad():
        model.eval()
        output=model(input_batch2)
    prob=torch.nn.functional.softmax(output,dim=1)
    # Read the categories
    with open(classesFile, "r") as f:
       categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    probmax,idx=prob.topk(k=1,dim=1)


    return idx, categories[idx] , probmax


def predictSentiment(onehot,model,textField):   
    model.freeze=False
    onehot=onehot.to(device)
    onehot.unsqueeze_(1)
    prediction = torch.sigmoid(model(onehot, lengthTensor)) > 0.5
    if prediction:
         return prediction ,'positive' 
    else:
          return prediction, 'negative'
