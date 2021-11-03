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

def ImageToTensor(filename,preprocess=True):

    input_image = Image.open(filename)
    if preprocess:
        input_image2=transforms.Resize((224,224))(input_image)
    input_tensor=transforms.ToTensor()(input_image2)
    return input_tensor,input_image