from __future__ import print_function, division
import os
import urllib
import torch
from PIL import Image
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def get_model(modelname='googlenet'):
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

    model= torch.hub.load('pytorch/vision:v0.10.0', modelname, pretrained=True)
    return model
