
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
def visualize(baseline, image,grads, IG,cat,fname,cmap='PiYG',alpha=0.4):
  cmap_bound=np.abs(IG).max()
  img=transforms.ToPILImage()(IG)
  grads=transforms.ToPILImage()(grads)
  fig,axes= plt.subplots(nrows=2, ncols=3, figsize=(13, 13))
  plt.subplot(231)
  plt.title(f'original_image -predicted as {cat}')
  plt.imshow(image)
  plt.subplot(232)
  plt.title('baseline_image')
  plt.imshow(transforms.ToPILImage()(baseline))
  plt.subplot(233)
  plt.title('Attribution')
  plt.imshow(transforms.Grayscale()(img),vmin=-cmap_bound, vmax=cmap_bound, cmap=cmap) 
  plt.subplot(234)
  plt.title('overLay')
  plt.imshow(transforms.Grayscale()(img),vmin=-cmap_bound, vmax=cmap_bound, cmap=cmap) 
  plt.imshow(image,alpha=alpha)
  plt.subplot(235)
  plt.title('gradiant')
  plt.imshow(transforms.Grayscale()(grads),vmin=-cmap_bound, vmax=cmap_bound, cmap=cmap)
  fig.delaxes(axes[1][2])
  plt.savefig(fname)
  plt.tight_layout()
  return fig



def  hlstr(string, color='white'):
    """
    Return HTML markup highlighting text with the desired color.
    """
    return f"<mark style=background-color:{color}>{string} </mark>"
def colorize(attrs, cmap='PiYG'):
    """
    Compute hex colors based on the attributions for a single instance.
    Uses a diverging colorscale by default and normalizes and scales
    the colormap so that colors are consistent with the attributions.
    """
 
    cmap_bound = np.abs(attrs).max()
    norm = mpl.colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
    cmap = mpl.cm.get_cmap(cmap)

    # now compute hex values of colors
    colors = list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))
    return colors

def visualizeText(IG,tokens):
  colors = colorize(IG.detach().cpu().numpy())
  HTML("".join(list(map(hlstr, tokens, colors))))

