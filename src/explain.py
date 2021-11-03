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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def interpolate(baseline, input, steps, plot=False):
    assert input.shape[1] == baseline.shape[1]
    interpolates = torch.empty((steps, *input.shape))
    plt.figure(figsize=(10, 10))
    for idx in range(steps):
        alpha = idx / steps
        interpolated = baseline + (alpha * (input - baseline))
        if plot:
            plt.subplot(int(steps / 2), steps - int(steps / 2), idx + 1)

            plt.imshow(transforms.ToPILImage()(interpolated))

        interpolates[idx, ...] = interpolated

    return interpolates


def computeGradiant(input, model, target_idx=None, isText=False):
    gradient = torch.empty_like(input)
    input.requires_grad = True

    model.to(device)

    model.zero_grad()

    input_batch = input.unsqueeze(0)

    if isText:
        model.freeze = True
        input_batch = input_batch.permute(1, 0, 2)

        output = model(input_batch.to(device), text_lengths=lengthTensor)

        output = torch.sigmoid(output)

    else:

        output = model(input_batch.to(device))
        # find target if none is provided
        prob = torch.nn.functional.softmax(output, dim=1)

        if target_idx:
            outputs = output[:, target_idx].squeeze(0)
        else:
            outputs = output

    gradient = torch.autograd.grad(outputs=outputs, inputs=input_batch, )[0]
    if isText:
        gradient = gradient.permute(1, 0, 2)
    return gradient.squeeze_(0)


# class model , n_steps , internal_batch_size , method
# Methods : explain return attributions , parameters X , baseline , target
def generate_IG(input, baseline, model, n_steps, target_idx, isText):
    norm = input - baseline
    interpol = interpolate(baseline, input, n_steps)
    gradient = torch.empty(*interpol.shape)
    for idx, i in enumerate(interpol):
        gradient[idx, ...] = computeGradiant(i, model, target_idx)
    IG = torch.mean(gradient[:-1], dim=0) * norm
    return IG, gradient[-1]


def explain(input_image, model, n_steps, target_idx, isText, baseline=None, preprocess=True):
    if preprocess:
        input_image = transforms.Resize((224, 224))(input_image)
    input_tensor = transforms.ToTensor()(input_image)
    if not baseline == None:
        baseline = torch.zeros(*input_tensor.shape)

    IG, grads = generate_IG(input_tensor, baseline, model, n_steps=n_steps, target_idx=target_idx, isText=isText)
    return IG, grads
