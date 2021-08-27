
import torch
import argparse

from src.config import args 
from models.model_generator import get_model
from src.explain import explain
from src.predict import predict
from utils.utils import ImageToTensor
from utils.visualize import visualize

def main():
  input_tensor,input_image=ImageToTensor(args.input_image)
  baseline=torch.zeros_like(input_tensor)
  n_steps=args.n_steps
  model=get_model()
  target_idx,cat,score=predict(input_image, model,classesFile=args.classesFile)
  IG,grads=explain(input_image,model,n_steps, target_idx,baseline,preprocess=True)
  visualize(baseline, input_image,grads, IG,cat,fname=f"{args.output_result} {cat}.jpg",cmap='PiYG',alpha=0.4)

if __name__ == "__main__":
  main()