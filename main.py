
import torch
import argparse
import torch.nn as nn
from src.config import args 
from models.model_generator import get_model
from src.explain import explain
from src.predict import predict
from utils.utils import ImageToTensor
from utils.visualize import visualize
from src.dataloader import generate_IMDB_loaders
from src.train import Trainer
#main.py

from src.dataloader import preprocess
torch.backends.cudnn.enabled = False

def main():
  experiment=args.experiment
  if experiment=='image':

    #Image Experiment
    input_tensor,input_image=ImageToTensor(args.input_image)
    baseline=torch.zeros_like(input_tensor)
    n_steps=args.n_steps
    model=get_model()
    target_idx,cat,score=predict(input_image, model,classesFile=args.classesFile)
    IG,grads=explain(input_image,model,n_steps, target_idx,isText=False,baseline=baseline,preprocess=True)
    visualize(baseline, input_image,grads, IG,cat,fname=f"{args.output_result} {cat}.jpg",cmap='PiYG',alpha=0.4)
    print('finished')
  elif experiment=='text':
    print('here')
    #Text Experiment
    trainLoader,validLoader,testLoader,TEXT,_ =generate_IMDB_loaders(args.embedding,args.size_of_vocab,args.validation_split,args.batch_size)

    model=get_model(modelname='Textclassifier',size_of_vocab=args.size_of_vocab, embedding_dim=args.embedding_dim, num_hidden_nodes=args.num_hidden_nodes,num_output_nodes=args.num_output_nodes, num_layers=args.num_layers, 
                    bidirectional = args.bidirectional, dropout= args.dropout ,pad_idx=TEXT.vocab.stoi[TEXT.pad_token])
    model.to(device)
    best_checkpoint= Trainer(args.n_epochs,model,args.model_path, trainLoader, validLoader, optimizor= torch.optim.Adam(),
      criterion=nn.BCEWithLogitsLoss() ,TEXT=TEXT)
    pretrained_model=model.load_state_dict(torch.load(best_checkpoint))
    
    onehot , tokens =preprocess(args.input_sent)
    embedded=model.embedding(onehot.to(device))
    baseline=torch.zeros_like(embedded)
    IG,grads=explain(embedded,model,n_steps=args.n_steps, target_idx=None,isText=True,baseline=baseline,preprocess=False)
    visualizeText(torch.sum(IG,dim=1),tokens)
if __name__ == "__main__":
  main()
