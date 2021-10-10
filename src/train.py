#Train.py

def trainText(model, iterator, optimizer, criterion):
    
    #initialize every epoch 
    epoch_loss = 0
    epoch_acc = 0
    
    #set the model in training phase
    model.train()  
    
    for batch in iterator:
        
        #resets the gradients after every batch
        optimizer.zero_grad()   
        
        #retrieve text and no. of words
        text,  text_lengths= batch.text   
        
        #print(text.shape, text_lengths.shape)
        
        #convert to 1D tensor
        predictions = model(text, text_lengths).squeeze(1)  
        #print(predictions)
        
        #compute the loss
        loss = criterion(predictions, batch.label)        
       
        #compute the binary accuracy
        acc =accuracy(predictions, batch.label)   
        
        #backpropage the loss and compute the gradients
        loss.backward()       
        
        #update the weights
        optimizer.step()      
        
        #loss and accuracy
        epoch_loss += loss.item()  
        epoch_acc += acc.item()    
        
    return epoch_loss / len(iterator),epoch_acc / len(iterator)
    
def evaluateText(model, iterator, criterion):
      epoch_loss=0
      epoch_acc=0
      model.eval()
      with torch.no_grad():
        for batch in iterator:
          text, text_lengths=batch.text
          predictions=model(text, text_lengths).squeeze()  
          #compute the loss
          loss = criterion(predictions, batch.label)        
            
            #compute the binary accuracy
          acc =accuracy(predictions, batch.label) 
          #loss and accuracy
          epoch_loss += loss.item()  
          epoch_acc += acc.item()    
            
      return epoch_loss / len(iterator),   epoch_acc / len(iterator)

def Trainer(n_epochs,model,model_path, trainLoader, validLoader, optimizor,criterion ,TEXT):
    best_loss=float('inf')
    model_init(TEXT,model)
    for epoch in range(n_epochs):
      loss_train, acc_train=trainText(model, trainLoader, optimizor,criterion)
      print(f'train_acc {acc_train}')
      loss_valid, acc_valid=evaluateText(model, validLoader,criterion)
      print(f'valid_acc {acc_valid}')
      if loss_valid <best_loss:
        best_loss=loss_valid
        torch.save(model.state_dict(),f'{model_path}/SentimentModel.pt')
      return f'{model_path}/SentimentModel.pt'
      

def Tester(model,model_path,testLoader,criterion):
    model.load_state_dict(torch.load(f'{model_path}/SentimentModel.pt'))
    loss_test ,acc_test=evaluate(model, testLoader,criterion)

    print(f'test_acc {acc_test}')
    return loss_test ,acc_tes

def accuracy(preds,true):
  preds=torch.round(torch.sigmoid(preds))
  return (preds==true).sum()/len(true)

def model_init(TEXT,model):
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(embedding_dim)
    model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)
    
            